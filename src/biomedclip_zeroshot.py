"""
Zero-Shot Classification with BiomedCLIP on HAM10000 (7-class).

Uses the same lesion-level stratified test split (seed=42) as all other
experiments so results are directly comparable with the linear-probe baseline
(Macro-F1: 0.420, MEL Recall: 0.509).

Two strategies evaluated:
  1. Averaged prompts  — encode all 5 prompts per class, L2-norm each,
                         average, L2-norm again → one class embedding;
                         predict argmax of cosine similarity.
  2. Max-sim ensemble  — keep prompts separate; for each class take the max
                         cosine similarity across the 5 prompts; predict argmax.

Run from project root:
    python src/biomedclip_zeroshot.py \\
        --metadata-csv data/HAM10000_metadata.csv \\
        --images-dir data \\
        --output-dir results
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import Config
from dataset import SkinLesionDataset, lesion_level_split
from helpers import get_device, set_seed

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPTS_PER_CLASS: dict[str, list[str]] = {
    "mel": [
        "This is a melanoma",
        "This is a malignant melanoma",
        "This is a skin cancer called melanoma",
        "A dermoscopy image of melanoma",
        "This is a cancerous pigmented skin lesion",
    ],
    "nv": [
        "This is a nevus",
        "This is a benign mole",
        "This is a melanocytic nevus",
        "A dermoscopy image of a common mole",
        "This is a benign pigmented skin lesion",
    ],
    "bkl": [
        "This is a benign keratosis",
        "This is a seborrheic keratosis",
        "This is a solar lentigo",
        "A dermoscopy image of benign keratosis",
        "This is a benign keratosis-like lesion",
    ],
    "bcc": [
        "This is a basal cell carcinoma",
        "This is a basal cell cancer",
        "This is a skin cancer called basal cell carcinoma",
        "A dermoscopy image of basal cell carcinoma",
        "This is a non-melanoma skin cancer",
    ],
    "akiec": [
        "This is an actinic keratosis",
        "This is a solar keratosis",
        "This is a precancerous skin lesion",
        "A dermoscopy image of actinic keratosis",
        "This is an intraepithelial carcinoma",
    ],
    "vasc": [
        "This is a vascular lesion",
        "This is an angioma",
        "This is a vascular skin growth",
        "A dermoscopy image of a vascular lesion",
        "This is a benign vascular skin lesion",
    ],
    "df": [
        "This is a dermatofibroma",
        "This is a benign skin fibroma",
        "This is a dermal fibroma",
        "A dermoscopy image of dermatofibroma",
        "This is a fibrous skin lesion",
    ],
}

# HAM10000 sorted label map (same as all other scripts)
LABEL_MAP: dict[str, int] = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}
IDX_TO_CLASS: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}
CLASS_ORDER: list[str] = [IDX_TO_CLASS[i] for i in range(len(LABEL_MAP))]
MEL_IDX: int = LABEL_MAP["mel"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_multiclass_table(
    csv_path: Path,
    col_image_id: str,
    col_lesion_id: str,
    col_diagnosis: str,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {col_image_id, col_lesion_id, col_diagnosis}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = df.copy()
    df["_binary_label"] = df[col_diagnosis].astype(str).map(LABEL_MAP).astype(int)
    return df


# ---------------------------------------------------------------------------
# Text embedding helpers
# ---------------------------------------------------------------------------

def build_averaged_text_embeddings(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """
    For each class: encode all prompts → L2-norm each → average → L2-norm.
    Returns shape (num_classes, embed_dim).
    """
    class_embeddings: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for cls_name in CLASS_ORDER:
            prompts = PROMPTS_PER_CLASS[cls_name]
            tokens = tokenizer(prompts).to(device)          # (P, seq_len)
            text_feats = model.encode_text(tokens)          # (P, D)
            text_feats = F.normalize(text_feats, dim=-1)    # L2-norm each prompt
            avg = text_feats.mean(dim=0)                    # (D,)
            avg = F.normalize(avg, dim=-1)                  # L2-norm average
            class_embeddings.append(avg)
    return torch.stack(class_embeddings, dim=0)             # (C, D)


def build_per_prompt_text_embeddings(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """
    For each class: encode all prompts → L2-norm each.
    Returns shape (num_classes, num_prompts, embed_dim).
    """
    all_embeddings: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for cls_name in CLASS_ORDER:
            prompts = PROMPTS_PER_CLASS[cls_name]
            tokens = tokenizer(prompts).to(device)
            text_feats = model.encode_text(tokens)          # (P, D)
            text_feats = F.normalize(text_feats, dim=-1)
            all_embeddings.append(text_feats)
    return torch.stack(all_embeddings, dim=0)               # (C, P, D)


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_test_images(
    model: torch.nn.Module,
    df_test: pd.DataFrame,
    images_dir: Path,
    col_image_id: str,
    preprocess,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (image_features, true_labels); features are L2-normalised."""
    ds = SkinLesionDataset(df_test, images_dir, col_image_id, transform=preprocess)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    all_feats: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Encoding test images"):
            images = images.to(device)
            feats = model.encode_image(images)              # (B, D)
            feats = F.normalize(feats, dim=-1)
            all_feats.append(feats.cpu())
            all_labels.append(labels)
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


# ---------------------------------------------------------------------------
# Prediction strategies
# ---------------------------------------------------------------------------

def predict_averaged(
    image_feats: torch.Tensor,       # (N, D)  L2-normalised
    text_embeddings: torch.Tensor,   # (C, D)  L2-normalised
) -> np.ndarray:
    """Cosine sim with averaged class embeddings → argmax."""
    sims = image_feats @ text_embeddings.T          # (N, C)
    return sims.argmax(dim=-1).numpy()


def predict_max_sim(
    image_feats: torch.Tensor,          # (N, D)
    per_prompt_embeddings: torch.Tensor,  # (C, P, D)
) -> np.ndarray:
    """
    For each image and each class, take the max cosine similarity across prompts,
    then predict the argmax class.
    """
    num_classes = per_prompt_embeddings.shape[0]
    # sims_per_prompt: (N, C, P)
    sims = torch.einsum("nd,cpd->ncp", image_feats, per_prompt_embeddings)
    max_sims = sims.max(dim=-1).values      # (N, C)  max over prompts
    return max_sims.argmax(dim=-1).numpy()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strategy_name: str,
    linear_probe_macro_f1: float = 0.420,
    linear_probe_mel_recall: float = 0.509,
) -> str:
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mel_recall = recall_score(y_true, y_pred, labels=[MEL_IDX], average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_ORDER))))
    cls_report = classification_report(
        y_true, y_pred,
        labels=list(range(len(CLASS_ORDER))),
        target_names=CLASS_ORDER,
        zero_division=0,
    )

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"  Strategy: {strategy_name}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Confusion Matrix (rows = true class, cols = predicted class):")
    lines.append("  " + " ".join(f"{c:>6}" for c in CLASS_ORDER))
    for i, row in enumerate(cm):
        lines.append(f"{CLASS_ORDER[i]:>6}  " + " ".join(f"{v:>6d}" for v in row))
    lines.append("")
    lines.append("Per-class metrics:")
    lines.append(cls_report)
    lines.append(f"  Macro-F1      : {macro_f1:.4f}")
    lines.append(f"  MEL Recall    : {mel_recall:.4f}")
    lines.append("")
    lines.append("Comparison with BiomedCLIP linear probe:")
    lines.append(f"  Linear probe  Macro-F1 = {linear_probe_macro_f1:.3f}   "
                 f"MEL Recall = {linear_probe_mel_recall:.3f}")
    lines.append(f"  Zero-shot     Macro-F1 = {macro_f1:.3f}   "
                 f"MEL Recall = {mel_recall:.3f}")
    delta_f1 = macro_f1 - linear_probe_macro_f1
    delta_rec = mel_recall - linear_probe_mel_recall
    lines.append(f"  Delta         Macro-F1 = {delta_f1:+.3f}   "
                 f"MEL Recall = {delta_rec:+.3f}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BiomedCLIP zero-shot classification on HAM10000"
    )
    p.add_argument("--metadata-csv", type=Path, default=None)
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config()
    if args.metadata_csv is not None:
        cfg.metadata_csv = args.metadata_csv
    if args.images_dir is not None:
        cfg.images_dir = args.images_dir
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.seed is not None:
        cfg.seed = args.seed

    set_seed(cfg.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "biomedclip_zeroshot_results.txt"

    # --- Metadata & test split -----------------------------------------------
    print("Loading metadata …")
    df = load_multiclass_table(
        cfg.metadata_csv,
        cfg.col_image_id,
        cfg.col_lesion_id,
        cfg.col_diagnosis,
    )
    print(f"Label map: {LABEL_MAP}")

    _, _, df_test = lesion_level_split(
        df,
        cfg.col_lesion_id,
        cfg.train_ratio,
        cfg.val_ratio,
        cfg.test_ratio,
        cfg.seed,
    )
    print(f"Test set size: {len(df_test)} images")

    # --- Load BiomedCLIP ------------------------------------------------------
    device = get_device()
    print(f"Loading BiomedCLIP ViT-B/16 on {device} …")
    try:
        from open_clip import create_model_from_pretrained, get_tokenizer
    except ImportError:
        raise SystemExit(
            "open_clip_torch is not installed.\n"
            "Run:  pip install open_clip_torch"
        )

    model, preprocess = create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    tokenizer = get_tokenizer(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model = model.to(device)
    model.eval()

    # --- Encode text prompts --------------------------------------------------
    print("\nEncoding text prompts …")
    avg_text_emb = build_averaged_text_embeddings(model, tokenizer, device)   # (C, D)
    pp_text_emb = build_per_prompt_text_embeddings(model, tokenizer, device)  # (C, P, D)
    print(f"  Averaged embeddings : {tuple(avg_text_emb.shape)}")
    print(f"  Per-prompt embeddings: {tuple(pp_text_emb.shape)}")

    # --- Encode test images ---------------------------------------------------
    print("\nEncoding test images …")
    image_feats, y_true_tensor = encode_test_images(
        model,
        df_test,
        cfg.images_dir,
        cfg.col_image_id,
        preprocess,
        args.batch_size,
        cfg.num_workers,
        device,
    )
    y_true = y_true_tensor.numpy()
    print(f"  Image features: {tuple(image_feats.shape)}")

    # --- Predictions ----------------------------------------------------------
    print("\nRunning zero-shot predictions …")
    # Move text embeddings to CPU for matmul (image_feats already on CPU)
    avg_text_emb_cpu = avg_text_emb.cpu()
    pp_text_emb_cpu = pp_text_emb.cpu()

    y_pred_avg = predict_averaged(image_feats, avg_text_emb_cpu)
    y_pred_max = predict_max_sim(image_feats, pp_text_emb_cpu)

    # --- Build reports --------------------------------------------------------
    report_avg = build_report(y_true, y_pred_avg, "Averaged Prompts (per-class mean embedding)")
    report_max = build_report(y_true, y_pred_max, "Max-Sim Ensemble (per-prompt max similarity)")

    header = textwrap.dedent(f"""\
        BiomedCLIP Zero-Shot Classification — HAM10000 (7-class)
        =========================================================
        Model  : hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
        Split  : test set, lesion-level stratified split (seed={cfg.seed})
        Images : {len(df_test)}
        Classes: {CLASS_ORDER}
        Prompts: {len(next(iter(PROMPTS_PER_CLASS.values())))} per class

        Reference (BiomedCLIP linear probe, same test split):
          Macro-F1   = 0.420
          MEL Recall = 0.509

    """)

    full_report = header + report_avg + "\n" + report_max

    print("\n" + full_report)

    out_file.write_text(full_report, encoding="utf-8")
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
