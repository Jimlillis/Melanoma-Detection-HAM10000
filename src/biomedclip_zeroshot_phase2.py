"""
BiomedCLIP Zero-Shot Classification — Phase 2 (clinically-informed prompts).

Uses the EXACT same model, split, and evaluation logic as Phase 1
(src/biomedclip_zeroshot.py).  The only change is PROMPTS_PER_CLASS, which
now contains detailed dermoscopic descriptions generated from representative
HAM10000 images via MedGPT.

Phase 1 baseline (for comparison printed at the end):
  Averaged strategy: Macro-F1=0.201, MEL Recall=0.401
  Max-Sim strategy:  Macro-F1=0.157, MEL Recall=0.443

Run from project root:
    python src/biomedclip_zeroshot_phase2.py \\
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
# Phase 1 results (hard-coded for side-by-side comparison)
# ---------------------------------------------------------------------------

PHASE1_AVERAGED_MACRO_F1: float = 0.201
PHASE1_AVERAGED_MEL_RECALL: float = 0.401
PHASE1_MAXSIM_MACRO_F1: float = 0.157
PHASE1_MAXSIM_MEL_RECALL: float = 0.443

# ---------------------------------------------------------------------------
# Phase 2 prompts — clinically-informed dermoscopic descriptions
# ---------------------------------------------------------------------------

PROMPTS_PER_CLASS: dict[str, list[str]] = {
    "mel": [
        "A dermoscopy image of an asymmetric melanoma with irregular borders, multicomponent pattern, and marked color variegation including brown, black, red, and white",
        "A dermoscopy image of melanoma showing chaotic architecture, atypical pigment network, eccentric black blotches, and regression structures",
        "A dermoscopy image of a pigmented melanoma with irregular structureless areas, dark clods, whitish scar-like depigmentation, and gray peppering",
        "A dermoscopy image of melanoma with asymmetrically distributed colors, ill-defined margins, hemorrhagic red area, and black structureless pigmentation",
        "A dermoscopy image of cutaneous melanoma demonstrating multicolor asymmetry, irregular blotches, focal regression, and a disorganized dermoscopic pattern",
    ],
    "nv": [
        "A dermoscopy image of a small oval brown melanocytic nevus with a darker central structureless blotch and softer peripheral pigmentation",
        "A dermoscopy image of a benign-appearing melanocytic nevus showing a light-to-medium brown reticuloglobular pattern with mild central hyperpigmentation",
        "A dermoscopy image of an oval tan-brown nevus with delicate pigment network, subtle globular areas, and overall limited color variation",
        "A dermoscopy image of a melanocytic nevus with symmetric brown pigmentation, a faint network, and a central darker blotch on pink skin",
        "A dermoscopy image of a common melanocytic nevus demonstrating uniform brown shades, soft borders, and a reticular to structureless central pattern",
    ],
    "bkl": [
        "A dermoscopy image of a well-circumscribed light-to-dark brown seborrheic keratosis with milia-like cysts and comedo-like openings",
        "A dermoscopy image of a keratotic tan-brown plaque showing cerebriform fissures and ridges on a pink background",
        "A dermoscopy image of a sharply demarcated benign keratosis with moth-eaten borders, surface scale, and multiple keratin plugs",
        "A dermoscopy image of a lobulated brown seborrheic keratosis with structureless pigmentation and whitish-yellow clods",
        "A dermoscopy image of a papillomatous benign keratosis displaying fissures, milia-like cysts, and absent pigment network",
    ],
    "bcc": [
        "A dermoscopy image of basal cell carcinoma with pink translucent stroma, arborizing telangiectasia, and blue-gray ovoid nests",
        "A dermoscopy image of pigmented basal cell carcinoma showing a structureless pink background, brown-gray globules, and focal ulceration",
        "A dermoscopy image of basal cell carcinoma with fine branching vessels, shiny white blotches, and blue-gray pigmented clods",
        "A dermoscopy image of basal cell carcinoma lacking a pigment network and containing arborizing vessels with gray-brown ovoid foci",
        "A dermoscopy image of pigmented basal cell carcinoma with asymmetric pink-white areas, microarborizing telangiectasia, and focal erosive crusts",
    ],
    "akiec": [
        "A dermoscopy image of a pink scaly actinic keratosis with a follicular pseudonetwork and perifollicular white halos",
        "A dermoscopy image of intraepithelial carcinoma showing erythematous structureless areas, adherent keratin, and enlarged follicular openings",
        "A dermoscopy image of pigmented actinic keratosis with focal brown-gray blotches on a pink keratinizing background",
        "A dermoscopy image of Bowen disease with surface scale, white structureless zones, and subtle dotted to linear irregular vessels",
        "A dermoscopy image of a non-melanocytic keratinizing plaque with follicular accentuation and a strawberry-pattern appearance",
    ],
    "vasc": [
        "A dermoscopy image of a well-circumscribed symmetric red-purple angioma with multiple lacunae on a pink background",
        "A dermoscopy image of a dome-shaped cherry angioma showing violaceous to red clods separated by pale septa",
        "A dermoscopy image of a round vascular papule with homogeneous reddish-purple coloration and a classic lacunar pattern",
        "A dermoscopy image of an angioma with sharply defined borders, lobulated red-purple lagoons, and no pigment network",
        "A dermoscopy image of a benign-appearing vascular lesion composed of clustered dark red to violaceous lacunae",
    ],
    "df": [
        "A dermoscopy image of a symmetric dermatofibroma with a central white scar-like patch and faint peripheral pigment network",
        "A dermoscopy image of a pink-brown dermatofibroma showing a structureless fibrotic white center and delicate light brown rim",
        "A dermoscopy image of a dermatofibroma with central shiny white scar-like areas and subtle symmetric peripheral reticular pigmentation",
        "A dermoscopy image of a round pink dermatofibroma with a pale central stellate white area and a thin brown peripheral network",
        "A dermoscopy image of a benign-appearing dermatofibroma with central white fibrosis, mild erythema, and symmetric peripheral pigmentation",
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
# Data helpers (identical to biomedclip_zeroshot.py)
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
# Text embedding helpers (identical logic, uses Phase 2 PROMPTS_PER_CLASS)
# ---------------------------------------------------------------------------

def build_averaged_text_embeddings(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """(C, D) — one L2-normed average embedding per class."""
    class_embeddings: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for cls_name in CLASS_ORDER:
            prompts = PROMPTS_PER_CLASS[cls_name]
            tokens = tokenizer(prompts).to(device)
            text_feats = model.encode_text(tokens)
            text_feats = F.normalize(text_feats, dim=-1)
            avg = text_feats.mean(dim=0)
            avg = F.normalize(avg, dim=-1)
            class_embeddings.append(avg)
    return torch.stack(class_embeddings, dim=0)


def build_per_prompt_text_embeddings(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """(C, P, D) — L2-normed embedding per prompt per class."""
    all_embeddings: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for cls_name in CLASS_ORDER:
            prompts = PROMPTS_PER_CLASS[cls_name]
            tokens = tokenizer(prompts).to(device)
            text_feats = model.encode_text(tokens)
            text_feats = F.normalize(text_feats, dim=-1)
            all_embeddings.append(text_feats)
    return torch.stack(all_embeddings, dim=0)


# ---------------------------------------------------------------------------
# Image encoding (identical to biomedclip_zeroshot.py)
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
            feats = model.encode_image(images)
            feats = F.normalize(feats, dim=-1)
            all_feats.append(feats.cpu())
            all_labels.append(labels)
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


# ---------------------------------------------------------------------------
# Prediction strategies (identical to biomedclip_zeroshot.py)
# ---------------------------------------------------------------------------

def predict_averaged(
    image_feats: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> np.ndarray:
    sims = image_feats @ text_embeddings.T
    return sims.argmax(dim=-1).numpy()


def predict_max_sim(
    image_feats: torch.Tensor,
    per_prompt_embeddings: torch.Tensor,
) -> np.ndarray:
    sims = torch.einsum("nd,cpd->ncp", image_feats, per_prompt_embeddings)
    max_sims = sims.max(dim=-1).values
    return max_sims.argmax(dim=-1).numpy()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strategy_name: str,
) -> tuple[str, float, float]:
    """Returns (report_text, macro_f1, mel_recall)."""
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mel_recall = recall_score(
        y_true, y_pred, labels=[MEL_IDX], average="macro", zero_division=0
    )

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
    lines.append(f"  Macro-F1   : {macro_f1:.4f}")
    lines.append(f"  MEL Recall : {mel_recall:.4f}")
    lines.append("")

    return "\n".join(lines), macro_f1, mel_recall


def build_comparison_table(
    p2_avg_f1: float,
    p2_avg_rec: float,
    p2_max_f1: float,
    p2_max_rec: float,
) -> str:
    sep = "-" * 52
    lines = [
        "",
        "=" * 52,
        "  Phase 1 vs Phase 2 — Side-by-Side Comparison",
        "=" * 52,
        f"{'Strategy':<22} | {'Macro-F1':>9} | {'MEL Recall':>10}",
        sep,
        f"{'Phase1 Averaged':<22} | {PHASE1_AVERAGED_MACRO_F1:>9.3f} | {PHASE1_AVERAGED_MEL_RECALL:>10.3f}",
        f"{'Phase1 Max-Sim':<22} | {PHASE1_MAXSIM_MACRO_F1:>9.3f} | {PHASE1_MAXSIM_MEL_RECALL:>10.3f}",
        sep,
        f"{'Phase2 Averaged':<22} | {p2_avg_f1:>9.3f} | {p2_avg_rec:>10.3f}",
        f"{'Phase2 Max-Sim':<22} | {p2_max_f1:>9.3f} | {p2_max_rec:>10.3f}",
        sep,
        f"{'Delta Averaged':<22} | {p2_avg_f1 - PHASE1_AVERAGED_MACRO_F1:>+9.3f} | {p2_avg_rec - PHASE1_AVERAGED_MEL_RECALL:>+10.3f}",
        f"{'Delta Max-Sim':<22} | {p2_max_f1 - PHASE1_MAXSIM_MACRO_F1:>+9.3f} | {p2_max_rec - PHASE1_MAXSIM_MEL_RECALL:>+10.3f}",
        "=" * 52,
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BiomedCLIP zero-shot Phase 2 (clinically-informed prompts)"
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
    out_file = out_dir / "biomedclip_zeroshot_phase2_results.txt"

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
    print("\nEncoding Phase 2 text prompts …")
    avg_text_emb = build_averaged_text_embeddings(model, tokenizer, device)
    pp_text_emb = build_per_prompt_text_embeddings(model, tokenizer, device)
    print(f"  Averaged embeddings  : {tuple(avg_text_emb.shape)}")
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
    avg_text_emb_cpu = avg_text_emb.cpu()
    pp_text_emb_cpu = pp_text_emb.cpu()

    y_pred_avg = predict_averaged(image_feats, avg_text_emb_cpu)
    y_pred_max = predict_max_sim(image_feats, pp_text_emb_cpu)

    # --- Build reports --------------------------------------------------------
    report_avg, p2_avg_f1, p2_avg_rec = build_report(
        y_true, y_pred_avg, "Phase 2 Averaged Prompts (per-class mean embedding)"
    )
    report_max, p2_max_f1, p2_max_rec = build_report(
        y_true, y_pred_max, "Phase 2 Max-Sim Ensemble (per-prompt max similarity)"
    )

    comparison = build_comparison_table(p2_avg_f1, p2_avg_rec, p2_max_f1, p2_max_rec)

    header = textwrap.dedent(f"""\
        BiomedCLIP Zero-Shot Classification — Phase 2 (Clinically-Informed Prompts)
        ============================================================================
        Model  : hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
        Split  : test set, lesion-level stratified split (seed={cfg.seed})
        Images : {len(df_test)}
        Classes: {CLASS_ORDER}
        Prompts: {len(next(iter(PROMPTS_PER_CLASS.values())))} per class (dermoscopic descriptions via MedGPT)

        Phase 1 reference (generic prompts):
          Averaged  Macro-F1={PHASE1_AVERAGED_MACRO_F1:.3f}  MEL Recall={PHASE1_AVERAGED_MEL_RECALL:.3f}
          Max-Sim   Macro-F1={PHASE1_MAXSIM_MACRO_F1:.3f}  MEL Recall={PHASE1_MAXSIM_MEL_RECALL:.3f}

    """)

    full_report = header + report_avg + "\n" + report_max + comparison

    print("\n" + full_report)

    out_file.write_text(full_report, encoding="utf-8")
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
