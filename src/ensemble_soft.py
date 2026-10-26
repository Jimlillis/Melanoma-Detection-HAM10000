"""
Strategy B — Soft Ensemble: equal-weight average of softmax probabilities
from EfficientNet-B0 (7-class) and DINOv2 ViT-S/14 linear probe (7-class).

Inference is deterministic — no TTA, no MC Dropout.

EfficientNet runs on raw test images (224×224, ImageNet normalisation).
DINOv2 probabilities are produced by passing pre-extracted CLS-token features
(data/processed/dinov2/dinov2_feats_test.pt) through the trained linear head,
which avoids re-running the backbone.

Ensemble rule:
    final_probs = (efficientnet_probs + dinov2_probs) / 2
    prediction  = argmax(final_probs)

Run from project root:
    python src/ensemble_soft.py \\
        --metadata-csv data/HAM10000_metadata.csv \\
        --images-dir data \\
        --output-dir results

Override checkpoint paths if needed:
    --efficientnet-ckpt outputs/efficientnet_mc/best_efficientnet.pth
    --dinov2-head-ckpt  outputs/dinov2_linear/best_dino_head.pth
    --dinov2-feats-dir  data/processed/dinov2
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
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
from dataset import SkinLesionDataset, build_transforms, lesion_level_split
from helpers import get_device, set_seed
from model import build_efficientnet_b0

# ---------------------------------------------------------------------------
# Label constants (sorted alphabetical — same as all other scripts)
# ---------------------------------------------------------------------------

LABEL_MAP: dict[str, int] = {
    "akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6,
}
IDX_TO_CLASS: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}
CLASS_ORDER: list[str] = [IDX_TO_CLASS[i] for i in range(len(LABEL_MAP))]
NUM_CLASSES: int = len(LABEL_MAP)
MEL_IDX: int = LABEL_MAP["mel"]

# Hard-coded individual baselines for the comparison table
_EFFNET_MACRO_F1:  float = 0.692
_EFFNET_MEL_REC:   float = 0.713
_DINOV2_MACRO_F1:  float = 0.545
_DINOV2_MEL_REC:   float = 0.689


# ---------------------------------------------------------------------------
# Metadata helper
# ---------------------------------------------------------------------------

def load_multiclass_table(csv_path: Path, cfg: Config):
    import pandas as pd
    df = pd.read_csv(csv_path)
    required = {cfg.col_image_id, cfg.col_lesion_id, cfg.col_diagnosis}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = df.copy()
    df["_binary_label"] = df[cfg.col_diagnosis].astype(str).map(LABEL_MAP).astype(int)
    return df


# ---------------------------------------------------------------------------
# EfficientNet inference
# ---------------------------------------------------------------------------

def load_efficientnet(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    print(f"  Loading EfficientNet checkpoint: {ckpt_path.resolve()}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def infer_efficientnet(
    model: torch.nn.Module,
    df_test,
    cfg: Config,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """Return (softmax_probs (N,7), y_true (N,)) for the test split."""
    transform = build_transforms(cfg.image_size, is_train=False)
    ds = SkinLesionDataset(df_test, cfg.images_dir, cfg.col_image_id, transform=transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    all_probs: list[torch.Tensor] = []
    all_labels: list[int] = []

    for imgs, labels in tqdm(loader, desc="EfficientNet inference"):
        logits = model(imgs.to(device))
        probs = F.softmax(logits, dim=1).cpu()
        all_probs.append(probs)
        all_labels.extend(labels.tolist())

    return torch.cat(all_probs, dim=0), np.array(all_labels)


# ---------------------------------------------------------------------------
# DINOv2 linear probe inference (uses pre-extracted features)
# ---------------------------------------------------------------------------

def load_dinov2_head(ckpt_path: Path, device: torch.device):
    import torch.nn as nn
    print(f"  Loading DINOv2 head checkpoint: {ckpt_path.resolve()}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feat_dim: int = ckpt["feat_dim"]
    head = nn.Linear(feat_dim, NUM_CLASSES)
    head.load_state_dict(ckpt["model_state_dict"])
    head.to(device).eval()
    print(f"  DINOv2 head: Linear({feat_dim} -> {NUM_CLASSES})")
    return head


@torch.no_grad()
def infer_dinov2(
    head: torch.nn.Module,
    feats_path: Path,
    device: torch.device,
    y_true_ref: np.ndarray,
) -> torch.Tensor:
    """
    Load pre-extracted test features, run through linear head, return softmax probs (N,7).
    Verifies that the stored labels match y_true_ref (EfficientNet test split).
    """
    print(f"  Loading DINOv2 test features: {feats_path.resolve()}")
    data = torch.load(feats_path, map_location="cpu", weights_only=False)
    feats: torch.Tensor = data["features"]   # (N, 384)
    labels: torch.Tensor = data["labels"]    # (N,)

    # Sanity-check: same test split → same labels in the same order
    if not np.array_equal(labels.numpy(), y_true_ref):
        raise RuntimeError(
            "DINOv2 feature file labels do not match the EfficientNet test split labels.\n"
            "Ensure both were produced with the same seed and split ratios (seed=42, 70/15/15)."
        )

    feats = feats.to(device)
    all_probs: list[torch.Tensor] = []
    batch_size = 512
    for start in range(0, len(feats), batch_size):
        batch = feats[start : start + batch_size]
        logits = head(batch)
        all_probs.append(F.softmax(logits, dim=1).cpu())

    return torch.cat(all_probs, dim=0)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def format_cls_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    rep = classification_report(
        y_true, y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_ORDER,
        zero_division=0,
        output_dict=True,
    )
    lines = ["              precision    recall  f1-score   support", ""]
    for name in CLASS_ORDER:
        r = rep[name]
        lines.append(
            f"{name:>12} {r['precision']:>10.2f} {r['recall']:>9.2f}"
            f" {r['f1-score']:>9.2f} {int(r['support']):>9d}"
        )
    lines.append("")
    lines.append(
        f"{'accuracy':>12} {'':>10} {'':>9}"
        f" {rep['accuracy']:>9.2f} {int(rep['macro avg']['support']):>9d}"
    )
    for agg in ("macro avg", "weighted avg"):
        r = rep[agg]
        lines.append(
            f"{agg:>12} {r['precision']:>10.2f} {r['recall']:>9.2f}"
            f" {r['f1-score']:>9.2f} {int(r['support']):>9d}"
        )
    return "\n".join(lines)


def build_confusion_block(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    lines = ["Confusion Matrix (rows = true class, cols = predicted class):"]
    lines.append("         " + "  ".join(f"{c:>6}" for c in CLASS_ORDER))
    for i, row in enumerate(cm):
        lines.append(f"{CLASS_ORDER[i]:>8} " + "  ".join(f"{v:>6d}" for v in row))
    return "\n".join(lines)


def build_comparison_table(
    ens_f1: float, ens_rec: float,
) -> str:
    sep = "-" * 50
    rows = [
        ("EfficientNet (alone)", _EFFNET_MACRO_F1, _EFFNET_MEL_REC),
        ("DINOv2 (alone)",       _DINOV2_MACRO_F1, _DINOV2_MEL_REC),
        ("Soft Ensemble (B)",    ens_f1,            ens_rec),
    ]
    lines = [
        "",
        "=" * 50,
        "  Soft Ensemble — Model Comparison",
        "=" * 50,
        f"{'Model':<25} | {'Macro-F1':>9} | {'MEL Recall':>10}",
        sep,
    ]
    for name, f1, rec in rows:
        lines.append(f"{name:<25} | {f1:>9.3f} | {rec:>10.3f}")
    lines.append("=" * 50)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Strategy B: soft ensemble of EfficientNet-B0 and DINOv2 linear probe"
    )
    p.add_argument("--metadata-csv", type=Path, default=None)
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument(
        "--efficientnet-ckpt",
        type=Path,
        default=Path("outputs/efficientnet_improved/best_model.pth"),
    )
    p.add_argument(
        "--dinov2-head-ckpt",
        type=Path,
        default=Path("outputs/dinov2_linear/best_dino_head.pth"),
    )
    p.add_argument(
        "--dinov2-feats-dir",
        type=Path,
        default=Path("data/processed/dinov2"),
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # The Greek-locale (cp1253) Windows console cannot encode arrows; all
    # print() calls here are plain ASCII, this is the second line of defense.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
    device = get_device()
    print(f"Device : {device}")
    print(f"Seed   : {cfg.seed}\n")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Reproduce test split (seed=42, 70/15/15)
    # -------------------------------------------------------------------------
    print("Loading metadata and reproducing test split …")
    df_all = load_multiclass_table(cfg.metadata_csv, cfg)
    _, _, df_test = lesion_level_split(
        df_all,
        cfg.col_lesion_id,
        cfg.train_ratio, cfg.val_ratio, cfg.test_ratio,
        cfg.seed,
    )
    print(f"Test split: {len(df_test)} images\n")

    # -------------------------------------------------------------------------
    # EfficientNet inference
    # -------------------------------------------------------------------------
    print("[1/2] EfficientNet-B0 inference …")
    eff_model = load_efficientnet(args.efficientnet_ckpt, device)
    eff_probs, y_true = infer_efficientnet(
        eff_model, df_test, cfg, args.batch_size, device
    )
    print(f"  EfficientNet probs : {tuple(eff_probs.shape)}\n")

    # -------------------------------------------------------------------------
    # DINOv2 linear probe inference (pre-extracted features)
    # -------------------------------------------------------------------------
    print("[2/2] DINOv2 linear probe inference …")
    dino_head = load_dinov2_head(args.dinov2_head_ckpt, device)
    feats_test_path = Path(args.dinov2_feats_dir) / "dinov2_feats_test.pt"
    dino_probs = infer_dinov2(dino_head, feats_test_path, device, y_true)
    print(f"  DINOv2 probs       : {tuple(dino_probs.shape)}\n")

    # -------------------------------------------------------------------------
    # Soft ensemble
    # -------------------------------------------------------------------------
    ensemble_probs = (eff_probs + dino_probs) / 2.0     # (N, 7)

    y_pred_eff  = eff_probs.argmax(dim=1).numpy()
    y_pred_dino = dino_probs.argmax(dim=1).numpy()
    y_pred_ens  = ensemble_probs.argmax(dim=1).numpy()

    eff_f1   = f1_score(y_true, y_pred_eff,  average="macro", zero_division=0)
    eff_rec  = recall_score(y_true, y_pred_eff,  labels=[MEL_IDX], average="macro", zero_division=0)
    dino_f1  = f1_score(y_true, y_pred_dino, average="macro", zero_division=0)
    dino_rec = recall_score(y_true, y_pred_dino, labels=[MEL_IDX], average="macro", zero_division=0)
    ens_f1   = f1_score(y_true, y_pred_ens,  average="macro", zero_division=0)
    ens_rec  = recall_score(y_true, y_pred_ens,  labels=[MEL_IDX], average="macro", zero_division=0)

    # -------------------------------------------------------------------------
    # Build report
    # -------------------------------------------------------------------------
    comparison = build_comparison_table(ens_f1, ens_rec)

    header = textwrap.dedent(f"""\
        Strategy B — Soft Ensemble (EfficientNet-B0 + DINOv2 linear probe)
        ====================================================================
        EfficientNet checkpoint : {args.efficientnet_ckpt.resolve()}
        DINOv2 head checkpoint  : {args.dinov2_head_ckpt.resolve()}
        DINOv2 feature file     : {feats_test_path.resolve()}
        Split                   : test set, lesion-level (seed={cfg.seed}, 70/15/15)
        Test images             : {len(df_test)}

        Ensemble rule: final_probs = (efficientnet_probs + dinov2_probs) / 2
                       prediction  = argmax(final_probs)

    """)

    eff_section = "\n".join([
        "=" * 64,
        "  EfficientNet-B0 (alone) — test-set results",
        "=" * 64,
        "",
        build_confusion_block(y_true, y_pred_eff),
        "",
        "Per-class metrics:",
        format_cls_report(y_true, y_pred_eff),
        "",
        f"  Macro-F1   : {eff_f1:.4f}",
        f"  MEL Recall : {eff_rec:.4f}",
        "",
    ])

    dino_section = "\n".join([
        "=" * 64,
        "  DINOv2 linear probe (alone) — test-set results",
        "=" * 64,
        "",
        build_confusion_block(y_true, y_pred_dino),
        "",
        "Per-class metrics:",
        format_cls_report(y_true, y_pred_dino),
        "",
        f"  Macro-F1   : {dino_f1:.4f}",
        f"  MEL Recall : {dino_rec:.4f}",
        "",
    ])

    ens_section = "\n".join([
        "=" * 64,
        "  Soft Ensemble (B) — test-set results",
        "=" * 64,
        "",
        build_confusion_block(y_true, y_pred_ens),
        "",
        "Per-class metrics:",
        format_cls_report(y_true, y_pred_ens),
        "",
        f"  Macro-F1   : {ens_f1:.4f}",
        f"  MEL Recall : {ens_rec:.4f}",
        "",
    ])

    full_report = header + eff_section + "\n" + dino_section + "\n" + ens_section + comparison

    print("\n" + full_report)

    out_file = out_dir / "ensemble_soft_results.txt"
    out_file.write_text(full_report, encoding="utf-8")
    print(f"\nResults saved -> {out_file}")


if __name__ == "__main__":
    main()
