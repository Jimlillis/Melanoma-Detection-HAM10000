"""
Strategy C — Weighted Ensemble: optimised-weight average of softmax
probabilities from EfficientNet-B0 and DINOv2 ViT-S/14 linear probe.

Ensemble rule:
    final_probs = w1 * efficientnet_probs + w2 * dinov2_probs
    where w1 + w2 = 1.0,  prediction = argmax(final_probs)

Weight optimisation:
    Grid search over w1 ∈ {0.00, 0.05, …, 1.00} (21 candidates, w2 = 1 − w1)
    is performed on the VALIDATION set — never touching the test set — to avoid
    data leakage.  Two optimal configurations are selected:
      (a) weights that maximise Val Macro-F1
      (b) weights that maximise Val MEL Recall
    Both are then evaluated on the held-out TEST set.

Inference notes:
    EfficientNet runs on raw images (224×224, ImageNet normalisation).
    DINOv2 probabilities come from pre-extracted CLS-token features passed
    through the frozen linear head — the ViT backbone is not re-run.

Run from project root:
    python src/ensemble_weighted.py \\
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
# Label constants
# ---------------------------------------------------------------------------

LABEL_MAP: dict[str, int] = {
    "akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6,
}
IDX_TO_CLASS: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}
CLASS_ORDER: list[str] = [IDX_TO_CLASS[i] for i in range(len(LABEL_MAP))]
NUM_CLASSES: int = len(LABEL_MAP)
MEL_IDX: int = LABEL_MAP["mel"]

# Hard-coded baselines for the comparison table
_EFFNET_F1:   float = 0.692;  _EFFNET_REC:   float = 0.713
_DINOV2_F1:   float = 0.545;  _DINOV2_REC:   float = 0.689
_SOFT_ENS_F1: float = 0.696;  _SOFT_ENS_REC: float = 0.731

# Grid search step
_GRID_STEP: float = 0.05

# ---------------------------------------------------------------------------
# Metadata
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
# EfficientNet
# ---------------------------------------------------------------------------

def load_efficientnet(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    print(f"  EfficientNet checkpoint : {ckpt_path.resolve()}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def infer_efficientnet(
    model: torch.nn.Module,
    df_split,
    cfg: Config,
    batch_size: int,
    device: torch.device,
    desc: str = "EfficientNet",
) -> tuple[torch.Tensor, np.ndarray]:
    """Return (softmax_probs (N,7), y_true (N,))."""
    transform = build_transforms(cfg.image_size, is_train=False)
    ds = SkinLesionDataset(df_split, cfg.images_dir, cfg.col_image_id, transform=transform)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )
    probs_list: list[torch.Tensor] = []
    labels_list: list[int] = []
    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        probs_list.append(F.softmax(model(imgs.to(device)), dim=1).cpu())
        labels_list.extend(labels.tolist())
    return torch.cat(probs_list, dim=0), np.array(labels_list)


# ---------------------------------------------------------------------------
# DINOv2 linear probe (pre-extracted features)
# ---------------------------------------------------------------------------

def load_dinov2_head(ckpt_path: Path, device: torch.device):
    import torch.nn as nn
    print(f"  DINOv2 head checkpoint  : {ckpt_path.resolve()}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head = nn.Linear(ckpt["feat_dim"], NUM_CLASSES)
    head.load_state_dict(ckpt["model_state_dict"])
    print(f"  DINOv2 head : Linear({ckpt['feat_dim']} -> {NUM_CLASSES})")
    return head.to(device).eval()


@torch.no_grad()
def infer_dinov2_feats(
    head: torch.nn.Module,
    feats_path: Path,
    device: torch.device,
    y_ref: np.ndarray,
) -> torch.Tensor:
    """
    Load pre-extracted features, run through linear head → softmax probs (N,7).
    Verifies stored labels match y_ref to confirm identical split.
    """
    print(f"  DINOv2 features         : {feats_path.resolve()}")
    data = torch.load(feats_path, map_location="cpu", weights_only=False)
    feats: torch.Tensor = data["features"]
    labels: torch.Tensor = data["labels"]

    if not np.array_equal(labels.numpy(), y_ref):
        raise RuntimeError(
            f"Label mismatch between {feats_path.name} and the reconstructed split.\n"
            "Ensure both use seed=42 with 70/15/15 ratios."
        )

    probs_list: list[torch.Tensor] = []
    for start in range(0, len(feats), 512):
        batch = feats[start : start + 512].to(device)
        probs_list.append(F.softmax(head(batch), dim=1).cpu())
    return torch.cat(probs_list, dim=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mel_rec  = recall_score(y_true, y_pred, labels=[MEL_IDX], average="macro", zero_division=0)
    return macro_f1, mel_rec


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

GridRow = tuple[float, float, float, float]   # (w_eff, w_dino, macro_f1, mel_rec)


def grid_search(
    eff_probs:  torch.Tensor,   # (N, 7)
    dino_probs: torch.Tensor,   # (N, 7)
    y_true:     np.ndarray,
) -> list[GridRow]:
    """
    Vectorised grid search: build all 21 ensemble predictions at once,
    then compute sklearn metrics per weight pair.
    """
    W1 = torch.arange(0.0, 1.0 + _GRID_STEP / 2, _GRID_STEP)  # (21,)

    # (21, N, 7) — broadcast weight over samples and classes
    all_ens = (
        W1[:, None, None] * eff_probs.unsqueeze(0)
        + (1 - W1)[:, None, None] * dino_probs.unsqueeze(0)
    )
    all_preds = all_ens.argmax(dim=-1)  # (21, N)

    results: list[GridRow] = []
    for i, w1 in enumerate(W1.tolist()):
        y_pred = all_preds[i].numpy()
        f1, rec = compute_metrics(y_true, y_pred)
        results.append((round(w1, 2), round(1.0 - w1, 2), f1, rec))
    return results


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_grid_table(results: list[GridRow]) -> str:
    sep = "-" * 54
    lines = [
        "=" * 54,
        "  Grid Search — Validation Set",
        "=" * 54,
        f"  {'w_eff':>6}  {'w_dino':>6}  {'Val Macro-F1':>13}  {'Val MEL Recall':>14}",
        sep,
    ]
    for w1, w2, f1, rec in results:
        lines.append(f"  {w1:>6.2f}  {w2:>6.2f}  {f1:>13.4f}  {rec:>14.4f}")
    lines.append("=" * 54)
    return "\n".join(lines)


def format_top5(results: list[GridRow], metric: str) -> str:
    """metric: 'f1' or 'rec'"""
    col_idx = 2 if metric == "f1" else 3
    label   = "Macro-F1" if metric == "f1" else "MEL Recall"
    top5 = sorted(results, key=lambda r: r[col_idx], reverse=True)[:5]
    sep = "-" * 44
    lines = [
        f"  Top 5 by Val {label}:",
        f"  {'Rank':>4}  {'w_eff':>6}  {'w_dino':>6}  {label:>12}",
        "  " + sep,
    ]
    for rank, row in enumerate(top5, 1):
        lines.append(f"  {rank:>4}  {row[0]:>6.2f}  {row[1]:>6.2f}  {row[col_idx]:>12.4f}")
    lines.append("  " + sep)
    return "\n".join(lines)


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
    lines += ["",
              f"{'accuracy':>12} {'':>10} {'':>9}"
              f" {rep['accuracy']:>9.2f} {int(rep['macro avg']['support']):>9d}"]
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


def build_test_section(
    title: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    w1: float,
    w2: float,
) -> str:
    f1, rec = compute_metrics(y_true, y_pred)
    return "\n".join([
        "=" * 68,
        f"  {title}",
        f"  Weights: w_eff={w1:.2f}  w_dino={w2:.2f}",
        "=" * 68,
        "",
        build_confusion_block(y_true, y_pred),
        "",
        "Per-class metrics:",
        format_cls_report(y_true, y_pred),
        "",
        f"  Macro-F1   : {f1:.4f}",
        f"  MEL Recall : {rec:.4f}",
        "",
    ])


def build_comparison_table(
    best_f1_w1:   float, test_f1_at_bestf1:  float, test_rec_at_bestf1:  float,
    best_rec_w1:  float, test_f1_at_bestrec: float, test_rec_at_bestrec: float,
) -> str:
    sep = "-" * 58
    rows = [
        ("EfficientNet (alone)",              _EFFNET_F1,    _EFFNET_REC),
        ("DINOv2 (alone)",                    _DINOV2_F1,    _DINOV2_REC),
        ("Soft Ensemble B (w=0.50/0.50)",     _SOFT_ENS_F1,  _SOFT_ENS_REC),
        (f"Weighted C — best F1  (w={best_f1_w1:.2f}/{1-best_f1_w1:.2f})",
                                              test_f1_at_bestf1,  test_rec_at_bestf1),
        (f"Weighted C — best MEL (w={best_rec_w1:.2f}/{1-best_rec_w1:.2f})",
                                              test_f1_at_bestrec, test_rec_at_bestrec),
    ]
    lines = [
        "",
        "=" * 58,
        "  Strategy C — Weighted Ensemble: Final Comparison",
        "=" * 58,
        f"{'Model':<38} | {'Macro-F1':>8} | {'MEL Recall':>10}",
        sep,
    ]
    for name, f1, rec in rows:
        lines.append(f"{name:<38} | {f1:>8.3f} | {rec:>10.3f}")
    lines.append("=" * 58)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Strategy C: weighted ensemble with val-set weight optimisation"
    )
    p.add_argument("--metadata-csv",    type=Path, default=None)
    p.add_argument("--images-dir",      type=Path, default=None)
    p.add_argument("--output-dir",      type=Path, default=Path("results"))
    p.add_argument("--efficientnet-ckpt",
                   type=Path, default=Path("outputs/efficientnet_mc/best_efficientnet.pth"))
    p.add_argument("--dinov2-head-ckpt",
                   type=Path, default=Path("outputs/dinov2_linear/best_dino_head.pth"))
    p.add_argument("--dinov2-feats-dir",
                   type=Path, default=Path("data/processed/dinov2"))
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--num-workers", type=int,   default=None)
    p.add_argument("--seed",        type=int,   default=None)
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
    if args.metadata_csv  is not None: cfg.metadata_csv  = args.metadata_csv
    if args.images_dir    is not None: cfg.images_dir    = args.images_dir
    if args.num_workers   is not None: cfg.num_workers   = args.num_workers
    if args.seed          is not None: cfg.seed          = args.seed

    set_seed(cfg.seed)
    device = get_device()
    print(f"Device : {device}")
    print(f"Seed   : {cfg.seed}\n")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats_dir = Path(args.dinov2_feats_dir)

    # -------------------------------------------------------------------------
    # Reproduce val + test splits
    # -------------------------------------------------------------------------
    print("Loading metadata and reproducing lesion-level splits …")
    df_all = load_multiclass_table(cfg.metadata_csv, cfg)
    _, df_val, df_test = lesion_level_split(
        df_all, cfg.col_lesion_id,
        cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, cfg.seed,
    )
    print(f"  Val  : {len(df_val)} images")
    print(f"  Test : {len(df_test)} images\n")

    # -------------------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------------------
    print("Loading models …")
    eff_model = load_efficientnet(args.efficientnet_ckpt, device)
    dino_head = load_dinov2_head(args.dinov2_head_ckpt, device)
    print()

    # -------------------------------------------------------------------------
    # Inference — validation split
    # -------------------------------------------------------------------------
    print("[1/4] EfficientNet inference on VAL split …")
    eff_val_probs, y_val = infer_efficientnet(
        eff_model, df_val, cfg, args.batch_size, device, desc="EfficientNet val"
    )
    print(f"  shape={tuple(eff_val_probs.shape)}\n")

    print("[2/4] DINOv2 inference on VAL split …")
    dino_val_probs = infer_dinov2_feats(
        dino_head, feats_dir / "dinov2_feats_val.pt", device, y_val
    )
    print(f"  shape={tuple(dino_val_probs.shape)}\n")

    # -------------------------------------------------------------------------
    # Inference — test split
    # -------------------------------------------------------------------------
    print("[3/4] EfficientNet inference on TEST split …")
    eff_test_probs, y_test = infer_efficientnet(
        eff_model, df_test, cfg, args.batch_size, device, desc="EfficientNet test"
    )
    print(f"  shape={tuple(eff_test_probs.shape)}\n")

    print("[4/4] DINOv2 inference on TEST split …")
    dino_test_probs = infer_dinov2_feats(
        dino_head, feats_dir / "dinov2_feats_test.pt", device, y_test
    )
    print(f"  shape={tuple(dino_test_probs.shape)}\n")

    # -------------------------------------------------------------------------
    # Grid search on VAL set
    # -------------------------------------------------------------------------
    print(f"Running grid search (w_eff in [0.00, 1.00] step {_GRID_STEP}) on VAL set …")
    grid_results = grid_search(eff_val_probs, dino_val_probs, y_val)

    grid_table = format_grid_table(grid_results)
    top5_f1    = format_top5(grid_results, "f1")
    top5_rec   = format_top5(grid_results, "rec")

    print("\n" + grid_table)
    print("\n" + top5_f1)
    print("\n" + top5_rec)

    # -------------------------------------------------------------------------
    # Optimal weights
    # -------------------------------------------------------------------------
    best_f1_row  = max(grid_results, key=lambda r: r[2])
    best_rec_row = max(grid_results, key=lambda r: r[3])

    best_f1_w1,  best_f1_w2  = best_f1_row[0],  best_f1_row[1]
    best_rec_w1, best_rec_w2 = best_rec_row[0], best_rec_row[1]

    print(f"\nOptimal weights (val):")
    print(f"  Best Macro-F1   -> w_eff={best_f1_w1:.2f}, w_dino={best_f1_w2:.2f}"
          f"  (val F1={best_f1_row[2]:.4f}, val MEL={best_f1_row[3]:.4f})")
    print(f"  Best MEL Recall -> w_eff={best_rec_w1:.2f}, w_dino={best_rec_w2:.2f}"
          f"  (val F1={best_rec_row[2]:.4f}, val MEL={best_rec_row[3]:.4f})")

    # -------------------------------------------------------------------------
    # Test evaluation with both optimal weight pairs
    # -------------------------------------------------------------------------
    print("\nEvaluating on TEST set …")

    # (a) weights optimal for Macro-F1
    ens_test_bestf1  = best_f1_w1  * eff_test_probs + best_f1_w2  * dino_test_probs
    y_pred_bestf1    = ens_test_bestf1.argmax(dim=1).numpy()
    tf1_f1, tf1_rec  = compute_metrics(y_test, y_pred_bestf1)

    # (b) weights optimal for MEL Recall
    ens_test_bestrec = best_rec_w1 * eff_test_probs + best_rec_w2 * dino_test_probs
    y_pred_bestrec   = ens_test_bestrec.argmax(dim=1).numpy()
    tr_f1, tr_rec    = compute_metrics(y_test, y_pred_bestrec)

    # -------------------------------------------------------------------------
    # Assemble full report
    # -------------------------------------------------------------------------
    header = textwrap.dedent(f"""\
        Strategy C — Weighted Ensemble (EfficientNet-B0 + DINOv2 linear probe)
        =======================================================================
        EfficientNet checkpoint : {args.efficientnet_ckpt.resolve()}
        DINOv2 head checkpoint  : {args.dinov2_head_ckpt.resolve()}
        DINOv2 val features     : {(feats_dir / "dinov2_feats_val.pt").resolve()}
        DINOv2 test features    : {(feats_dir / "dinov2_feats_test.pt").resolve()}
        Split (seed={cfg.seed})         : 70 % train / 15 % val / 15 % test (lesion-level)
        Val images              : {len(df_val)}
        Test images             : {len(df_test)}

        Ensemble rule: final_probs = w_eff * efficientnet_probs + w_dino * dinov2_probs
        Grid search  : w_eff ∈ {{0.00, 0.05, …, 1.00}}  (step {_GRID_STEP}, {len(grid_results)} candidates)
        Optimised on : validation set only (test set kept blind)

    """)

    section_bestf1  = build_test_section(
        f"Weighted Ensemble C — Best Macro-F1 weights (w_eff={best_f1_w1:.2f}, w_dino={best_f1_w2:.2f})",
        y_test, y_pred_bestf1, best_f1_w1, best_f1_w2,
    )
    section_bestrec = build_test_section(
        f"Weighted Ensemble C — Best MEL Recall weights (w_eff={best_rec_w1:.2f}, w_dino={best_rec_w2:.2f})",
        y_test, y_pred_bestrec, best_rec_w1, best_rec_w2,
    )
    comparison = build_comparison_table(
        best_f1_w1,  tf1_f1, tf1_rec,
        best_rec_w1, tr_f1,  tr_rec,
    )

    full_report = (
        header
        + grid_table + "\n\n"
        + top5_f1    + "\n\n"
        + top5_rec   + "\n\n"
        + section_bestf1  + "\n"
        + section_bestrec + "\n"
        + comparison
    )

    print("\n" + comparison)

    out_file = out_dir / "ensemble_weighted_results.txt"
    out_file.write_text(full_report, encoding="utf-8")
    print(f"\nFull report saved -> {out_file}")


if __name__ == "__main__":
    main()
