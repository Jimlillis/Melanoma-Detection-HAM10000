"""
Strategy A v2 — Feature Concatenation Ensemble (3 MEL-aware variants).

Same architecture and features as ensemble_feature_concat.py (Strategy A):
  EfficientNet-B0 (1280-dim) + DINOv2 ViT-S/14 (384-dim) → 1664-dim MLP

Three variants that try to balance Macro-F1 and MEL Recall:

  Variant 1 — MEL-boosted Focal Loss
    class_weights = inverse_frequency_weights
    class_weights[mel] *= 3.0          ← extra 3× boost
    Best-model criterion: val Macro-F1
    Checkpoint: outputs/ensemble_concat/best_concat_v2_melboost.pth

  Variant 2 — Composite criterion
    class_weights = inverse_frequency_weights (no boost)
    Best-model criterion: 0.5 × val_Macro-F1 + 0.5 × val_MEL_Recall
    Checkpoint: outputs/ensemble_concat/best_concat_v2_composite.pth

  Variant 3 — Threshold tuning (post-hoc, built on Variant 1)
    Load best_concat_v2_melboost.pth
    Grid-search mel threshold ∈ {0.10, 0.15, …, 0.50} on VALIDATION set
      if mel_prob > t → predict mel
      else            → argmax of the remaining 6 classes
    Criterion: 0.5 × Macro-F1 + 0.5 × MEL_Recall
    Apply best threshold on TEST set

Prerequisites (features must already be extracted):
  python src/extract_efficientnet_features.py  ...  (→ data/processed/efficientnet/)
  python src/extract_features_v2.py            ...  (→ data/processed/dinov2/)

Run from project root:
  python src/ensemble_concat_v2.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from helpers import get_device, set_seed


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Multi-class Focal Loss with optional per-class alpha weights."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_softmax = F.log_softmax(logits, dim=1)
        log_pt = log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------------
# MLP classifier  (identical architecture to Strategy A)
# ---------------------------------------------------------------------------

class ConcatMLP(nn.Module):
    """
    Two-hidden-layer MLP for concatenated EfficientNet+DINOv2 features.

    Input  : (B, 1664)   — 1280-dim effnet  ||  384-dim dinov2
    Output : (B, 7)      — raw logits for 7 HAM10000 classes
    """

    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_class_weights(
    labels: torch.Tensor, num_classes: int, device: torch.device
) -> torch.Tensor:
    """Inverse-frequency weights: w_c = N / (C * n_c)."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = labels.numel() / (num_classes * counts)
    weights[counts == 0] = 0.0
    return weights.to(device)


def make_weighted_sampler(
    labels: torch.Tensor, num_classes: int
) -> WeightedRandomSampler:
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights_per_class = 1.0 / counts.clamp(min=1.0)
    sample_weights = weights_per_class[labels].double()
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean_loss, y_true, y_pred, probs)."""
    model.eval()
    losses: list[float] = []
    all_y:     list[int]   = []
    all_pred:  list[int]   = []
    all_probs: list[np.ndarray] = []

    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        logits = model(feats)
        losses.append(criterion(logits, labels).item())
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_y.extend(labels.cpu().tolist())
        all_pred.extend(logits.argmax(dim=1).cpu().tolist())

    return (
        float(np.mean(losses)),
        np.array(all_y),
        np.array(all_pred),
        np.vstack(all_probs),
    )


def apply_mel_threshold(
    probs: np.ndarray, mel_idx: int, threshold: float
) -> np.ndarray:
    """
    Override argmax with MEL when mel_prob > threshold.
    For samples where mel_prob ≤ threshold, predict argmax over the remaining 6 classes.
    """
    preds = np.empty(len(probs), dtype=np.int64)
    for i, p in enumerate(probs):
        if p[mel_idx] > threshold:
            preds[i] = mel_idx
        else:
            masked = p.copy()
            masked[mel_idx] = -1.0
            preds[i] = int(np.argmax(masked))
    return preds


def composite_score(
    y_true: np.ndarray, y_pred: np.ndarray, mel_idx: int
) -> float:
    """0.5 × Macro-F1 + 0.5 × MEL Recall."""
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mel_recall = recall_score(
        y_true, y_pred, labels=[mel_idx], average="micro", zero_division=0
    )
    return 0.5 * macro_f1 + 0.5 * mel_recall


def format_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> str:
    rep = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    lines = ["              precision    recall  f1-score   support", ""]
    for name in class_names:
        row = rep[name]
        lines.append(
            f"{name:>12} {row['precision']:>10.2f} {row['recall']:>9.2f}"
            f" {row['f1-score']:>9.2f} {int(row['support']):>9d}"
        )
    lines.append("")
    lines.append(
        f"{'accuracy':>12} {'':>10} {'':>9} {rep['accuracy']:>9.2f}"
        f" {int(rep['macro avg']['support']):>9d}"
    )
    for agg in ("macro avg", "weighted avg"):
        row = rep[agg]
        lines.append(
            f"{agg:>12} {row['precision']:>10.2f} {row['recall']:>9.2f}"
            f" {row['f1-score']:>9.2f} {int(row['support']):>9d}"
        )
    return "\n".join(lines)


def confusion_matrix_str(
    cm: np.ndarray, class_names: list[str]
) -> str:
    lines = ["Confusion Matrix  (rows = true, cols = predicted):"]
    lines.append("         " + "  ".join(f"{c:>6}" for c in class_names))
    for i, row in enumerate(cm):
        lines.append(f"{class_names[i]:>8} " + "  ".join(f"{v:>6}" for v in row))
    return "\n".join(lines)


def load_and_concat(
    effnet_dir: Path,
    dinov2_dir: Path,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    eff  = torch.load(effnet_dir / f"effnet_feats_{split}.pt",  map_location="cpu", weights_only=False)
    dino = torch.load(dinov2_dir / f"dinov2_feats_{split}.pt",  map_location="cpu", weights_only=False)

    if not torch.equal(eff["labels"], dino["labels"]):
        raise ValueError(
            f"Label mismatch between EfficientNet and DINOv2 for split '{split}'."
        )
    if eff["label_map"] != dino["label_map"]:
        raise ValueError("label_map mismatch between feature files.")

    concat = torch.cat([eff["features"], dino["features"]], dim=1)
    return concat, eff["labels"].long(), eff["label_map"]


def train_one_variant(
    *,
    tag: str,
    class_weights: torch.Tensor,
    criterion_key: str,   # "macro_f1" | "composite"
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val:   torch.Tensor,
    y_val:   torch.Tensor,
    in_dim: int,
    num_classes: int,
    mel_idx: int,
    label_map: dict[str, int],
    device: torch.device,
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[ConcatMLP, str, float, float]:
    """
    Train one MLP variant.

    Returns (trained_model, checkpoint_path, best_val_macro_f1, best_val_mel_recall).
    """
    print(f"\n{'=' * 68}")
    print(f"VARIANT: {tag}  |  criterion: {criterion_key}")
    print(f"{'=' * 68}")

    sampler = make_weighted_sampler(y_train, num_classes)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    focal = FocalLoss(gamma=args.gamma, weight=class_weights)
    model = ConcatMLP(in_dim, num_classes, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = -1.0
    best_macro_f1 = 0.0
    best_mel_recall = 0.0
    ckpt_path = str(out_dir / f"best_concat_v2_{tag}.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = focal(model(feats), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        scheduler.step()

        val_loss, y_true_v, y_pred_v, _ = evaluate(model, val_loader, device, focal)
        val_macro_f1 = f1_score(y_true_v, y_pred_v, average="macro", zero_division=0)
        val_mel_recall = (
            recall_score(y_true_v, y_pred_v, labels=[mel_idx], average="micro", zero_division=0)
            if mel_idx >= 0 else float("nan")
        )

        if criterion_key == "macro_f1":
            score = val_macro_f1
        else:  # composite
            score = 0.5 * val_macro_f1 + 0.5 * val_mel_recall

        print(
            f"Epoch {epoch:3d}  "
            f"train_loss={running_loss / n_batches:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_macro_f1={val_macro_f1:.4f}  "
            f"val_mel_recall={val_mel_recall:.4f}"
            + (f"  composite={score:.4f}" if criterion_key == "composite" else "")
        )

        if score > best_score:
            best_score = score
            best_macro_f1 = val_macro_f1
            best_mel_recall = val_mel_recall
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_map":   label_map,
                    "in_dim":      in_dim,
                    "num_classes": num_classes,
                    "dropout":     args.dropout,
                    "epoch":       epoch,
                    "val_macro_f1":    val_macro_f1,
                    "val_mel_recall":  val_mel_recall,
                },
                ckpt_path,
            )
            print(f"  -> New best checkpoint ({criterion_key}={best_score:.4f})")

    print(
        f"\n{tag} training complete.  "
        f"Best val macro_f1={best_macro_f1:.4f}  mel_recall={best_mel_recall:.4f}"
    )
    return model, ckpt_path, best_macro_f1, best_mel_recall


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Strategy A v2 — Feature Concat Ensemble with 3 MEL-aware variants"
    )
    p.add_argument("--effnet-feats-dir",  type=Path, default=Path("data/processed/efficientnet"))
    p.add_argument("--dinov2-feats-dir",  type=Path, default=Path("data/processed/dinov2"))
    p.add_argument("--output-dir",        type=Path, default=Path("outputs/ensemble_concat"))
    p.add_argument("--results-dir",       type=Path, default=Path("results"))
    p.add_argument("--epochs",            type=int,  default=50)
    p.add_argument("--batch-size",        type=int,  default=128)
    p.add_argument("--lr",                type=float, default=1e-3)
    p.add_argument("--weight-decay",      type=float, default=1e-4)
    p.add_argument("--gamma",             type=float, default=2.0)
    p.add_argument("--dropout",           type=float, default=0.3)
    p.add_argument("--mel-boost",         type=float, default=3.0,
                   help="Extra multiplier applied to MEL weight in Variant 1")
    p.add_argument("--num-workers",       type=int,  default=4)
    p.add_argument("--seed",              type=int,  default=42)
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
    set_seed(args.seed)
    device = get_device()

    out_dir = args.output_dir
    results_dir = args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Load & concatenate features ------------------------------------------
    print("Loading and concatenating features …")
    X_train, y_train, label_map = load_and_concat(args.effnet_feats_dir, args.dinov2_feats_dir, "train")
    X_val,   y_val,   _         = load_and_concat(args.effnet_feats_dir, args.dinov2_feats_dir, "val")
    X_test,  y_test,  _         = load_and_concat(args.effnet_feats_dir, args.dinov2_feats_dir, "test")

    in_dim      = X_train.shape[1]
    num_classes = len(label_map)
    idx_to_class: dict[int, str] = {v: k for k, v in label_map.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]
    mel_idx = label_map.get("mel", -1)
    if mel_idx < 0:
        raise ValueError("'mel' class not found in label_map.")

    print(f"Feature dim : {in_dim}  (EfficientNet 1280 + DINOv2 384)")
    print(f"Classes     : {num_classes}  ->  {label_map}")
    print(f"Train: {len(y_train)}   Val: {len(y_val)}   Test: {len(y_test)}")

    # Shared test loader
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # -------------------------------------------------------------------------
    # VARIANT 1 — MEL-boosted weights
    # -------------------------------------------------------------------------
    cw_boost = compute_class_weights(y_train, num_classes, device).clone()
    cw_boost[mel_idx] *= args.mel_boost
    print(f"\nVariant 1 — MEL boost x{args.mel_boost}")
    print(f"  Raw weights: {[f'{w:.3f}' for w in cw_boost.cpu().tolist()]}")

    model_v1, ckpt_v1, _, _ = train_one_variant(
        tag="melboost",
        class_weights=cw_boost,
        criterion_key="macro_f1",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        in_dim=in_dim, num_classes=num_classes, mel_idx=mel_idx,
        label_map=label_map,
        device=device, out_dir=out_dir, args=args,
    )

    # -------------------------------------------------------------------------
    # VARIANT 2 — Composite criterion (standard weights)
    # -------------------------------------------------------------------------
    cw_std = compute_class_weights(y_train, num_classes, device)
    print(f"\nVariant 2 — Composite criterion")
    print(f"  Raw weights: {[f'{w:.3f}' for w in cw_std.cpu().tolist()]}")

    model_v2, ckpt_v2, _, _ = train_one_variant(
        tag="composite",
        class_weights=cw_std,
        criterion_key="composite",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        in_dim=in_dim, num_classes=num_classes, mel_idx=mel_idx,
        label_map=label_map,
        device=device, out_dir=out_dir, args=args,
    )

    # -------------------------------------------------------------------------
    # TEST EVALUATION — Variants 1 & 2
    # -------------------------------------------------------------------------
    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for variant_tag, model, ckpt_path in [
        ("melboost",  model_v1, ckpt_v1),
        ("composite", model_v2, ckpt_v2),
    ]:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        focal_eval = FocalLoss(
            gamma=args.gamma,
            weight=(cw_boost if variant_tag == "melboost" else cw_std),
        )
        _, y_true_t, y_pred_t, probs_t = evaluate(model, test_loader, device, focal_eval)
        results[variant_tag] = (y_true_t, y_pred_t, probs_t)

    # -------------------------------------------------------------------------
    # VARIANT 3 — Threshold tuning (built on Variant 1)
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 68}")
    print("VARIANT: threshold_tuned  |  based on Variant 1 (melboost)")
    print(f"{'=' * 68}")

    # Collect val probabilities from Variant 1
    focal_v1 = FocalLoss(gamma=args.gamma, weight=cw_boost)
    ckpt_v1_data = torch.load(ckpt_v1, map_location=device, weights_only=False)
    model_v1.load_state_dict(ckpt_v1_data["model_state_dict"])

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    _, y_true_val, _, probs_val = evaluate(model_v1, val_loader, device, focal_v1)

    thresholds = [round(t, 2) for t in np.arange(0.10, 0.55, 0.05)]
    print(f"\nGrid-searching MEL threshold on validation set over {thresholds} …")
    print(f"{'Threshold':>12}  {'Macro-F1':>10}  {'MEL Recall':>11}  {'Composite':>11}")

    best_t = thresholds[0]
    best_t_score = -1.0
    threshold_rows: list[str] = []

    for t in thresholds:
        y_pred_t_val = apply_mel_threshold(probs_val, mel_idx, t)
        mf1 = f1_score(y_true_val, y_pred_t_val, average="macro", zero_division=0)
        mr  = recall_score(y_true_val, y_pred_t_val, labels=[mel_idx], average="micro", zero_division=0)
        comp = 0.5 * mf1 + 0.5 * mr
        row = f"{t:>12.2f}  {mf1:>10.4f}  {mr:>11.4f}  {comp:>11.4f}"
        print(row)
        threshold_rows.append(row)
        if comp > best_t_score:
            best_t_score = comp
            best_t = t

    print(f"\nBest threshold on val: t={best_t:.2f}  (composite={best_t_score:.4f})")

    # Apply best threshold on test set
    _, y_true_test_v1, _, probs_test_v1 = evaluate(model_v1, test_loader, device, focal_v1)
    y_pred_thresh = apply_mel_threshold(probs_test_v1, mel_idx, best_t)
    results["threshold_tuned"] = (y_true_test_v1, y_pred_thresh, probs_test_v1)

    # -------------------------------------------------------------------------
    # Print & collect test results for all 3 variants
    # -------------------------------------------------------------------------
    summary: dict[str, tuple[float, float]] = {}

    output_sections: list[str] = []

    for vtag, (y_t, y_p, _) in results.items():
        mf1 = f1_score(y_t, y_p, average="macro", zero_division=0)
        mr  = recall_score(y_t, y_p, labels=[mel_idx], average="micro", zero_division=0)
        summary[vtag] = (mf1, mr)

        cm = confusion_matrix(y_t, y_p, labels=list(range(num_classes)))
        cm_str = confusion_matrix_str(cm, class_names)
        report_str = format_classification_report(y_t, y_p, class_names)

        header = {
            "melboost":       "VARIANT 1 — MEL-Boosted Focal Loss",
            "composite":      "VARIANT 2 — Composite Criterion",
            "threshold_tuned": f"VARIANT 3 — Threshold Tuned (t={best_t:.2f})",
        }[vtag]

        section = (
            f"\n{'=' * 68}\n"
            f"TEST SET EVALUATION — {header}\n"
            f"{'=' * 68}\n\n"
            f"{cm_str}\n\n"
            f"Per-class Classification Report:\n"
            f"{report_str}\n\n"
            f"{'=' * 68}\n"
            f"  Macro-F1        : {mf1:.4f}\n"
            f"  Melanoma Recall : {mr:.4f}\n"
            f"{'=' * 68}"
        )
        print(section)
        output_sections.append(section)

    # -------------------------------------------------------------------------
    # Comparison table
    # -------------------------------------------------------------------------
    sep = "-" * 62
    hdr = f"{'Model':<42} {'Macro-F1':>9} {'MEL Recall':>11}"

    baselines = [
        ("EfficientNet (alone)",                0.692, 0.713),
        ("Soft Ensemble B (0.50/0.50)",          0.696, 0.731),
        ("Feature Concat A (original)",          0.724, 0.575),
        ("Feature Concat A v2 — MEL boost",      *summary["melboost"]),
        ("Feature Concat A v2 — Composite crit.", *summary["composite"]),
        (f"Feature Concat A v2 — Threshold (t={best_t:.2f})", *summary["threshold_tuned"]),
    ]

    rows = "\n".join(
        f"{name:<42} {f1:>9.3f} {mel:>11.3f}"
        for name, f1, mel in baselines
    )

    table = (
        f"\n{'=' * 62}\n"
        f"ENSEMBLE COMPARISON — HAM10000 Test Set\n"
        f"{sep}\n"
        f"{hdr}\n"
        f"{sep}\n"
        f"{rows}\n"
        f"{sep}\n"
        f"{'=' * 62}\n"
    )
    print(table)

    # Threshold grid table (for saved output)
    tgrid_header = f"{'Threshold':>12}  {'Macro-F1':>10}  {'MEL Recall':>11}  {'Composite':>11}"
    threshold_grid = (
        "VARIANT 3 — Threshold Grid Search on Validation Set\n"
        f"{tgrid_header}\n"
        + "\n".join(threshold_rows)
        + f"\n\nBest threshold: t={best_t:.2f}  (composite={best_t_score:.4f})\n"
    )

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    results_path = results_dir / "ensemble_concat_v2_results.txt"

    content = (
        "STRATEGY A v2 — FEATURE CONCATENATION ENSEMBLE (3 MEL-AWARE VARIANTS)\n"
        "EfficientNet-B0 (1280-dim) + DINOv2 ViT-S/14 (384-dim) → 1664-dim MLP\n\n"
        "Configuration\n"
        f"  MLP      : Linear(1664→512)→BN→GELU→Dropout({args.dropout})\n"
        f"             → Linear(512→256)→BN→GELU→Dropout({args.dropout}) → Linear(256→7)\n"
        f"  Loss     : Focal (gamma={args.gamma})\n"
        f"  Optimiser: AdamW (lr={args.lr}, weight_decay={args.weight_decay})\n"
        f"  Scheduler: CosineAnnealingLR (T_max={args.epochs})\n"
        f"  Epochs   : {args.epochs}   Batch size: {args.batch_size}   Seed: {args.seed}\n"
        f"  MEL boost (Variant 1): ×{args.mel_boost}\n\n"
        + "\n".join(output_sections)
        + "\n\n"
        + threshold_grid
        + table
    )

    results_path.write_text(content, encoding="utf-8")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
