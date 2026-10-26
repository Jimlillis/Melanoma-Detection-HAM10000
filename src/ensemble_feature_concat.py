"""
Strategy A — Feature Concatenation Ensemble.

Concatenates frozen EfficientNet-B0 (1280-dim) and DINOv2 ViT-S/14 (384-dim)
penultimate-layer features into a single 1664-dim representation, then trains a
small MLP classifier on top.

No backbone re-runs are needed at train time — features must be pre-extracted:
  python src/extract_efficientnet_features.py  ...  (→ data/processed/efficientnet/)
  python src/extract_features_v2.py            ...  (→ data/processed/dinov2/)

MLP architecture:
  Linear(1664 → 512) → BatchNorm1d → GELU → Dropout(0.3)
  Linear(512  → 256) → BatchNorm1d → GELU → Dropout(0.3)
  Linear(256  → 7)

Training:
  Loss      : Focal Loss (gamma=2) with inverse-frequency class weights
  Optimiser : AdamW (lr=1e-3, weight_decay=1e-4)
  Scheduler : CosineAnnealingLR (T_max = epochs)
  Epochs    : 50   |   Batch size: 128
  Best model: highest val Macro-F1

Outputs:
  outputs/ensemble_concat/best_concat_head.pth
  results/ensemble_concat_results.txt

Comparison table (hard-coded baselines from prior experiments):
  EfficientNet alone         Macro-F1=0.692  MEL Recall=0.713
  DINOv2 alone               Macro-F1=0.545  MEL Recall=0.689
  Soft Ensemble B (0.50/0.50)Macro-F1=0.696  MEL Recall=0.731
  Weighted Ensemble C best-F1Macro-F1=0.696  MEL Recall=0.719
  Feature Concat A (1664-MLP) measured here

Run from project root:
  python src/ensemble_feature_concat.py
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
# Focal Loss  (identical to train_dino_head.py)
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
        pt  = log_pt.exp()
        ce  = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------------
# MLP classifier
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
            # Block 1
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            # Block 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            # Output
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
    counts  = torch.bincount(labels, minlength=num_classes).float()
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
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (mean_loss, y_true, y_pred)."""
    model.eval()
    losses: list[float] = []
    all_y:    list[int] = []
    all_pred: list[int] = []

    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        logits = model(feats)
        losses.append(criterion(logits, labels).item())
        all_y.extend(labels.cpu().tolist())
        all_pred.extend(logits.argmax(dim=1).cpu().tolist())

    return float(np.mean(losses)), np.array(all_y), np.array(all_pred)


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


def load_and_concat(
    effnet_dir: Path,
    dinov2_dir: Path,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """
    Load EfficientNet and DINOv2 features for *split* (train/val/test),
    verify label consistency, and concatenate along the feature dimension.

    Returns (concat_feats, labels, label_map).
    """
    eff  = torch.load(effnet_dir / f"effnet_feats_{split}.pt",  map_location="cpu", weights_only=False)
    dino = torch.load(dinov2_dir / f"dinov2_feats_{split}.pt",  map_location="cpu", weights_only=False)

    if not torch.equal(eff["labels"], dino["labels"]):
        raise ValueError(
            f"Label mismatch between EfficientNet and DINOv2 for split '{split}'. "
            "Ensure both were extracted with the same seed and CSV."
        )
    if eff["label_map"] != dino["label_map"]:
        raise ValueError("label_map mismatch between feature files.")

    concat = torch.cat([eff["features"], dino["features"]], dim=1)  # (N, 1664)
    return concat, eff["labels"].long(), eff["label_map"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Strategy A — Feature Concatenation Ensemble (EfficientNet + DINOv2 → MLP)"
    )
    p.add_argument(
        "--effnet-feats-dir",
        type=Path,
        default=Path("data/processed/efficientnet"),
        help="Directory containing effnet_feats_{train,val,test}.pt",
    )
    p.add_argument(
        "--dinov2-feats-dir",
        type=Path,
        default=Path("data/processed/dinov2"),
        help="Directory containing dinov2_feats_{train,val,test}.pt",
    )
    p.add_argument("--output-dir",  type=Path, default=Path("outputs/ensemble_concat"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--epochs",      type=int,  default=50)
    p.add_argument("--batch-size",  type=int,  default=128)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight-decay",type=float, default=1e-4)
    p.add_argument("--gamma",       type=float, default=2.0, help="Focal loss gamma")
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--num-workers", type=int,  default=4)
    p.add_argument("--seed",        type=int,  default=42)
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

    out_dir     = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Load & concatenate features ------------------------------------------
    print("Loading and concatenating features …")
    X_train, y_train, label_map = load_and_concat(
        args.effnet_feats_dir, args.dinov2_feats_dir, "train"
    )
    X_val,   y_val,   _         = load_and_concat(
        args.effnet_feats_dir, args.dinov2_feats_dir, "val"
    )
    X_test,  y_test,  _         = load_and_concat(
        args.effnet_feats_dir, args.dinov2_feats_dir, "test"
    )

    in_dim      = X_train.shape[1]   # 1664
    num_classes = len(label_map)
    idx_to_class: dict[int, str] = {v: k for k, v in label_map.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]
    mel_idx     = label_map.get("mel", -1)

    print(f"Feature dim : {in_dim}  (EfficientNet 1280 + DINOv2 384)")
    print(f"Classes     : {num_classes}  ->  {label_map}")
    print(f"Train: {len(y_train)}   Val: {len(y_val)}   Test: {len(y_test)}")

    # --- DataLoaders ----------------------------------------------------------
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
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --- Model, loss, optimiser, scheduler ------------------------------------
    class_weights = compute_class_weights(y_train, num_classes, device)
    weight_display = {idx_to_class[i]: f"{w:.3f}" for i, w in enumerate(class_weights.cpu().tolist())}
    print(f"Class weights (alpha): {weight_display}")

    model     = ConcatMLP(in_dim, num_classes, dropout=args.dropout).to(device)
    criterion = FocalLoss(gamma=args.gamma, weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(
        f"\nMLP: Linear({in_dim}->512)->BN->GELU->Dropout({args.dropout})"
        f" -> Linear(512->256)->BN->GELU->Dropout({args.dropout}) -> Linear(256->{num_classes})"
    )
    print(f"Loss: Focal (gamma={args.gamma})  |  Optimiser: AdamW (lr={args.lr}, wd={args.weight_decay})")
    print(f"Scheduler: CosineAnnealingLR (T_max={args.epochs})")
    print(f"Epochs: {args.epochs}  |  Batch size: {args.batch_size}\n")

    # --- Training loop --------------------------------------------------------
    best_val_f1 = -1.0
    best_path   = out_dir / "best_concat_head.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches    = 0

        for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches    += 1

        scheduler.step()

        val_loss, y_true_val, y_pred_val = evaluate(model, val_loader, device, criterion)
        val_macro_f1   = f1_score(y_true_val, y_pred_val, average="macro", zero_division=0)
        val_mel_recall = (
            recall_score(y_true_val, y_pred_val, labels=[mel_idx], average="micro", zero_division=0)
            if mel_idx >= 0 else float("nan")
        )

        print(
            f"Epoch {epoch:3d}  "
            f"train_loss={running_loss / n_batches:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_macro_f1={val_macro_f1:.4f}  "
            f"val_mel_recall={val_mel_recall:.4f}"
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_map":  label_map,
                    "in_dim":     in_dim,
                    "num_classes":num_classes,
                    "dropout":    args.dropout,
                    "epoch":      epoch,
                    "val_macro_f1": val_macro_f1,
                },
                best_path,
            )
            print(f"  -> New best checkpoint (val macro_f1={best_val_f1:.4f})")

    print(f"\nTraining complete.  Best val macro_f1 = {best_val_f1:.4f}")

    # --- Test evaluation ------------------------------------------------------
    print("\n" + "=" * 68)
    print("TEST SET EVALUATION")
    print("=" * 68)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    _, y_true_test, y_pred_test = evaluate(model, test_loader, device, criterion)

    # Confusion matrix
    cm = confusion_matrix(y_true_test, y_pred_test, labels=list(range(num_classes)))
    print("\nConfusion Matrix  (rows = true, cols = predicted):")
    col_header = "         " + "  ".join(f"{c:>6}" for c in class_names)
    print(col_header)
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>8} " + "  ".join(f"{v:>6}" for v in row))

    # Per-class report
    print("\nPer-class Classification Report:")
    report_str = format_classification_report(y_true_test, y_pred_test, class_names)
    print(report_str)

    # Summary metrics
    macro_f1   = f1_score(y_true_test, y_pred_test, average="macro",  zero_division=0)
    mel_recall = (
        recall_score(y_true_test, y_pred_test, labels=[mel_idx], average="micro", zero_division=0)
        if mel_idx >= 0 else float("nan")
    )

    print("=" * 68)
    print(f"  Macro-F1        : {macro_f1:.4f}")
    print(
        f"  Melanoma Recall : {mel_recall:.4f}"
        f"  (class '{idx_to_class.get(mel_idx, '?')}', index {mel_idx})"
    )
    print("=" * 68)

    # --- Comparison table (with hard-coded baselines) -------------------------
    baselines = [
        ("EfficientNet (alone)",              0.692, 0.713),
        ("DINOv2 (alone)",                    0.545, 0.689),
        ("Soft Ensemble B (0.50/0.50)",        0.696, 0.731),
        ("Weighted Ensemble C (0.60/0.40)",    0.696, 0.719),
        ("Feature Concat A (1664-dim MLP)",    macro_f1, mel_recall),
    ]

    sep  = "-" * 62
    hdr  = f"{'Model':<38} {'Macro-F1':>10} {'MEL Recall':>11}"
    rows = "\n".join(
        f"{name:<38} {f1:>10.3f} {mel:>11.3f}"
        for name, f1, mel in baselines
    )

    table = f"""
{'=' * 62}
ENSEMBLE COMPARISON — HAM10000 Test Set
{sep}
{hdr}
{sep}
{rows}
{sep}
Last row: Feature Concat A (this script)
{'=' * 62}
"""
    print(table)

    # --- Save results ---------------------------------------------------------
    results_path = results_dir / "ensemble_concat_results.txt"
    cm_lines = [
        "Confusion Matrix  (rows = true, cols = predicted):",
        "         " + "  ".join(f"{c:>6}" for c in class_names),
    ]
    for i, row in enumerate(cm):
        cm_lines.append(f"{class_names[i]:>8} " + "  ".join(f"{v:>6}" for v in row))

    content = f"""STRATEGY A — FEATURE CONCATENATION ENSEMBLE
EfficientNet-B0 (1280-dim) + DINOv2 ViT-S/14 (384-dim) → 1664-dim MLP

Configuration
  MLP      : Linear(1664→512)→BN→GELU→Dropout({args.dropout})
             → Linear(512→256)→BN→GELU→Dropout({args.dropout}) → Linear(256→7)
  Loss     : Focal (gamma={args.gamma}) + inverse-frequency class weights
  Optimiser: AdamW (lr={args.lr}, weight_decay={args.weight_decay})
  Scheduler: CosineAnnealingLR (T_max={args.epochs})
  Epochs   : {args.epochs}   Batch size: {args.batch_size}   Seed: {args.seed}
  Best val Macro-F1: {best_val_f1:.4f}  (epoch {ckpt['epoch']})
  Checkpoint: {best_path}

{chr(10).join(cm_lines)}

Per-class Classification Report:
{report_str}

{'=' * 68}
  Macro-F1        : {macro_f1:.4f}
  Melanoma Recall : {mel_recall:.4f}
{'=' * 68}
{table}"""

    results_path.write_text(content, encoding="utf-8")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
