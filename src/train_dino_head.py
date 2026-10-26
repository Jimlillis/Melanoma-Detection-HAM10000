"""
Train a single linear head on frozen DINO ViT-S/16 features for 7-class HAM10000.

Imbalance strategy (both active simultaneously):
  - Selectable loss via --loss: focal (default), dice, or combined (0.5 focal + 0.5 dice)
  - All losses use inverse-frequency class weights from the train split
  - WeightedRandomSampler so every batch is approximately class-balanced

Best checkpoint is selected by macro-F1 on the validation split.
Final test evaluation prints:
  - 7-class confusion matrix
  - Per-class classification report
  - Highlighted Melanoma recall, macro-F1, and which loss was used

Run from project root:
  python src/train_dino_head.py \\
    --features-dir data/processed \\
    --output-dir outputs/dino_linear \\
    --loss focal   # or dice, combined
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
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
    """
    Multi-class Focal Loss with optional per-class alpha weights.

    FL(x, t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    ``p_t`` is derived directly from log-softmax so the focal modulation
    factor is always the raw class probability, independent of whether
    class weights are used.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        # register_buffer moves the weight tensor with .to(device) automatically
        self.register_buffer("weight", weight)  # shape (C,) or None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # p_t: probability assigned to the true class — computed from raw
        # log-softmax so it is unaffected by the class-weight scaling.
        log_softmax = F.log_softmax(logits, dim=1)                    # (B, C)
        log_pt = log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        pt = log_pt.exp()                                              # (B,)

        # Cross-entropy, optionally weighted by alpha_t
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")  # (B,)

        focal = (1.0 - pt) ** self.gamma * ce
        return focal.mean()


# ---------------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Soft multi-class Dice Loss.

    DL = 1 - mean_c( (2 * sum_b p_bc * t_bc + smooth) / (sum_b p_bc + sum_b t_bc + smooth) )

    where p_bc = softmax probability for class c on sample b,
    and   t_bc = one-hot indicator.

    Optional class weights (same inverse-frequency weights as FocalLoss):
    weighted mean over classes instead of simple mean.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.smooth = smooth
        self.register_buffer("weight", weight)  # shape (C,) or None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)                               # (B, C)
        one_hot = F.one_hot(targets, num_classes=num_classes).float()  # (B, C)

        # Sum over batch dimension → (C,)
        intersection = (probs * one_hot).sum(dim=0)
        union = probs.sum(dim=0) + one_hot.sum(dim=0)
        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (C,)

        loss_per_class = 1.0 - dice_per_class  # (C,)

        if self.weight is not None:
            return (loss_per_class * self.weight).sum() / self.weight.sum().clamp(min=1e-8)
        return loss_per_class.mean()


# ---------------------------------------------------------------------------
# Combined Loss (Focal + Dice, equal weighting)
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """0.5 * FocalLoss + 0.5 * DiceLoss."""

    def __init__(self, focal: FocalLoss, dice: DiceLoss) -> None:
        super().__init__()
        self.focal = focal
        self.dice = dice

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.focal(logits, targets) + 0.5 * self.dice(logits, targets)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_class_weights(
    labels: torch.Tensor, num_classes: int, device: torch.device
) -> torch.Tensor:
    """Inverse-frequency weights: w_c = N / (C * n_c), on ``device``."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = labels.numel() / (num_classes * counts)
    weights[counts == 0] = 0.0  # guard against absent classes
    return weights.to(device)


def make_weighted_sampler(
    labels: torch.Tensor, num_classes: int
) -> WeightedRandomSampler:
    """One sample weight per example; minority classes upsampled to ~50-50."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights_per_class = 1.0 / counts.clamp(min=1.0)
    sample_weights = weights_per_class[labels].double()
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# Same targets as train_efficientnet.py's _REBALANCE_TARGETS -- kept in sync
# deliberately, so a probe run using --rebalance-train applies literally the
# same train-split resampling scheme as the EfficientNet pipeline, on
# features instead of images. None = keep original count.
_REBALANCE_TARGETS: dict[str, int | None] = {
    "akiec": 1000, "bcc": 1000, "df": 1000, "vasc": 1000,
    "nv": 1300, "bkl": None, "mel": None,
}


def rebalance_features(
    X: torch.Tensor, y: torch.Tensor, label_map: dict[str, int], seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-class resampling of pre-extracted TRAIN features/labels, mirroring
    train_efficientnet.py's rebalance_train_split() but operating on tensor
    rows (by index) instead of DataFrame rows -- there is no image to
    re-load, so this is index resampling with the identical semantics:
    oversample (with replacement) classes below their target count,
    undersample (without replacement) classes above it, leave others as-is,
    then shuffle so classes aren't block-ordered for the DataLoader.
    """
    idx_to_class = {v: k for k, v in label_map.items()}
    rng = np.random.RandomState(seed)
    y_np = y.numpy()
    chosen: list[np.ndarray] = []
    for cls_idx in sorted(idx_to_class):
        cls_name = idx_to_class[cls_idx]
        target = _REBALANCE_TARGETS.get(cls_name)
        positions = np.where(y_np == cls_idx)[0]
        n = len(positions)
        if target is None or target == n:
            chosen.append(positions)
        elif n < target:
            extra = rng.choice(positions, size=target - n, replace=True)
            chosen.append(np.concatenate([positions, extra]))
        else:
            chosen.append(rng.choice(positions, size=target, replace=False))
    all_idx = np.concatenate(chosen)
    rng.shuffle(all_idx)
    idx_tensor = torch.from_numpy(all_idx).long()
    return X[idx_tensor], y[idx_tensor]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (mean_loss, y_true, y_pred) over the full loader."""
    model.eval()
    losses: list[float] = []
    all_y: list[int] = []
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
    """
    Build a stable text report from classification_report(output_dict=True).
    This avoids sporadic duplicated lines seen in some Windows terminals.
    """
    rep = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    lines = ["              precision    recall  f1-score   support", ""]
    for name in class_names:
        row = rep[name]
        lines.append(
            f"{name:>12} {row['precision']:>10.2f} {row['recall']:>9.2f} {row['f1-score']:>9.2f} {int(row['support']):>9d}"
        )

    lines.append("")
    lines.append(
        f"{'accuracy':>12} {'':>10} {'':>9} {rep['accuracy']:>9.2f} {int(rep['macro avg']['support']):>9d}"
    )
    for agg in ("macro avg", "weighted avg"):
        row = rep[agg]
        lines.append(
            f"{agg:>12} {row['precision']:>10.2f} {row['recall']:>9.2f} {row['f1-score']:>9.2f} {int(row['support']):>9d}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train linear head on DINO ViT-S/16 features (7-class HAM10000)"
    )
    p.add_argument("--features-dir", type=Path, default=Path("data/processed"))
    p.add_argument(
        "--features-prefix",
        type=str,
        default="dino_feats",
        help="Filename prefix for feature files (default: dino_feats). "
             "Use dinov2_feats for DINOv2 features.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs/dino_linear"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma")
    p.add_argument(
        "--loss",
        choices=["focal", "dice", "combined"],
        default="focal",
        help="Loss function: focal (default), dice, or combined (0.5 focal + 0.5 dice)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--rebalance-train", action="store_true",
                   help="Apply train_efficientnet.py-style train-split rebalancing "
                        "(oversample akiec/bcc/df/vasc to 1000, undersample nv to "
                        "1300, leave bkl/mel unchanged) to the cached TRAIN features "
                        "before training. Off by default -- old behaviour unchanged.")
    p.add_argument("--no-sampler", action="store_true",
                   help="Disable the WeightedRandomSampler; use a plain shuffled "
                        "DataLoader instead. Off by default -- old behaviour unchanged.")
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
    run_start = time.time()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load pre-extracted features -----------------------------------------
    print("Loading features …")
    train_data = torch.load(
        args.features_dir / f"{args.features_prefix}_train.pt", map_location="cpu", weights_only=False
    )
    val_data = torch.load(
        args.features_dir / f"{args.features_prefix}_val.pt", map_location="cpu", weights_only=False
    )
    test_data = torch.load(
        args.features_dir / f"{args.features_prefix}_test.pt", map_location="cpu", weights_only=False
    )

    label_map: dict[str, int] = train_data["label_map"]
    idx_to_class: dict[int, str] = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    class_names = [idx_to_class[i] for i in range(num_classes)]
    feat_dim: int = train_data["features"].shape[1]

    print(f"Feature dim : {feat_dim}")
    print(f"Classes     : {num_classes}  ->  {label_map}")

    X_train, y_train = train_data["features"], train_data["labels"].long()
    X_val,   y_val   = val_data["features"],   val_data["labels"].long()
    X_test,  y_test  = test_data["features"],  test_data["labels"].long()

    if args.rebalance_train:
        before = {idx_to_class[i]: int((y_train == i).sum()) for i in range(num_classes)}
        X_train, y_train = rebalance_features(X_train, y_train, label_map, args.seed)
        after = {idx_to_class[i]: int((y_train == i).sum()) for i in range(num_classes)}
        print(f"Train-split rebalancing applied. Before: {before}")
        print(f"                                 After:  {after}")

    # --- DataLoaders ---------------------------------------------------------
    if args.no_sampler:
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print("WeightedRandomSampler : disabled (--no-sampler); plain shuffle")
    else:
        sampler = make_weighted_sampler(y_train, num_classes)
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=args.batch_size,
            sampler=sampler,          # shuffle must be False when sampler is given
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

    # --- Loss & model --------------------------------------------------------
    class_weights = compute_class_weights(y_train, num_classes, device)

    weight_display = {idx_to_class[i]: f"{w:.3f}" for i, w in enumerate(class_weights.cpu().tolist())}
    print(f"Class weights (alpha): {weight_display}")

    _focal = FocalLoss(gamma=args.gamma, weight=class_weights)
    _dice  = DiceLoss(weight=class_weights)
    if args.loss == "focal":
        criterion = _focal
    elif args.loss == "dice":
        criterion = _dice
    else:  # combined
        criterion = CombinedLoss(_focal, _dice)

    print(f"Loss function : {args.loss}")

    model = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    mel_idx = label_map.get("mel", -1)

    # --- Training loop -------------------------------------------------------
    best_val_f1 = -1.0
    best_path = out_dir / "best_dino_head.pth"
    log_csv_path = out_dir / "training_log.csv"
    print(f"Per-epoch log : {log_csv_path}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for feats, labels in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False
        ):
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), labels)
            loss.backward()     # calculate the gradient
            optimizer.step()    # update the model parameters student.weights = student.weights - lr × gradient
            running_loss += loss.item()
            n_batches += 1

        val_loss, y_true_val, y_pred_val = evaluate(model, val_loader, device, criterion)

        val_macro_f1 = f1_score(y_true_val, y_pred_val, average="macro", zero_division=0)
        val_mel_recall = (
            recall_score(
                y_true_val, y_pred_val, labels=[mel_idx], average="micro", zero_division=0
            )
            if mel_idx >= 0
            else float("nan")
        )

        print(
            f"Epoch {epoch:3d}  "
            f"train_loss={running_loss / n_batches:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_macro_f1={val_macro_f1:.4f}  "
            f"val_mel_recall={val_mel_recall:.4f}"
        )

        elapsed = time.time() - run_start
        is_new_csv = not log_csv_path.exists()
        with open(log_csv_path, "a", newline="", encoding="utf-8") as _f:
            _w = csv.writer(_f)
            if is_new_csv:
                _w.writerow(["epoch", "train_loss", "val_loss", "val_macro_f1",
                             "val_mel_recall", "elapsed_seconds"])
            _w.writerow([epoch, f"{running_loss / n_batches:.6f}", f"{val_loss:.6f}",
                         f"{val_macro_f1:.6f}", f"{val_mel_recall:.6f}", f"{elapsed:.2f}"])

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_map": label_map,
                    "feat_dim": feat_dim,
                    "epoch": epoch,
                    "val_macro_f1": val_macro_f1,
                    "val_mel_recall": val_mel_recall,
                    "seed": args.seed,
                    "loss_name": args.loss,
                    "wall_clock_seconds": elapsed,
                    "rebalance_train": args.rebalance_train,
                    "used_sampler": not args.no_sampler,
                },
                best_path,
            )
            print(f"  -> New best checkpoint (val macro_f1={best_val_f1:.4f})")

    print(f"\nTraining complete.  Best val macro_f1 = {best_val_f1:.4f}")

    # --- Test evaluation -----------------------------------------------------
    print("\n" + "=" * 64)
    print("TEST SET EVALUATION")
    print("=" * 64)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    _, y_true_test, y_pred_test = evaluate(model, test_loader, device, criterion)

    # Confusion matrix
    cm = confusion_matrix(y_true_test, y_pred_test, labels=list(range(num_classes)))
    print("\nConfusion Matrix  (rows = true class, cols = predicted class):")
    col_header = "         " + "  ".join(f"{c:>6}" for c in class_names)
    print(col_header)
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>8} " + "  ".join(f"{v:>6}" for v in row))

    # Per-class report
    print("\nPer-class Classification Report:")
    report = format_classification_report(y_true_test, y_pred_test, class_names)
    print(report)

    # Highlighted summary
    macro_f1 = f1_score(y_true_test, y_pred_test, average="macro", zero_division=0)
    mel_recall = (
        recall_score(
            y_true_test, y_pred_test, labels=[mel_idx], average="micro", zero_division=0
        )
        if mel_idx >= 0
        else float("nan")
    )

    print("=" * 64)
    print(f"  Loss Function   : {args.loss}")
    print(f"  Macro-F1        : {macro_f1:.4f}")
    print(
        f"  Melanoma Recall : {mel_recall:.4f}"
        f"  (class '{idx_to_class.get(mel_idx, '?')}', index {mel_idx})"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()
