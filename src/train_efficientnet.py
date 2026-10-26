"""
Train EfficientNet-B0 (ImageNet pre-trained) for 7-class HAM10000 classification.

Three enhancements over the plain baseline:
  1. Train-split rebalancing — minority classes oversampled to 1000 images,
     nv downsampled to 1300, bkl/mel kept at their original counts.
     Val and test are never touched.
  2. Two-phase training — Phase 1: backbone frozen, head only (15 epochs).
                          Phase 2: all layers unfrozen (15 epochs, lr=1e-4).
     Best checkpoint tracked by val macro-F1 across both phases.
  3. Test-time augmentation — 5 deterministic views per image (original,
     h-flip, v-flip, colour-jitter, 20° rotation); softmax probabilities
     averaged before predicting.

Output format is identical to train_dino_head.py for direct comparison.

Run from project root:
  python src/train_efficientnet.py \\
    --metadata-csv data/HAM10000_metadata.csv \\
    --images-dir data \\
    --output-dir outputs/efficientnet_7class
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import Config
from dataset import SkinLesionDataset, build_transforms, lesion_level_split
from helpers import get_device, resolve_image_path, set_seed
from model import build_efficientnet_b0
# Reuse the exact same loss implementations as the DINO/BiomedCLIP linear
# probes (not reimplemented here) so loss-function comparisons are apples
# to apples across both training pipelines.
from train_dino_head import CombinedLoss, DiceLoss, FocalLoss


# ---------------------------------------------------------------------------
# Rebalancing targets  (applied to train split only)
# ---------------------------------------------------------------------------

# None = keep original count; int = resample to this exact count
_REBALANCE_TARGETS: dict[str, int | None] = {
    "akiec": 1000,   # oversample  (≈229 in train)
    "bcc":   1000,   # oversample  (≈360 in train)
    "df":    1000,   # oversample  (≈81  in train)
    "vasc":  1000,   # oversample  (≈99  in train)
    "nv":    1300,   # undersample (≈4694 in train)
    "bkl":   None,   # keep original (≈769)
    "mel":   None,   # keep original (≈779)
}


# ---------------------------------------------------------------------------
# Metadata loading — 7-class, compatible with extract_features.py split
# ---------------------------------------------------------------------------

def load_multiclass_table(
    csv_path: Path,
    col_image_id: str,
    col_lesion_id: str,
    col_diagnosis: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Read HAM10000 CSV and add ``_binary_label`` = 7-class integer label.
    Column named ``_binary_label`` for compatibility with ``lesion_level_split``.
    Alphabetical sort: akiec=0, bcc=1, bkl=2, df=3, mel=4, nv=5, vasc=6.
    """
    df = pd.read_csv(csv_path)
    missing = {col_image_id, col_lesion_id, col_diagnosis} - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    classes = sorted(df[col_diagnosis].astype(str).unique().tolist())
    label_map = {c: i for i, c in enumerate(classes)}
    df = df.copy()
    df["_binary_label"] = df[col_diagnosis].astype(str).map(label_map).astype(int)
    return df, label_map


# ---------------------------------------------------------------------------
# Train-split rebalancing
# ---------------------------------------------------------------------------

def rebalance_train_split(
    df: pd.DataFrame,
    col_diagnosis: str,
    seed: int,
) -> pd.DataFrame:
    """
    Per-class resampling applied only to the train DataFrame.

    For classes with a target count in ``_REBALANCE_TARGETS``:
      - count < target: oversample with replacement.  The same image_id can
        appear multiple times; each appearance gets a fresh random augmentation
        at DataLoader time.
      - count > target: undersample without replacement.
    Classes mapped to None (bkl, mel) are passed through unchanged.
    Any class absent from the dict is also kept intact (forward-compatible).
    """
    parts: list[pd.DataFrame] = []
    for dx, group in df.groupby(col_diagnosis):
        target = _REBALANCE_TARGETS.get(str(dx))
        n = len(group)
        if target is None or target == n:
            parts.append(group)
        elif n < target:
            extra = group.sample(n=target - n, replace=True, random_state=seed)
            parts.append(pd.concat([group, extra], ignore_index=True))
        else:
            parts.append(group.sample(n=target, replace=False, random_state=seed))

    rebalanced = pd.concat(parts, ignore_index=True)
    # Shuffle so that classes are not block-ordered in the DataLoader
    return rebalanced.sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Class weights — same formula as train.py
# ---------------------------------------------------------------------------

def compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Inverse-frequency weights: w_c = N / (C * n_c)."""
    counts = np.bincount(labels)
    n_classes = len(counts)
    weights = len(labels) / (n_classes * counts.astype(float))
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

_NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def build_train_transforms(image_size: int) -> transforms.Compose:
    """
    Stronger augmentation for training: flips + rotation + colour jitter.
    Replaces the plain ``build_transforms(is_train=True)`` which has no rotation.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(**_NORM),
    ])


def build_tta_transforms(image_size: int) -> list[transforms.Compose]:
    """
    Five TTA views applied at test time:
      0: original (no augmentation)
      1: horizontal flip  (p=1.0 → deterministic)
      2: vertical flip    (p=1.0 → deterministic)
      3: colour jitter
      4: 20° rotation     (degrees=(20,20) → always exactly +20°)
    """
    resize_norm = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(**_NORM),
    ]
    return [
        # 0 — original
        transforms.Compose(resize_norm),
        # 1 — h-flip
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(**_NORM),
        ]),
        # 2 — v-flip
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(**_NORM),
        ]),
        # 3 — colour jitter
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(**_NORM),
        ]),
        # 4 — 20° rotation (exact)
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=(20, 20)),
            transforms.ToTensor(),
            transforms.Normalize(**_NORM),
        ]),
    ]


# ---------------------------------------------------------------------------
# TTA Dataset
# ---------------------------------------------------------------------------

class TTASkinLesionDataset(Dataset):
    """
    Applies T transforms to each image and returns a stacked tensor of shape
    (T, C, H, W) alongside the integer label.  T is len(tta_transforms).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        images_dir: Path,
        col_image_id: str,
        tta_transforms: list[transforms.Compose],
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.col_image_id = col_image_id
        self.tta_transforms = tta_transforms

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image_id = str(row[self.col_image_id])
        label = int(row["_binary_label"])
        img = Image.open(resolve_image_path(self.images_dir, image_id)).convert("RGB")
        views = torch.stack([t(img) for t in self.tta_transforms])  # (T, C, H, W)
        return views, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Single-view evaluation for validation. Same signature as train_dino_head.py."""
    
    model.eval() # dropout OFF 
    '''
    # if you want to use dropout, you can do it like this:
    for layer in model.modules():
    if isinstance(layer, nn.Dropout):
        layer.train()  # dropout ON μόνο στα dropout layers

    # Και τρέχεις 10-20 φορές:
    predictions = [model(x) for _ in range(10)]
    final = torch.stack(predictions).mean(dim=0)
    '''
    
    losses: list[float] = []
    all_y: list[int] = []
    all_pred: list[int] = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(criterion(logits, y).item())
        all_y.extend(y.cpu().tolist())
        all_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return float(np.mean(losses)), np.array(all_y), np.array(all_pred)


@torch.no_grad()
def evaluate_tta(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    TTA evaluation for the test set.

    Each batch item is (B, T, C, H, W).  The T views are run through the model
    one at a time (memory-safe), softmax probabilities are averaged across views,
    and the winning class is taken via argmax.
    """
    model.eval()
    all_y: list[int] = []
    all_pred: list[int] = []
    for views, labels in tqdm(loader, desc="TTA inference"):
        B, T, C, H, W = views.shape
        probs_list: list[torch.Tensor] = []
        for t in range(T):
            view_t = views[:, t].to(device)          # (B, C, H, W)
            logits_t = model(view_t)                  # (B, num_classes)
            probs_list.append(torch.softmax(logits_t, dim=1))  # (B, num_classes)
        # Stack → (T, B, num_classes), mean over T → (B, num_classes)
        avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)
        pred = avg_probs.argmax(dim=1)
        all_y.extend(labels.tolist())
        all_pred.extend(pred.cpu().tolist())
    return np.array(all_y), np.array(all_pred)


def evaluate_tta_mc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mc_passes: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    TTA + MC Dropout evaluation for the test set.

    Inference mode:
      - model.eval()          → BatchNorm uses running stats (no leakage)
      - Dropout layers → train()  → stochastic activations enabled

    For each image: loop over T TTA views; for each view run ``mc_passes``
    stochastic forward passes.  All T × mc_passes softmax vectors are averaged
    to obtain the final prediction.

    Uncertainty is estimated as the mean *variance* across the T × mc_passes
    probability vectors for each image, then averaged over the test set.
    Higher variance = the model is less sure about that image.

    Returns
    -------
    y_true       : (N,) ground-truth class indices
    y_pred       : (N,) predicted class indices
    mean_unc     : scalar — average per-image uncertainty over the test set
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    all_y: list[int] = []
    all_pred: list[int] = []
    all_uncertainty: list[float] = []

    n_views = None  # filled on first batch for the progress description
    with torch.no_grad():
        for views, labels in tqdm(loader, desc=f"TTA+MC Dropout ({mc_passes} passes/view)"):
            B, T, C, H, W = views.shape
            if n_views is None:
                n_views = T

            # Collect all T × mc_passes softmax vectors on CPU to save VRAM
            all_probs: list[torch.Tensor] = []
            for t in range(T):
                view_t = views[:, t].to(device)          # (B, C, H, W)
                for _ in range(mc_passes):
                    logits = model(view_t)                # (B, num_classes)
                    all_probs.append(
                        torch.softmax(logits, dim=1).cpu()
                    )

            # stacked: (K, B, num_classes)  where K = T × mc_passes
            stacked = torch.stack(all_probs, dim=0)
            avg_probs = stacked.mean(dim=0)              # (B, num_classes)
            pred = avg_probs.argmax(dim=1)               # (B,)

            # Per-image uncertainty: mean variance across K prediction vectors
            # var(K, B, C) → mean over C → (B,)
            var_per_image = stacked.var(dim=0).mean(dim=1)  # (B,)

            all_y.extend(labels.tolist())
            all_pred.extend(pred.tolist())
            all_uncertainty.extend(var_per_image.tolist())

    return np.array(all_y), np.array(all_pred), float(np.mean(all_uncertainty))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    desc: str,
) -> float:
    """One training epoch; returns mean loss over batches."""
    model.train()
    running_loss = 0.0
    n_batches = 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()     # calculate the gradient
        optimizer.step()    # update the model parameters student.weights = student.weights - lr × gradient
        running_loss += loss.item()
        n_batches += 1
    return running_loss / n_batches


def append_csv_row(log_csv_path: Path, row: list) -> None:
    """Append one row to the per-run training-log CSV, writing the header
    first if the file doesn't exist yet. Opened/closed per call (not kept
    open for the whole run) so a row is durably on disk the instant an
    epoch finishes -- important for an unattended run that might be killed
    mid-training."""
    is_new = not log_csv_path.exists()
    with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                ["phase", "epoch", "train_loss", "val_loss",
                 "val_macro_f1", "val_mel_recall", "elapsed_seconds"]
            )
        writer.writerow(row)


def val_step(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label_map: dict[str, int],
    mel_idx: int,
    best_val_f1: float,
    best_path: Path,
    epoch_tag: str,
    *,
    phase: int,
    epoch: int,
    train_loss: float,
    seed: int,
    split_seed: int,
    loss_name: str,
    run_start: float,
    log_csv_path: Path,
) -> float:
    """
    Evaluate on val, print per-epoch metrics, append a per-epoch CSV log row
    (every epoch, not just improvements), and save checkpoint if new best.
    Returns updated best_val_f1.
    """
    val_loss, y_true_val, y_pred_val = evaluate(model, val_loader, device, criterion)
    val_macro_f1 = f1_score(y_true_val, y_pred_val, average="macro", zero_division=0)
    val_mel_recall = (
        recall_score(y_true_val, y_pred_val, labels=[mel_idx], average="micro", zero_division=0)
        if mel_idx >= 0 else float("nan")
    )
    elapsed = time.time() - run_start
    print(
        f"{epoch_tag}  "
        f"val_loss={val_loss:.4f}  "
        f"val_macro_f1={val_macro_f1:.4f}  "
        f"val_mel_recall={val_mel_recall:.4f}"
    )
    append_csv_row(
        log_csv_path,
        [phase, epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
         f"{val_macro_f1:.6f}", f"{val_mel_recall:.6f}", f"{elapsed:.2f}"],
    )
    if val_macro_f1 > best_val_f1:
        best_val_f1 = val_macro_f1
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "label_map": label_map,
                "val_macro_f1": val_macro_f1,
                "val_mel_recall": val_mel_recall,
                "epoch": epoch,
                "phase": phase,
                "seed": seed,
                "split_seed": split_seed,
                "loss_name": loss_name,
                "wall_clock_seconds": elapsed,
            },
            best_path,
        )
        print(f"  -> New best checkpoint (val macro_f1={best_val_f1:.4f})")
    return best_val_f1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EfficientNet-B0 (7-class) with rebalancing, two-phase training, and TTA"
    )
    p.add_argument("--metadata-csv", type=Path, default=None)
    p.add_argument("--images-dir",   type=Path, default=None)
    p.add_argument("--output-dir",   type=Path, default=Path("outputs/efficientnet_7class"))
    p.add_argument("--phase1-epochs", type=int,   default=15,
                   help="Epochs with frozen backbone (head-only training)")
    p.add_argument("--phase2-epochs", type=int,   default=15,
                   help="Epochs with all layers unfrozen (full fine-tune)")
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--phase1-lr",     type=float, default=1e-3,
                   help="Learning rate for Phase 1 (classifier head only)")
    p.add_argument("--phase2-lr",     type=float, default=1e-4,
                   help="Learning rate for Phase 2 (full network)")
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--seed",          type=int,   default=None)
    p.add_argument("--split-seed",    type=int,   default=None,
                   help="Seed for the lesion-level train/val/test split, decoupled "
                        "from --seed (training RNG). Defaults to --seed's value if "
                        "omitted, i.e. identical to old behaviour. Set this explicitly "
                        "(to the shared seed, normally 42) when comparing multiple "
                        "--seed values against a FIXED test set -- otherwise each "
                        "--seed also silently reshuffles which images are test images, "
                        "and a model's 'test' set can overlap with its own train/val.")
    p.add_argument("--num-workers",   type=int,   default=None)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument(
        "--loss",
        choices=["ce", "focal", "dice", "combined"],
        default="ce",
        help="Loss function: ce (weighted cross-entropy, default -- reproduces "
             "the original behaviour exactly), focal, dice, or combined "
             "(0.5 focal + 0.5 dice). Uses the same FocalLoss/DiceLoss/CombinedLoss "
             "classes as train_dino_head.py.",
    )
    p.add_argument("--gamma", type=float, default=2.0,
                    help="Focal loss gamma (used when --loss is focal or combined)")
    p.add_argument("--log-csv", type=Path, default=None,
                   help="Per-epoch CSV log path (default: <output-dir>/training_log.csv)")
    p.add_argument("--mc-passes", type=int, default=10,
                   help="Number of stochastic MC Dropout forward passes per TTA view "
                        "(default: 10 -> 5 views x 10 passes = 50 predictions per image)")
    p.add_argument("--no-mc", action="store_true",
                   help="Disable MC Dropout; run TTA-only inference (original behaviour)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Defensive: this repo's console is a Greek-locale (cp1253) Windows
    # console that cannot encode arrows/multiplication signs -- a bare
    # print() of one crashes the whole run with UnicodeEncodeError. All
    # print() calls below are kept plain-ASCII; this is a second line of
    # defense in case that's ever missed.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    cfg = Config()
    if args.metadata_csv is not None:
        cfg.metadata_csv = args.metadata_csv
    if args.images_dir is not None:
        cfg.images_dir = args.images_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    set_seed(cfg.seed)
    device = get_device()
    run_start = time.time()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Metadata & split (identical to extract_features.py) -----------------
    print("Loading metadata …")
    df, label_map = load_multiclass_table(
        cfg.metadata_csv, cfg.col_image_id, cfg.col_lesion_id, cfg.col_diagnosis,
    )
    idx_to_class: dict[int, str] = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    class_names = [idx_to_class[i] for i in range(num_classes)]
    print(f"Classes     : {num_classes}  ->  {label_map}")

    # The lesion-level split and the training RNG (weight init, batch order,
    # augmentation) are deliberately separate knobs. --split-seed defaults to
    # cfg.seed so every existing invocation is unaffected; it exists so a
    # seed-variance study can vary --seed (training randomness) while pinning
    # --split-seed (which images are train/val/test) -- otherwise a different
    # --seed silently produces a *different* train/val/test partition, and a
    # model's "test" set can overlap heavily with what it was itself trained
    # on. (Discovered exactly this way: seed=1337 without this fix leaked 86%
    # of the seed=42 test set into seed=1337's own train+val split.)
    split_seed = args.split_seed if args.split_seed is not None else cfg.seed
    df_train, df_val, df_test = lesion_level_split(
        df, cfg.col_lesion_id, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, split_seed,
    )
    print(f"Split sizes — train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")

    # --- Rebalance train split (val and test are untouched) ------------------
    df_train_rb = rebalance_train_split(df_train, cfg.col_diagnosis, cfg.seed)
    before = df_train[cfg.col_diagnosis].value_counts().sort_index().to_dict()
    after  = df_train_rb[cfg.col_diagnosis].value_counts().sort_index().to_dict()
    print(f"Train counts before rebalancing: {before}")
    print(f"Train counts after  rebalancing: {after}")

    # --- DataLoaders ---------------------------------------------------------
    train_loader = DataLoader(
        SkinLesionDataset(
            df_train_rb, cfg.images_dir, cfg.col_image_id,
            transform=build_train_transforms(cfg.image_size),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        SkinLesionDataset(
            df_val, cfg.images_dir, cfg.col_image_id,
            transform=build_transforms(cfg.image_size, is_train=False),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    tta_loader = DataLoader(
        TTASkinLesionDataset(
            df_test, cfg.images_dir, cfg.col_image_id,
            tta_transforms=build_tta_transforms(cfg.image_size),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --- Loss & model --------------------------------------------------------
    # Weights from the rebalanced split -- what the model actually trains on
    class_weights = compute_class_weights(df_train_rb["_binary_label"].values, device)

    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif args.loss == "focal":
        criterion = FocalLoss(gamma=args.gamma, weight=class_weights).to(device)
    elif args.loss == "dice":
        criterion = DiceLoss(weight=class_weights).to(device)
    else:  # combined
        criterion = CombinedLoss(
            FocalLoss(gamma=args.gamma, weight=class_weights),
            DiceLoss(weight=class_weights),
        ).to(device)
    gamma_note = f" (gamma={args.gamma})" if args.loss in ("focal", "combined") else ""
    print(f"Loss function : {args.loss}{gamma_note}")

    weight_display = {idx_to_class[i]: f"{w:.3f}" for i, w in enumerate(class_weights.cpu().tolist())}
    print(f"Class weights : {weight_display}")

    pretrained = not args.no_pretrained
    model = build_efficientnet_b0(num_classes=num_classes, pretrained=pretrained).to(device)
    print(f"Model         : EfficientNet-B0 ({'pretrained' if pretrained else 'random weights'})")

    mel_idx = label_map.get("mel", -1)
    best_val_f1 = -1.0
    best_path = out_dir / "best_efficientnet.pth"
    log_csv_path = args.log_csv if args.log_csv is not None else (out_dir / "training_log.csv")
    print(f"Per-epoch log : {log_csv_path}")

    # =========================================================================
    # Phase 1 — frozen backbone, classifier head only
    # =========================================================================
    print(f"\n{'=' * 56}")
    print(f"Phase 1 : frozen backbone, head only  "
          f"(lr={args.phase1_lr}, {args.phase1_epochs} epochs)")
    print(f"{'=' * 56}")

    for param in model.features.parameters():
        param.requires_grad = False
    # avgpool has no learnable params; classifier is already unfrozen

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.phase1_lr,
        weight_decay=args.weight_decay,
    )

    for epoch in range(1, args.phase1_epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, criterion, device,
            desc=f"P1 Epoch {epoch}/{args.phase1_epochs}",
        )
        best_val_f1 = val_step(
            model, val_loader, criterion, device, label_map,
            mel_idx, best_val_f1, best_path,
            epoch_tag=f"P1 Epoch {epoch:3d}  train_loss={train_loss:.4f}",
            phase=1, epoch=epoch, train_loss=train_loss, seed=cfg.seed, split_seed=split_seed,
            loss_name=args.loss, run_start=run_start, log_csv_path=log_csv_path,
        )

    print(f"\nPhase 1 complete.  Best val macro_f1 so far = {best_val_f1:.4f}")

    # =========================================================================
    # Phase 2 — all layers unfrozen, full fine-tune
    # =========================================================================
    print(f"\n{'=' * 56}")
    print(f"Phase 2 : full network fine-tune  "
          f"(lr={args.phase2_lr}, {args.phase2_epochs} epochs)")
    print(f"{'=' * 56}")

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.phase2_lr,
        weight_decay=args.weight_decay,
    )

    for epoch in range(1, args.phase2_epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, criterion, device,
            desc=f"P2 Epoch {epoch}/{args.phase2_epochs}",
        )
        best_val_f1 = val_step(
            model, val_loader, criterion, device, label_map,
            mel_idx, best_val_f1, best_path,
            epoch_tag=f"P2 Epoch {epoch:3d}  train_loss={train_loss:.4f}",
            phase=2, epoch=epoch, train_loss=train_loss, seed=cfg.seed, split_seed=split_seed,
            loss_name=args.loss, run_start=run_start, log_csv_path=log_csv_path,
        )

    print(f"\nTraining complete.  Best val macro_f1 = {best_val_f1:.4f}")

    # =========================================================================
    # Test evaluation — TTA or TTA + MC Dropout
    # =========================================================================
    n_tta_views = len(build_tta_transforms(cfg.image_size))
    if args.no_mc:
        mode_label = f"5-view TTA only  (--no-mc)"
    else:
        total_preds = n_tta_views * args.mc_passes
        mode_label = (
            f"5-view TTA + MC Dropout  "
            f"({n_tta_views} views x {args.mc_passes} passes = {total_preds} predictions/image)"
        )

    print("\n" + "=" * 64)
    print(f"TEST SET EVALUATION  ({mode_label})")
    print("=" * 64)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if args.no_mc:
        y_true_test, y_pred_test = evaluate_tta(model, tta_loader, device)
        mean_uncertainty = None
    else:
        y_true_test, y_pred_test, mean_uncertainty = evaluate_tta_mc(
            model, tta_loader, device, mc_passes=args.mc_passes
        )

    # Confusion matrix
    cm = confusion_matrix(y_true_test, y_pred_test, labels=list(range(num_classes)))
    print("\nConfusion Matrix  (rows = true class, cols = predicted class):")
    col_header = "         " + "  ".join(f"{c:>6}" for c in class_names)
    print(col_header)
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>8} " + "  ".join(f"{v:>6}" for v in row))

    # Per-class report
    print("\nPer-class Classification Report:")
    print(classification_report(y_true_test, y_pred_test, target_names=class_names, zero_division=0))

    # Highlighted summary
    macro_f1 = f1_score(y_true_test, y_pred_test, average="macro", zero_division=0)
    mel_recall = (
        recall_score(y_true_test, y_pred_test, labels=[mel_idx], average="micro", zero_division=0)
        if mel_idx >= 0 else float("nan")
    )

    print("=" * 64)
    print(f"  Macro-F1        : {macro_f1:.4f}")
    print(
        f"  Melanoma Recall : {mel_recall:.4f}"
        f"  (class '{idx_to_class.get(mel_idx, '?')}', index {mel_idx})"
    )
    if mean_uncertainty is not None:
        print(
            f"  MC Uncertainty  : {mean_uncertainty:.6f}"
            f"  (mean per-image variance across {n_tta_views * args.mc_passes} predictions)"
        )
    print("=" * 64)


if __name__ == "__main__":
    main()
