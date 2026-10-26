"""
Train EfficientNet-B0 on HAM10000 binary labels with lesion-level split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running as `python src/train.py` from project root
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import Config
from dataset import (
    SkinLesionDataset,
    build_transforms,
    lesion_level_split,
    load_metadata_table,
    make_weighted_sampler,
)
from model import build_efficientnet_b0
from helpers import classification_metrics, format_metrics, get_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train melanoma vs non-melanoma baseline")
    p.add_argument("--metadata-csv", type=Path, default=None)
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--split-seed", type=int, default=None,
                   help="Seed for the lesion-level train/val/test split, decoupled "
                        "from --seed (training RNG). Defaults to --seed's resolved "
                        "value if omitted, i.e. identical to old behaviour.")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--skip-test", action="store_true", help="Do not evaluate on test set after training")
    p.add_argument("--use-sampler", action="store_true",
                   help="Use WeightedRandomSampler instead of class-weighted loss")
    return p.parse_args()


def apply_args_to_config(cfg: Config, args: argparse.Namespace) -> Config:
    if args.metadata_csv is not None:
        cfg.metadata_csv = args.metadata_csv
    if args.images_dir is not None:
        cfg.images_dir = args.images_dir
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.seed is not None:
        cfg.seed = args.seed
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.no_pretrained:
        cfg.pretrained = False
    return cfg


def compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Inverse-frequency weights for CrossEntropyLoss.
    w_c = n_total / (n_classes * n_c)  → rare class gets a higher weight.
    Returns a 1-D tensor of shape (n_classes,) on `device`.
    """
    counts = np.bincount(labels)  # [n_neg, n_pos]
    n_classes = len(counts)
    weights = len(labels) / (n_classes * counts.astype(float))
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, dict]:
    model.eval()
    losses = []
    all_y: list[int] = []
    all_pred: list[int] = []
    all_prob: list[float] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item())
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
        all_y.extend(y.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())
        all_prob.extend(prob.cpu().numpy().tolist())

    y_true = np.array(all_y)
    y_pred = np.array(all_pred)
    y_score = np.array(all_prob)
    metrics = classification_metrics(y_true, y_pred, y_score)
    return float(np.mean(losses)), metrics


def train() -> None:
    args = parse_args()
    cfg = apply_args_to_config(Config(), args)
    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata_table(
        cfg.metadata_csv,
        cfg.col_image_id,
        cfg.col_lesion_id,
        cfg.col_diagnosis,
        cfg.melanoma_class,
    )
    # Decoupled from cfg.seed (training RNG) so a seed-variance study can vary
    # --seed while pinning --split-seed to keep the train/val/test partition
    # fixed; --split-seed defaults to cfg.seed, so old invocations are unaffected.
    split_seed = args.split_seed if args.split_seed is not None else cfg.seed
    df_train, df_val, df_test = lesion_level_split(
        df,
        cfg.col_lesion_id,
        cfg.train_ratio,
        cfg.val_ratio,
        cfg.test_ratio,
        split_seed,
    )

    train_ds = SkinLesionDataset(
        df_train,
        cfg.images_dir,
        cfg.col_image_id,
        transform=build_transforms(cfg.image_size, is_train=True),
    )
    val_ds = SkinLesionDataset(
        df_val,
        cfg.images_dir,
        cfg.col_image_id,
        transform=build_transforms(cfg.image_size, is_train=False),
    )

    # --- Imbalance strategy ------------------------------------------------
    # Default: weighted loss — melanoma gradient is amplified ~9×.
    # --use-sampler: WeightedRandomSampler — each batch is ~50/50 instead.
    # The two approaches are mutually exclusive; pick one per experiment.
    device = get_device()

    if args.use_sampler:
        sampler = make_weighted_sampler(df_train)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,          # shuffle must be False when sampler is set
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        criterion = nn.CrossEntropyLoss()
        print("Imbalance strategy: WeightedRandomSampler")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        class_weights = compute_class_weights(df_train["_binary_label"].values, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Imbalance strategy: class weights {class_weights.tolist()}")
    # -----------------------------------------------------------------------

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = build_efficientnet_b0(num_classes=2, pretrained=cfg.pretrained)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_val_f1 = -1.0
    best_path = cfg.checkpoint_path()

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=running_loss / n_batches)

        val_loss, val_metrics = evaluate_epoch(model, val_loader, device, criterion)
        print(
            f"Epoch {epoch}: val_loss={val_loss:.4f}, {format_metrics(val_metrics)}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
            print(f"  Saved new best checkpoint (macro_f1={best_val_f1:.4f}) -> {best_path}")

    print(f"Training finished. Best val macro_f1={best_val_f1:.4f}")

    if not args.skip_test:
        test_ds = SkinLesionDataset(
            df_test,
            cfg.images_dir,
            cfg.col_image_id,
            transform=build_transforms(cfg.image_size, is_train=False),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        try:
            checkpoint = torch.load(
                best_path, map_location=device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        _, test_metrics = evaluate_epoch(model, test_loader, device, criterion)
        print(f"Test: {format_metrics(test_metrics)}")


if __name__ == "__main__":
    train()
