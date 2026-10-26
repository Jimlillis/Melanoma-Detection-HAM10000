"""
Load a saved checkpoint and evaluate on the held-out test split (lesion-level).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import Config
from dataset import (
    SkinLesionDataset,
    build_transforms,
    lesion_level_split,
    load_metadata_table,
)
from model import build_efficientnet_b0
from helpers import classification_metrics, format_metrics, get_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate saved model on test split")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to .pth from training")
    p.add_argument("--metadata-csv", type=Path, default=None)
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    return p.parse_args()


def apply_args(cfg: Config, args: argparse.Namespace) -> Config:
    if args.metadata_csv is not None:
        cfg.metadata_csv = args.metadata_csv
    if args.images_dir is not None:
        cfg.images_dir = args.images_dir
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.seed is not None:
        cfg.seed = args.seed
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    return cfg


@torch.no_grad()
def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_y: list[int] = []
    all_pred: list[int] = []
    all_prob: list[float] = []

    for x, y in tqdm(loader, desc="Evaluating"):
        x = x.to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
        all_y.extend(y.numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())
        all_prob.extend(prob.cpu().numpy().tolist())

    y_true = np.array(all_y)
    y_pred = np.array(all_pred)
    y_score = np.array(all_prob)
    return classification_metrics(y_true, y_pred, y_score)


def main() -> None:
    args = parse_args()
    cfg = apply_args(Config(), args)
    set_seed(cfg.seed)

    try:
        checkpoint = torch.load(
            args.checkpoint, map_location=get_device(), weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=get_device())
    saved = checkpoint.get("config", {})
    if saved:
        # Restore paths from checkpoint if not overridden
        if args.metadata_csv is None and "metadata_csv" in saved:
            cfg.metadata_csv = Path(saved["metadata_csv"])
        if args.images_dir is None and "images_dir" in saved:
            cfg.images_dir = Path(saved["images_dir"])

    df = load_metadata_table(
        cfg.metadata_csv,
        cfg.col_image_id,
        cfg.col_lesion_id,
        cfg.col_diagnosis,
        cfg.melanoma_class,
    )
    _, _, df_test = lesion_level_split(
        df,
        cfg.col_lesion_id,
        cfg.train_ratio,
        cfg.val_ratio,
        cfg.test_ratio,
        cfg.seed,
    )

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

    device = get_device()
    model = build_efficientnet_b0(num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    metrics = run_eval(model, test_loader, device)
    print(format_metrics(metrics))


if __name__ == "__main__":
    main()
