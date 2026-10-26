"""
Extract penultimate-layer (1280-dim) features from the trained EfficientNet-B0
for all three HAM10000 splits (train / val / test).

The "penultimate layer" is the 1280-dim global-average-pooled representation
produced BEFORE the dropout and final linear classifier:

    model.features(x)          →  (B, 1280, 7, 7)   [backbone convolutions]
    model.avgpool(...)          →  (B, 1280, 1, 1)   [adaptive avg pool]
    flatten                     →  (B, 1280)          ← this is what we save

Uses the identical lesion-level split (seed=42) as all other experiments to
guarantee no data leakage and direct comparability.

Output — one file per split saved under data/processed/efficientnet/:
  effnet_feats_train.pt
  effnet_feats_val.pt
  effnet_feats_test.pt

Each file is a dict:
  {
    "features":  FloatTensor  (N, 1280),
    "labels":    LongTensor   (N,),
    "label_map": dict[str, int],   # e.g. {"akiec": 0, "mel": 4, ...}
  }

Run from project root:
  python src/extract_efficientnet_features.py \\
    --checkpoint outputs/efficientnet_mc/best_efficientnet.pth \\
    --metadata-csv data/HAM10000_metadata.csv \\
    --images-dir data \\
    --output-dir data/processed/efficientnet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
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
# Metadata loading — 7-class variant (identical to other extract_features scripts)
# ---------------------------------------------------------------------------

def load_multiclass_table(
    csv_path: Path,
    col_image_id: str,
    col_lesion_id: str,
    col_diagnosis: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    df = pd.read_csv(csv_path)
    required = {col_image_id, col_lesion_id, col_diagnosis}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    classes = sorted(df[col_diagnosis].astype(str).unique().tolist())
    label_map = {c: i for i, c in enumerate(classes)}

    df = df.copy()
    df["_binary_label"] = df[col_diagnosis].astype(str).map(label_map).astype(int)
    return df, label_map


# ---------------------------------------------------------------------------
# Feature extractor wrapper
# ---------------------------------------------------------------------------

class EfficientNetFeatureExtractor(nn.Module):
    """
    Wraps a trained EfficientNet-B0 and returns 1280-dim penultimate features.

    Forward pass:
        features(x) → avgpool(x) → flatten   [no dropout, no final linear]
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.features = backbone.features   # Conv backbone
        self.avgpool  = backbone.avgpool    # AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)           # (B, 1280, 7, 7)
        x = self.avgpool(x)            # (B, 1280, 1, 1)
        x = x.flatten(start_dim=1)    # (B, 1280)
        return x


# ---------------------------------------------------------------------------
# Per-split extraction
# ---------------------------------------------------------------------------

def extract_split(
    extractor: nn.Module,
    df_split: pd.DataFrame,
    images_dir: Path,
    col_image_id: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (features (N,1280), labels (N,)) for one split."""
    transform = build_transforms(224, is_train=False)
    ds = SkinLesionDataset(df_split, images_dir, col_image_id, transform=transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_features: list[torch.Tensor] = []
    all_labels:   list[torch.Tensor] = []

    extractor.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device)
            feats  = extractor(images)
            all_features.append(feats.cpu())
            all_labels.append(labels)   # already CPU from SkinLesionDataset

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract EfficientNet-B0 penultimate features (1280-dim) for HAM10000"
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/efficientnet_mc/best_efficientnet.pth"),
        help="Path to the trained EfficientNet-B0 checkpoint.",
    )
    p.add_argument("--metadata-csv",  type=Path, default=None)
    p.add_argument("--images-dir",    type=Path, default=None)
    p.add_argument("--output-dir",    type=Path, default=Path("data/processed/efficientnet"))
    p.add_argument("--batch-size",    type=int,  default=64)
    p.add_argument("--num-workers",   type=int,  default=None)
    p.add_argument("--seed",          type=int,  default=None)
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

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # --- Load checkpoint & build model ----------------------------------------
    device = get_device()
    print(f"Device : {device}")
    print(f"Loading checkpoint : {checkpoint_path}")

    ckpt        = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_label_map: dict[str, int] = ckpt["label_map"]
    num_classes = len(ckpt_label_map)

    backbone  = build_efficientnet_b0(num_classes=num_classes, pretrained=False)
    backbone.load_state_dict(ckpt["model_state_dict"])
    backbone.to(device)

    extractor = EfficientNetFeatureExtractor(backbone).to(device)
    extractor.eval()

    val_f1 = ckpt.get("val_macro_f1", float("nan"))
    print(f"Checkpoint val_macro_f1 : {val_f1:.4f}")
    print(f"Label map : {ckpt_label_map}")

    # --- Metadata & lesion-level split -----------------------------------------
    print("\nLoading metadata …")
    df, label_map = load_multiclass_table(
        cfg.metadata_csv,
        cfg.col_image_id,
        cfg.col_lesion_id,
        cfg.col_diagnosis,
    )

    # Sanity-check: label map must match what the checkpoint was trained with
    if label_map != ckpt_label_map:
        raise ValueError(
            f"Label map mismatch!\n"
            f"  CSV     : {label_map}\n"
            f"  Checkpoint: {ckpt_label_map}"
        )

    df_train, df_val, df_test = lesion_level_split(
        df,
        cfg.col_lesion_id,
        cfg.train_ratio,
        cfg.val_ratio,
        cfg.test_ratio,
        cfg.seed,
    )
    print(
        f"Split sizes — train: {len(df_train)}, "
        f"val: {len(df_val)}, test: {len(df_test)}"
    )

    # --- Extract & save --------------------------------------------------------
    splits = [
        ("train", df_train, "effnet_feats_train.pt"),
        ("val",   df_val,   "effnet_feats_val.pt"),
        ("test",  df_test,  "effnet_feats_test.pt"),
    ]

    for split_name, df_split, fname in splits:
        feats, labels = extract_split(
            extractor,
            df_split,
            cfg.images_dir,
            cfg.col_image_id,
            args.batch_size,
            cfg.num_workers,
            device,
            desc=f"Extracting {split_name}",
        )
        out_path = out_dir / fname
        torch.save(
            {"features": feats, "labels": labels, "label_map": label_map},
            out_path,
        )
        print(
            f"Saved {out_path}  "
            f"(features {tuple(feats.shape)}, labels {tuple(labels.shape)})"
        )

    print("\nDone. Feature dimension: 1280 (EfficientNet-B0 penultimate layer)")


if __name__ == "__main__":
    main()
