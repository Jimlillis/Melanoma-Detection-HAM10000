"""
Extract DINOv2 ViT-S/14 *with registers* CLS-token features for all three
HAM10000 splits.

Identical to extract_features_v2.py except it uses the register variant of
DINOv2 ViT-S/14 (dinov2_vits14_reg) as suggested by:
  Darcet et al., "Vision Transformers Need Registers", ICLR 2024.

Reuses the same lesion-level train/val/test split as the EfficientNet baseline
(dataset.lesion_level_split with identical seed and ratios) so there is no
data leakage and the splits can be directly compared with other feature sets.

Output — one file per split:
  data/processed/dinov2_reg/dinov2_reg_feats_train.pt
  data/processed/dinov2_reg/dinov2_reg_feats_val.pt
  data/processed/dinov2_reg/dinov2_reg_feats_test.pt

Each file is a dict:
  {
    "features":  FloatTensor  (N, 384),   # CLS token from ViT-S/14-reg
    "labels":    LongTensor   (N,),       # 7-class integer label
    "label_map": dict[str, int],          # e.g. {"akiec": 0, "mel": 4, ...}
  }

DINOv2 ViT-S/14 (with or without registers) outputs 384-dim embeddings, so
train_dino_head.py works unchanged — just point --features-dir to the new files:

  python src/train_dino_head.py \\
    --features-dir data/processed/dinov2_reg \\
    --output-dir outputs/dinov2_reg_linear

Run from project root:
  python src/extract_features_v2reg.py \\
    --metadata-csv data/HAM10000_metadata.csv \\
    --images-dir data \\
    --output-dir data/processed/dinov2_reg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import Config
from dataset import SkinLesionDataset, build_transforms, lesion_level_split
from helpers import get_device, set_seed


# ---------------------------------------------------------------------------
# Metadata loading — 7-class variant
# ---------------------------------------------------------------------------

def load_multiclass_table(
    csv_path: Path,
    col_image_id: str,
    col_lesion_id: str,
    col_diagnosis: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Read the HAM10000 CSV and add a ``_binary_label`` column containing the
    7-class integer label (column is named ``_binary_label`` so that it is
    compatible with ``lesion_level_split`` without any modification).

    Labels are assigned by sorted alphabetical order of the ``dx`` values,
    which for HAM10000 gives:
        akiec=0, bcc=1, bkl=2, df=3, mel=4, nv=5, vasc=6

    Returns the augmented DataFrame and the label map.
    """
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
# Feature extraction
# ---------------------------------------------------------------------------

def extract_split(
    model: torch.nn.Module,
    df_split: pd.DataFrame,
    images_dir: Path,
    col_image_id: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (features, labels) tensors for one split."""
    # No augmentation — same transform used for val/test in the baseline
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
    all_labels: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device)
            # DINOv2 hub models return the CLS token directly → shape (B, 384)
            feats = model(images)
            all_features.append(feats.cpu())
            all_labels.append(labels)  # already on CPU from SkinLesionDataset

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract DINOv2 ViT-S/14 with registers CLS features for HAM10000"
    )
    p.add_argument("--metadata-csv", type=Path, default=None)
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("data/processed/dinov2_reg"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Start from Config defaults and override only what was supplied
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

    # --- Metadata & split ----------------------------------------------------
    print("Loading metadata …")
    df, label_map = load_multiclass_table(
        cfg.metadata_csv,
        cfg.col_image_id,
        cfg.col_lesion_id,
        cfg.col_diagnosis,
    )
    print(f"Label map : {label_map}")

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

    # --- DINOv2 with registers model -----------------------------------------
    device = get_device()
    print(f"Loading DINOv2 ViT-S/14 with registers on {device} …")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
    model = model.to(device)
    model.eval()

    # --- Extract & save ------------------------------------------------------
    splits = [
        ("train", df_train, "dinov2_reg_feats_train.pt"),
        ("val",   df_val,   "dinov2_reg_feats_val.pt"),
        ("test",  df_test,  "dinov2_reg_feats_test.pt"),
    ]
    for split_name, df_split, fname in splits:
        feats, labels = extract_split(
            model,
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


if __name__ == "__main__":
    main()
