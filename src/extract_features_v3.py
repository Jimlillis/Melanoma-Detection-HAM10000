"""
Extract DINOv3 ViT-S/16 features for all three HAM10000 splits.

Reuses the same lesion-level train/val/test split as the EfficientNet baseline
(dataset.lesion_level_split with identical seed and ratios) so there is no
data leakage and the splits can be directly compared with DINOv1/v2 features.

Pooling (--global-pool, passed straight to timm.create_model(global_pool=...)):

  token (default)  the CLS token: token 0 of the sequence after the model's
                   final LayerNorm. This is upstream DINOv3's x_norm_clstoken
                   and the same kind of feature the DINOv1/DINOv2/DINOv2-reg
                   extractors return.
  avg              the mean of the PATCH tokens only, after the same final
                   LayerNorm; the CLS token and the 4 register tokens are
                   excluded. This is timm's default for this model, and it is
                   what this script silently extracted before --global-pool
                   existed: those files were labelled "CLS" but were mean-pooled
                   patch tokens (kept as dinov3_feats_meanpool_*.pt).

Output — one file per split:
  token:  data/processed/dinov3/dinov3_feats_{train,val,test}.pt
  avg:    data/processed/dinov3/dinov3_feats_meanpool_{train,val,test}.pt

Each file is a dict:
  {
    "features":  FloatTensor  (N, 384),   # pooled features, see --global-pool
    "labels":    LongTensor   (N,),       # 7-class integer label
    "label_map": dict[str, int],          # e.g. {"akiec": 0, "mel": 4, ...}
  }

DINOv3 ViT-S/16 outputs 384-dim embeddings (same dim as DINOv1 ViT-S/16 and
DINOv2 ViT-S/14), so train_dino_head.py works unchanged (the loss and seed are
the ones the reported DINOv3 probe was trained with):

  python src/train_dino_head.py \\
    --features-dir data/processed/dinov3 \\
    --features-prefix dinov3_feats \\
    --output-dir outputs/dinov3_linear \\
    --loss combined --seed 42

Run from project root:
  python src/extract_features_v3.py \\
    --metadata-csv data/HAM10000_metadata.csv \\
    --images-dir data \\
    --output-dir data/processed/dinov3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import timm
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
            # With num_classes=0 the classifier is an Identity, so model(x) returns
            # the POOLED features, shape (B, 384). Which pooling is decided by the
            # global_pool argument given to timm.create_model (NOT always the CLS
            # token): 'token' -> CLS token, 'avg' -> mean of the patch tokens. In
            # both cases the final LayerNorm has already been applied to all tokens.
            feats = model(images)
            all_features.append(feats.cpu())
            all_labels.append(labels)  # already on CPU from SkinLesionDataset

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract DINOv3 ViT-S/16 features (CLS token or mean-pooled "
                    "patch tokens) for HAM10000"
    )
    p.add_argument("--metadata-csv", type=Path, default=None)
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("data/processed/dinov3"))
    p.add_argument(
        "--global-pool",
        choices=["token", "avg"],
        default="token",
        help="Pooling of the ViT token sequence, passed to timm as global_pool. "
             "'token' = CLS token (default; matches upstream DINOv3 and the "
             "DINOv1/v2 extractors) -> dinov3_feats_*.pt. 'avg' = mean of the "
             "patch tokens only (CLS and register tokens excluded) -> "
             "dinov3_feats_meanpool_*.pt, so it never overwrites the CLS files.",
    )
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

    # --- DINOv3 model (timm) -------------------------------------------------
    device = get_device()
    print(f"Loading DINOv3 ViT-S/16 via timm on {device} "
          f"(global_pool={args.global_pool}) …")
    model = timm.create_model(
        "vit_small_patch16_dinov3",
        pretrained=True,
        num_classes=0,                  # Identity head: model(x) returns the pooled 384-dim features
        global_pool=args.global_pool,   # 'token' = CLS token, 'avg' = mean of patch tokens
    )
    # Guard against a silent fall-back to timm's default ('avg'), which is exactly
    # how the original features ended up mean-pooled while labelled as CLS.
    assert model.global_pool == args.global_pool, (
        f"timm built the model with global_pool={model.global_pool!r}, "
        f"expected {args.global_pool!r}"
    )
    model = model.to(device)
    model.eval()

    # --- Extract & save ------------------------------------------------------
    # 'token' keeps the file names train_dino_head.py expects for DINOv3
    # (--features-prefix dinov3_feats); 'avg' goes to a separate _meanpool prefix.
    prefix = "dinov3_feats" if args.global_pool == "token" else "dinov3_feats_meanpool"
    print(f"Output files : {out_dir / prefix}_<train|val|test>.pt")
    splits = [
        ("train", df_train, f"{prefix}_train.pt"),
        ("val",   df_val,   f"{prefix}_val.pt"),
        ("test",  df_test,  f"{prefix}_test.pt"),
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
