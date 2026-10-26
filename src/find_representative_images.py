"""
Find high-confidence correct predictions from the EfficientNet 7-class model.

For each of the 7 HAM10000 classes, selects the top-2 images where:
  - the prediction is correct
  - confidence (max softmax probability) > 0.85

Copies selected images to outputs/representative_samples/ with filename:
  {class}_{confidence:.2f}_{image_id}.jpg

Run from project root:
  python src/find_representative_images.py \\
    --checkpoint outputs/efficientnet_improved/best_model.pth \\
    --metadata-csv data/HAM10000_metadata.csv \\
    --images-dir data \\
    --output-dir outputs/representative_samples
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import Config
from dataset import SkinLesionDataset, build_transforms, lesion_level_split
from helpers import get_device, resolve_image_path, set_seed
from model import build_efficientnet_b0


# ---------------------------------------------------------------------------
# Data helpers — identical to gradcam.py / train_efficientnet.py
# ---------------------------------------------------------------------------

def _load_multiclass_table(
    csv_path: Path,
    cfg: Config,
) -> tuple[pd.DataFrame, dict[str, int]]:
    df = pd.read_csv(csv_path)
    classes = sorted(df[cfg.col_diagnosis].astype(str).unique().tolist())
    label_map = {c: i for i, c in enumerate(classes)}
    df = df.copy()
    df["_binary_label"] = df[cfg.col_diagnosis].astype(str).map(label_map).astype(int)
    return df, label_map


# ---------------------------------------------------------------------------
# Inference — single forward pass, no TTA, no MC Dropout
# ---------------------------------------------------------------------------

def run_inference(
    model: torch.nn.Module,
    df_test: pd.DataFrame,
    cfg: Config,
    device: torch.device,
) -> pd.DataFrame:
    """
    Run a single deterministic forward pass over the test split.

    Returns a DataFrame with columns:
        image_id, true_label_idx, pred_label_idx, confidence
    """
    transform = build_transforms(cfg.image_size, is_train=False)
    dataset = SkinLesionDataset(
        df_test, cfg.images_dir, cfg.col_image_id, transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    all_image_ids: list[str] = []
    all_true: list[int] = []
    all_pred: list[int] = []
    all_conf: list[float] = []

    image_ids = df_test[cfg.col_image_id].tolist()
    idx = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Inference", unit="batch"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1).cpu()
            conf, pred = probs.max(dim=1)

            batch_size = imgs.size(0)
            all_image_ids.extend(image_ids[idx : idx + batch_size])
            all_true.extend(labels.tolist())
            all_pred.extend(pred.tolist())
            all_conf.extend(conf.tolist())
            idx += batch_size

    return pd.DataFrame(
        {
            "image_id": all_image_ids,
            "true_label_idx": all_true,
            "pred_label_idx": all_pred,
            "confidence": all_conf,
        }
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select top-2 high-confidence correct predictions per class."
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/efficientnet_improved/best_model.pth",
        help="Path to the EfficientNet checkpoint (.pth).",
    )
    parser.add_argument(
        "--metadata-csv",
        default="data/HAM10000_metadata.csv",
        help="Path to HAM10000_metadata.csv.",
    )
    parser.add_argument(
        "--images-dir",
        default="data",
        help="Root directory that contains the image folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/representative_samples",
        help="Destination folder for the copied images.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.85,
        help="Minimum softmax confidence to consider an image (default: 0.85).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of images to pick per class (default: 2).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # Build the default Config just to reuse its column-name and split constants
    cfg = Config()
    cfg.images_dir = Path(args.images_dir)

    # -----------------------------------------------------------------------
    # Load metadata + reproduce the identical test split
    # -----------------------------------------------------------------------
    csv_path = Path(args.metadata_csv)
    df_all, label_map = _load_multiclass_table(csv_path, cfg)
    idx_to_class = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    _, _, df_test = lesion_level_split(
        df_all,
        col_lesion_id=cfg.col_lesion_id,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
    )
    print(f"Test split: {len(df_test)} images")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    ckpt_path = Path(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = build_efficientnet_b0(num_classes=num_classes, pretrained=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)

    # -----------------------------------------------------------------------
    # Run inference
    # -----------------------------------------------------------------------
    results = run_inference(model, df_test, cfg, device)

    # Attach human-readable class names
    results["true_label"] = results["true_label_idx"].map(idx_to_class)
    results["predicted_label"] = results["pred_label_idx"].map(idx_to_class)

    # -----------------------------------------------------------------------
    # Filter: correct predictions above confidence threshold
    # -----------------------------------------------------------------------
    correct_mask = results["true_label_idx"] == results["pred_label_idx"]
    conf_mask = results["confidence"] > args.confidence_threshold
    filtered = results[correct_mask & conf_mask].copy()

    print(
        f"\nImages passing filter (correct & conf > {args.confidence_threshold:.2f}): "
        f"{len(filtered)} / {len(results)}"
    )

    # -----------------------------------------------------------------------
    # Pick top-k per class
    # -----------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict] = []
    class_names = sorted(label_map.keys())  # alphabetical — deterministic

    for cls in class_names:
        cls_df = filtered[filtered["true_label"] == cls].sort_values(
            "confidence", ascending=False
        )
        top = cls_df.head(args.top_k)

        if len(top) == 0:
            print(f"  [{cls}] WARNING: no images passed the filter — skipping.")
            continue

        if len(top) < args.top_k:
            print(
                f"  [{cls}] WARNING: only {len(top)} image(s) available "
                f"(wanted {args.top_k})."
            )

        for _, row in top.iterrows():
            image_id = row["image_id"]
            conf = float(row["confidence"])
            src_path = resolve_image_path(cfg.images_dir, image_id)
            dst_name = f"{cls}_{conf:.2f}_{image_id}.jpg"
            dst_path = output_dir / dst_name
            shutil.copy2(src_path, dst_path)
            selected_rows.append(
                {
                    "class": cls,
                    "image_id": image_id,
                    "confidence": conf,
                    "filename": dst_name,
                }
            )

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"{'CLASS':<10} {'IMAGE_ID':<22} {'CONFIDENCE':>10}  FILENAME")
    print("=" * 65)
    for r in selected_rows:
        print(
            f"{r['class']:<10} {r['image_id']:<22} {r['confidence']:>10.4f}  {r['filename']}"
        )
    print("=" * 65)
    print(f"\n{len(selected_rows)} images copied to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
