"""
DINO ViT-S/16 self-attention map explainability for HAM10000.

Loads the frozen DINO ViT-S/16 from torch.hub and extracts the CLS-token
self-attention of the last transformer block.  For each of the 7 HAM10000
classes the script selects the same three test examples as gradcam.py
(correct / wrong / uncertain, identified via EfficientNet MC Dropout) so the
two sets of visualisations are directly comparable.

Architecture facts (ViT-S/16, 224 × 224 input):
  patch_size  = 16 → 14 × 14 = 196 spatial patches
  num_tokens  = 197  (CLS + 196 patches)
  num_heads   = 6
  get_last_selfattention() → (1, 6, 197, 197)
  CLS-to-patch attention   → [:, :, 0, 1:] → (6, 196) → reshape (6, 14, 14)

For each example the saved file shows:
  Row 1 : original image | mean-across-heads attention overlay
  Row 2 : individual overlays for each of the 6 heads

The summary grid (outputs/dino_attention/summary_grid.png) has 7 rows × 3
columns (correct / wrong / uncertain), each cell showing
original | mean-attention overlay — identical format to the Grad-CAM grid.

Colormap and overlay opacity match gradcam.py (COLORMAP_JET, 40%) for a
fair side-by-side visual comparison.

Run from the project root:
  python src/dino_attention.py \\
    --metadata-csv data/HAM10000_metadata.csv \\
    --images-dir data \\
    --efficientnet-checkpoint outputs/efficientnet_mc/best_efficientnet.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import Config
from dataset import SkinLesionDataset, build_transforms, lesion_level_split
from helpers import get_device, resolve_image_path, set_seed
from model import build_efficientnet_b0


# ---------------------------------------------------------------------------
# Data helpers  (mirrors gradcam.py — kept self-contained)
# ---------------------------------------------------------------------------

def _load_multiclass_table(
    csv_path: Path,
    cfg: Config,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Read HAM10000 CSV, add ``_binary_label`` = 7-class integer label.
    akiec=0, bcc=1, bkl=2, df=3, mel=4, nv=5, vasc=6  (alphabetical).
    """
    df = pd.read_csv(csv_path)
    classes = sorted(df[cfg.col_diagnosis].astype(str).unique().tolist())
    label_map = {c: i for i, c in enumerate(classes)}
    df = df.copy()
    df["_binary_label"] = df[cfg.col_diagnosis].astype(str).map(label_map).astype(int)
    return df, label_map


def load_image_tensor(
    image_id: str,
    images_dir: Path,
    image_size: int,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Returns:
      rgb_np : (H_orig, W_orig, 3) uint8 — original pixels for display
      tensor : (1, 3, image_size, image_size) float32 on device — model input
    """
    path = resolve_image_path(images_dir, image_id)
    img_pil = Image.open(path).convert("RGB")
    rgb_np = np.array(img_pil)
    transform = build_transforms(image_size, is_train=False)
    tensor = transform(img_pil).unsqueeze(0).to(device)
    return rgb_np, tensor


# ---------------------------------------------------------------------------
# EfficientNet MC Dropout inference  (same logic as gradcam.py)
# ---------------------------------------------------------------------------

def run_mc_inference(
    model: nn.Module,
    df_test: pd.DataFrame,
    cfg: Config,
    mc_passes: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MC Dropout inference on the full test split.

    Returns y_true (N,), y_pred (N,), uncertainty (N,).
    Uncertainty = mean per-image variance of softmax probabilities across passes.
    """
    from torch.utils.data import DataLoader

    transform = build_transforms(cfg.image_size, is_train=False)
    dataset = SkinLesionDataset(df_test, cfg.images_dir, cfg.col_image_id, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))

    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    all_y_true: list[int] = []
    all_y_pred: list[int] = []
    all_unc: list[float] = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"EfficientNet MC Dropout ({mc_passes} passes)"):
            x = x.to(device)
            probs_list = []
            for _ in range(mc_passes):
                logits = model(x)
                probs_list.append(torch.softmax(logits, dim=1).cpu())
            stacked    = torch.stack(probs_list, dim=0)        # (K, B, C)
            mean_probs = stacked.mean(dim=0)                   # (B, C)
            preds      = mean_probs.argmax(dim=1).tolist()
            unc        = stacked.var(dim=0).mean(dim=1).tolist()
            all_y_true.extend(y.tolist())
            all_y_pred.extend(preds)
            all_unc.extend(unc)

    return np.array(all_y_true), np.array(all_y_pred), np.array(all_unc)


# ---------------------------------------------------------------------------
# Example selection  (identical logic to gradcam.py)
# ---------------------------------------------------------------------------

def select_examples(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    class_names: list[str],
    col_image_id: str,
) -> dict[str, dict[str, dict | None]]:
    """
    For each class select:
      correct   – true==pred==class, lowest uncertainty (most confident hit)
      wrong     – true==class, pred!=class, highest uncertainty (worst miss)
      uncertain – true==class, highest MC variance regardless of correctness

    Returns result[class_name][ex_type] → dict(image_id, true_idx, pred_idx,
    uncertainty, df_idx) | None.
    """
    result: dict[str, dict[str, dict | None]] = {}
    df_reset = df_test.reset_index(drop=True)

    for cls_idx, cls_name in enumerate(class_names):
        examples: dict[str, dict | None] = {"correct": None, "wrong": None, "uncertain": None}
        cls_indices = np.where(y_true == cls_idx)[0]

        if len(cls_indices) == 0:
            print(f"  [WARN] No test examples for class '{cls_name}'")
            result[cls_name] = examples
            continue

        def _entry(idx: int) -> dict:
            row = df_reset.iloc[idx]
            return dict(
                image_id=str(row[col_image_id]),
                true_idx=cls_idx,
                pred_idx=int(y_pred[idx]),
                uncertainty=float(uncertainty[idx]),
                df_idx=int(idx),
            )

        # correct
        correct_idx = np.where((y_true == cls_idx) & (y_pred == cls_idx))[0]
        if len(correct_idx) > 0:
            examples["correct"] = _entry(correct_idx[uncertainty[correct_idx].argmin()])
        else:
            print(f"  [WARN] No correct predictions for class '{cls_name}'")

        # wrong
        wrong_idx = np.where((y_true == cls_idx) & (y_pred != cls_idx))[0]
        if len(wrong_idx) > 0:
            examples["wrong"] = _entry(wrong_idx[uncertainty[wrong_idx].argmax()])
        else:
            print(f"  [WARN] No misclassifications for class '{cls_name}'")

        # uncertain
        examples["uncertain"] = _entry(cls_indices[uncertainty[cls_indices].argmax()])

        result[cls_name] = examples

    return result


# ---------------------------------------------------------------------------
# DINO attention extraction
# ---------------------------------------------------------------------------

def extract_attention_maps(
    dino_model: nn.Module,
    tensor: torch.Tensor,      # (1, 3, H, W) on device
    patch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract CLS-token self-attention of the last DINO transformer block.

    Returns
    -------
    mean_map   : (14, 14) float32 in [0, 1] — attention averaged across heads
    head_maps  : (6, 14, 14) float32 in [0, 1] — per-head attention maps

    Each head's map is normalised independently to [0, 1] so that
    low-activation heads are not washed out by high-activation ones.

    The attention tensor from get_last_selfattention() has shape
    (1, num_heads, num_tokens, num_tokens); num_tokens = 1 + H/p * W/p.
    We take the CLS row (index 0) and discard the CLS-to-CLS weight (index 0
    of the key axis), leaving 196 patch weights that reshape to 14 × 14.
    """
    dino_model.eval()
    with torch.no_grad():
        # shape: (1, num_heads, num_tokens, num_tokens)
        attn = dino_model.get_last_selfattention(tensor)

    B, num_heads, num_tokens, _ = attn.shape
    h = w = int((num_tokens - 1) ** 0.5)   # 14 for ViT-S/16 + 224px input

    # CLS token attending to spatial patches: (num_heads, h*w)
    cls_attn = attn[0, :, 0, 1:].cpu().float()   # (6, 196)
    cls_attn = cls_attn.reshape(num_heads, h, w)  # (6, 14, 14)

    # Normalise each head independently
    head_maps = np.zeros((num_heads, h, w), dtype=np.float32)
    for i in range(num_heads):
        a = cls_attn[i].numpy()
        a_min, a_max = a.min(), a.max()
        head_maps[i] = (a - a_min) / (a_max - a_min + 1e-8)

    # Mean across heads, re-normalised
    mean_map = head_maps.mean(axis=0)
    mean_map = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min() + 1e-8)

    return mean_map, head_maps


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def make_overlay(
    original_rgb: np.ndarray,  # (H, W, 3) uint8 RGB
    attn_map: np.ndarray,      # (h, w) float32 in [0, 1] — any spatial size
    heatmap_alpha: float = 0.4,
    display_size: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize original and attention map to display_size, blend with COLORMAP_JET.

    Returns orig_display (H, H, 3) uint8 and overlay (H, H, 3) uint8.
    Uses the same colormap and alpha as gradcam.py for fair visual comparison.
    """
    orig_display = cv2.resize(
        original_rgb, (display_size, display_size), interpolation=cv2.INTER_LINEAR,
    )
    attn_uint8 = (attn_map * 255).astype(np.uint8)
    attn_resized = cv2.resize(
        attn_uint8, (display_size, display_size), interpolation=cv2.INTER_LINEAR,
    )
    heatmap_bgr = cv2.applyColorMap(attn_resized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig_display, 1.0 - heatmap_alpha, heatmap_rgb, heatmap_alpha, 0)
    return orig_display, overlay


def save_attention_figure(
    out_path: Path,
    orig_display: np.ndarray,       # (S, S, 3) uint8
    mean_overlay: np.ndarray,       # (S, S, 3) uint8
    head_overlays: list[np.ndarray],# 6 × (S, S, 3) uint8
    title: str,
    subtitle: str = "",
) -> None:
    """
    Save a two-row panel:
      Row 1 (2 panels) : original | mean-head attention overlay
      Row 2 (6 panels) : individual head overlays (head 0 … 5)

    Layout keeps the two 'summary' panels larger so they are easy to scan,
    while still showing per-head detail below.
    """
    num_heads = len(head_overlays)   # 6

    # Use a grid with 6 columns; row 1 spans first 3 + last 3 columns
    fig = plt.figure(figsize=(num_heads * 2.2, 5.0))
    fig.suptitle(title, fontsize=8, fontweight="bold", y=1.01)
    if subtitle:
        fig.text(0.5, -0.01, subtitle, ha="center", fontsize=7, color="#555555")

    # Row 1: original (cols 0-2) | mean overlay (cols 3-5) — each 3 cols wide
    ax_orig = fig.add_subplot(2, 6, (1, 3))
    ax_mean = fig.add_subplot(2, 6, (4, 6))

    ax_orig.imshow(orig_display)
    ax_orig.set_title("Original", fontsize=8)
    ax_orig.axis("off")

    ax_mean.imshow(mean_overlay)
    ax_mean.set_title("Mean attention (all heads)", fontsize=8)
    ax_mean.axis("off")

    # Row 2: one panel per head
    for h_idx, h_overlay in enumerate(head_overlays):
        ax = fig.add_subplot(2, 6, 6 + h_idx + 1)
        ax.imshow(h_overlay)
        ax.set_title(f"Head {h_idx}", fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_summary_grid(
    composites: dict[str, dict[str, tuple[np.ndarray, np.ndarray] | None]],
    class_names: list[str],
    idx_to_class: dict[int, str],
    selected: dict[str, dict[str, dict | None]],
    out_path: Path,
) -> None:
    """
    7 × 3 summary grid identical in format to the Grad-CAM grid:
      rows = classes, columns = correct / wrong / uncertain
      each cell = original | mean-attention overlay (concatenated horizontally)
    """
    type_keys   = ["correct", "wrong", "uncertain"]
    type_labels = ["Correct", "Wrong", "Uncertain"]
    n_classes   = len(class_names)

    fig, axes = plt.subplots(
        n_classes, 3,
        figsize=(3 * 5, n_classes * 2.5),
        squeeze=False,
    )

    for col, label in enumerate(type_labels):
        axes[0, col].set_title(label, fontsize=11, fontweight="bold", pad=6)

    for row, cls_name in enumerate(class_names):
        axes[row, 0].set_ylabel(
            cls_name, fontsize=10, rotation=0, ha="right", va="center", labelpad=50,
        )
        for col, key in enumerate(type_keys):
            ax = axes[row, col]
            pair = composites[cls_name].get(key)

            if pair is not None:
                orig, overlay = pair
                composite = np.concatenate([orig, overlay], axis=1)  # (S, 2S, 3)
                ax.imshow(composite)
                ex = selected[cls_name].get(key)
                if ex is not None:
                    pred_name = idx_to_class[ex["pred_idx"]]
                    unc_val   = ex["uncertainty"]
                    color = "green" if key == "correct" else ("red" if key == "wrong" else "orange")
                    ax.set_xlabel(
                        f"pred: {pred_name}  unc: {unc_val:.4f}",
                        fontsize=6.5, color=color, labelpad=2,
                    )
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="gray")
                ax.set_facecolor("#eeeeee")

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary grid saved -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DINO ViT-S/16 attention-map explainability for HAM10000"
    )
    p.add_argument(
        "--efficientnet-checkpoint",
        type=Path,
        default=Path("outputs/efficientnet_mc/best_efficientnet.pth"),
        help="EfficientNet checkpoint used to select examples via MC Dropout",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dino_attention"),
    )
    p.add_argument("--metadata-csv", type=Path, default=None)
    p.add_argument("--images-dir",   type=Path, default=None)
    p.add_argument(
        "--seed", type=int, default=None,
        help="Must match the training seed (default: Config.seed = 42)",
    )
    p.add_argument(
        "--mc-passes", type=int, default=20,
        help="MC Dropout passes for EfficientNet uncertainty estimation",
    )
    p.add_argument(
        "--display-size", type=int, default=224,
        help="Pixel size of each panel in saved composites",
    )
    p.add_argument(
        "--heatmap-alpha", type=float, default=0.4,
        help="Opacity of the attention heatmap overlay (same as gradcam.py)",
    )
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

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    eff_ckpt_path = Path(args.efficientnet_checkpoint)
    if not eff_ckpt_path.exists():
        sys.exit(f"[ERROR] EfficientNet checkpoint not found: {eff_ckpt_path}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.metadata_csv is not None:
        cfg.metadata_csv = Path(args.metadata_csv)
    if args.images_dir is not None:
        cfg.images_dir = Path(args.images_dir)

    set_seed(cfg.seed)
    device = get_device()
    print(f"Device : {device}")
    print(f"Seed   : {cfg.seed}")

    # -------------------------------------------------------------------------
    # Load EfficientNet (example selection only — not used for attention maps)
    # -------------------------------------------------------------------------
    print(f"\nLoading EfficientNet checkpoint: {eff_ckpt_path}")
    ckpt = torch.load(eff_ckpt_path, map_location=device, weights_only=False)
    label_map: dict[str, int] = ckpt["label_map"]
    num_classes   = len(label_map)
    idx_to_class  = {v: k for k, v in label_map.items()}
    class_names   = [idx_to_class[i] for i in range(num_classes)]
    val_f1        = ckpt.get("val_macro_f1", float("nan"))
    print(f"Classes ({num_classes}): {label_map}")
    print(f"EfficientNet saved val macro-F1 = {val_f1:.4f}")

    eff_model = build_efficientnet_b0(num_classes=num_classes, pretrained=False).to(device)
    eff_model.load_state_dict(ckpt["model_state_dict"])

    # -------------------------------------------------------------------------
    # Rebuild the identical test split
    # -------------------------------------------------------------------------
    print(f"\nBuilding test split from {cfg.metadata_csv} ...")
    df, _ = _load_multiclass_table(cfg.metadata_csv, cfg)
    _, _, df_test = lesion_level_split(
        df, cfg.col_lesion_id,
        cfg.train_ratio, cfg.val_ratio, cfg.test_ratio,
        cfg.seed,
    )
    df_test = df_test.reset_index(drop=True)
    print(f"Test set : {len(df_test)} images")

    # -------------------------------------------------------------------------
    # EfficientNet MC Dropout → predictions + uncertainty
    # -------------------------------------------------------------------------
    print(f"\nRunning EfficientNet MC Dropout ({args.mc_passes} passes) ...")
    y_true, y_pred, uncertainty = run_mc_inference(
        eff_model, df_test, cfg, args.mc_passes, device,
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"Test macro-F1 : {macro_f1:.4f}")

    # Free EfficientNet memory before loading DINO
    del eff_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Select 3 examples per class (same logic as gradcam.py)
    # -------------------------------------------------------------------------
    print("\nSelecting examples per class ...")
    selected = select_examples(
        df_test, y_true, y_pred, uncertainty, class_names, cfg.col_image_id,
    )

    # -------------------------------------------------------------------------
    # Load DINO ViT-S/16
    # -------------------------------------------------------------------------
    print("\nLoading DINO ViT-S/16 from torch.hub ...")
    dino_model = torch.hub.load(
        "facebookresearch/dino:main", "dino_vits16", pretrained=True, verbose=False,
    )
    dino_model = dino_model.to(device)
    dino_model.eval()

    # Verify the model exposes get_last_selfattention
    if not hasattr(dino_model, "get_last_selfattention"):
        sys.exit(
            "[ERROR] Loaded DINO model does not have get_last_selfattention(). "
            "Check the torch.hub source tag."
        )

    patch_size = 16
    num_heads  = 6
    feat_size  = cfg.image_size // patch_size   # 14

    print(f"DINO ViT-S/{patch_size}: {num_heads} heads, "
          f"{feat_size}x{feat_size} = {feat_size**2} patches per image")

    # -------------------------------------------------------------------------
    # Generate attention maps for every selected example
    # -------------------------------------------------------------------------
    # composites[cls_name][ex_type] = (orig_display, mean_overlay) | None
    composites: dict[str, dict[str, tuple[np.ndarray, np.ndarray] | None]] = {}

    for cls_name in class_names:
        print(f"\n-- {cls_name} --")
        cls_dir = out_root / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        composites[cls_name] = {}

        for ex_type in ("correct", "wrong", "uncertain"):
            ex = selected[cls_name].get(ex_type)
            if ex is None:
                composites[cls_name][ex_type] = None
                continue

            image_id  = ex["image_id"]
            pred_idx  = ex["pred_idx"]
            unc_val   = ex["uncertainty"]
            pred_name = idx_to_class[pred_idx]

            print(
                f"  {ex_type:9s}  {image_id}  "
                f"true={cls_name}  pred={pred_name}  unc={unc_val:.5f}"
            )

            try:
                rgb_np, tensor = load_image_tensor(
                    image_id, cfg.images_dir, cfg.image_size, device,
                )
            except FileNotFoundError as exc:
                print(f"    [WARN] {exc} — skipping")
                composites[cls_name][ex_type] = None
                continue

            # --- Extract DINO attention maps ---------------------------------
            mean_map, head_maps = extract_attention_maps(dino_model, tensor, patch_size)

            # --- Build per-head overlays for the detailed figure -------------
            orig_display, mean_overlay = make_overlay(
                rgb_np, mean_map,
                heatmap_alpha=args.heatmap_alpha,
                display_size=args.display_size,
            )
            head_overlays = []
            for h in range(num_heads):
                _, h_overlay = make_overlay(
                    rgb_np, head_maps[h],
                    heatmap_alpha=args.heatmap_alpha,
                    display_size=args.display_size,
                )
                head_overlays.append(h_overlay)

            # Store mean composite for the summary grid
            composites[cls_name][ex_type] = (orig_display, mean_overlay)

            # --- Build title / subtitle --------------------------------------
            if ex_type == "correct":
                verdict = f"CORRECT  true={cls_name}  pred={pred_name}"
            elif ex_type == "wrong":
                verdict = f"WRONG    true={cls_name}  pred={pred_name}"
            else:
                verdict = f"UNCERTAIN  true={cls_name}  pred={pred_name}"

            title    = f"{cls_name.upper()} | {verdict}"
            subtitle = f"image: {image_id}   MC uncertainty: {unc_val:.5f}"

            # --- Save per-example file (2-row layout with all heads) ---------
            out_file = cls_dir / f"{ex_type}_{image_id}.png"
            save_attention_figure(
                out_file,
                orig_display,
                mean_overlay,
                head_overlays,
                title,
                subtitle,
            )
            print(f"    -> {out_file}")

    # -------------------------------------------------------------------------
    # Summary grid (same format as gradcam.py)
    # -------------------------------------------------------------------------
    print("\nBuilding summary grid ...")
    make_summary_grid(
        composites, class_names, idx_to_class, selected,
        out_root / "summary_grid.png",
    )

    print(f"\nAll outputs saved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
