"""
Grad-CAM explainability for the 7-class EfficientNet-B0.

Loads the best checkpoint from outputs/efficientnet_mc/ (two-phase training +
TTA + MC Dropout), rebuilds the identical lesion-level test split (seed=42),
and for each of the 7 HAM10000 classes produces three annotated images:

  correct   – an example the model classified correctly (lowest uncertainty)
  wrong     – an example the model misclassified       (highest uncertainty wrong)
  uncertain – the example with the highest MC Dropout variance

Each image is saved as a side-by-side composite (original | Grad-CAM overlay)
in outputs/gradcam/<class_name>/.  A 7×3 summary grid is also written to
outputs/gradcam/summary_grid.png.

Grad-CAM target layer: model.features[-1]  (final Conv2dNormActivation,
outputs (B, 1280, 7, 7) for 224×224 input).  The heatmap explains the
*predicted* class — what the model "saw" when making its decision.

Run from the project root:
  python src/gradcam.py
  python src/gradcam.py --checkpoint outputs/efficientnet_mc/best_efficientnet.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for servers without display
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
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Grad-CAM (Selvaraju et al., 2017) using forward/backward hooks.

    The CAM is: ReLU( sum_c( alpha_c * A_c ) )
    where alpha_c = global-average-pool( dScore / dA_c ).

    Usage::

        gradcam = GradCAM(model, model.features[-1])
        cam = gradcam(input_tensor, class_idx)   # (H_feat, W_feat) in [0, 1]
        gradcam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, output) -> None:
        self._activations = output

    def _save_gradients(self, module, grad_in, grad_out) -> None:
        # grad_out[0]: gradient of the loss w.r.t. this layer's output
        self._gradients = grad_out[0]

    def __call__(
        self,
        input_tensor: torch.Tensor,  # (1, C, H, W) on correct device
        class_idx: int,
    ) -> np.ndarray:
        """
        Returns a float32 numpy array of shape (H_feat, W_feat) in [0, 1].
        For model.features[-1] with 224×224 input this is (7, 7).
        """
        self.model.eval()
        self.model.zero_grad()

        with torch.enable_grad():
            logits = self.model(input_tensor)   # (1, num_classes)
            score = logits[0, class_idx]
            score.backward()

        grads = self._gradients.detach()               # (1, C, H_f, W_f)
        acts  = self._activations.detach()             # (1, C, H_f, W_f)
        alpha = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam   = (alpha * acts).sum(dim=1).squeeze(0)   # (H_f, W_f)
        cam   = torch.relu(cam).cpu().numpy().astype(np.float32)

        cam_max = cam.max()
        if cam_max > 0:
            cam /= cam_max
        return cam

    def remove_hooks(self) -> None:
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_multiclass_table(
    csv_path: Path,
    cfg: Config,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Read HAM10000 CSV and add ``_binary_label`` = 7-class integer label.
    Alphabetical sort: akiec=0, bcc=1, bkl=2, df=3, mel=4, nv=5, vasc=6.
    (Column is named ``_binary_label`` for compatibility with lesion_level_split.)
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
    Load a single image and return:
      rgb_np : (H_orig, W_orig, 3) uint8 — original resolution, for display
      tensor : (1, 3, image_size, image_size) float32 on device — model input
    """
    path = resolve_image_path(images_dir, image_id)
    img_pil = Image.open(path).convert("RGB")
    rgb_np = np.array(img_pil)
    transform = build_transforms(image_size, is_train=False)
    tensor = transform(img_pil).unsqueeze(0).to(device)
    return rgb_np, tensor


# ---------------------------------------------------------------------------
# MC Dropout inference
# ---------------------------------------------------------------------------

def run_mc_inference(
    model: nn.Module,
    df_test: pd.DataFrame,
    cfg: Config,
    num_classes: int,
    mc_passes: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run MC Dropout inference on the full test split.

    Activates Dropout layers during inference for stochastic predictions.
    EfficientNet-B0's classifier has Dropout(p=0.2), giving well-calibrated
    uncertainty estimates.

    Returns
    -------
    y_true      : (N,) int — ground-truth 7-class labels
    y_pred      : (N,) int — predicted class (argmax of mean MC probabilities)
    uncertainty : (N,) float — mean per-image variance across mc_passes
    """
    from torch.utils.data import DataLoader

    transform = build_transforms(cfg.image_size, is_train=False)
    dataset = SkinLesionDataset(df_test, cfg.images_dir, cfg.col_image_id, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # model.eval() stops BatchNorm from updating running stats;
    # enabling Dropout layers keeps them stochastic for MC sampling.
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    all_y_true: list[int] = []
    all_y_pred: list[int] = []
    all_unc: list[float] = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"MC Dropout ({mc_passes} passes)"):
            x = x.to(device)
            probs_list = []
            for _ in range(mc_passes):
                logits = model(x)
                probs_list.append(torch.softmax(logits, dim=1).cpu())

            stacked   = torch.stack(probs_list, dim=0)        # (K, B, C)
            mean_probs = stacked.mean(dim=0)                   # (B, C)
            preds      = mean_probs.argmax(dim=1).tolist()     # B ints
            unc        = stacked.var(dim=0).mean(dim=1).tolist()  # B floats

            all_y_true.extend(y.tolist())
            all_y_pred.extend(preds)
            all_unc.extend(unc)

    return np.array(all_y_true), np.array(all_y_pred), np.array(all_unc)


# ---------------------------------------------------------------------------
# Example selection
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
    For each class select up to 3 representative test examples:

      correct   – true==pred==class; pick the *most confident* one (lowest uncertainty)
      wrong     – true==class, pred!=class; pick the *most uncertain* wrong prediction
      uncertain – true==class; pick the highest MC Dropout variance (may overlap correct/wrong)

    Returns
    -------
    result[class_name][example_type] -> dict with keys:
        image_id, true_idx, pred_idx, uncertainty, df_idx
    or None if no such example exists.
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

        def _make_entry(idx: int) -> dict:
            row = df_reset.iloc[idx]
            return {
                "image_id": str(row[col_image_id]),
                "true_idx": cls_idx,
                "pred_idx": int(y_pred[idx]),
                "uncertainty": float(uncertainty[idx]),
                "df_idx": int(idx),
            }

        # Correct: true==pred==cls_idx, lowest uncertainty
        correct_idx = np.where((y_true == cls_idx) & (y_pred == cls_idx))[0]
        if len(correct_idx) > 0:
            best = correct_idx[uncertainty[correct_idx].argmin()]
            examples["correct"] = _make_entry(best)
        else:
            print(f"  [WARN] No correct predictions for class '{cls_name}'")

        # Wrong: true==cls_idx, pred!=cls_idx, highest uncertainty (most interesting case)
        wrong_idx = np.where((y_true == cls_idx) & (y_pred != cls_idx))[0]
        if len(wrong_idx) > 0:
            best = wrong_idx[uncertainty[wrong_idx].argmax()]
            examples["wrong"] = _make_entry(best)
        else:
            print(f"  [WARN] No misclassifications for class '{cls_name}'")

        # Uncertain: true==cls_idx, highest MC variance (regardless of correctness)
        best = cls_indices[uncertainty[cls_indices].argmax()]
        examples["uncertain"] = _make_entry(best)

        result[cls_name] = examples

    return result


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def make_overlay(
    original_rgb: np.ndarray,  # (H, W, 3) uint8 RGB
    cam: np.ndarray,           # (H_f, W_f) float32 in [0, 1]
    heatmap_alpha: float = 0.4,
    display_size: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize original and CAM to display_size, then blend.

    Colormap: cv2.COLORMAP_JET — blue=low activation, red=high activation.
    This is the standard Grad-CAM colormap (high-importance regions appear red/warm).

    Returns
    -------
    orig_display : (display_size, display_size, 3) uint8 RGB
    overlay      : (display_size, display_size, 3) uint8 RGB
    """
    orig_display = cv2.resize(
        original_rgb, (display_size, display_size), interpolation=cv2.INTER_LINEAR
    )

    cam_uint8 = (cam * 255).astype(np.uint8)
    cam_resized = cv2.resize(cam_uint8, (display_size, display_size), interpolation=cv2.INTER_LINEAR)

    # applyColorMap expects single-channel; returns BGR
    heatmap_bgr = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(orig_display, 1.0 - heatmap_alpha, heatmap_rgb, heatmap_alpha, 0)
    return orig_display, overlay


def save_example(
    out_path: Path,
    orig: np.ndarray,
    overlay: np.ndarray,
    title: str,
    subtitle: str = "",
) -> None:
    """Save a side-by-side original | Grad-CAM composite with annotation."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))

    axes[0].imshow(orig)
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM overlay", fontsize=9)
    axes[1].axis("off")

    fig.suptitle(title, fontsize=8, fontweight="bold", y=1.01)
    if subtitle:
        fig.text(0.5, -0.01, subtitle, ha="center", fontsize=7, color="#555555")

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
    7 × 3 grid: rows = classes, columns = correct / wrong / uncertain.
    Each cell shows (original | Grad-CAM overlay) side by side.
    Column headers and row class labels are drawn with matplotlib.
    """
    type_keys   = ["correct", "wrong", "uncertain"]
    type_labels = ["Correct", "Wrong", "Uncertain"]
    n_classes   = len(class_names)

    # Each cell = two images side-by-side → 2 × display_size wide
    fig, axes = plt.subplots(
        n_classes, 3,
        figsize=(3 * 5, n_classes * 2.5),
        squeeze=False,
    )

    for col, label in enumerate(type_labels):
        axes[0, col].set_title(label, fontsize=11, fontweight="bold", pad=6)

    for row, cls_name in enumerate(class_names):
        # Row label on the left-most axis
        axes[row, 0].set_ylabel(
            cls_name, fontsize=10, rotation=0, ha="right", va="center", labelpad=50,
        )

        for col, key in enumerate(type_keys):
            ax = axes[row, col]
            pair = composites[cls_name].get(key)

            if pair is not None:
                orig, overlay = pair
                composite = np.concatenate([orig, overlay], axis=1)  # (H, 2W, 3)
                ax.imshow(composite)

                # Small annotation: predicted class
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
                ax.text(
                    0.5, 0.5, "N/A",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color="gray",
                )
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
        description="Grad-CAM explainability for the 7-class EfficientNet-B0"
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/efficientnet_mc/best_efficientnet.pth"),
        help="Path to best_efficientnet.pth saved by train_efficientnet.py",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gradcam"),
    )
    p.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Override config default (data/HAM10000_metadata.csv)",
    )
    p.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Override config default (data/)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed; must match the seed used during training (default: Config.seed=42)",
    )
    p.add_argument(
        "--mc-passes",
        type=int,
        default=20,
        help="MC Dropout passes for uncertainty estimation (default: 20)",
    )
    p.add_argument(
        "--display-size",
        type=int,
        default=224,
        help="Pixel size of each image in saved composites (default: 224)",
    )
    p.add_argument(
        "--heatmap-alpha",
        type=float,
        default=0.4,
        help="Opacity of the Grad-CAM heatmap overlay (default: 0.4)",
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
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        sys.exit(f"[ERROR] Checkpoint not found: {checkpoint_path}")

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
    # Load checkpoint
    # -------------------------------------------------------------------------
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    label_map: dict[str, int] = ckpt["label_map"]
    num_classes = len(label_map)
    idx_to_class: dict[int, str] = {v: k for k, v in label_map.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]
    print(f"Classes ({num_classes}): {label_map}")

    model = build_efficientnet_b0(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    val_f1 = ckpt.get("val_macro_f1", float("nan"))
    print(f"Model loaded  (saved val macro-F1 = {val_f1:.4f})")

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
    # MC Dropout inference → predictions + uncertainty for every test image
    # -------------------------------------------------------------------------
    print(f"\nRunning MC Dropout inference ({args.mc_passes} passes) ...")
    y_true, y_pred, uncertainty = run_mc_inference(
        model, df_test, cfg, num_classes, args.mc_passes, device,
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"Test macro-F1 : {macro_f1:.4f}")

    # Per-class accuracy summary
    print("\nPer-class prediction counts (test set):")
    for cls_idx, cls_name in enumerate(class_names):
        mask = y_true == cls_idx
        n_total   = mask.sum()
        n_correct = ((y_true == cls_idx) & (y_pred == cls_idx)).sum()
        print(f"  {cls_name:6s}  {n_correct:3d}/{n_total:3d} correct")

    # -------------------------------------------------------------------------
    # Select 3 examples per class
    # -------------------------------------------------------------------------
    print("\nSelecting examples per class ...")
    selected = select_examples(
        df_test, y_true, y_pred, uncertainty, class_names, cfg.col_image_id,
    )

    # -------------------------------------------------------------------------
    # Grad-CAM setup
    # -------------------------------------------------------------------------
    # Target: last convolutional block of EfficientNet-B0
    # model.features[-1] = Conv2dNormActivation → output (B, 1280, 7, 7)
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)
    print(f"\nGrad-CAM target layer: model.features[-1]  "
          f"({type(target_layer).__name__})")

    # -------------------------------------------------------------------------
    # Generate Grad-CAM composites for every selected example
    # -------------------------------------------------------------------------
    # composites[cls_name][ex_type] = (orig_display, overlay) | None
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
            true_idx  = ex["true_idx"]
            unc_val   = ex["uncertainty"]
            pred_name = idx_to_class[pred_idx]

            print(
                f"  {ex_type:9s}  {image_id}  "
                f"true={cls_name}  pred={pred_name}  unc={unc_val:.5f}"
            )

            # Load image (original + normalised tensor)
            try:
                rgb_np, tensor = load_image_tensor(
                    image_id, cfg.images_dir, cfg.image_size, device,
                )
            except FileNotFoundError as exc:
                print(f"    [WARN] {exc} — skipping")
                composites[cls_name][ex_type] = None
                continue

            # Grad-CAM: explain the *predicted* class score
            cam = gradcam(tensor, pred_idx)

            # Build overlay composite
            orig_display, overlay = make_overlay(
                rgb_np, cam,
                heatmap_alpha=args.heatmap_alpha,
                display_size=args.display_size,
            )
            composites[cls_name][ex_type] = (orig_display, overlay)

            # Title lines for the saved file
            if ex_type == "correct":
                verdict = f"CORRECT  true={cls_name}  pred={pred_name}"
            elif ex_type == "wrong":
                verdict = f"WRONG    true={cls_name}  pred={pred_name}"
            else:
                verdict = f"UNCERTAIN  true={cls_name}  pred={pred_name}"

            title    = f"{cls_name.upper()} | {verdict}"
            subtitle = f"image: {image_id}   MC uncertainty: {unc_val:.5f}"

            out_file = cls_dir / f"{ex_type}_{image_id}.png"
            save_example(out_file, orig_display, overlay, title, subtitle)
            print(f"    -> {out_file}")

    gradcam.remove_hooks()

    # -------------------------------------------------------------------------
    # Summary grid
    # -------------------------------------------------------------------------
    print("\nBuilding summary grid ...")
    make_summary_grid(
        composites, class_names, idx_to_class, selected,
        out_root / "summary_grid.png",
    )

    print(f"\nAll outputs saved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()