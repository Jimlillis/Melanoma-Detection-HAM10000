"""
Utilities: reproducibility, image path resolution, and evaluation metrics.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)


def set_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch (CPU and CUDA if available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic behavior can slow training; enable if you need full reproducibility on GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def resolve_image_path(images_dir: Path, image_id: str) -> Path:
    """
    Resolve path to a HAM10000 image file given image_id (without extension).

    Tries, in order:
    - images_dir / f"{image_id}.jpg"
    - common subfolders used in HAM10000 releases (part_1, part_2, etc.)
    """
    candidates = [
        images_dir / f"{image_id}.jpg",
        images_dir / "HAM10000_images_part_1" / f"{image_id}.jpg",
        images_dir / "HAM10000_images_part_2" / f"{image_id}.jpg",
        images_dir / "part1" / f"{image_id}.jpg",
        images_dir / "part2" / f"{image_id}.jpg",
    ]
    for path in candidates:
        if path.is_file():
            return path
    # Fallback: shallow search under images_dir
    for root, _, files in os.walk(images_dir):
        name = f"{image_id}.jpg"
        if name in files:
            return Path(root) / name
    raise FileNotFoundError(
        f"Could not find image for image_id={image_id!r} under {images_dir}"
    )


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    """
    Compute accuracy, macro-F1, melanoma recall (positive class), and ROC-AUC.

    Labels are expected as integers {0: non-melanoma, 1: melanoma}.
    y_score should be predicted probability of class 1 (melanoma), shape (n_samples,).
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    mel_recall = float(
        recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    )
    try:
        roc_auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        # e.g. only one class present in y_true
        roc_auc = float("nan")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "melanoma_recall": mel_recall,
        "roc_auc": roc_auc,
    }


def format_metrics(m: Dict[str, float]) -> str:
    parts = []
    for k in ("accuracy", "macro_f1", "melanoma_recall", "roc_auc"):
        v = m.get(k, float("nan"))
        if k == "roc_auc" and np.isnan(v):
            parts.append(f"{k}=nan")
        else:
            parts.append(f"{k}={v:.4f}")
    return ", ".join(parts)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
