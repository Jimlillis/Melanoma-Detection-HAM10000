"""
HAM10000 metadata loading, lesion-level split, and PyTorch Dataset.
Binary task: melanoma (1) vs non-melanoma (0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from helpers import resolve_image_path


def load_metadata_table(
    csv_path: Path,
    col_image_id: str,
    col_lesion_id: str,
    col_diagnosis: str,
    melanoma_class: str,
) -> pd.DataFrame:
    """Read HAM10000 CSV and add binary label: 1 = melanoma, 0 = other."""
    df = pd.read_csv(csv_path)
    required = {col_image_id, col_lesion_id, col_diagnosis}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df.copy()
    df["_binary_label"] = (df[col_diagnosis].astype(str) == melanoma_class).astype(int)
    return df


def lesion_level_split(
    df: pd.DataFrame,
    col_lesion_id: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split at lesion level so all images of a lesion stay in one fold.
    Uses stratification on the lesion's binary label (majority = single dx per lesion in HAM10000).
    """
    # One row per lesion: label should be consistent across images of the same lesion
    g = df.groupby(col_lesion_id, as_index=False).agg(
        _label=("_binary_label", "max"),
    )
    lesion_ids = g[col_lesion_id].values
    y = g["_label"].values

    # First split: train vs (val + test)
    val_plus_test = val_ratio + test_ratio
    if val_plus_test <= 0:
        raise ValueError("val_ratio + test_ratio must be positive")

    s1 = train_test_split(
        lesion_ids,
        y,
        test_size=val_plus_test,
        stratify=y,
        random_state=seed,
    )
    train_ids, temp_ids, y_train, y_temp = s1[0], s1[1], s1[2], s1[3]

    # Second split: val vs test
    rel_test = test_ratio / val_plus_test
    s2 = train_test_split(
        temp_ids,
        y_temp,
        test_size=rel_test,
        stratify=y_temp,
        random_state=seed,
    )
    val_ids, test_ids = s2[0], s2[1]

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    df_train = df[df[col_lesion_id].isin(train_set)].reset_index(drop=True)
    df_val = df[df[col_lesion_id].isin(val_set)].reset_index(drop=True)
    df_test = df[df[col_lesion_id].isin(test_set)].reset_index(drop=True)

    return df_train, df_val, df_test


def build_transforms(image_size: int, is_train: bool) -> transforms.Compose:
    """Resize to image_size, augment lightly on train."""
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Return a WeightedRandomSampler that upsamples the minority class so each
    epoch sees roughly equal numbers of melanoma and non-melanoma images.

    Weight per sample = 1 / count_of_its_class.
    num_samples is set to len(df) so epoch length stays the same as unweighted.
    """
    labels = df["_binary_label"].values
    class_counts = np.bincount(labels)  # [n_neg, n_pos]
    weights_per_class = 1.0 / class_counts.astype(float)
    sample_weights = torch.tensor(
        [weights_per_class[lbl] for lbl in labels], dtype=torch.double
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


class SkinLesionDataset(Dataset):
    """Loads images by image_id from a resolved path; labels from _binary_label column."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        images_dir: Path,
        col_image_id: str,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.col_image_id = col_image_id
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image_id = str(row[self.col_image_id])
        label = int(row["_binary_label"])
        path = resolve_image_path(self.images_dir, image_id)
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)
