"""
Default configuration for the HAM10000 melanoma vs non-melanoma baseline.
Override via command-line arguments in train.py / evaluate.py.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class Config:
    # --- Data paths (set to your local layout) ---
    metadata_csv: Path = Path("data/HAM10000_metadata.csv")
    images_dir: Path = Path("data/images")
    output_dir: Path = Path("outputs/baseline")

    # --- CSV column names (HAM10000 defaults) ---
    col_image_id: str = "image_id"
    col_lesion_id: str = "lesion_id"
    col_diagnosis: str = "dx"
    melanoma_class: str = "mel"

    # --- Split (lesion-level, stratified) ---
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # --- Training ---
    seed: int = 42
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    image_size: int = 224

    # --- Model ---
    pretrained: bool = True

    # --- Checkpoint ---
    checkpoint_name: str = "best_model.pth"

    def __post_init__(self) -> None:
        ratios: Tuple[float, float, float] = (
            self.train_ratio,
            self.val_ratio,
            self.test_ratio,
        )
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must sum to 1.0, got {sum(ratios)}"
            )

    def checkpoint_path(self) -> Path:
        return self.output_dir / self.checkpoint_name
