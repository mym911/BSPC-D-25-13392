from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

@dataclass
class ModelConfig:
    lr: float = 1e-4
    epochs_pretrain: int = 100
    epochs_gan: int = 200
    batch_size: int = 8
    mask_rate: float = 0.1
    cond_dim: int = 2

@dataclass
class DatasetConfig:
    phenotypic_csv: Path
    roots: Dict[str, Path]
    atlas_type: str  # "aal" or "schaefer200"
    model_dir: Path

    # phenotype column mapping (ABIDE/ADHD compatible)
    id_col: str = "image_id"
    age_col: str = "entry_age"
    sex_col: str = "PTGENDER"
    sex_male_values: Optional[List[str]] = None

    def __post_init__(self):
        if self.sex_male_values is None:
            self.sex_male_values = ["M", "MALE", "1"]
