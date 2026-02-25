from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class PathConfig:
    phenotypic_csv: Path
    ts_dir_aal: Path
    ts_dir_sch: Path

@dataclass
class TrainConfig:
    pca_components: int = 120
    init_lr: float = 1e-3
    weight_decay: float = 5e-4
    k: int = 10
    num_rounds: int = 200
    local_epochs: int = 2
    num_clients: int = 20
    n_splits: int = 50
    seeds: List[int] = None

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [0, 1, 2]
