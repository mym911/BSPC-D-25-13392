from __future__ import annotations
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .configs import ModelConfig, DatasetConfig
from .data import ROICSVLoader
from .models import ROICompletionGenerator
from .utils import get_device, get_logger

logger = get_logger(__name__)
device = get_device()

class BrainCompletionTester:
    def __init__(self, model_cfg: ModelConfig, data_cfg: DatasetConfig):
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.loader = ROICSVLoader(data_cfg)
        self.model = ROICompletionGenerator(model_cfg.cond_dim, len(self.loader.custom_labels)).to(device)
        self.model.eval()
        self.mean = None
        self.std = None

    def load_model(self, model_path: Path):
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def load_norm(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std

    def impute_real_missing(self) -> Dict[str, np.ndarray]:
        test_ds, _ = self.loader.load_dataset("test")
        results = {}

        for sid, subj in test_ds.items():
            features = subj["session_1"]["rest_features"]      # (R,T)
            missing_mask = np.asarray(subj["session_1"]["missing_mask"]).reshape(-1)  # 1=available
            missing_roi = (missing_mask == 0)
            all_zero_roi = np.all(features == 0, axis=1)
            combined_missing = missing_roi | all_zero_roi

            if self.mean is not None and self.std is not None:
                norm = (features - self.mean[:, None]) / (self.std[:, None] + 1e-8)
            else:
                norm = features.copy()

            masked = norm.copy()
            masked[combined_missing, :] = 0.0

            roi_mask_2d = (missing_mask[:, None] == 1)
            test_mask = torch.tensor(roi_mask_2d, dtype=torch.bool).unsqueeze(0).to(device)
            test_mask = test_mask.expand(1, test_mask.shape[1], masked.shape[1])

            x = torch.tensor(masked, dtype=torch.float32).unsqueeze(0).to(device)
            cond_np = self.loader.phenotypes.get(sid, np.zeros(self.model_cfg.cond_dim, dtype=np.float32))
            cond = torch.tensor(cond_np, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                completed = self.model(x, cond=cond, mask=test_mask).squeeze(0).cpu().numpy()

            if self.mean is not None and self.std is not None:
                completed = completed * (self.std[:, None] + 1e-8) + self.mean[:, None]

            final = features.copy()
            final[combined_missing, :] = completed[combined_missing, :]
            results[sid] = final

        return results
