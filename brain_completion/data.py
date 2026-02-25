from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .configs import DatasetConfig
from .utils import get_logger

logger = get_logger(__name__)

AAL_CUSTOM_LABELS = [
    '2001','2002','2101','2102','2111','2112','2201','2202','2211','2212',
    '2301','2302','2311','2312','2321','2322','2331','2332','2401','2402',
    '2501','2502','2601','2602','2611','2612','2701','2702','3001','3002',
    '4001','4002','4011','4012','4021','4022','4101','4102','4111','4112',
    '4201','4202','5001','5002','5011','5012','5021','5022','5101','5102',
    '5201','5202','5301','5302','5401','5402','6001','6002','6101','6102',
    '6201','6202','6211','6212','6221','6222','6301','6302','6401','6402',
    '7001','7002','7011','7012','7021','7022','7101','7102','8101','8102',
    '8111','8112','8121','8122','8201','8202','8211','8212','8301','8302',
    '9001','9002','9011','9012','9021','9022','9031','9032','9041','9042',
    '9051','9052','9061','9062','9071','9072','9081','9082','9100','9110',
    '9120','9130','9140','9150','9160','9170'
]

def collate_fn_with_dynamic_padding(batch):
    max_time = max(x.shape[-1] for x, _, _ in batch)
    B = len(batch)
    R = batch[0][0].shape[1]
    cond_dim = batch[0][2].shape[0]

    feats = torch.zeros((B, R, max_time))
    masks = torch.zeros((B, R, max_time), dtype=torch.bool)
    conds = torch.zeros((B, cond_dim))

    for i, (x, m, c) in enumerate(batch):
        T = x.shape[-1]
        feats[i, :, :T] = x.squeeze(0)
        masks[i, :, :T] = m.squeeze(0)
        conds[i] = c
    return feats, masks, conds

class ROISequenceDataset(Dataset):
    def __init__(self, data_dict: Dict[str, Any], phenotypic_data: Dict[str, np.ndarray]):
        self.data_dict = data_dict
        self.phenotypic_data = phenotypic_data
        self.subject_ids = list(data_dict.keys())

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        sid = self.subject_ids[idx]
        subj = self.data_dict[sid]
        features = subj["session_1"]["rest_features"]          # (R,T)
        roi_avail = subj["session_1"]["missing_mask"]          # (R,)

        T = features.shape[-1]
        m2d = (np.tile(roi_avail[:, None], (1, T)) == 1)
        mask = torch.tensor(m2d, dtype=torch.bool)
        cond = torch.tensor(self.phenotypic_data[sid], dtype=torch.float32)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        return x, mask, cond

class ROICSVLoader:
    """Load per-subject ROI time series from {sid}.csv (T×R with header)."""
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.phenotypes = self._load_phenotypic_data()

        example_csv = next(self.cfg.roots["train"].glob("*.csv"), None)
        if not example_csv:
            raise RuntimeError(f"No CSV under {self.cfg.roots['train']}")
        n_rois = pd.read_csv(example_csv, header=0).shape[1]

        if self.cfg.atlas_type == "aal":
            self.custom_labels = [int(x) for x in AAL_CUSTOM_LABELS]
        else:
            self.custom_labels = list(range(1, n_rois + 1))

        logger.info(f"Loaded phenotypes: {len(self.phenotypes)} subjects")

    def _load_phenotypic_data(self) -> Dict[str, np.ndarray]:
        df = pd.read_csv(self.cfg.phenotypic_csv)
        df.columns = [c.strip() for c in df.columns]
        req = [self.cfg.id_col, self.cfg.age_col, self.cfg.sex_col]
        miss = set(req) - set(df.columns)
        if miss:
            raise ValueError(f"Phenotypic CSV missing columns: {miss}")

        male_set = set([str(x).upper() for x in (self.cfg.sex_male_values or [])])
        phen = {}
        for _, row in df.iterrows():
            sid = str(row[self.cfg.id_col])
            age = float(row[self.cfg.age_col])

            sex_val = row[self.cfg.sex_col]
            if pd.isna(sex_val):
                sex = 0.0
            else:
                s = str(sex_val).upper().strip()
                sex = 1.0 if (s in male_set or s.startswith("M")) else 0.0
            phen[sid] = np.array([age, sex], dtype=np.float32)
        return phen

    def load_dataset(self, mode: str) -> Tuple[Dict[str, Any], List[str]]:
        root = self.cfg.roots.get(mode)
        if root is None:
            raise ValueError(f"Unknown mode={mode}")
        data, missing = {}, []
        for sid in self.phenotypes:
            item = self._load_subject(root, sid)
            if item:
                data[sid] = item
            else:
                missing.append(sid)
        return data, missing

    def _load_subject(self, root: Path, sid: str) -> Dict[str, Any]:
        p = root / f"{sid}.csv"
        if not p.exists():
            return {}
        df = pd.read_csv(p, header=0)
        features = df.values.astype(np.float32).T   # (R,T)
        missing_mask = (~np.all(features == 0, axis=1)).astype(np.int32)
        return {"session_1": {"rest_features": features, "missing_mask": missing_mask}}

def compute_global_normalization_params(train_dataset: Dict[str, Any], num_rois: int):
    roi_data = [[] for _ in range(num_rois)]
    for subj in train_dataset.values():
        feats = subj["session_1"]["rest_features"]
        for r in range(num_rois):
            roi_data[r].extend(feats[r, :].reshape(-1).tolist())
    mean = np.zeros(num_rois); std = np.zeros(num_rois)
    for r in range(num_rois):
        arr = np.asarray(roi_data[r], dtype=np.float32)
        mean[r] = float(arr.mean())
        std[r] = float(arr.std())
    return mean, std
