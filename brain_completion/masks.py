from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import torch

class DynamicMaskGenerator:
    """MCAR ROI-column mask."""
    def __init__(self, mask_rate: float = 0.3):
        self.mask_rate = float(mask_rate)

    def generate_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        if len(shape) != 3:
            raise ValueError("Expected [B,R,T]")
        B, R, T = shape
        mask = torch.zeros((B, R, T), dtype=torch.bool)
        for b in range(B):
            roi_mask = torch.rand(R) < self.mask_rate
            mask[b, roi_mask, :] = True
        return mask

class StructuredMaskGenerator:
    """mcar / roi_weighted / cluster / block"""
    def __init__(
        self,
        roi_weights: Optional[np.ndarray] = None,
        roi_groups: Optional[List[List[int]]] = None,
        corr_affinity: Optional[np.ndarray] = None,
        seed: int = 0,
        block_size: int = 20,
    ):
        self.roi_weights = roi_weights
        self.roi_groups = roi_groups
        self.corr_affinity = corr_affinity
        self.rng = np.random.RandomState(seed)
        self.block_size = int(block_size)

    def _pick_uniform(self, candidates: np.ndarray, k: int) -> np.ndarray:
        k = min(k, len(candidates))
        return self.rng.choice(candidates, size=k, replace=False)

    def _pick_weighted(self, candidates: np.ndarray, k: int) -> np.ndarray:
        k = min(k, len(candidates))
        if self.roi_weights is None:
            return self._pick_uniform(candidates, k)
        w = self.roi_weights[candidates].astype(np.float64)
        w = w / (w.sum() + 1e-12)
        if np.all(w <= 0) or np.isnan(w).any():
            return self._pick_uniform(candidates, k)
        return self.rng.choice(candidates, size=k, replace=False, p=w)

    def _pick_cluster(self, candidates: np.ndarray, k: int) -> np.ndarray:
        k = min(k, len(candidates))
        seed_roi = self._pick_uniform(candidates, 1)[0]
        if self.corr_affinity is not None:
            aff = self.corr_affinity[seed_roi].copy()
            aff_mask = np.full_like(aff, -np.inf, dtype=np.float64)
            aff_mask[candidates] = aff[candidates]
            return np.argsort(-aff_mask)[:k]
        cand_sorted = np.sort(candidates)
        pos = np.searchsorted(cand_sorted, seed_roi)
        left = right = pos
        chosen = [seed_roi]
        while len(chosen) < k:
            left -= 1; right += 1
            if left >= 0:
                chosen.append(cand_sorted[left])
                if len(chosen) >= k: break
            if right < len(cand_sorted):
                chosen.append(cand_sorted[right])
        return np.array(chosen[:k], dtype=int)

    def _pick_block(self, candidates: np.ndarray, k: int) -> np.ndarray:
        k = min(k, len(candidates))
        if candidates.size == 0:
            return candidates
        if self.roi_groups is None:
            max_idx = int(candidates.max())
            groups = [list(range(i, i + self.block_size)) for i in range(0, max_idx + 1, self.block_size)]
        else:
            groups = self.roi_groups
        cand_set = set(candidates.tolist())
        valid = []
        for g in groups:
            gg = [x for x in g if x in cand_set]
            if gg:
                valid.append(gg)
        if not valid:
            return self._pick_uniform(candidates, k)
        self.rng.shuffle(valid)
        chosen = []
        for g in valid:
            chosen.extend(g)
            if len(chosen) >= k:
                break
        return np.array(chosen[:k], dtype=int)

    def generate_mask(self, features: torch.Tensor, roi_avail_mask: torch.Tensor, rate: float, pattern: str) -> torch.Tensor:
        assert 0 < rate < 1
        if features.dim() != 3:
            raise ValueError("features must be [B,R,T]")
        B, R, T = features.shape
        roi_available = roi_avail_mask.any(dim=-1)
        mask = torch.zeros((B, R, T), dtype=torch.bool, device=features.device)
        for b in range(B):
            candidates = torch.where(roi_available[b])[0].detach().cpu().numpy()
            if candidates.size == 0:
                continue
            k = max(1, int(np.floor(rate * len(candidates))))
            if pattern == "mcar":
                rois = self._pick_uniform(candidates, k)
            elif pattern == "roi_weighted":
                rois = self._pick_weighted(candidates, k)
            elif pattern == "cluster":
                rois = self._pick_cluster(candidates, k)
            elif pattern == "block":
                rois = self._pick_block(candidates, k)
            else:
                raise ValueError(f"Unknown pattern={pattern}")
            mask[b, rois, :] = True
        return mask
