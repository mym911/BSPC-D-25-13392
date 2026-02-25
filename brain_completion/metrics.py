import numpy as np

def r2_global(feats_np: np.ndarray, pred_np: np.ndarray, sim_mask_np: np.ndarray) -> float:
    ss_res = np.sum((feats_np[sim_mask_np] - pred_np[sim_mask_np]) ** 2)
    ss_tot = np.sum((feats_np - feats_np.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-8))
