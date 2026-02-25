from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def brier_score_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(np.float32)
    y_prob = y_prob.astype(np.float32)
    return float(np.mean((y_prob - y_true) ** 2))

def ece_score_binary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = y_true.astype(np.int32)
    y_prob = y_prob.astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        cnt = int(mask.sum())
        if cnt == 0:
            continue

        acc_bin = float(np.mean(y_true[mask] == (y_prob[mask] >= 0.5)))
        conf_bin = float(np.mean(y_prob[mask]))
        ece += (cnt / n) * abs(acc_bin - conf_bin)

    return float(ece)

def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, save_path: Path, n_bins: int = 10):
    y_true = y_true.astype(np.int32)
    y_prob = y_prob.astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys = [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        if mask.sum() == 0:
            continue

        conf_bin = float(np.mean(y_prob[mask]))
        pred_bin = (y_prob[mask] >= 0.5).astype(np.int32)
        acc_bin = float(np.mean(pred_bin == y_true[mask]))

        xs.append(conf_bin)
        ys.append(acc_bin)

    plt.figure(figsize=(4.5, 4.5))
    plt.plot([0, 1], [0, 1])
    plt.scatter(xs, ys)
    plt.xlabel("Confidence (mean predicted probability)")
    plt.ylabel("Accuracy (empirical)")
    plt.title("Reliability Diagram")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def sens_spec_from_binary(y_true: np.ndarray, y_pred: np.ndarray):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sen = tp / (tp + fn + 1e-12)
    spe = tn / (tn + fp + 1e-12)
    return float(sen), float(spe)
