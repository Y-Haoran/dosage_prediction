from __future__ import annotations

import numpy as np


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    positives = int(y_true.sum())
    negatives = int((1 - y_true).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_true = y_true[order]
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    tpr = tps / positives
    fpr = fps / negatives
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    return float(np.trapz(tpr, fpr))


def binary_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_true = y_true[order]
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / positives
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.trapz(precision, recall))


def binary_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    return float(np.mean((y_true - y_prob) ** 2))

