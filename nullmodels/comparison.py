import numpy as np


def d_accuracy(y_true: np.ndarray, y0: np.ndarray, y1: np.ndarray) -> float:
    return np.mean(y0 == y_true) - np.mean(y1 == y_true)


def d_precision(y_true: np.ndarray, y0: np.ndarray, y1: np.ndarray) -> float:
    idx0, idx1 = y0 == 1, y1 == 1
    return np.mean(y0[idx0] == y_true[idx0]) - np.mean(y1[idx1] == y_true[idx1])


def d_recall(y_true: np.ndarray, y0: np.ndarray, y1: np.ndarray) -> float:
    idx = y_true == 1
    return np.mean(y0[idx] == y_true[idx]) - np.mean(y1[idx] == y_true[idx])


__all__ = [
    "d_recall",
    "d_accuracy",
    "d_precision",
]
