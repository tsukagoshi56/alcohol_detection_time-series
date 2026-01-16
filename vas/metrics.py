"""Metrics utilities for classification."""

from typing import Dict, List, Tuple

import numpy as np


def confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision_recall_f1(cm: np.ndarray, class_idx: int) -> Tuple[float, float, float]:
    tp = float(cm[class_idx, class_idx])
    fp = float(cm[:, class_idx].sum() - tp)
    fn = float(cm[class_idx, :].sum() - tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def macro_f1(cm: np.ndarray) -> float:
    f1s = []
    for i in range(cm.shape[0]):
        _, _, f1 = precision_recall_f1(cm, i)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def accuracy(cm: np.ndarray) -> float:
    total = cm.sum()
    if total == 0:
        return 0.0
    return float(np.trace(cm) / total)


def classification_report(cm: np.ndarray) -> Dict[str, float]:
    acc = accuracy(cm)
    macro = macro_f1(cm)
    p0, r0, f10 = precision_recall_f1(cm, 0)
    return {
        "accuracy": acc,
        "macro_f1": macro,
        "class0_precision": p0,
        "class0_recall": r0,
        "class0_f1": f10,
    }
