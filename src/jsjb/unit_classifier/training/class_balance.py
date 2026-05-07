from __future__ import annotations

import numpy as np
import torch


def effective_num_class_weights(
    labels,
    num_classes: int,
    beta: float = 0.9999,
    power: float = 1.0,
    min_weight: float = 0.2,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """Class-Balanced weights from effective number of samples.

    Formula from "Class-Balanced Loss Based on Effective Number of Samples":
    weight_c = (1 - beta) / (1 - beta ** n_c), then normalized to mean 1.
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_arr, minlength=int(num_classes)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    beta = float(np.clip(beta, 0.0, 0.999999))
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    weights = weights / max(weights.mean(), 1e-12)
    if power != 1.0:
        weights = np.power(weights, float(power))
        weights = weights / max(weights.mean(), 1e-12)
    weights = np.clip(weights, float(min_weight), float(max_weight))
    return torch.tensor(weights, dtype=torch.float32)


def balanced_class_weights(
    labels,
    num_classes: int,
    power: float = 0.6,
    min_weight: float = 0.5,
    max_weight: float = 5.0,
) -> torch.Tensor:
    labels_arr = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_arr, minlength=int(num_classes)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    total = counts.sum()
    weights = total / (float(num_classes) * counts)
    weights = np.power(weights, float(power))
    weights = np.clip(weights, float(min_weight), float(max_weight))
    return torch.tensor(weights, dtype=torch.float32)


def build_class_weights(
    labels,
    num_classes: int,
    strategy: str = "balanced",
    power: float = 0.6,
    min_weight: float = 0.5,
    max_weight: float = 5.0,
    effective_beta: float = 0.9999,
) -> torch.Tensor:
    strategy = str(strategy or "balanced").lower()
    if strategy in {"none", "off", "disabled"}:
        return torch.ones(int(num_classes), dtype=torch.float32)
    if strategy in {"effective_num", "class_balanced", "cb"}:
        return effective_num_class_weights(
            labels,
            num_classes=int(num_classes),
            beta=effective_beta,
            power=power,
            min_weight=min_weight,
            max_weight=max_weight,
        )
    return balanced_class_weights(
        labels,
        num_classes=int(num_classes),
        power=power,
        min_weight=min_weight,
        max_weight=max_weight,
    )
