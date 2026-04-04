from __future__ import annotations

import torch
from torch import Tensor


def sigmoid_confidence(logits: Tensor) -> Tensor:
    return torch.sigmoid(logits)


def binary_accuracy(logits: Tensor, labels: Tensor, threshold: float = 0.5) -> Tensor:
    predictions = sigmoid_confidence(logits) >= threshold
    return (predictions == labels.bool()).float().mean()