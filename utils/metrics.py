from __future__ import annotations

import torch
from torch import Tensor


def sigmoid_confidence(logits: Tensor) -> Tensor:
    return torch.sigmoid(logits)


def binary_accuracy(logits: Tensor, labels: Tensor, threshold: float = 0.5) -> Tensor:
    predictions = sigmoid_confidence(logits) >= threshold
    return (predictions == labels.bool()).float().mean()


def retrieval_topk_accuracy(similarity: Tensor, positive_mask: Tensor, k: int = 1, dim: int = 1) -> Tensor:
    if similarity.ndim != 2 or positive_mask.shape != similarity.shape:
        raise ValueError("similarity and positive_mask must both be [B, B] tensors.")

    if dim == 0:
        similarity = similarity.transpose(0, 1)
        positive_mask = positive_mask.transpose(0, 1)

    topk_indices = similarity.topk(k=min(k, similarity.size(1)), dim=1).indices
    hits = positive_mask.gather(dim=1, index=topk_indices)
    return hits.any(dim=1).float().mean()


def compute_recall_at_k(similarity: Tensor, positive_mask: Tensor, ks: tuple[int, ...]) -> dict[int, float]:
    return {
        k: float(retrieval_topk_accuracy(similarity, positive_mask, k=k).item())
        for k in ks
    }