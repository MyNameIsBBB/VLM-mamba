from __future__ import annotations

from torch import Tensor, nn

from models.interfaces import PredictionHead


class ZeroShotMatchingHead(PredictionHead):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, fused_embedding: Tensor) -> Tensor:
        return self.projection(fused_embedding).squeeze(-1)