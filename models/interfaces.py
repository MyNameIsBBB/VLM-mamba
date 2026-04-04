from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class TextBackboneOutput:
    sequence: Tensor
    pooled: Tensor


class VisionBackbone(nn.Module, ABC):
    @abstractmethod
    def forward(self, images: Tensor) -> Tensor:
        raise NotImplementedError


class SequenceBackbone(nn.Module, ABC):
    @abstractmethod
    def forward(self, tokens: Tensor) -> Tensor:
        raise NotImplementedError


class TextBackbone(nn.Module, ABC):
    @abstractmethod
    def forward(self, token_ids: Tensor, attention_mask: Tensor | None = None) -> TextBackboneOutput:
        raise NotImplementedError


class FusionModule(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        vision_tokens: Tensor,
        text_tokens: Tensor,
        text_attention_mask: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError


class PredictionHead(nn.Module, ABC):
    @abstractmethod
    def forward(self, fused_embedding: Tensor) -> Tensor:
        raise NotImplementedError