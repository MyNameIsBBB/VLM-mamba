from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class SequenceBackbone(nn.Module, ABC):
    @abstractmethod
    def forward(self, tokens: Tensor) -> Tensor:
        raise NotImplementedError