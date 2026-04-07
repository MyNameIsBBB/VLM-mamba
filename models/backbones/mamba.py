from __future__ import annotations

from typing import Final
import warnings

import torch
from torch import Tensor, nn

from models.interfaces import SequenceBackbone

try:
    from mamba_ssm import Mamba as MambaLayer
except ImportError:
    MambaLayer = None


class FallbackSelectiveSSM(nn.Module):
    def __init__(self, dim: int, conv_kernel: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(dim, dim)
        self.depthwise_conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=dim,
        )
        self.gate = nn.Linear(dim, dim)
        self.output_projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        projected = self.input_projection(tokens)
        mixed = self.depthwise_conv(projected.transpose(1, 2)).transpose(1, 2)
        mixed = mixed[:, : tokens.size(1), :]
        gated = torch.tanh(mixed) * torch.sigmoid(self.gate(tokens))
        return self.dropout(self.output_projection(gated))


class MambaSequenceBackbone(SequenceBackbone):
    _warned: Final[bool] = False

    def __init__(
        self,
        dim: int,
        depth: int,
        state_dim: int,
        conv_kernel: int,
        expand: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.layers = nn.ModuleList()
        self.cpu_fallback_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.using_mamba = MambaLayer is not None
        self._cpu_fallback_warning_shown = False

        if MambaLayer is None:
            warnings.warn(
                "mamba_ssm is not installed. Falling back to a lightweight selective SSM stub.",
                stacklevel=2,
            )
            for _ in range(depth):
                self.layers.append(FallbackSelectiveSSM(dim=dim, conv_kernel=conv_kernel, dropout=dropout))
        else:
            for _ in range(depth):
                self.layers.append(
                    MambaLayer(
                        d_model=dim,
                        d_state=state_dim,
                        d_conv=conv_kernel,
                        expand=expand,
                    )
                )
                self.cpu_fallback_layers.append(
                    FallbackSelectiveSSM(dim=dim, conv_kernel=conv_kernel, dropout=dropout)
                )

        self.output_norm = nn.LayerNorm(dim)

    def forward(self, tokens: Tensor) -> Tensor:
        hidden = tokens
        layers = self.layers
        if self.using_mamba and not hidden.is_cuda:
            if not self._cpu_fallback_warning_shown:
                warnings.warn(
                    "mamba_ssm CUDA kernels require CUDA tensors. Using CPU fallback selective SSM layers instead.",
                    stacklevel=2,
                )
                self._cpu_fallback_warning_shown = True
            layers = self.cpu_fallback_layers

        for norm, layer in zip(self.norms, layers):
            hidden = hidden + self.dropout(layer(norm(hidden)))
        return self.output_norm(hidden)