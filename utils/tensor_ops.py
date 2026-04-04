from __future__ import annotations

from torch import Tensor


def spatial_to_sequence(features: Tensor) -> tuple[Tensor, tuple[int, int]]:
    batch_size, channels, height, width = features.shape
    # CNN emits [B, C, H, W], while Mamba expects [B, N, C]. Rasterizing H x W keeps local structure explicit.
    tokens = features.flatten(start_dim=2).transpose(1, 2).contiguous()
    return tokens, (height, width)


def sequence_to_spatial(tokens: Tensor, spatial_size: tuple[int, int]) -> Tensor:
    batch_size, sequence_length, channels = tokens.shape
    height, width = spatial_size
    expected_length = height * width
    if sequence_length != expected_length:
        raise ValueError(
            f"Cannot restore [B, N, C] sequence of length {sequence_length} into ({height}, {width})."
        )
    # This inverts spatial_to_sequence so later heads can recover the 2D layout when needed.
    return tokens.transpose(1, 2).contiguous().view(batch_size, channels, height, width)