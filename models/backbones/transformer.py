from __future__ import annotations

import torch
from torch import Tensor, nn

from models.interfaces import TextBackbone, TextBackboneOutput


class TextTransformerEncoder(TextBackbone):
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        dim: int,
        depth: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_length, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, token_ids: Tensor, attention_mask: Tensor | None = None) -> TextBackboneOutput:
        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.max_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds the configured maximum of {self.max_length}."
            )

        positions = torch.arange(sequence_length, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        key_padding_mask = attention_mask == 0 if attention_mask is not None else None
        encoded = self.output_norm(self.encoder(hidden, src_key_padding_mask=key_padding_mask))

        if attention_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            weights = attention_mask.unsqueeze(-1).to(dtype=encoded.dtype)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        return TextBackboneOutput(sequence=encoded, pooled=pooled)