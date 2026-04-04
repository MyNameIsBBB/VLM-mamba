from __future__ import annotations

import torch
from torch import Tensor, nn

from models.interfaces import FusionModule


class MultimodalTransformerFusion(FusionModule):
    def __init__(self, dim: int, depth: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.modality_embedding = nn.Parameter(torch.randn(2, dim) * 0.02)
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

    def forward(
        self,
        vision_tokens: Tensor,
        text_tokens: Tensor,
        text_attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, vision_length, dim = vision_tokens.shape
        text_length = text_tokens.size(1)
        cls_token = self.cls_token.expand(batch_size, -1, -1)

        fused_tokens = torch.cat(
            [
                cls_token,
                vision_tokens + self.modality_embedding[0].view(1, 1, dim),
                text_tokens + self.modality_embedding[1].view(1, 1, dim),
            ],
            dim=1,
        )

        if text_attention_mask is None:
            key_padding_mask = None
        else:
            cls_mask = torch.ones((batch_size, 1), device=text_attention_mask.device, dtype=text_attention_mask.dtype)
            vision_mask = torch.ones(
                (batch_size, vision_length),
                device=text_attention_mask.device,
                dtype=text_attention_mask.dtype,
            )
            valid_mask = torch.cat([cls_mask, vision_mask, text_attention_mask], dim=1)
            key_padding_mask = valid_mask == 0

        encoded = self.output_norm(self.encoder(fused_tokens, src_key_padding_mask=key_padding_mask))
        return encoded[:, 0]