from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from models.backbones.mamba import MambaSequenceBackbone
from models.interfaces import SequenceBackbone


class HybridTextModel(nn.Module):
    def __init__(
        self,
        mamba_backbone: SequenceBackbone,
        dim: int,
        vocab_size: int,
        transformer_depth: int,
        transformer_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.mamba_backbone = mamba_backbone
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=transformer_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_depth,
            enable_nested_tensor=False,
        )
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        token_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        # Embed text
        hidden = self.embedding(token_ids)
        
        # Mamba context extraction
        mamba_out = self.mamba_backbone(hidden)
        
        # Transformer processing
        key_padding_mask = attention_mask == 0 if attention_mask is not None else None
        
        # Causal mask for the transformer to ensure it only attends to past tokens
        seq_len = mamba_out.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=mamba_out.device)

        transformer_out = self.transformer(
            mamba_out, 
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
            is_causal=True
        )
        
        # Output logits
        logits = self.lm_head(transformer_out)
        return logits, transformer_out


def build_svlb_from_config(config: dict[str, Any]) -> HybridTextModel:
    model_config = config["model"]
    dim = model_config["dim"]
    vocab_size = model_config["vocab_size"]

    mamba_backbone = MambaSequenceBackbone(
        dim=dim,
        depth=model_config["mamba"]["depth"],
        state_dim=model_config["mamba"]["state_dim"],
        conv_kernel=model_config["mamba"]["conv_kernel"],
        expand=model_config["mamba"]["expand"],
        dropout=model_config["mamba"].get("dropout", 0.0),
    )

    return HybridTextModel(
        mamba_backbone=mamba_backbone,
        dim=dim,
        vocab_size=vocab_size,
        transformer_depth=model_config["transformer"]["depth"],
        transformer_heads=model_config["transformer"]["num_heads"],
        dropout=model_config["transformer"].get("dropout", 0.0),
    )