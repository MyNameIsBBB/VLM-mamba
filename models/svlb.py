from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.backbones.cnn import MobileNetV3Encoder
from models.backbones.mamba import MambaSequenceBackbone
from models.backbones.transformer import PretrainedMultilingualTextEncoder, TextTransformerEncoder
from models.interfaces import SequenceBackbone, TextBackbone, VisionBackbone
from utils.tensor_ops import spatial_to_sequence


@dataclass
class SelectiveVisionLanguageOutput:
    match_logit: Tensor
    vision_tokens: Tensor
    text_tokens: Tensor
    fused_embedding: Tensor | None
    image_embedding: Tensor
    text_embedding: Tensor
    similarity_matrix: Tensor
    spatial_size: tuple[int, int]


class SelectiveVisionLanguageBackbone(nn.Module):
    def __init__(
        self,
        vision_backbone: VisionBackbone,
        sequence_backbone: SequenceBackbone,
        text_backbone: TextBackbone,
        dim: int,
        vision_grid_size: tuple[int, int],
        temperature_init: float,
    ) -> None:
        super().__init__()
        self.vision_backbone = vision_backbone
        self.sequence_backbone = sequence_backbone
        self.text_backbone = text_backbone
        self.vision_grid_size = vision_grid_size
        self.vision_position_embedding = nn.Parameter(torch.zeros(1, vision_grid_size[0] * vision_grid_size[1], dim))
        self.image_projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.text_projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature_init), dtype=torch.float32))

        nn.init.normal_(self.vision_position_embedding, std=0.02)

    def _vision_position_tokens(self, spatial_size: tuple[int, int], device: torch.device, dtype: torch.dtype) -> Tensor:
        base_height, base_width = self.vision_grid_size
        target_height, target_width = spatial_size

        if (base_height, base_width) == (target_height, target_width):
            return self.vision_position_embedding.to(device=device, dtype=dtype)

        position_grid = self.vision_position_embedding.view(1, base_height, base_width, -1).permute(0, 3, 1, 2)
        resized = F.interpolate(position_grid, size=(target_height, target_width), mode="bilinear", align_corners=False)
        return resized.flatten(start_dim=2).transpose(1, 2).to(device=device, dtype=dtype)

    def encode_image_tokens(self, images: Tensor) -> tuple[Tensor, tuple[int, int]]:
        device = next(self.parameters()).device
        images = images.to(device)
        spatial_features = self.vision_backbone(images)
        vision_tokens, spatial_size = spatial_to_sequence(spatial_features)
        vision_tokens = vision_tokens + self._vision_position_tokens(spatial_size, device, vision_tokens.dtype)
        return self.sequence_backbone(vision_tokens), spatial_size

    def encode_image_embedding(self, images: Tensor) -> tuple[Tensor, Tensor, tuple[int, int]]:
        vision_tokens, spatial_size = self.encode_image_tokens(images)
        pooled = vision_tokens.mean(dim=1)
        image_embedding = F.normalize(self.image_projection(pooled), dim=-1)
        return image_embedding, vision_tokens, spatial_size

    def encode_text_embedding(self, token_ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        text_output = self.text_backbone(token_ids, attention_mask)
        text_embedding = F.normalize(self.text_projection(text_output.pooled), dim=-1)
        return text_embedding, text_output.sequence

    def encode_image(self, images: Tensor) -> tuple[Tensor, tuple[int, int]]:
        vision_tokens, spatial_size = self.encode_image_tokens(images)
        return vision_tokens, spatial_size

    def forward(
        self,
        images: Tensor,
        token_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> SelectiveVisionLanguageOutput:
        image_embedding, vision_tokens, spatial_size = self.encode_image_embedding(images)
        text_embedding, text_tokens = self.encode_text_embedding(token_ids, attention_mask)
        similarity_matrix = self.logit_scale.exp() * image_embedding @ text_embedding.transpose(0, 1)
        match_logit = similarity_matrix.diagonal()
        return SelectiveVisionLanguageOutput(
            match_logit=match_logit,
            vision_tokens=vision_tokens,
            text_tokens=text_tokens,
            fused_embedding=None,
            image_embedding=image_embedding,
            text_embedding=text_embedding,
            similarity_matrix=similarity_matrix,
            spatial_size=spatial_size,
        )


def build_svlb_from_config(config: dict[str, Any]) -> SelectiveVisionLanguageBackbone:
    model_config = config["model"]
    dim = model_config["vision"]["out_channels"]
    text_dim = model_config["text"]["dim"]

    if dim != text_dim:
        raise ValueError("vision.out_channels and text.dim must match for retrieval alignment.")

    vision_backbone = MobileNetV3Encoder(
        out_channels=dim,
        pretrained=model_config["vision"].get("pretrained", True),
        freeze_stem=model_config["vision"].get("freeze_stem", False),
    )
    sequence_backbone = MambaSequenceBackbone(
        dim=dim,
        depth=model_config["mamba"]["depth"],
        state_dim=model_config["mamba"]["state_dim"],
        conv_kernel=model_config["mamba"]["conv_kernel"],
        expand=model_config["mamba"]["expand"],
        dropout=model_config["mamba"].get("dropout", 0.0),
    )
    pretrained_model_name = model_config["text"].get("pretrained_model_name")
    if pretrained_model_name:
        text_backbone = PretrainedMultilingualTextEncoder(
            model_name=pretrained_model_name,
            max_length=model_config["text"]["max_length"],
            dim=text_dim,
            dropout=model_config["text"].get("dropout", 0.0),
            freeze_backbone=model_config["text"].get("freeze_backbone", False),
        )
    else:
        text_backbone = TextTransformerEncoder(
            vocab_size=model_config["text"]["vocab_size"],
            max_length=model_config["text"]["max_length"],
            dim=text_dim,
            depth=model_config["text"]["depth"],
            num_heads=model_config["text"]["num_heads"],
            dropout=model_config["text"].get("dropout", 0.0),
        )

    return SelectiveVisionLanguageBackbone(
        vision_backbone=vision_backbone,
        sequence_backbone=sequence_backbone,
        text_backbone=text_backbone,
        dim=dim,
        vision_grid_size=tuple(model_config["vision"].get("spatial_grid_size", [7, 7])),
        temperature_init=config["train"].get("temperature", 0.07),
    )