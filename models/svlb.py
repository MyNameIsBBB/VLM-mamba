from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from models.backbones.cnn import MobileNetV3Encoder
from models.backbones.mamba import MambaSequenceBackbone
from models.backbones.transformer import TextTransformerEncoder
from models.fusion.multimodal import MultimodalTransformerFusion
from models.heads.matching import ZeroShotMatchingHead
from models.interfaces import FusionModule, PredictionHead, SequenceBackbone, TextBackbone, VisionBackbone
from utils.tensor_ops import spatial_to_sequence


@dataclass
class SelectiveVisionLanguageOutput:
    match_logit: Tensor
    vision_tokens: Tensor
    text_tokens: Tensor
    fused_embedding: Tensor
    spatial_size: tuple[int, int]


class SelectiveVisionLanguageBackbone(nn.Module):
    def __init__(
        self,
        vision_backbone: VisionBackbone,
        sequence_backbone: SequenceBackbone,
        text_backbone: TextBackbone,
        fusion_module: FusionModule,
        prediction_head: PredictionHead,
    ) -> None:
        super().__init__()
        self.vision_backbone = vision_backbone
        self.sequence_backbone = sequence_backbone
        self.text_backbone = text_backbone
        self.fusion_module = fusion_module
        self.prediction_head = prediction_head

    def encode_image(self, images: Tensor) -> tuple[Tensor, tuple[int, int]]:
        device = next(self.parameters()).device
        images = images.to(device)
        spatial_features = self.vision_backbone(images)
        vision_tokens, spatial_size = spatial_to_sequence(spatial_features)
        vision_tokens = vision_tokens.to(device)
        return self.sequence_backbone(vision_tokens), spatial_size
    def forward(
        self,
        images: Tensor,
        token_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> SelectiveVisionLanguageOutput:
        vision_tokens, spatial_size = self.encode_image(images)
        text_output = self.text_backbone(token_ids, attention_mask)
        fused_embedding = self.fusion_module(vision_tokens, text_output.sequence, attention_mask)
        match_logit = self.prediction_head(fused_embedding)
        return SelectiveVisionLanguageOutput(
            match_logit=match_logit,
            vision_tokens=vision_tokens,
            text_tokens=text_output.sequence,
            fused_embedding=fused_embedding,
            spatial_size=spatial_size,
        )


def build_svlb_from_config(config: dict[str, Any]) -> SelectiveVisionLanguageBackbone:
    model_config = config["model"]
    dim = model_config["vision"]["out_channels"]
    text_dim = model_config["text"]["dim"]
    fusion_dim = model_config["fusion"]["dim"]

    if len({dim, text_dim, fusion_dim}) != 1:
        raise ValueError("vision.out_channels, text.dim, and fusion.dim must match for plug-and-play fusion.")

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
    text_backbone = TextTransformerEncoder(
        vocab_size=model_config["text"]["vocab_size"],
        max_length=model_config["text"]["max_length"],
        dim=text_dim,
        depth=model_config["text"]["depth"],
        num_heads=model_config["text"]["num_heads"],
        dropout=model_config["text"].get("dropout", 0.0),
    )
    fusion_module = MultimodalTransformerFusion(
        dim=fusion_dim,
        depth=model_config["fusion"]["depth"],
        num_heads=model_config["fusion"]["num_heads"],
        dropout=model_config["fusion"].get("dropout", 0.0),
    )
    prediction_head = ZeroShotMatchingHead(
        dim=fusion_dim,
        dropout=model_config["head"].get("dropout", 0.0),
    )
    return SelectiveVisionLanguageBackbone(
        vision_backbone=vision_backbone,
        sequence_backbone=sequence_backbone,
        text_backbone=text_backbone,
        fusion_module=fusion_module,
        prediction_head=prediction_head,
    )