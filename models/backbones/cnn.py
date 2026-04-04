from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from models.interfaces import VisionBackbone


class MobileNetV3Encoder(VisionBackbone):
    def __init__(
        self,
        out_channels: int,
        pretrained: bool = True,
        freeze_stem: bool = False,
    ) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        in_channels = backbone.classifier[0].in_features
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.Hardswish()

        if freeze_stem:
            stem = self.features[:2]
            for parameter in stem.parameters():
                parameter.requires_grad = False

    def forward(self, images: Tensor) -> Tensor:
        spatial_features = self.features(images)
        projected = self.projection(spatial_features)
        return self.activation(self.norm(projected))