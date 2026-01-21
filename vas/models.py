"""Siamese ResNet+GRU model for VAS classification."""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class SiameseResNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = False,
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()

        if backbone == "resnet18":
            net = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
        elif backbone == "resnet34":
            net = models.resnet34(weights=None if not pretrained else models.ResNet34_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.feature_dim = net.fc.in_features

        self.head = nn.Sequential(
            nn.Linear(self.feature_dim * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        feat = self.backbone(x)
        feat = torch.flatten(feat, 1)  # (B, feature_dim)
        return feat

    def forward(self, anchor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # anchor: (B, C, H, W), target: (B, C, H, W)
        anchor_feat = self._encode(anchor)
        target_feat = self._encode(target)
        diff = torch.abs(target_feat - anchor_feat)
        
        # Concatenate: (B, feature_dim * 3)
        combined = torch.cat([anchor_feat, target_feat, diff], dim=1)
        return self.head(combined)


def split_params(model: nn.Module) -> Tuple[list, list]:
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    return backbone_params, head_params
