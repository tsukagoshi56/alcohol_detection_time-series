"""Siamese ResNet+GRU model for VAS classification."""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class SiameseResNetGRU(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = False,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.5,
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

        self.rnn = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        rnn_out_dim = rnn_hidden * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.Linear(rnn_out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feat = self.backbone(x)
        feat = feat.view(b, t, self.feature_dim)
        _, h_n = self.rnn(feat)

        if self.rnn.bidirectional:
            # last layer forward/backward
            forward = h_n[-2]
            backward = h_n[-1]
            return torch.cat([forward, backward], dim=1)
        return h_n[-1]

    def forward(self, anchor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        anchor_feat = self._encode(anchor)
        target_feat = self._encode(target)
        diff = torch.abs(target_feat - anchor_feat)
        return self.head(diff)


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
