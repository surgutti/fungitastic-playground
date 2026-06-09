from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ResNet50_Weights,
    ResNet101_Weights,
    Wide_ResNet50_2_Weights,
    resnet50,
    resnet101,
    wide_resnet50_2,
)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        super().__init__(*layers)


class SqueezeExcite2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.block = ConvNormAct(in_channels + skip_channels, out_channels, dropout=dropout)
        self.attention = SqueezeExcite2d(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return self.attention(x)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rates: tuple[int, ...] = (1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for rate in rates:
            if rate == 1:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.SiLU(inplace=True),
                    )
                )
            else:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=3,
                            padding=rate,
                            dilation=rate,
                            bias=False,
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.SiLU(inplace=True),
                    )
                )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(rates), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class ResNetUNetBackbone(nn.Module):
    """ImageNet-pretrained ResNet encoder with a U-Net/FPN style decoder.

    The module returns dense feature embeddings at the input resolution. Pair it
    with `AdvancedSegmentationModel` or the repository's existing segmentation
    wrapper to get final class logits.

    Backbones:
    - "resnet50": good first strong model.
    - "resnet101": better accuracy if training time is acceptable.
    - "wide_resnet50_2": heavier; often a good single-GPU high-capacity model.
    """

    def __init__(
        self,
        backbone_name: Literal["resnet50", "resnet101", "wide_resnet50_2"] = "resnet50",
        out_channels: int = 96,
        decoder_channels: tuple[int, int, int, int] = (512, 256, 128, 96),
        pretrained: bool = True,
        dropout: float = 0.05,
        freeze_stem: bool = False,
        use_aspp: bool = True,
    ):
        super().__init__()

        encoder = self._make_encoder(backbone_name, pretrained)
        self.backbone_name = backbone_name

        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        encoder_channels = self._encoder_channels(backbone_name)
        c1, c2, c3, c4, c5 = encoder_channels
        d4, d3, d2, d1 = decoder_channels

        self.center = ASPP(c5, d4) if use_aspp else ConvNormAct(c5, d4, dropout=dropout)
        self.dec4 = DecoderBlock(d4, c4, d4, dropout=dropout)
        self.dec3 = DecoderBlock(d4, c3, d3, dropout=dropout)
        self.dec2 = DecoderBlock(d3, c2, d2, dropout=dropout)
        self.dec1 = DecoderBlock(d2, c1, d1, dropout=dropout)
        self.out = nn.Sequential(
            ConvNormAct(d1, out_channels, dropout=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

        if freeze_stem:
            for param in self.stem.parameters():
                param.requires_grad = False

    @staticmethod
    def _make_encoder(
        backbone_name: Literal["resnet50", "resnet101", "wide_resnet50_2"],
        pretrained: bool,
    ) -> nn.Module:
        if backbone_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            return resnet50(weights=weights)
        if backbone_name == "resnet101":
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            return resnet101(weights=weights)
        if backbone_name == "wide_resnet50_2":
            weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            return wide_resnet50_2(weights=weights)

        raise ValueError(f"Unsupported backbone_name: {backbone_name}")

    @staticmethod
    def _encoder_channels(backbone_name: str) -> tuple[int, int, int, int, int]:
        # c1 is the stem feature map. Bottleneck ResNets have the same stage
        # output dimensions; wide_resnet50_2 widens the internal channels but
        # keeps layer outputs compatible.
        if backbone_name in {"resnet50", "resnet101", "wide_resnet50_2"}:
            return 64, 256, 512, 1024, 2048
        raise ValueError(f"Unsupported backbone_name: {backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        c1 = self.stem(x)                 # 1/2
        c2 = self.layer1(self.pool(c1))    # 1/4
        c3 = self.layer2(c2)               # 1/8
        c4 = self.layer3(c3)               # 1/16
        c5 = self.layer4(c4)               # 1/32

        x = self.center(c5)
        x = self.dec4(x, c4)
        x = self.dec3(x, c3)
        x = self.dec2(x, c2)
        x = self.dec1(x, c1)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return self.out(x)
