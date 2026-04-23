import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.conv_block import ConvBlock

class EncDecNetBackbone(nn.Module):
  
  def __init__(
      self,
      in_channels: int = 3,
      base_channels: int = 32,
      out_channels: int = 32
  ):
    super().__init__()

    self.enc1 = ConvBlock(in_channels, base_channels)
    self.pool1 = nn.MaxPool2d(kernel_size=2)

    self.enc2 = ConvBlock(base_channels, base_channels * 2)
    self.pool2 = nn.MaxPool2d(kernel_size=2)

    self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)

    self.up2 = nn.ConvTranspose2d(
      base_channels * 4,
      base_channels * 2,
      kernel_size=2,
      stride=2,
    )
    self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

    self.up1 = nn.ConvTranspose2d(
      base_channels * 2,
      base_channels,
      kernel_size=2,
      stride=2,
    )
    self.dec1 = ConvBlock(base_channels * 2, base_channels)

    self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

  @staticmethod
  def _match_size(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == reference.shape[-2:]:
      return x

    return F.interpolate(
      x,
      size=reference.shape[-2:],
      mode="bilinear",
      align_corners=False,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    enc1 = self.enc1(x)
    enc2 = self.enc2(self.pool1(enc1))

    x = self.bottleneck(self.pool2(enc2))

    x = self.up2(x)
    x = self._match_size(x, enc2)
    x = torch.cat([x, enc2], dim=1)
    x = self.dec2(x)

    x = self.up1(x)
    x = self._match_size(x, enc1)
    x = torch.cat([x, enc1], dim=1)
    x = self.dec1(x)

    return self.out(x)
