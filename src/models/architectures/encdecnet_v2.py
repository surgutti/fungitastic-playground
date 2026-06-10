import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.conv_block import ConvBlock
from src.models.modules.residual_block import ResidualBlock
from src.models.modules.attention_gate import AttentionGate # <-- Import the new gate

class EncDecNetBackbone(nn.Module):
  
  def __init__(
      self,
      in_channels: int = 3,
      base_channels: int = 32,
      out_channels: int = 32
  ):
    super().__init__()

    self.enc1 = ResidualBlock(in_channels, base_channels)
    self.pool1 = nn.MaxPool2d(kernel_size=2)

    self.enc2 = ResidualBlock(base_channels, base_channels * 2)
    self.pool2 = nn.MaxPool2d(kernel_size=2)

    self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4)
    self.pool3 = nn.MaxPool2d(kernel_size=2)

    self.enc4 = ResidualBlock(base_channels * 4, base_channels * 8)
    self.pool4 = nn.MaxPool2d(kernel_size=2)

    self.bottleneck = nn.Sequential(
        ResidualBlock(base_channels * 8, base_channels * 16),
        nn.Dropout2d(p=0.5)
    )

    self.ag4 = AttentionGate(F_g=base_channels * 8, F_l=base_channels * 8, F_int=base_channels * 4)
    self.ag3 = AttentionGate(F_g=base_channels * 4, F_l=base_channels * 4, F_int=base_channels * 2)
    self.ag2 = AttentionGate(F_g=base_channels * 2, F_l=base_channels * 2, F_int=base_channels)
    self.ag1 = AttentionGate(F_g=base_channels, F_l=base_channels, F_int=base_channels // 2)

    self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
    self.dec4 = ResidualBlock(base_channels * 16, base_channels * 8)

    self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
    self.dec3 = ResidualBlock(base_channels * 8, base_channels * 4)

    self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
    self.dec2 = ResidualBlock(base_channels * 4, base_channels * 2)

    self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
    self.dec1 = ResidualBlock(base_channels * 2, base_channels)

    self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

  @staticmethod
  def _match_size(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == reference.shape[-2:]:
      return x
    return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    enc1 = self.enc1(x)
    enc2 = self.enc2(self.pool1(enc1))
    enc3 = self.enc3(self.pool2(enc2))
    enc4 = self.enc4(self.pool3(enc3))
    
    x = self.bottleneck(self.pool4(enc4))

    x = self.up4(x)
    x = self._match_size(x, enc4)
    enc4_attended = self.ag4(g=x, x=enc4) 
    x = torch.cat([x, enc4_attended], dim=1)
    x = self.dec4(x)

    x = self.up3(x)
    x = self._match_size(x, enc3)
    enc3_attended = self.ag3(g=x, x=enc3) 
    x = torch.cat([x, enc3_attended], dim=1)
    x = self.dec3(x)

    x = self.up2(x)
    x = self._match_size(x, enc2)
    enc2_attended = self.ag2(g=x, x=enc2) 
    x = torch.cat([x, enc2_attended], dim=1)
    x = self.dec2(x)

    x = self.up1(x)
    x = self._match_size(x, enc1)
    enc1_attended = self.ag1(g=x, x=enc1) 
    x = torch.cat([x, enc1_attended], dim=1)
    x = self.dec1(x)

    return self.out(x)