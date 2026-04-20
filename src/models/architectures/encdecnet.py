import torch.nn as nn

class EncDecNetBackbone(nn.Module):
  def __init__(self,
    )
  def __init__(self, in_channels=3, num_classes=2):
    super().__init__()

    # (B, 3, 300, 300)
    self.enc1 = nn.Sequential(
      nn.Conv2d(in_channels, 32, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 32, 3, padding=1),
      nn.ReLU(inplace=True)
    )
    # (B, 32, 300, 300)
    self.pool1 = nn.MaxPool2d(2)
    # (B, 32, 150, 150)
    self.enc2 = nn.Sequential(
      nn.Conv2d(32, 64, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, 3, padding=1),
      nn.ReLU(inplace=True)
    )
    # (B, 64, 150, 150)
    self.pool2 = nn.MaxPool2d(2)
    # (B, 64, 75, 75)
    self.bottleneck = nn.Sequential(
      nn.Conv2d(64, 128, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 128, 3, padding=1),
      nn.ReLU(inplace=True)
    )
    # (B, 128, 75, 75)
    self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
    # (B, 64, 150, 150)
    self.dec1 = nn.Sequential(
      nn.Conv2d(64, 64, 3, padding=1),
      nn.ReLU(inplace=True)
    )
    # (B, 64, 150, 150)
    self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
    # (B, 32, 300, 300)
    self.dec2 = nn.Sequential(
      nn.Conv2d(32, 32, 3, padding=1),
      nn.ReLU(inplace=True)
    )
    # (B, 32, 300, 300)

  def forward(self, x):
    x = self.enc1(x)
    x = self.pool1(x)
    x = self.enc2(x)
    x = self.pool2(x)
    x = self.bottleneck(x)
    x = self.up1(x)
    x = self.dec1(x)
    x = self.up2(x)
    x = self.dec2(x)
    x = self.head(x)
    return x