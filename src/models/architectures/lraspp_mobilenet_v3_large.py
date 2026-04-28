import torch
import torch.nn as nn

from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights

class MobileNetV3(nn.Module):

  def __init__(
      self,
      in_channels: int = 3,
      out_channels: int = 32
  ):
    super().__init__()

    self.net = lraspp_mobilenet_v3_large(LRASPP_MobileNet_V3_Large_Weights.DEFAULT)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    
    return self.net(x)