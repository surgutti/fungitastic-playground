import torch
import torch.nn as nn

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights

class MobileNetV3(nn.Module):
  
  def __init__(
      self
  ):
    super().__init__()
    self.net = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)['out']