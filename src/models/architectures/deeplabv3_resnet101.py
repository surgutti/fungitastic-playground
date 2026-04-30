import torch
import torch.nn as nn

from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

class ResNet101(nn.Module):
  
  def __init__(
      self
  ):
    super().__init__()
    self.net = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)['out']