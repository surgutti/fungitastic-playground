import torch
import torch.nn as nn

from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights

class ResNet101(nn.Module):
  
  def __init__(
      self
  ):
    super().__init__()
    self.net = fcn_resnet101(weights=fcn_resnet101.DEFAULT)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)['out']