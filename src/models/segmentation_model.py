import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationModel(L.LightningModule):
  def __init__(
      self,
      backbone: nn.Module,
      embed_ch_dim: int,
      num_classes: int,
      lr: float = 1e-4
  ):
    super().__init__()
    self.backbone = backbone
    self.head = nn.Conv2d(embed_ch_dim, num_classes, kernel_size=1)
    self.lr = lr
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.head(self.backbone(x))
  
  def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
    images, masks = batch
    # TODO
    preds = self(images)
    return loss

  def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
    return self._shared_step(batch, "train")

  def validation_step(self, batch: tuple, batch_idx: int) -> None:
    self._shared_step(batch, "val")

  def test_step(self, batch: tuple, batch_idx: int) -> None:
    self._shared_step(batch, "test")

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.lr)