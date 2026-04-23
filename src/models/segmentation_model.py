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
      lr: float = 1e-4,
      ce_weight: float = 1.0,
      dice_weight: float = 1.0,
      dice_smooth: float = 1.0,
      weight_decay: float = 1e-4,
      ignore_index: int | None = None,
      include_background_in_dice: bool = False
  ):
    super().__init__()
    self.save_hyperparameters(ignore=["backbone"])
    self.backbone = backbone
    self.head = nn.Conv2d(embed_ch_dim, num_classes, kernel_size=1)
    self.num_classes = num_classes
    self.lr = lr
    self.ce_weight = ce_weight
    self.dice_weight = dice_weight
    self.dice_smooth = dice_smooth
    self.weight_decay = weight_decay
    self.ignore_index = ignore_index
    self.include_background_in_dice = include_background_in_dice
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.head(self.backbone(x))

  def _cross_entropy_loss(
      self,
      logits: torch.Tensor,
      masks: torch.Tensor
  ) -> torch.Tensor:
    if self.ignore_index is None:
      return F.cross_entropy(logits, masks)

    return F.cross_entropy(logits, masks, ignore_index=self.ignore_index)

  def _dice_loss(
      self,
      logits: torch.Tensor,
      masks: torch.Tensor
  ) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)

    if self.ignore_index is None:
      valid_pixels = torch.ones_like(masks, dtype=torch.bool)
      safe_masks = masks
    else:
      valid_pixels = masks != self.ignore_index
      safe_masks = torch.where(valid_pixels, masks, torch.zeros_like(masks))

    target = F.one_hot(safe_masks, num_classes=self.num_classes)
    target = target.permute(0, 3, 1, 2).type_as(probs)

    valid_pixels = valid_pixels.unsqueeze(1)
    probs = probs * valid_pixels
    target = target * valid_pixels

    if not self.include_background_in_dice and self.num_classes > 1:
      probs = probs[:, 1:]
      target = target[:, 1:]

    dims = (0, 2, 3)
    intersection = torch.sum(probs * target, dim=dims)
    cardinality = torch.sum(probs + target, dim=dims)
    dice = (2.0 * intersection + self.dice_smooth) / (
        cardinality + self.dice_smooth
    )

    return 1.0 - dice.mean()

  def _loss(
      self,
      logits: torch.Tensor,
      masks: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce_loss = self._cross_entropy_loss(logits, masks)
    dice_loss = self._dice_loss(logits, masks)
    loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
    return loss, ce_loss, dice_loss
  
  def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
    images, masks = batch

    logits = self(images)
    masks = masks.long()

    loss, ce_loss, dice_loss = self._loss(logits, masks)

    self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
    self.log(f"{stage}/ce_loss", ce_loss, on_step=False, on_epoch=True)
    self.log(f"{stage}/dice_loss", dice_loss, on_step=False, on_epoch=True)

    return loss

  def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
    return self._shared_step(batch, "train")

  def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
    return self._shared_step(batch, "val")

  def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
    return self._shared_step(batch, "test")

  def _optimizer_parameter_groups(self) -> list[dict]:
    decay_params = []
    no_decay_params = []

    for _, param in self.named_parameters():
      if not param.requires_grad:
        continue

      if param.ndim >= 2:
        decay_params.append(param)
      else:
        no_decay_params.append(param)

    return [
      {"params": decay_params, "weight_decay": self.weight_decay},
      {"params": no_decay_params, "weight_decay": 0.0},
    ]

  def configure_optimizers(self):
    return torch.optim.AdamW(self._optimizer_parameter_groups(), lr=self.lr)
