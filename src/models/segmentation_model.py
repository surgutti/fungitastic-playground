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

    for stage in ("train", "val", "test"):
      self.register_buffer(
          f"_{stage}_confusion_matrix",
          torch.zeros(num_classes, num_classes, dtype=torch.long),
          persistent=False,
      )
  
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

  def _confusion_matrix(self, stage: str) -> torch.Tensor:
    return getattr(self, f"_{stage}_confusion_matrix")

  def _update_confusion_matrix(
      self,
      stage: str,
      logits: torch.Tensor,
      masks: torch.Tensor
  ) -> None:
    if stage == "val" and self.trainer.sanity_checking:
      return

    with torch.no_grad():
      preds = logits.argmax(dim=1)

      if self.ignore_index is None:
        valid_pixels = torch.ones_like(masks, dtype=torch.bool)
      else:
        valid_pixels = masks != self.ignore_index

      preds = preds[valid_pixels]
      targets = masks[valid_pixels]
      if targets.numel() == 0:
        return

      indices = targets * self.num_classes + preds
      counts = torch.bincount(
          indices,
          minlength=self.num_classes * self.num_classes,
      )
      counts = counts.reshape(self.num_classes, self.num_classes)
      self._confusion_matrix(stage).add_(counts)

  def _segmentation_metrics(
      self,
      stage: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    confmat = self._confusion_matrix(stage)
    if getattr(self.trainer, "world_size", 1) > 1:
      gathered = self.all_gather(confmat)
      confmat = gathered.reshape(
          -1,
          self.num_classes,
          self.num_classes,
      ).sum(dim=0)

    confmat = confmat.float()
    total = confmat.sum()
    if total <= 0:
      zero = confmat.new_tensor(0.0)
      return zero, zero

    true_positives = confmat.diag()
    pixel_acc = true_positives.sum() / total
    false_positives = confmat.sum(dim=0) - true_positives
    false_negatives = confmat.sum(dim=1) - true_positives
    union = true_positives + false_positives + false_negatives

    class_ids = torch.arange(self.num_classes, device=confmat.device)
    first_class = 0 if self.include_background_in_dice else 1
    class_ids = class_ids[class_ids >= first_class]
    valid_classes = union[class_ids] > 0
    if not valid_classes.any():
      mean_iou = confmat.new_tensor(0.0)
    else:
      mean_iou = (
          true_positives[class_ids][valid_classes]
          / union[class_ids][valid_classes]
      ).mean()

    return pixel_acc, mean_iou

  def _reset_confusion_matrix(self, stage: str) -> None:
    self._confusion_matrix(stage).zero_()

  def _log_segmentation_metrics(self, stage: str) -> None:
    pixel_acc, mean_iou = self._segmentation_metrics(stage)
    self.log(f"{stage}/pixel_acc", pixel_acc, on_epoch=True)
    self.log(
        f"{stage}/mean_iou",
        mean_iou,
        prog_bar=True,
        on_epoch=True,
    )
    self._reset_confusion_matrix(stage)
  
  def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
    images, masks = batch

    logits = self(images)
    masks = masks.long()

    loss, ce_loss, dice_loss = self._loss(logits, masks)
    self._update_confusion_matrix(stage, logits, masks)
    batch_size = images.shape[0]

    self.log(
        f"{stage}/loss",
        loss,
        prog_bar=True,
        on_step=False,
        on_epoch=True,
        batch_size=batch_size,
    )
    self.log(
        f"{stage}/ce_loss",
        ce_loss,
        on_step=False,
        on_epoch=True,
        batch_size=batch_size,
    )
    self.log(
        f"{stage}/dice_loss",
        dice_loss,
        on_step=False,
        on_epoch=True,
        batch_size=batch_size,
    )
    return loss

  def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
    return self._shared_step(batch, "train")

  def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
    return self._shared_step(batch, "val")

  def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
    return self._shared_step(batch, "test")

  def on_train_epoch_end(self) -> None:
    self._log_segmentation_metrics("train")

  def on_validation_epoch_end(self) -> None:
    if self.trainer.sanity_checking:
      self._reset_confusion_matrix("val")
      return

    self._log_segmentation_metrics("val")

  def on_test_epoch_end(self) -> None:
    self._log_segmentation_metrics("test")

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
