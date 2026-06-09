from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        super().__init__(*layers)


class AdvancedSegmentationModel(L.LightningModule):
    """Lightning module for stronger semantic-segmentation experiments.

    It keeps compatibility with the repository convention: the backbone returns
    an embedding map of shape [B, embed_ch_dim, H, W], and this module adds the
    final segmentation head.

    The default loss is intentionally conservative for FungiTastic:
    weighted CE + foreground Dice. Optional focal/Tversky terms are available
    for rare classes like pores/ring.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_ch_dim: int,
        num_classes: int,
        lr: float = 2e-4,
        min_lr: float = 1e-6,
        ce_weight: float = 0.7,
        dice_weight: float = 0.9,
        focal_weight: float = 0.0,
        tversky_weight: float = 0.0,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.35,
        tversky_beta: float = 0.65,
        dice_smooth: float = 1.0,
        weight_decay: float = 1e-4,
        ignore_index: int | None = None,
        include_background_in_loss: bool = False,
        include_background_in_metric: bool = False,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        scheduler: Literal["cosine", "onecycle", "none"] = "cosine",
        head_dropout: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        self.head = nn.Sequential(
            ConvNormAct(embed_ch_dim, embed_ch_dim, dropout=head_dropout),
            nn.Conv2d(embed_ch_dim, num_classes, kernel_size=1),
        )

        self.num_classes = num_classes
        self.lr = lr
        self.min_lr = min_lr
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.focal_gamma = focal_gamma
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.dice_smooth = dice_smooth
        self.weight_decay = weight_decay
        self.ignore_index = ignore_index
        self.include_background_in_loss = include_background_in_loss
        self.include_background_in_metric = include_background_in_metric
        self.scheduler = scheduler

        ce_class_weights = self._build_class_weights(class_weights, num_classes)
        self.register_buffer("ce_class_weights", ce_class_weights, persistent=True)

        for stage in ("train", "val", "test"):
            self.register_buffer(
                f"_{stage}_confusion_matrix",
                torch.zeros(num_classes, num_classes, dtype=torch.long),
                persistent=False,
            )

    @staticmethod
    def _build_class_weights(
        class_weights: Sequence[float] | torch.Tensor | None,
        num_classes: int,
    ) -> torch.Tensor:
        if class_weights is None:
            return torch.empty(0, dtype=torch.float32)

        weights = torch.as_tensor(class_weights, dtype=torch.float32)
        if weights.ndim != 1:
            raise ValueError("class_weights must be a 1D sequence")
        if weights.numel() != num_classes:
            raise ValueError(f"Expected {num_classes} class weights, got {weights.numel()}")
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def _valid_mask_and_safe_target(self, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ignore_index is None:
            valid = torch.ones_like(masks, dtype=torch.bool)
            safe_masks = masks
        else:
            valid = masks != self.ignore_index
            safe_masks = torch.where(valid, masks, torch.zeros_like(masks))
        return valid, safe_masks

    def _cross_entropy_loss(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        weight = None
        if self.ce_class_weights.numel() > 0:
            weight = self.ce_class_weights.to(device=logits.device, dtype=logits.dtype)

        ignore_index = -100 if self.ignore_index is None else self.ignore_index
        return F.cross_entropy(logits, masks, weight=weight, ignore_index=ignore_index)

    def _focal_loss(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        weight = None
        if self.ce_class_weights.numel() > 0:
            weight = self.ce_class_weights.to(device=logits.device, dtype=logits.dtype)

        ignore_index = -100 if self.ignore_index is None else self.ignore_index
        ce = F.cross_entropy(
            logits,
            masks,
            weight=weight,
            ignore_index=ignore_index,
            reduction="none",
        )

        valid, _ = self._valid_mask_and_safe_target(masks)
        ce = ce[valid]
        if ce.numel() == 0:
            return logits.new_tensor(0.0)

        pt = torch.exp(-ce)
        return ((1.0 - pt).pow(self.focal_gamma) * ce).mean()

    def _target_prob_tensors(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(logits, dim=1)
        valid, safe_masks = self._valid_mask_and_safe_target(masks)

        target = F.one_hot(safe_masks, num_classes=self.num_classes)
        target = target.permute(0, 3, 1, 2).type_as(probs)

        valid = valid.unsqueeze(1)
        probs = probs * valid
        target = target * valid

        if not self.include_background_in_loss and self.num_classes > 1:
            probs = probs[:, 1:]
            target = target[:, 1:]

        return probs, target

    def _dice_loss(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        probs, target = self._target_prob_tensors(logits, masks)
        dims = (0, 2, 3)

        intersection = torch.sum(probs * target, dim=dims)
        cardinality = torch.sum(probs + target, dim=dims)
        dice = (2.0 * intersection + self.dice_smooth) / (cardinality + self.dice_smooth)
        return 1.0 - dice.mean()

    def _tversky_loss(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        probs, target = self._target_prob_tensors(logits, masks)
        dims = (0, 2, 3)

        true_pos = torch.sum(probs * target, dim=dims)
        false_pos = torch.sum(probs * (1.0 - target), dim=dims)
        false_neg = torch.sum((1.0 - probs) * target, dim=dims)

        score = (true_pos + self.dice_smooth) / (
            true_pos
            + self.tversky_alpha * false_pos
            + self.tversky_beta * false_neg
            + self.dice_smooth
        )
        return 1.0 - score.mean()

    def _loss(self, logits: torch.Tensor, masks: torch.Tensor) -> dict[str, torch.Tensor]:
        ce_loss = self._cross_entropy_loss(logits, masks)
        dice_loss = self._dice_loss(logits, masks)
        focal_loss = (
            self._focal_loss(logits, masks)
            if self.focal_weight > 0.0
            else logits.new_tensor(0.0)
        )
        tversky_loss = (
            self._tversky_loss(logits, masks)
            if self.tversky_weight > 0.0
            else logits.new_tensor(0.0)
        )

        total = (
            self.ce_weight * ce_loss
            + self.dice_weight * dice_loss
            + self.focal_weight * focal_loss
            + self.tversky_weight * tversky_loss
        )
        return {
            "loss": total,
            "ce_loss": ce_loss,
            "dice_loss": dice_loss,
            "focal_loss": focal_loss,
            "tversky_loss": tversky_loss,
        }

    def _confusion_matrix(self, stage: str) -> torch.Tensor:
        return getattr(self, f"_{stage}_confusion_matrix")

    def _update_confusion_matrix(
        self,
        stage: str,
        logits: torch.Tensor,
        masks: torch.Tensor,
    ) -> None:
        if stage == "val" and self.trainer.sanity_checking:
            return

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            valid, _ = self._valid_mask_and_safe_target(masks)
            preds = preds[valid]
            targets = masks[valid]

            if targets.numel() == 0:
                return

            indices = targets * self.num_classes + preds
            counts = torch.bincount(
                indices,
                minlength=self.num_classes * self.num_classes,
            ).reshape(self.num_classes, self.num_classes)
            self._confusion_matrix(stage).add_(counts)

    def _metric_class_ids(self, device: torch.device) -> torch.Tensor:
        first_class = 0 if self.include_background_in_metric else 1
        return torch.arange(first_class, self.num_classes, device=device)

    def _segmentation_metrics(self, stage: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        confmat = self._confusion_matrix(stage)
        if getattr(self.trainer, "world_size", 1) > 1:
            gathered = self.all_gather(confmat)
            confmat = gathered.reshape(-1, self.num_classes, self.num_classes).sum(dim=0)

        confmat = confmat.float()
        total = confmat.sum()
        if total <= 0:
            zero = confmat.new_tensor(0.0)
            return zero, zero, torch.zeros(self.num_classes, device=confmat.device)

        true_pos = confmat.diag()
        pixel_acc = true_pos.sum() / total
        false_pos = confmat.sum(dim=0) - true_pos
        false_neg = confmat.sum(dim=1) - true_pos
        union = true_pos + false_pos + false_neg

        per_class_iou = torch.where(union > 0, true_pos / union.clamp_min(1.0), torch.nan)
        class_ids = self._metric_class_ids(confmat.device)
        valid_classes = ~torch.isnan(per_class_iou[class_ids])
        mean_iou = (
            per_class_iou[class_ids][valid_classes].mean()
            if valid_classes.any()
            else confmat.new_tensor(0.0)
        )
        return pixel_acc, mean_iou, per_class_iou

    def _reset_confusion_matrix(self, stage: str) -> None:
        self._confusion_matrix(stage).zero_()

    def _log_segmentation_metrics(self, stage: str) -> None:
        pixel_acc, mean_iou, per_class_iou = self._segmentation_metrics(stage)
        self.log(f"{stage}/pixel_acc", pixel_acc, on_epoch=True)
        self.log(f"{stage}/mean_iou", mean_iou, prog_bar=True, on_epoch=True)

        for class_id, value in enumerate(per_class_iou):
            if torch.isfinite(value):
                self.log(f"{stage}/iou_class_{class_id}", value, on_epoch=True)

        self._reset_confusion_matrix(stage)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        images, masks = batch
        masks = masks.long()

        logits = self(images)
        losses = self._loss(logits, masks)
        self._update_confusion_matrix(stage, logits, masks)

        batch_size = images.shape[0]
        for name, value in losses.items():
            self.log(
                f"{stage}/{name}",
                value,
                prog_bar=(name == "loss"),
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        return losses["loss"]

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
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
        optimizer = torch.optim.AdamW(self._optimizer_parameter_groups(), lr=self.lr)

        if self.scheduler == "none":
            return optimizer

        if self.scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.08,
                div_factor=20.0,
                final_div_factor=200.0,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        if self.scheduler == "cosine":
            max_epochs = self.trainer.max_epochs
            if max_epochs is None or max_epochs < 1:
                max_epochs = 100
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                eta_min=self.min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        raise ValueError(f"Unknown scheduler: {self.scheduler}")
