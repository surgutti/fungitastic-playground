from collections.abc import Iterable

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class WandbEpochAxisCallback(L.Callback):
  def __init__(
      self,
      epoch_metric: str = "epoch",
      metric_prefixes: Iterable[str] = ("train", "val", "test"),
      summary_min_metrics: Iterable[str] = (
          "train/loss",
          "train/ce_loss",
          "train/dice_loss",
          "val/loss",
          "val/ce_loss",
          "val/dice_loss",
          "test/loss",
          "test/ce_loss",
          "test/dice_loss",
      ),
      summary_max_metrics: Iterable[str] = (
          "train/pixel_acc",
          "train/mean_iou",
          "val/pixel_acc",
          "val/mean_iou",
          "test/pixel_acc",
          "test/mean_iou",
      ),
  ):
    super().__init__()
    self.epoch_metric = epoch_metric
    self.metric_prefixes = tuple(metric_prefixes)
    self.summary_min_metrics = tuple(summary_min_metrics)
    self.summary_max_metrics = tuple(summary_max_metrics)
    self._defined = False
    self._last_logged_epoch_step: tuple[int, int] | None = None

  def setup(
      self,
      trainer: L.Trainer,
      pl_module: L.LightningModule,
      stage: str,
  ) -> None:
    self._define_metrics(trainer)
    self._log_epoch(trainer)

  def on_train_epoch_start(
      self,
      trainer: L.Trainer,
      pl_module: L.LightningModule,
  ) -> None:
    self._log_epoch(trainer)

  def on_validation_epoch_start(
      self,
      trainer: L.Trainer,
      pl_module: L.LightningModule,
  ) -> None:
    self._log_epoch(trainer)

  def on_test_epoch_start(
      self,
      trainer: L.Trainer,
      pl_module: L.LightningModule,
  ) -> None:
    self._log_epoch(trainer)

  @rank_zero_only
  def _define_metrics(self, trainer: L.Trainer) -> None:
    if self._defined:
      return

    for logger in self._wandb_loggers(trainer):
      run = logger.experiment
      if not hasattr(run, "define_metric"):
        continue

      run.define_metric(self.epoch_metric, hidden=True)
      run.define_metric("trainer/global_step", hidden=True)

      for prefix in self.metric_prefixes:
        run.define_metric(
            f"{prefix}/*",
            step_metric=self.epoch_metric,
            step_sync=True,
        )

      for metric in self.summary_min_metrics:
        run.define_metric(
            metric,
            step_metric=self.epoch_metric,
            step_sync=True,
            summary="min",
        )

      for metric in self.summary_max_metrics:
        run.define_metric(
            metric,
            step_metric=self.epoch_metric,
            step_sync=True,
            summary="max",
        )

    self._defined = True

  @rank_zero_only
  def _log_epoch(self, trainer: L.Trainer) -> None:
    epoch_step = (trainer.current_epoch, trainer.global_step)
    if self._last_logged_epoch_step == epoch_step:
      return

    metrics = {self.epoch_metric: trainer.current_epoch}
    for logger in self._wandb_loggers(trainer):
      logger.log_metrics(metrics, step=trainer.global_step)

    self._last_logged_epoch_step = epoch_step

  def _wandb_loggers(self, trainer: L.Trainer) -> list[WandbLogger]:
    loggers = getattr(trainer, "loggers", None)
    if loggers is None:
      logger = getattr(trainer, "logger", None)
      loggers = [] if logger is None else [logger]

    return [
        logger
        for logger in loggers
        if isinstance(logger, WandbLogger)
    ]
