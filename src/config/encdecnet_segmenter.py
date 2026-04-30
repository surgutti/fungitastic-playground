import fiddle as fdl

from lightning.pytorch.callbacks import ModelCheckpoint

from src.config.schemas import ExperimentConfig, TrainingConfig
from src.models.architectures.encdecnet import EncDecNetBackbone
from src.models.weighted_augmented_segmentation_model import WeightedAugmentedSegmentationModel
from src.datasets.fungitastic import FungiTasticDataModule
from src.datasets.augmentation import (
  FUNGITASTIC_CE_CLASS_WEIGHTS,
  build_eval_transform,
  build_train_transform,
)

from src.config.constants import WANDB_ENTITY, WANDB_PROJECT

from lightning.pytorch.loggers import WandbLogger

def build_config() -> fdl.Config[ExperimentConfig]:
  max_epochs = 10
  embed_ch_dim = 32
  num_classes = 6

  architecture = fdl.Config(
    EncDecNetBackbone,
    in_channels=3,
    base_channels=32,
    out_channels=embed_ch_dim
  )

  data_module = fdl.Config(
    FungiTasticDataModule,
    "data/FungiTastic",
    batch_size=32,
    train_transform=build_train_transform(),
    eval_transform=build_eval_transform(),
  )

  wandb_logger = fdl.Partial(
      WandbLogger,
      entity=WANDB_ENTITY,
      project=WANDB_PROJECT,
  )

  checkpoints_callback = fdl.Partial(
    ModelCheckpoint,
    monitor="val/mean_iou",
    every_n_epochs=1,
    save_top_k=1,
    mode="max"
  )

  model = fdl.Config(
    WeightedAugmentedSegmentationModel,
    architecture,
    embed_ch_dim=embed_ch_dim,
    num_classes=num_classes,
    lr=3e-4,
    weight_decay=1e-4,
    class_weights=FUNGITASTIC_CE_CLASS_WEIGHTS,
  )

  training_cfg = fdl.Config(
    TrainingConfig,
    wandb_logger=wandb_logger,
    checkpoint_callback=checkpoints_callback,
    max_epochs=max_epochs,
    callbacks=[]
  )

  return fdl.Config(
    ExperimentConfig,
    "weighted_augmented_encdecnet_segmenter",
    model,
    data_module,
    training_cfg
  )
