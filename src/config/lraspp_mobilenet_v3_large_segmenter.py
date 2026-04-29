import fiddle as fdl

from lightning.pytorch.callbacks import ModelCheckpoint

from src.config.schemas import ExperimentConfig, TrainingConfig
from src.models.segmentation_model import SegmentationModel
from src.datasets.fungitastic import FungiTasticDataModule
from src.models.architectures.mobilenet_v3 import MobileNetV3
from src.config.constants import WANDB_ENTITY, WANDB_PROJECT

from lightning.pytorch.loggers import WandbLogger

def build_config() -> fdl.Config[ExperimentConfig]:
  max_epochs = 20
  embed_ch_dim = 21
  num_classes = 8

  architecture = fdl.Config(
    MobileNetV3
  )

  data_module = fdl.Config(
    FungiTasticDataModule,
    "data/FungiTastic",
    batch_size=64,
    num_workers=2
  )

  wandb_logger = fdl.Partial(
      WandbLogger,
      entity=WANDB_ENTITY,
      project=WANDB_PROJECT,
  )

  checkpoints_callback = fdl.Partial(
    ModelCheckpoint,
    monitor="val/loss",
    every_n_epochs=1,
    save_top_k=1,
    mode="min"
  )

  model = fdl.Config(
    SegmentationModel,
    architecture,
    embed_ch_dim=embed_ch_dim,
    num_classes=num_classes,
    lr=1e-4,
    weight_decay=1e-4
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
    "lraspp_mobilenet_v3_large_segmenter",
    model,
    data_module,
    training_cfg
  )
