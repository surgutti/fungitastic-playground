import fiddle as fdl

from lightning.pytorch.callbacks import ModelCheckpoint

from src.config.schemas import ExperimentConfig, TrainingConfig
from src.models.architectures.encdecnet import EncDecNetBackbone
from src.models.segmentation_model import SegmentationModel
from src.datasets.fungitastic import FungiTasticDataModule

from src.config.constants import WANDB_ENTITY, WANDB_PROJECT

from lightning.pytorch.loggers import WandbLogger

def build_config() -> fdl.Config[ExperimentConfig]:
  max_epochs = 5
  embed_ch_dim = 32
  num_classes = 9

  architecture = fdl.Config(
    EncDecNetBackbone,
    in_channels=3,
    base_channels=32,
    out_channels=embed_ch_dim
  )

  data_module = fdl.Config(
    FungiTasticDataModule,
    "data/FungiTastic",
    batch_size=32
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
    lr=3e-4,
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
    "encdecnet_segmenter",
    model,
    data_module,
    training_cfg
  )
