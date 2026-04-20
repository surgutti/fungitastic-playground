import fiddle as fdl

from lightning.pytorch.callbacks import ModelCheckpoint

from src.config.schemas import ExperimentConfig
from src.models.architectures.encdecnet import EncDecNetBackbone
from src.models.segmentation_model import SegmentationModel
from src.datasets.fungitastic import FungiTasticDataModule

def build_config() -> fdl.Config[ExperimentConfig]:
  max_epochs = 100
  embed_ch_dim = 32
  num_classes = 8

  architecture = fdl.Config(
    EncDecNetBackbone,
    # TODO
  )

  data_module = fdl.Config(
    FungiTasticDataModule,
    "data",
    batch_size=128
  )

  checkpoints_callback = fdl.Partial(
    ModelCheckpoint,
    monitor="val/acc",
    every_n_epoch=1,
    save_top_k=1,
    mode="max"
  )

  model = fdl.Config(
    SegmentationModel,
    architecture,
    embed_ch_dim=embed_ch_dim,
    num_classes=num_classes,
    lr=1e-4
  )

  return fdl.Config(
    ExperimentConfig,
    "encdecnet_segmenter",
    model,
    data_module
  )
