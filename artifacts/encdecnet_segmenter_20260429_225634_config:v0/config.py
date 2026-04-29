import fiddle as fdl
from lightning.pytorch.callbacks import model_checkpoint
from lightning.pytorch.loggers import wandb
from src.config import schemas
from src.datasets import fungitastic
from src.models.architectures import encdecnet
from src.models import segmentation_model


def build_config():
  root = fdl.Config(schemas.ExperimentConfig)
  root.name = 'encdecnet_segmenter'
  
  root.model = fdl.Config(segmentation_model.SegmentationModel)
  root.model.embed_ch_dim = 32
  root.model.num_classes = 8
  root.model.lr = 0.0003
  root.model.weight_decay = 0.0001
  
  root.model.backbone = fdl.Config(encdecnet.EncDecNetBackbone)
  root.model.backbone.in_channels = 3
  root.model.backbone.base_channels = 32
  root.model.backbone.out_channels = 32
  
  root.data_module = fdl.Config(fungitastic.FungiTasticDataModule)
  root.data_module.data_root = 'data/FungiTastic'
  root.data_module.batch_size = 64
  
  root.training_cfg = fdl.Config(schemas.TrainingConfig)
  root.training_cfg.max_epochs = 5
  root.training_cfg.callbacks = []
  
  root.training_cfg.wandb_logger = fdl.Partial(wandb.WandbLogger)
  root.training_cfg.wandb_logger.project = 'FungiTasticSegmentation'
  root.training_cfg.wandb_logger.entity = 'university-of-wroclaw-nntp-course'
  
  root.training_cfg.checkpoint_callback = fdl.Partial(model_checkpoint.ModelCheckpoint)
  root.training_cfg.checkpoint_callback.monitor = 'val/loss'
  root.training_cfg.checkpoint_callback.save_top_k = 1
  root.training_cfg.checkpoint_callback.mode = 'min'
  root.training_cfg.checkpoint_callback.every_n_epochs = 1
  
  return root