import fiddle as fdl

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config.constants import WANDB_ENTITY, WANDB_PROJECT
from src.config.schemas import ExperimentConfig, TrainingConfig
from src.datasets.augmentation import build_eval_transform, build_train_transform
from src.datasets.fungitastic import FungiTasticDataModule
from src.models.architectures.encdecnet import EncDecNetBackbone
from src.models.segmentation_model import SegmentationModel


def build_config() -> fdl.Config[ExperimentConfig]:
    """Legacy config for early EncDecNet checkpoints.

    Some April checkpoints were trained before the current 6-class compact mask
    setup. Their final head has 8 output channels and their Lightning module did
    not contain the `ce_class_weights` buffer. Use this config only for loading
    old checkpoints such as:

      logs/encdecnet_segmenter_20260430_011129/epoch=0-step=432.ckpt
    """

    max_epochs = 10
    embed_ch_dim = 32
    num_classes = 8

    architecture = fdl.Config(
        EncDecNetBackbone,
        in_channels=3,
        base_channels=32,
        out_channels=embed_ch_dim,
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

    checkpoint_callback = fdl.Partial(
        ModelCheckpoint,
        monitor="val/mean_iou",
        every_n_epochs=1,
        save_top_k=1,
        mode="max",
    )

    model = fdl.Config(
        SegmentationModel,
        architecture,
        embed_ch_dim=embed_ch_dim,
        num_classes=num_classes,
        lr=3e-4,
        weight_decay=1e-4,
    )

    training_cfg = fdl.Config(
        TrainingConfig,
        wandb_logger=wandb_logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=max_epochs,
        callbacks=[],
    )

    return fdl.Config(
        ExperimentConfig,
        "legacy_encdecnet_8class_segmenter",
        model,
        data_module,
        training_cfg,
    )
