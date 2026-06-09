import fiddle as fdl

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config.constants import WANDB_ENTITY, WANDB_PROJECT
from src.config.schemas import ExperimentConfig, TrainingConfig
from src.datasets.augmentation import (
    FUNGITASTIC_CE_CLASS_WEIGHTS,
    build_eval_transform,
    build_train_transform,
)
from src.datasets.fungitastic import FungiTasticDataModule
from src.models.advanced_segmentation_model import AdvancedSegmentationModel
from src.models.architectures.resnet_unet import ResNetUNetBackbone


def build_config() -> fdl.Config[ExperimentConfig]:
    max_epochs = 100
    embed_ch_dim = 128
    num_classes = 6

    architecture = fdl.Config(
        ResNetUNetBackbone,
        backbone_name="resnet101",
        out_channels=embed_ch_dim,
        decoder_channels=(512, 256, 128, 128),
        pretrained=True,
        dropout=0.05,
        freeze_stem=False,
        use_aspp=True,
    )

    data_module = fdl.Config(
        FungiTasticDataModule,
        "data/FungiTastic",
        batch_size=2,
        num_workers=8,
        train_transform=build_train_transform(image_size=640),
        eval_transform=build_eval_transform(),
        segmentation_root="data/FungiTastic/SegmentationDataset-Mini-720p-640px",
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
        save_top_k=3,
        save_last=True,
        mode="max",
    )

    callbacks = [
        fdl.Config(
            EarlyStopping,
            monitor="val/mean_iou",
            patience=20,
            mode="max",
        )
    ]

    model = fdl.Config(
        AdvancedSegmentationModel,
        architecture,
        embed_ch_dim=embed_ch_dim,
        num_classes=num_classes,
        lr=1e-4,
        min_lr=1e-6,
        ce_weight=0.6,
        dice_weight=1.0,
        focal_weight=0.1,
        tversky_weight=0.1,
        weight_decay=1e-4,
        class_weights=FUNGITASTIC_CE_CLASS_WEIGHTS,
        scheduler="cosine",
        head_dropout=0.05,
    )

    training_cfg = fdl.Config(
        TrainingConfig,
        wandb_logger=wandb_logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=max_epochs,
        callbacks=callbacks,
    )

    return fdl.Config(
        ExperimentConfig,
        "advanced_resnet101_unet_640",
        model,
        data_module,
        training_cfg,
    )
