from torchvision.transforms import v2

from src.datasets.transforms import (
  build_segmentation_eval_transform,
  build_segmentation_train_transform,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Class order: background, cap, stem, gills, pores, ring.
# These are intentionally softened rather than raw inverse-frequency weights.
FUNGITASTIC_CE_CLASS_WEIGHTS = (0.35, 1.0, 1.5, 1.5, 4.0, 5.0)


def build_train_transform(image_size: int = 300) -> v2.Compose:
  return build_segmentation_train_transform(
    image_size=image_size,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
  )


def build_eval_transform() -> v2.Compose:
  return build_segmentation_eval_transform(
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
  )
