from collections.abc import Sequence

from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode


def _size_tuple(size: int | Sequence[int]) -> tuple[int, int]:
  if isinstance(size, int):
    return size, size

  if len(size) != 2:
    raise ValueError(f"Expected size with two values, got {size}")

  return int(size[0]), int(size[1])


def build_segmentation_train_transform(
    image_size: int | Sequence[int],
    mean: Sequence[float],
    std: Sequence[float],
) -> v2.Compose:
  size = _size_tuple(image_size)

  return v2.Compose(
    [
      v2.RandomHorizontalFlip(p=0.5),
      v2.RandomApply(
        [
          v2.RandomAffine(
            degrees=(-20.0, 20.0),
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
            shear=(-5.0, 5.0),
            interpolation=InterpolationMode.BILINEAR,
            fill={
              tv_tensors.Image: 0.0,
              tv_tensors.Mask: 0,
            },
          ),
        ],
        p=0.6,
      ),
      v2.RandomApply(
        [
          v2.RandomResizedCrop(
            size=size,
            scale=(0.78, 1.0),
            ratio=(0.9, 1.1),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
          ),
        ],
        p=0.75,
      ),
      v2.RandomApply(
        [
          v2.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.03,
          ),
        ],
        p=0.8,
      ),
      v2.RandomApply(
        [
          v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ],
        p=0.15,
      ),
      v2.RandomApply(
        [
          v2.GaussianNoise(mean=0.0, sigma=0.02, clip=True),
        ],
        p=0.2,
      ),
      v2.Normalize(mean=mean, std=std),
    ]
  )


def build_segmentation_eval_transform(
    mean: Sequence[float],
    std: Sequence[float],
) -> v2.Compose:
  return v2.Compose(
    [
      v2.Normalize(mean=mean, std=std),
    ]
  )
