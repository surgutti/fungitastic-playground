from pathlib import Path
from typing import Callable

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, read_image


class FungiTasticDataset(Dataset):

  LABEL_TO_ID = {
    "background": 0,
    "cap": 1,
    "stem": 2,
    "gills": 3,
    "pores": 4,
    "ring": 5,
  }

  ID_TO_LABEL = {
    v: k for k, v in LABEL_TO_ID.items()
  }

  DEFAULT_SEGMENTATION_DIR = "SegmentationDataset"

  def __init__(
      self,
      data_root: str | Path,
      split: str,
      transform: Callable | None = None,
      image_transform: Callable | None = None,
      segmentation_root: str | Path | None = None,
  ):
    self.data_root = Path(data_root)
    self.split = split
    self.transform = transform
    self.image_transform = image_transform

    if segmentation_root is None:
      segmentation_root = self.data_root / self.DEFAULT_SEGMENTATION_DIR
    self.segmentation_root = Path(segmentation_root)
    
    split_path = self.segmentation_root / f"dataset-{split}.npz"
    if not split_path.is_file():
      raise FileNotFoundError(
        f"Prepared split file not found: {split_path}."
        "Run `uv run python scripts/prepare_dataset.py` first."
      )

    with np.load(split_path, allow_pickle=False) as data:
      self.filenames = data["image_filenames"].astype(str).tolist()
      self.masks = np.asarray(data["masks"], dtype=np.uint8)

    self.image_paths = [
      self.segmentation_root / split / filename
      for filename in self.filenames
    ]

    if len(self.image_paths) != len(self.masks):
      raise ValueError(
        f"Image/mask count mismatch for {split}: "
        f"{len(self.image_paths)} image vs {len(self.masks)} masks"
      )
    
  def __len__(self) -> int:
    return len(self.filenames)

  def __getitem__(self, idx: int):
    image = read_image(self.image_paths[idx], mode=ImageReadMode.RGB)
    image = image.float() / 255.0

    mask = torch.from_numpy(self.masks[idx]).long()

    if self.image_transform is not None:
      image = self.image_transform(image)

    if self.transform is not None:
      image = tv_tensors.Image(image)
      mask = tv_tensors.Mask(mask)
      image, mask = self.transform(image, mask)
      image = image.as_subclass(torch.Tensor)
      mask = mask.as_subclass(torch.Tensor).long()

    return image, mask

class FungiTasticDataModule(L.LightningDataModule):

  def __init__(
      self,
      data_root: str | Path,
      batch_size: int = 64,
      transform: Callable | None = None,
      image_transform: Callable | None = None,
      train_transform: Callable | None = None,
      eval_transform: Callable | None = None,
      train_image_transform: Callable | None = None,
      eval_image_transform: Callable | None = None,
      segmentation_root: str | Path | None = None,
      num_workers: int = 4,
      pin_memory: bool = True,
      persistent_workers: bool = True
  ):
    super().__init__()

    self.data_root = Path(data_root)
    self.segmentation_root = Path(segmentation_root) if segmentation_root is not None else None
    self.batch_size = batch_size
    self.train_image_transform = (
      train_image_transform
      if train_image_transform is not None
      else image_transform
    )
    self.eval_image_transform = (
      eval_image_transform
      if eval_image_transform is not None
      else image_transform
    )
    self.train_transform = (
      train_transform
      if train_transform is not None
      else transform
    )
    self.eval_transform = (
      eval_transform
      if eval_transform is not None
      else transform
    )
    self.num_workers = num_workers
    self.pin_memory = pin_memory
    self.persistent_workers = persistent_workers and num_workers > 0

    self.train_dataset: FungiTasticDataset | None = None
    self.val_dataset: FungiTasticDataset | None = None
    self.test_dataset: FungiTasticDataset | None = None

  def setup(self, stage: str | None = None):
    if stage in (None, "fit"):
      if self.train_dataset is None:
        self.train_dataset = FungiTasticDataset(
          self.data_root,
          "train",
          image_transform=self.train_image_transform,
          transform=self.train_transform,
          segmentation_root=self.segmentation_root,
        )

      if self.val_dataset is None:
        self.val_dataset = FungiTasticDataset(
          self.data_root,
          "val",
          image_transform=self.eval_image_transform,
          transform=self.eval_transform,
          segmentation_root=self.segmentation_root,
        )

    if stage in (None, "validate"):
      if self.val_dataset is None:
        self.val_dataset = FungiTasticDataset(
          self.data_root,
          "val",
          image_transform=self.eval_image_transform,
          transform=self.eval_transform,
          segmentation_root=self.segmentation_root,
        )

    if stage in (None, "test"):
      if self.test_dataset is None:
        self.test_dataset = FungiTasticDataset(
          self.data_root,
          "test",
          image_transform=self.eval_image_transform,
          transform=self.eval_transform,
          segmentation_root=self.segmentation_root,
        )

  def train_dataloader(self):
    if self.train_dataset is None:
      self.setup("fit")
    assert self.train_dataset is not None

    return self._dataloader(
      self.train_dataset,
      shuffle=True,
    )

  def val_dataloader(self):
    if self.val_dataset is None:
      self.setup("validate")
    assert self.val_dataset is not None

    return self._dataloader(
      self.val_dataset,
      shuffle=False,
    )

  def test_dataloader(self):
    if self.test_dataset is None:
      self.setup("test")
    assert self.test_dataset is not None

    return self._dataloader(
      self.test_dataset,
      shuffle=False,
    )

  def _dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
    return DataLoader(
      dataset,
      batch_size=self.batch_size,
      shuffle=shuffle,
      num_workers=self.num_workers,
      pin_memory=self.pin_memory,
      persistent_workers=self.persistent_workers,
    )
