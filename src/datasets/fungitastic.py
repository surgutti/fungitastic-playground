from pathlib import Path
from typing import Callable

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image


class FungiTasticDataset(Dataset):
  """Dataset backed by compact per-split NPZ mask files.

  The image files still come from the regular 300p FungiTastic Mini download.
  The masks are precomputed semantic masks produced by
  scripts/compact_segmentation_dataset.py.
  """

  LABEL_TO_ID = {
    "background": 0,
    "cap": 1,
    "stem": 2,
    "gills": 3,
    "pores": 4,
    "ring": 5,
    "ridges": 6,
    "teeth": 7,
  }

  DEFAULT_SEGMENTATION_DIR = "SegmentationDataset"

  def __init__(
      self,
      data_root: str | Path,
      split: str,
      transform: Callable | None = None,
      segmentation_root: str | Path | None = None,
  ):
    self.data_root = Path(data_root)
    self.split = split
    self.transform = transform

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
      self.segmentation_root / split / Path(filename).with_suffix(".png").name
      for filename in self.filenames
    ]

    if len(self.imaeg_paths) != len(self.masks):
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

    if self.transform is not None:
      image, mask = self.transform(image, mask)

    return image, mask

class FungiTasticDataModule(L.LightningDataModule):

  def __init__(
      self,
      data_root: str | Path,
      batch_size: int = 64,
      transform: Callable | None = None,
      segmentation_root: str | Path | None = None,
      num_workers: int = 8,
      pin_memory: bool = True,
      persistent_workers: bool = True
  ):
    super().__init__()

    self.data_root = Path(data_root)
    self.segmentation_root = Path(segmentation_root) if segmentation_root is not None else None
    self.batch_size = batch_size
    self.transform = transform
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
          transform=self.transform,
          segmentation_root=self.segmentation_root,
        )

      if self.val_dataset is None:
        self.val_dataset = FungiTasticDataset(
          self.data_root,
          "val",
          transform=self.transform,
          compact_root=self.compact_root,
        )

    if stage in (None, "validate"):
      if self.val_dataset is None:
        self.val_dataset = FungiTasticDataset(
          self.data_root,
          "val",
          transform=self.transform,
          segmentation_root=self.segmentation_root,
        )

    if stage in (None, "test"):
      if self.test_dataset is None:
        self.test_dataset = FungiTasticDataset(
          self.data_root,
          "test",
          transform=self.transform,
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
