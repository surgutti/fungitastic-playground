from pathlib import Path
from typing import Callable

import lightning as L
import torch
from torchvision.transforms import Transform

from torch.utils.data import DataLoader, Dataset, random_split

class FungiTasticDataset(Dataset):
  def __init__(
      self,
      image_dir: Path,
      masks_file: Path,
      transform: Callable [[torch.Tensor], torch.Tensor] | None = None,
      image_transform: Callable [[torch.Tensor], torch.Tensor] | None = None
  ):
    super().__init__()
    # TODO
    pass

  def __getitem__(self, idx: int):
    # TODO
    pass

class FungiTasticDataModule(L.LightningDataModule):

  """
  Args:
    data_dir:         Place where the download.py script put data
    batch_size:       Size of a batch
    transform:        Transforms applied to both images and masks (resizing, crop, ...)
    image_transforms: Transforms applied only to images (distortion, normalization)
  """

  def __init__(
      self,
      data_dir: str | Path,
      batch_size: int = 32,
      transform: Callable [[torch.Tensor], torch.Tensor] | None = None,
      image_transform: Callable [[torch.Tensor], torch.Tensor] | None = None
  ):
    super().__init__()

    self.data_dir = Path(data_dir)
    self.batch_size = batch_size

    image_dir = self.data_dir / "FungiTastic" / "FungiTastic"
    masks_dir = self.data_dir / "FungiTastic"
    
    self.train_dataset = FungiTasticDataset(
      image_dir / "train" / "300p", 
      masks_dir / "FungiTastic-Mini-TrainMasks.parquet",
      transform=transform,
      image_transform=image_transform
    )
    
    self.val_dataset = FungiTasticDataset(
      image_dir / "val" / "300p", 
      masks_dir / "FungiTastic-Mini-ValidationMasks.parquet",
      transform=transform,
      image_transform=image_transform
    )
    
    self.test_dataset = FungiTasticDataset(
      image_dir / "test" / "300p", 
      masks_dir / "FungiTastic-Mini-TestMasks.parquet",
      transform=transform,
      image_transform=image_transform
    )
    
  def train_dataloader(self):
    return DataLoader(
      self.train_dataset, 
      self.batch_size,
      shuffle=True,
      num_workers=8,
      pin_memory=True,
      persistent_workers=True
    )

  def val_dataloader(self):
    return DataLoader(
      self.val_dataset, 
      self.batch_size,
      shuffle=False,
      num_workers=8,
      pin_memory=True,
      persistent_workers=True
    )

  def test_dataloader(self):
    return DataLoader(
      self.test_dataset, 
      self.batch_size,
      shuffle=False,
      num_workers=8,
      pin_memory=True,
      persistent_workers=True
    )