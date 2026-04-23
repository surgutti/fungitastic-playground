from pathlib import Path
from typing import Callable

import lightning as L
import torch
from torchvision.transforms import Transform

from torch.utils.data import DataLoader, Dataset, random_split

import numpy as np
import pandas as pd
from PIL import Image
import cv2

class FungiTasticDataset(Dataset):

  def __init__(
      self,
      data_root: Path,
      split: str,
      transform: Callable [[torch.Tensor], torch.Tensor] | None = None,
      image_transform: Callable [[torch.Tensor], torch.Tensor] | None = None
  ):
    super().__init__()
    
    self.data_root = data_root
    self.split = split
    self.transform = transform
    self.image_transform = image_transform

    mask_naming = {
      'train': 'Train',
      'val': 'Validation',
      'test': 'Test'
    }

    meta_naming = {
      'train': 'Train',
      'val': 'Val',
      'test': 'Test'
    }

    mask_path = \
      data_root /\
        f'FungiTastic-Mini-{mask_naming[split]}Masks.parquet'
    
    meta_path = \
      data_root /\
        'metadata' /\
          'FungiTastic-Mini' /\
              f'FungiTastic-Mini-{meta_naming[split]}.csv'
    
    self.df = pd.read_csv(meta_path)
    self.df["image_path"] = self.df.filename.apply(
      lambda x: str((data_root / 'FungiTastic' / split / '300p' / x).resolve())
    )

    self.gt_masks = pd.read_parquet(mask_path)
    self.gt_masks.rename(columns={'file_name': 'filename'}, inplace=True)

    self.df = self.df.merge(self.gt_masks, on='filename', how='inner')
    self.uniq_df = self.df['filename'].unique()

    self.df = self.df.label.apply(lambda x:
      if x == 'gills':
        return 1
      elif 
    )

  def __len__(self) -> int:
    return 

  @staticmethod
  def _rle_to_mask(rle_points, height, width):
    mask = np.zeros(height * width, dtype=np.uint8)
    rle_counts = rle_points[:-4]
    current_position = 0
    current_value = 0
    for rle_count in rle_counts:
      mask[current_position:current_position + rle_count] = current_value
      current_value ^= 1
      current_position += rle_count
    mask = mask.reshape((height, width))
    return mask

  def __getitem__(self, idx: int):
    filename = self.uniq_df.iloc[idx]

    rows = self.df.loc[self.df['filename'] == filename]
    image_path = rows['image_path'].iloc[0]

    image = Image.open(image_path)
    width, height = image.size

    mask = torch.zeros((height, width, 3), dtype=torch.uint8)
    for row in rows[['rle', 'label', 'width', 'height']].itertuples():
      if row.label == 'fruiting_body':
        continue
      
      mask = self._rle_to_mask(row.rle, row.height, row.width)
      mask = cv2.resize(mask.astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST).astype(bool)



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