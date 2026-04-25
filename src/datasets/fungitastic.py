from pathlib import Path
from typing import Callable

import lightning as L
import torch
import pandas as pd
# from torchvision.transforms import Transform
# import torchvision.transforms as transforms
from torchvision.transforms import v2
from PIL import Image
import numpy as np
import re
import torchvision.transforms.v2.functional as F

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
    self.image_dir = Path(image_dir)
    self.masks_file = Path(masks_file)

    self.path_map = {}
    for p in self.image_dir.glob("*.JPG"):
        numbers = re.findall(r'\d+', p.stem)
        if numbers:
            longest_id = max(numbers, key=len)
            self.path_map[longest_id] = p
        
    self.transform = transform
    self.image_transform = image_transform
    self.df = pd.read_parquet(masks_file, columns=['label', 'file_name', 'width', 'height', 'rle'])
    self.label_map = {'cap': 1, 'fruiting_body': 2, 'gills': 3, 'pores': 4, 'ring': 5, 'stem':6, 'unknown underside': 7}
  def __len__(self):
    return len(self.df)
  

  def rle_to_mask(self, width, height, rle, mask_label):
    mask_flat = np.zeros(width * height, dtype=np.uint8)
    current_pos = 0
    for i, length in enumerate(rle):
        if i % 2 == 1:
            mask_flat[current_pos : current_pos + length] = self.label_map[mask_label]
        current_pos += length
            
    mask_2d = mask_flat.reshape((height, width), order='F')
    return Image.fromarray(mask_2d)

  def __getitem__(self, idx: int):
    metadata = self.df.iloc[idx]

    original_name = metadata['file_name']
    print("Original name: ", original_name)

    numbers = re.findall(r'\d+', original_name)
    if not numbers:
        raise ValueError(f"No digits found in filename: {original_name}")
        
    obs_id = max(numbers, key=len)
    
    img_path = self.path_map.get(obs_id)
    
    if img_path is None:
        raise FileNotFoundError(f"ID {obs_id} not found in path_map. Check prefixes/suffixes.")

    image = Image.open(img_path).convert("RGB")
    mask = self.rle_to_mask(metadata['width'], metadata['height'], metadata['rle'], metadata['label'])

    if self.transform is not None:
      image, mask = self.transform(image, mask)
    if self.image_transform is not None:
      image = self.image_transform(image)

    if not isinstance(image, torch.Tensor):
      image = F.to_dtype(F.to_image(image), torch.float32, scale=True)
  
    if not isinstance(mask, torch.Tensor):
      mask = torch.as_tensor(np.array(mask), dtype=torch.long)

    return image, mask

class FungiTasticDataModule(L.LightningDataModule):

  """
  Args:
    data_dir:         Place where the download.py script put data
    batch_size:       Size of a batch
    transform:        Transforms (v2) applied to both images and masks (resizing, crop, ...)
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
