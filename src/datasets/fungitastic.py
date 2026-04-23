from pathlib import Path
from typing import Callable

import lightning as L
import torch

from torch.utils.data import DataLoader, Dataset, random_split

import cv2
import numpy as np
import pandas as pd
from torchvision.io import read_image, ImageReadMode

class FungiTasticDataset(Dataset):

  LABEL_TO_ID = {
    "cap": 1,
    "stem": 2,
    "gills": 3,
    "pores": 4,
    "ring": 5,
    "ridges": 6,
    "teeth": 7,
    "unknown underside": 8
  }

  IGNORE_LABELS = {
    "fruiting_body",
    "microscopic"
  }

  def __init__(self, data_root: Path, split: str, transforms=None):
    self.data_root = Path(data_root)
    self.split = split
    self.transform = transforms

    image_root = self.data_root / "FungiTastic" / split / "300p"

    mask_name = {
      "train": "Train",
      "val" : "Validation",
      "test": "Test"
    }[split]

    meta_name = {
      "train": "Train",
      "val": "Val",
      "test": "Test"
    }[split]

    meta_path = (
      self.data_root
      / "metadata"
      / "FungiTastic-Mini"
      / f"FungiTastic-Mini-{meta_name}.csv"
    )

    mask_path = self.data_root / f"FungiTastic-Mini-{mask_name}Masks.parquet"

    meta = pd.read_csv(meta_path, usecols=["filename"])
    masks = pd.read_parquet(
      mask_path,
      columns=["file_name", "label", "width", "height", "rle"]
    ).rename(columns={"file_name": "filename"})

    valid_filenames = set(meta["filename"])
    masks = masks[masks["filename"].isin(valid_filenames)].copy()
    
    masks = masks[~masks["label"].isin(self.IGNORE_LABELS)].copy()
    masks["label_id"] = masks["label"].map(self.LABEL_TO_ID)

    unknown_labels = masks.loc[masks["label_id"].isna(), "label"].unique()
    # print(f'{unknown_labels=}')
    assert len(unknown_labels) == 0

    masks["label_id"] = masks["label_id"].astype(np.uint8)

    masks = masks.sort_values("filename", kind="stable").reset_index(drop=True)

    filenames = masks["filename"].to_numpy()

    starts = np.flatnonzero(
      np.r_[True, filenames[1:] != filenames[:-1]]
    )

    self.part_offsets = np.r_[starts, len(masks)].astype(np.int64)

    self.filenames = filenames[starts].tolist()
    self.image_paths = [
      str(image_root / filename)
      for filename in self.filenames
    ]

    self.part_rles = masks["rle"].to_numpy(dtype=object)
    self.part_label_ids = masks["label_id"].to_numpy(np.uint8)
    self.part_widths = masks["width"].to_numpy(np.int32)
    self.part_heights = masks["height"].to_numpy(np.int32)

    del meta
    del masks

  def __len__(self) -> int:
    return len(self.filenames)

  def __getitem__(self, idx: int):
    image = read_image(self.image_paths[idx], mode=ImageReadMode.RGB)

    start = self.part_offsets[idx]
    end = self.part_offsets[idx + 1]

    mask = self._build_semantic_mask(start, end)

    image_h, image_w = image.shape[-2:]
    mask = cv2.resize(
      mask,
      (image_w, image_h),
      interpolation=cv2.INTER_NEAREST
    )

    mask = torch.from_numpy(mask).long()

    image, mask = self._pad_crop_300(image, mask)

    image = image.float() / 255.0

    if self.transform is not None:
      image, mask = self.transform(image, mask)
    
    return image, mask

  def _build_semantic_mask(self, start: int, end: int) -> np.ndarray:
    height = int(self.part_heights[start])
    width = int(self.part_widths[start])

    semantic = np.zeros((height, width), dtype=np.uint8)

    for part_idx in range(start, end):
      part_mask = self._rle_to_mask(
        self.part_rles[part_idx],
        int(self.part_heights[part_idx]),
        int(self.part_widths[part_idx])
      )

      label_id = self.part_label_ids[part_idx]
      semantic[part_mask.astype(bool)] = label_id
    
    return semantic

  @staticmethod
  def _rle_to_mask(rle_points, height: int, width: int) -> np.ndarray:
    mask = np.zeros(height * width, dtype=np.uint8)

    position = 0
    value = 0
    for count in rle_points[:-4]:
      if value == 1:
        mask[position:position + count] = 1
      
      position += count
      value ^= 1
    
    return mask.reshape(height, width)
  
  @staticmethod
  def _pad_crop_300(image: torch.Tensor, mask: torch.Tensor):
    target = 300
    _, h, w = image.shape

    pad_h = max(0, target - h)
    pad_w = max(0, target - w)

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    image = torch.nn.functional.pad(
      image,
      (left, right, top, bottom),
      value=0
    )

    mask = torch.nn.functional.pad(
      mask,
      (left, right, top, bottom),
      value=0
    )

    _, h, w = image.shape

    top = max(0, (h - target) // 2)
    left = max(0, (w - target) // 2)

    image = image[:, top:top + target, left:left + target]
    mask = mask[top:top + target, left:left + target]

    return image, mask


class FungiTasticDataModule(L.LightningDataModule):

  def __init__(
      self,
      data_root: str | Path,
      batch_size: int = 32,
      transform: Callable [[torch.Tensor], torch.Tensor] | None = None
  ):
    super().__init__()

    self.data_root = Path(data_root)
    self.batch_size = batch_size

    self.train_dataset = FungiTasticDataset(
      data_root,
      "train",
      transform=transform,
    )
    
    self.val_dataset = FungiTasticDataset(
      data_root,
      "val",
      transforms=transform
    )
    
    self.test_dataset = FungiTasticDataset(
      data_root,
      "test",
      transforms=transform
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