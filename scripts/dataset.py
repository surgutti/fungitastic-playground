
import torch as T
import torchvision as TV
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
import pandas as pd
import numpy as np
import ast

class SegmetationDataset(Dataset):
  def __init__(self, root, split="train", size="300", transforms=None):
    self.root = Path(root)
    self.split = split
    self.transforms = transforms

    split_to_csv = {
      "train": "FungiTastic-Mini-ClosedSet-Train.csv",
      "val": "FungiTastic-Mini-ClosedSet-Val.csv",
      "test": "FungiTastic-Mini-ClosedSet-Test.csv",
    }

    split_to_masks = {
      "train": "FungiTastic-Mini-TrainMasks.parquet",
      "val": "FungiTastic-Mini-ValMasks.parquet",
      "test": "FungiTastic-Mini-TestMasks.parquet"
    }

    csv_path = self.root / "metadata" / "FungiTastic-Mini" / split_to_csv[split]
    mask_path = self.root / "masks" / split_to_masks[split]

    df = pd.read_csv(csv_path)
    df["image_path"] = df["filename"].apply(
      lambda x: str(self.root / "FungiTastic-Mini" / split / f"{size}p" / x)
    )

    gt_masks = pd.read_parquet(mask_path).rename(columns={"file_name": "filename"}, inplace=True)

    if 'rle' in gt_masks.columns and isinstance(gt_masks['rle'].iloc[0], str):
      gt_masks['rle]'] = gt_masks['rle'].apply(ast.literal_eval)

    self.df = df.merge(gt_masks, on="filename", how="inner").reset_index(drop=True)
  
  def __len__(self):
    return len(self.df)

  @staticmethod
  def rle_to_mask(rle_points, height, width):
    mask = np.zeros(height * width, dtype=np.uint8)

    rle_counts = rle_points[:-4]

    current_position = 0
    current_value = 0
    for rle_count in rle_counts:
      mask[current_position:current_position+rle_count] = current_value
      current_position += rle_count
      current_value ^= 1
    
    return mask.reshape((height, width))

  def __getitem__(self, idx):
    row = self.df.iloc[idx]

    image = read_image(row["image_path"], mode=ImageReadMode.RGB)
    mask = self.build_binary_mask(row["rle"], int(row["height"]), int(row["width"]))

    image = TV.Image(image)
    mask = TV.Mask(mask)

    if self.transforms is not None:
      image, mask = self.transforms(image, mask)
    
    image = image.float() / 255.0
    mask = mask.to(T.long)

    return image, mask


def get_train_and_valid_loaders():
  train_dataset = SegmetationDataset("data/FungiTastic", split="train")
  valid_dataset = SegmetationDataset("data/FungiTastic", split="valid")

  train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
  )

  valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
  )