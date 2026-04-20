from pathlib import Path

import lightning as L
import torch

from torch.utils.data import DataLoader, Dataset, random_split

class FungiTasticDataset(Dataset):
  def __init__(
      self,
      image_dir: Path,
      masks_file: Path
  ):
    super().__init__()
    # TODO
    pass

  @staticmethod
  def _normalize(img: torch.Tensor) -> torch.Tensor:
    # TODO
    pass

  def __getitem__(self, idx: int):
    # TODO
    pass

class FungiTasticDataModule(L.LightningDataModule):
  def __init__(
      self,
      data_dir: str | Path,
      batch_size: int = 32,
  ):
    super().__init__()

    self.data_dir = Path(data_dir)
    self.batch_size = batch_size

    image_dir = self.data_dir / "FungiTastic" / "FungiTastic"
    masks_dir = self.data_dir / "FungiTastic"
    
    self.train_dataset = FungiTasticDataset(
      image_dir / "train" / "300p", 
      masks_dir / "FungiTastic-Mini-TrainMasks.parquet"
    )
    
    self.val_dataset = FungiTasticDataset(
      image_dir / "val" / "300p", 
      masks_dir / "FungiTastic-Mini-ValidationMasks.parquet"
    )
    
    self.test_dataset = FungiTasticDataset(
      image_dir / "test" / "300p", 
      masks_dir / "FungiTastic-Mini-TestMasks.parquet"
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