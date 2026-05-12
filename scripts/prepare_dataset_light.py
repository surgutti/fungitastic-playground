from pathlib import Path
import torch
import cv2
import pandas as pd
from torchvision.io import read_image, write_png, ImageReadMode
import numpy as np
from tqdm import tqdm
import pyarrow.parquet as pq  # <-- Added this import

TARGET_SIZE = 300
DATA_ROOT = Path("data/FungiTastic")
OUTPUT_ROOT = DATA_ROOT / "SegmentationDataset"

IGNORE_LABELS = {
    "fruiting_body",
    "microscopic",
    "unknown underside",
    # these two only are occuring less than a 0.001% in the dataset.
    "ridges",
    "teeth",
}

LABEL_TO_ID = {
    "background": 0,
    "cap": 1,
    "stem": 2,
    "gills": 3,
    "pores": 4,
    "ring": 5,
}

def rle_to_mask(rle_points, height, width):
    mask = np.zeros(height * width, dtype=np.uint8)

    position, value = 0, 0
    for count in rle_points[:-4]:
        if value == 1:
            mask[position:position + count] = 1
        position += count
        value ^= 1
    
    return mask.reshape(height, width)

def pad_crop(image, mask):
    _, height, width = image.shape

    pad_h = max(0, TARGET_SIZE - height)
    pad_w = max(0, TARGET_SIZE - width)

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
        torch.Tensor(mask),
        (left, right, top, bottom),
        value=0
    ).numpy()

    _, height, width = image.shape

    top = max(0, (height - TARGET_SIZE) // 2)
    left = max(0, (width - TARGET_SIZE) // 2)

    image = image[:, top:top + TARGET_SIZE, left:left + TARGET_SIZE]
    mask = mask[top:top + TARGET_SIZE, left:left + TARGET_SIZE]

    return image, mask

def main():
    for split in ["train", "val", "test"]:

        mask_name = {
            "train": "Train",
            "val": "Validation",
            "test": "Test"
        }[split]

        mask_path = (
            DATA_ROOT
            / f"FungiTastic-Mini-{mask_name}Masks.parquet"
        )

        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)

        meta_name = {
            "train": "Train",
            "val": "Val",
            "test": "Test"
        }[split]

        meta_path = (
            DATA_ROOT
            / "metadata"
            / "FungiTastic-Mini"
            / f"FungiTastic-Mini-{meta_name}.csv"
        )

        if not meta_path.is_file():
            raise FileNotFoundError(meta_path)

        image_root = DATA_ROOT / "FungiTastic-Mini" / split / "300p"

        if not image_root.is_dir():
            raise FileNotFoundError(image_root)

        meta = pd.read_csv(meta_path, usecols=["filename"])

        valid_filenames = set()
        for _, row in meta.iterrows():
            filename = row["filename"]
            if not (image_root / filename).is_file():
                raise FileNotFoundError(image_root / filename)
            valid_filenames.add(filename)

        print(f"{len(valid_filenames)=}")

        
        print(f"Reading and filtering {mask_path.name} in chunks to save RAM...")
        parquet_file = pq.ParquetFile(mask_path)
        filtered_chunks = []
        
        
        for batch in tqdm(parquet_file.iter_batches(batch_size=20000), desc="Processing chunks"):
            
            chunk_db = batch.to_pandas().rename(columns={"file_name": "filename"})
            chunk_db = chunk_db[chunk_db["filename"].isin(valid_filenames)]
            chunk_db = chunk_db[~chunk_db["label"].isin(IGNORE_LABELS)]
            if not chunk_db.empty:
                filtered_chunks.append(chunk_db)
        
        masks_db = pd.concat(filtered_chunks, ignore_index=True)
        print(f"Success! Loaded {len(masks_db)} relevant rows into RAM.")



        unique_filenames = masks_db["filename"].unique()
        num_images = len(unique_filenames)

        image_filenames = []
        

        temp_mask_path = OUTPUT_ROOT / f"temp_{split}_masks.npy"
        masks_on_disk = np.lib.format.open_memmap(
            temp_mask_path, 
            mode='w+', 
            dtype=np.uint8, 
            shape=(num_images, TARGET_SIZE, TARGET_SIZE)
        )
    
        current_idx = 0

        (OUTPUT_ROOT / f"{split}").mkdir(parents=True, exist_ok=True)

        for filename, group in tqdm(masks_db.groupby("filename"), desc="Generating Masks"):
            height = int(group["height"].iloc[0])
            width = int(group["width"].iloc[0])
            
            mask = np.zeros((height, width), dtype=np.uint8)

            for _, row in group.iterrows():
                label_id = LABEL_TO_ID[row["label"]]
                rle_mask = rle_to_mask(row["rle"], height, width)
                mask[rle_mask == 1] = label_id

            image = read_image(image_root / filename, mode=ImageReadMode.RGB)
            image_height, image_width = image.shape[-2:]

            mask = cv2.resize(
                mask,
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST
            )

            image, mask = pad_crop(image, mask)

            png_name = Path(filename).with_suffix(".png").name
            write_png(image, OUTPUT_ROOT / split / png_name)

            masks_on_disk[current_idx] = mask
            image_filenames.append(png_name)
            
            current_idx += 1

        print("Images and masks processed safely to disk.")
        
        del masks_on_disk 

        safe_masks = np.load(temp_mask_path, mmap_mode='r')

        print("Compressing into .npz...")
        np.savez_compressed(
            OUTPUT_ROOT / f"dataset-{split}.npz", 
            image_filenames=np.asarray(image_filenames), 
            masks=safe_masks
        )
        
        temp_mask_path.unlink()
        
        print(f"Masks done for {split}\n---")

if __name__ == "__main__":
    main()