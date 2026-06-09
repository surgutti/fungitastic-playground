from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.io import ImageReadMode, read_image, write_png
from tqdm import tqdm

IGNORE_LABELS = {
    "fruiting_body",
    "microscopic",
    "unknown underside",
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

SUBSET_TO_NAME = {
    "m": "Mini",
    "fs": "FewShot",
    "full": "",
}


def rle_to_mask(rle_points, height: int, width: int) -> np.ndarray:
    mask = np.zeros(height * width, dtype=np.uint8)
    position, value = 0, 0
    for count in rle_points[:-4]:
        if value == 1:
            mask[position : position + count] = 1
        position += count
        value ^= 1
    return mask.reshape(height, width)


def _resize_image(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    image = image.unsqueeze(0).float()
    image = torch.nn.functional.interpolate(
        image,
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    return image.squeeze(0).round().clamp(0, 255).to(torch.uint8)


def letterbox_square(image: torch.Tensor, mask: np.ndarray, target_size: int) -> tuple[torch.Tensor, np.ndarray]:
    _, height, width = image.shape
    scale = min(target_size / height, target_size / width)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))

    image = _resize_image(image, (new_height, new_width))
    mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    pad_h = target_size - new_height
    pad_w = target_size - new_width
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    image = torch.nn.functional.pad(image, (left, right, top, bottom), value=0)
    mask = np.pad(mask, ((top, bottom), (left, right)), mode="constant", constant_values=0)
    return image, mask


def center_crop_or_pad(image: torch.Tensor, mask: np.ndarray, target_size: int) -> tuple[torch.Tensor, np.ndarray]:
    _, height, width = image.shape

    pad_h = max(0, target_size - height)
    pad_w = max(0, target_size - width)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    image = torch.nn.functional.pad(image, (left, right, top, bottom), value=0)
    mask = np.pad(mask, ((top, bottom), (left, right)), mode="constant", constant_values=0)

    _, height, width = image.shape
    top = max(0, (height - target_size) // 2)
    left = max(0, (width - target_size) // 2)

    image = image[:, top : top + target_size, left : left + target_size]
    mask = mask[top : top + target_size, left : left + target_size]
    return image, mask


def dataset_prefix(subset: str) -> str:
    subset_name = SUBSET_TO_NAME[subset]
    return "FungiTastic" if subset == "full" else f"FungiTastic-{subset_name}"


def split_names(split: str) -> tuple[str, str]:
    mask_name = {"train": "Train", "val": "Validation", "test": "Test"}[split]
    meta_name = {"train": "Train", "val": "Val", "test": "Test"}[split]
    return mask_name, meta_name


@click.command()
@click.option("--data_root", type=click.Path(exists=True, file_okay=False, dir_okay=True), default="data/FungiTastic", show_default=True)
@click.option("--subset", type=click.Choice(["m", "fs", "full"]), default="m", show_default=True)
@click.option("--image_size", type=click.Choice(["300", "500", "720", "fullsize"]), default="500", show_default=True)
@click.option("--target_size", type=int, default=512, show_default=True)
@click.option("--resize_mode", type=click.Choice(["letterbox", "center_crop"]), default="letterbox", show_default=True)
@click.option("--output_root", type=click.Path(file_okay=False, dir_okay=True), default=None)
def main(data_root: str, subset: str, image_size: str, target_size: int, resize_mode: str, output_root: str | None):
    """Prepare compact segmentation NPZ files at a chosen resolution.

    Recommended first high-quality setup:
      uv run scripts/download.py --subset m --metadata --masks --images --size 500 --save_path ./data
      uv run python -m scripts.prepare_dataset_configurable --subset m --image_size 500 --target_size 512
    """

    data_root_path = Path(data_root)
    prefix = dataset_prefix(subset)
    size_dir = f"{image_size}p" if image_size != "fullsize" else "fullsize"

    if output_root is None:
        output_root_path = data_root_path / f"SegmentationDataset-{prefix.replace('FungiTastic-', '')}-{size_dir}-{target_size}px"
    else:
        output_root_path = Path(output_root)

    resize_fn = letterbox_square if resize_mode == "letterbox" else center_crop_or_pad

    for split in ["train", "val", "test"]:
        mask_name, meta_name = split_names(split)
        mask_path = data_root_path / f"{prefix}-{mask_name}Masks.parquet"
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)

        meta_path = data_root_path / "metadata" / prefix / f"{prefix}-{meta_name}.csv"
        if not meta_path.is_file():
            raise FileNotFoundError(meta_path)

        image_root = data_root_path / prefix / split / size_dir
        if not image_root.is_dir():
            raise FileNotFoundError(image_root)

        meta = pd.read_csv(meta_path, usecols=["filename"])
        valid_filenames = set()
        for filename in meta["filename"]:
            if not (image_root / filename).is_file():
                raise FileNotFoundError(image_root / filename)
            valid_filenames.add(filename)

        masks_db = pd.read_parquet(
            mask_path,
            columns=["file_name", "label", "width", "height", "rle"],
        ).rename(columns={"file_name": "filename"})

        masks_db = masks_db[masks_db["filename"].isin(valid_filenames)].copy()
        masks_db = masks_db[~masks_db["label"].isin(IGNORE_LABELS)]
        masks_db = masks_db[masks_db["label"].isin(LABEL_TO_ID)]

        masks = []
        image_filenames = []
        (output_root_path / split).mkdir(parents=True, exist_ok=True)

        for filename, group in tqdm(masks_db.groupby("filename"), desc=f"prepare {split}"):
            height = int(group["height"].iloc[0])
            width = int(group["width"].iloc[0])
            mask = np.zeros((height, width), dtype=np.uint8)

            for _, row in group.iterrows():
                label_id = LABEL_TO_ID[row["label"]]
                rle_mask = rle_to_mask(row["rle"], height, width)
                mask[rle_mask == 1] = label_id

            image = read_image(image_root / filename, mode=ImageReadMode.RGB)
            image_height, image_width = image.shape[-2:]

            mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
            image, mask = resize_fn(image, mask, target_size)

            png_name = Path(filename).with_suffix(".png").name
            write_png(image, output_root_path / split / png_name)
            masks.append(mask)
            image_filenames.append(png_name)

        np.savez_compressed(
            output_root_path / f"dataset-{split}.npz",
            image_filenames=np.asarray(image_filenames),
            masks=np.asarray(masks, dtype=np.uint8),
        )

        print(f"{split}: saved {len(masks)} samples to {output_root_path}")

    print(f"Done. Use segmentation_root={output_root_path}")


if __name__ == "__main__":
    main()
