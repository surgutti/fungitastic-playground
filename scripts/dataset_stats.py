from pathlib import Path

import click
import numpy as np

from src.datasets.fungitastic import FungiTasticDataset


@click.command()
@click.option("--segmentation_root", type=click.Path(exists=True, file_okay=False, dir_okay=True), default="data/FungiTastic/SegmentationDataset")
@click.option("--split", type=str, default="train", show_default=True)
@click.option("--clip_max", type=float, default=8.0, show_default=True)
@click.option("--background_weight", type=float, default=0.35, show_default=True)
def main(segmentation_root: str, split: str, clip_max: float, background_weight: float):
    """Print class pixel statistics and a reasonable CE-weight tuple.

    Example:
      uv run python -m scripts.dataset_stats --segmentation_root data/FungiTastic/SegmentationDataset-Mini-500p-512px
    """

    path = Path(segmentation_root) / f"dataset-{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        masks = np.asarray(data["masks"], dtype=np.uint8)

    num_classes = len(FungiTasticDataset.LABEL_TO_ID)
    counts = np.bincount(masks.reshape(-1), minlength=num_classes).astype(np.float64)
    freq = counts / counts.sum()

    foreground = freq[1:]
    median_fg = np.median(foreground[foreground > 0])
    weights = median_fg / np.maximum(freq, 1e-12)
    weights = np.clip(weights, 0.0, clip_max)
    weights[0] = background_weight
    weights = weights / weights[1]  # keep cap close to 1.0

    print(f"split={split}")
    print(f"num_images={masks.shape[0]}")
    print("class_id,label,pixels,frequency,ce_weight")
    for class_id in range(num_classes):
        label = FungiTasticDataset.ID_TO_LABEL[class_id]
        print(f"{class_id},{label},{int(counts[class_id])},{freq[class_id]:.8f},{weights[class_id]:.4f}")

    as_tuple = tuple(float(f"{w:.4f}") for w in weights)
    print()
    print(f"Suggested FUNGITASTIC_CE_CLASS_WEIGHTS = {as_tuple}")


if __name__ == "__main__":
    main()
