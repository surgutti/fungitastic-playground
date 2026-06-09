from pathlib import Path

import click
import fiddle as fdl
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets.fungitastic import FungiTasticDataset
from src.utils.config import parse_fiddle_config


def parse_scales(scales: str) -> list[float]:
    values = [float(x.strip()) for x in scales.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one scale is required")
    return values


@torch.inference_mode()
def predict_tta(
    model: torch.nn.Module,
    images: torch.Tensor,
    scales: list[float],
    hflip: bool,
) -> torch.Tensor:
    original_size = images.shape[-2:]
    logits_sum = None
    count = 0

    for scale in scales:
        if scale == 1.0:
            scaled = images
        else:
            scaled = F.interpolate(images, scale_factor=scale, mode="bilinear", align_corners=False)

        logits = model(scaled)
        logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)
        logits_sum = logits if logits_sum is None else logits_sum + logits
        count += 1

        if hflip:
            flipped = torch.flip(scaled, dims=[-1])
            flip_logits = model(flipped)
            flip_logits = torch.flip(flip_logits, dims=[-1])
            flip_logits = F.interpolate(flip_logits, size=original_size, mode="bilinear", align_corners=False)
            logits_sum = logits_sum + flip_logits
            count += 1

    assert logits_sum is not None
    return logits_sum / count


def update_confusion_matrix(
    confmat: torch.Tensor,
    preds: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    ignore_index: int | None,
) -> None:
    if ignore_index is None:
        valid = torch.ones_like(masks, dtype=torch.bool)
    else:
        valid = masks != ignore_index

    preds = preds[valid]
    targets = masks[valid]
    if targets.numel() == 0:
        return

    idx = targets * num_classes + preds
    counts = torch.bincount(idx, minlength=num_classes * num_classes)
    confmat += counts.reshape(num_classes, num_classes).cpu()


def metrics_from_confmat(confmat: torch.Tensor, include_background: bool) -> tuple[float, float, np.ndarray]:
    confmat = confmat.double()
    total = confmat.sum()
    true_pos = confmat.diag()
    pixel_acc = (true_pos.sum() / total).item() if total > 0 else 0.0

    false_pos = confmat.sum(dim=0) - true_pos
    false_neg = confmat.sum(dim=1) - true_pos
    union = true_pos + false_pos + false_neg
    per_class = torch.where(union > 0, true_pos / union.clamp_min(1.0), torch.nan)

    first = 0 if include_background else 1
    selected = per_class[first:]
    selected = selected[torch.isfinite(selected)]
    mean_iou = selected.mean().item() if selected.numel() > 0 else 0.0
    return pixel_acc, mean_iou, per_class.numpy()


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.argument("checkpoint_path", type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.option("--split", type=click.Choice(["val", "test"]), default="val", show_default=True)
@click.option("--scales", type=str, default="1.0", show_default=True, help="Comma-separated scales, e.g. 0.75,1.0,1.25")
@click.option("--hflip", is_flag=True, default=False)
@click.option("--precision", type=click.Choice(["32", "bf16", "fp16"]), default="bf16", show_default=True)
@click.option("--device", type=str, default="cuda", show_default=True)
def main(
    config_path: str,
    checkpoint_path: str,
    split: str,
    scales: str,
    hflip: bool,
    precision: str,
    device: str,
):
    """Evaluate a checkpoint with optional test-time augmentation.

    Examples:
      uv run python -m scripts.evaluate_tta src/config/resnet50_unet_512.py logs/.../last.ckpt --split val
      uv run python -m scripts.evaluate_tta src/config/resnet50_unet_512.py logs/.../last.ckpt --split val --scales 0.75,1.0,1.25 --hflip
    """

    cfg = parse_fiddle_config(config_path)
    built_cfg = fdl.build(cfg)

    model = built_cfg.model
    data_module = built_cfg.data_module

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().to(device)

    if split == "val":
        data_module.setup("validate")
        dataloader = data_module.val_dataloader()
    else:
        data_module.setup("test")
        dataloader = data_module.test_dataloader()

    num_classes = int(getattr(model, "num_classes", len(FungiTasticDataset.LABEL_TO_ID)))
    ignore_index = getattr(model, "ignore_index", None)
    include_background = bool(getattr(model, "include_background_in_metric", False))
    scale_values = parse_scales(scales)

    confmat = torch.zeros(num_classes, num_classes, dtype=torch.long)

    use_amp = device.startswith("cuda") and precision != "32"
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    for images, masks in tqdm(dataloader, desc=f"evaluate {split}"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
            logits = predict_tta(model, images, scale_values, hflip)

        preds = logits.argmax(dim=1)
        update_confusion_matrix(confmat, preds, masks, num_classes, ignore_index)

    pixel_acc, mean_iou, per_class_iou = metrics_from_confmat(confmat, include_background)

    print(f"checkpoint={Path(checkpoint_path)}")
    print(f"split={split}")
    print(f"scales={scale_values}, hflip={hflip}")
    print(f"pixel_acc={pixel_acc:.6f}")
    print(f"mean_iou={mean_iou:.6f}")
    print("class_id,label,iou")
    for class_id, iou in enumerate(per_class_iou):
        label = FungiTasticDataset.ID_TO_LABEL.get(class_id, f"class_{class_id}")
        value = "" if np.isnan(iou) else f"{iou:.6f}"
        print(f"{class_id},{label},{value}")


if __name__ == "__main__":
    main()
