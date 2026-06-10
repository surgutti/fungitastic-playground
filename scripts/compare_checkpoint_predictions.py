from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import click
import fiddle as fdl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from src.datasets.augmentation import IMAGENET_MEAN, IMAGENET_STD
from src.datasets.fungitastic import FungiTasticDataset
from src.utils.config import parse_fiddle_config

PALETTE = np.array(
    [
        [0, 0, 0],        # background
        [220, 20, 60],    # cap
        [34, 139, 34],    # stem
        [30, 144, 255],   # gills
        [255, 165, 0],    # pores
        [148, 0, 211],    # ring
    ],
    dtype=np.float32,
) / 255.0


@dataclass
class LoadedModel:
    name: str
    config_path: Path
    checkpoint_path: Path
    model: torch.nn.Module
    dataset: FungiTasticDataset
    filename_to_idx: dict[str, int]


def _split_csv_values(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def parse_model_spec(spec: str) -> tuple[str, Path, Path]:
    """Parse NAME=CONFIG:CKPT or CONFIG:CKPT."""
    if "=" in spec:
        name, rest = spec.split("=", 1)
        name = name.strip()
    else:
        name, rest = "", spec

    if ":" not in rest:
        raise click.BadParameter(
            "model spec must be NAME=CONFIG:CKPT or CONFIG:CKPT; "
            f"got {spec!r}"
        )

    config_str, ckpt_str = rest.split(":", 1)
    config_path = Path(config_str).expanduser()
    checkpoint_path = Path(ckpt_str).expanduser()

    if not name:
        name = checkpoint_path.parent.name or checkpoint_path.stem

    if not config_path.is_file():
        raise click.BadParameter(f"Config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise click.BadParameter(f"Checkpoint not found: {checkpoint_path}")

    return name, config_path, checkpoint_path


def _setup_dataset(data_module, split: str) -> FungiTasticDataset:
    if split == "val":
        data_module.setup("validate")
        dataset = data_module.val_dataset
    elif split == "test":
        data_module.setup("test")
        dataset = data_module.test_dataset
    else:
        data_module.setup("fit")
        dataset = data_module.train_dataset

    if dataset is None:
        raise RuntimeError(f"Dataset for split={split!r} was not created")
    return dataset


def load_model(spec: str, split: str, device: torch.device) -> LoadedModel:
    name, config_path, checkpoint_path = parse_model_spec(spec)
    cfg = parse_fiddle_config(str(config_path))
    built_cfg = fdl.build(cfg)

    model = built_cfg.model
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    dataset = _setup_dataset(built_cfg.data_module, split)
    filename_to_idx = {filename: idx for idx, filename in enumerate(dataset.filenames)}

    return LoadedModel(
        name=name,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model=model,
        dataset=dataset,
        filename_to_idx=filename_to_idx,
    )


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype, device=image.device).view(3, 1, 1)
    image = image * std + mean
    image = image.clamp(0.0, 1.0)
    return image.permute(1, 2, 0).cpu().numpy()


def colorize_mask(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask
    mask_np = np.asarray(mask_np, dtype=np.int64)
    mask_np = np.clip(mask_np, 0, len(PALETTE) - 1)
    return PALETTE[mask_np]


def overlay_mask(image: np.ndarray, mask: torch.Tensor | np.ndarray, alpha: float) -> np.ndarray:
    colors = colorize_mask(mask)
    mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    foreground = (mask_np > 0)[..., None]
    return np.where(foreground, (1.0 - alpha) * image + alpha * colors, image)


def resize_mask_to(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(mask.shape[-2:]) == tuple(size):
        return mask
    return F.interpolate(
        mask[None, None].float(),
        size=size,
        mode="nearest",
    )[0, 0].long()


def choose_filenames(
    base_dataset: FungiTasticDataset,
    explicit_filenames: list[str],
    explicit_indices: list[int],
    num_samples: int,
    prefer_classes: list[int],
    seed: int,
) -> list[str]:
    filenames: list[str] = []

    for filename in explicit_filenames:
        if filename not in base_dataset.filenames:
            raise click.BadParameter(f"Filename not found in base dataset: {filename}")
        filenames.append(filename)

    for idx in explicit_indices:
        if idx < 0 or idx >= len(base_dataset):
            raise click.BadParameter(f"Index out of range: {idx}")
        filenames.append(base_dataset.filenames[idx])

    if len(filenames) >= num_samples:
        return filenames[:num_samples]

    rng = np.random.default_rng(seed)

    candidates = np.arange(len(base_dataset))
    if prefer_classes:
        masks = base_dataset.masks
        keep = []
        prefer = set(prefer_classes)
        for idx, mask in enumerate(masks):
            labels = set(np.unique(mask).astype(int).tolist())
            if labels & prefer:
                keep.append(idx)
        if keep:
            candidates = np.asarray(keep, dtype=np.int64)

    rng.shuffle(candidates)
    for idx in candidates:
        filename = base_dataset.filenames[int(idx)]
        if filename not in filenames:
            filenames.append(filename)
        if len(filenames) >= num_samples:
            break

    return filenames


@torch.inference_mode()
def predict_for_filename(
    loaded: LoadedModel,
    filename: str,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    if filename not in loaded.filename_to_idx:
        raise KeyError(f"{filename!r} not found for model {loaded.name}")

    idx = loaded.filename_to_idx[filename]
    image, mask = loaded.dataset[idx]
    image = image.to(device, non_blocking=True)

    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        logits = loaded.model(image.unsqueeze(0))

    pred = logits.argmax(dim=1)[0].cpu()
    return denormalize_image(image.cpu()), mask.cpu().long(), pred.long()


def save_legend(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 1.4))
    ax.axis("off")
    for class_id, label in FungiTasticDataset.ID_TO_LABEL.items():
        if class_id >= len(PALETTE):
            continue
        ax.scatter([], [], color=PALETTE[class_id], label=f"{class_id}: {label}", s=80)
    ax.legend(ncol=6, loc="center", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_sample_figure(
    filename: str,
    base_image: np.ndarray,
    base_mask: torch.Tensor,
    predictions: list[tuple[str, torch.Tensor]],
    output_path: Path,
    alpha: float,
    plain_masks: bool,
) -> None:
    ncols = 2 + len(predictions)
    fig_w = max(10.0, 3.2 * ncols)
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 4.2))
    if ncols == 1:
        axes = [axes]

    axes[0].imshow(base_image)
    axes[0].set_title("image")
    axes[0].axis("off")

    gt_img = colorize_mask(base_mask) if plain_masks else overlay_mask(base_image, base_mask, alpha)
    axes[1].imshow(gt_img)
    axes[1].set_title("ground truth")
    axes[1].axis("off")

    base_size = tuple(base_mask.shape[-2:])
    for ax, (name, pred) in zip(axes[2:], predictions):
        pred = resize_mask_to(pred, base_size)
        pred_img = colorize_mask(pred) if plain_masks else overlay_mask(base_image, pred, alpha)
        ax.imshow(pred_img)
        ax.set_title(name, fontsize=9)
        ax.axis("off")

    fig.suptitle(filename)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


@click.command()
@click.option("--model", "model_specs", multiple=True, required=True, help="NAME=CONFIG:CKPT. Pass many times.")
@click.option("--split", type=click.Choice(["train", "val", "test"]), default="val", show_default=True)
@click.option("--filename", "filenames", multiple=True, help="Exact PNG filename from prepared dataset. Can be repeated or comma-separated.")
@click.option("--index", "indices", multiple=True, type=int, help="Sample index in the first/base model dataset.")
@click.option("--num_samples", type=int, default=8, show_default=True)
@click.option("--prefer_class", "prefer_classes", multiple=True, type=int, help="Prefer samples containing class id, e.g. 4 or 5. Can repeat.")
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--precision", type=click.Choice(["32", "bf16", "fp16"]), default="bf16", show_default=True)
@click.option("--alpha", type=float, default=0.55, show_default=True)
@click.option("--plain_masks", is_flag=True, default=False, help="Show color masks instead of overlays.")
@click.option("--output_dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=Path("reports/prediction_comparison"), show_default=True)
def main(
    model_specs: tuple[str, ...],
    split: str,
    filenames: tuple[str, ...],
    indices: tuple[int, ...],
    num_samples: int,
    prefer_classes: tuple[int, ...],
    seed: int,
    device: str,
    precision: str,
    alpha: float,
    plain_masks: bool,
    output_dir: Path,
) -> None:
    """Create side-by-side prediction figures for many checkpoints.

    Example:
      uv run python -m scripts.compare_checkpoint_predictions \
        --model resnet50=src/config/resnet50_unet_512.py:logs/advanced_resnet50_unet_512_.../last.ckpt \
        --model resnet101=src/config/resnet101_unet_512.py:logs/advanced_resnet101_unet_512_.../last.ckpt \
        --split val \
        --prefer_class 4 \
        --prefer_class 5 \
        --num_samples 10
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_amp = torch_device.type == "cuda" and precision != "32"

    loaded_models = [load_model(spec, split, torch_device) for spec in model_specs]
    base = loaded_models[0]

    selected_filenames = choose_filenames(
        base.dataset,
        explicit_filenames=_split_csv_values(filenames),
        explicit_indices=list(indices),
        num_samples=num_samples,
        prefer_classes=list(prefer_classes),
        seed=seed,
    )

    save_legend(output_dir / "legend.png")
    (output_dir / "selected_filenames.txt").write_text(
        "\n".join(selected_filenames) + "\n",
        encoding="utf-8",
    )

    for sample_idx, filename in enumerate(selected_filenames):
        base_image, base_mask, _ = predict_for_filename(base, filename, torch_device, amp_dtype, use_amp)
        predictions: list[tuple[str, torch.Tensor]] = []

        for loaded in loaded_models:
            try:
                _, _, pred = predict_for_filename(loaded, filename, torch_device, amp_dtype, use_amp)
            except KeyError:
                click.echo(f"Skipping {filename} for {loaded.name}: filename missing in that dataset")
                continue
            predictions.append((loaded.name, pred))

        safe_filename = Path(filename).stem.replace("/", "_")
        save_sample_figure(
            filename=filename,
            base_image=base_image,
            base_mask=base_mask,
            predictions=predictions,
            output_path=output_dir / f"{sample_idx:02d}_{safe_filename}.png",
            alpha=alpha,
            plain_masks=plain_masks,
        )

    click.echo(f"Saved {len(selected_filenames)} comparison figure(s) to {output_dir}")
    click.echo(f"Selected filenames written to {output_dir / 'selected_filenames.txt'}")


if __name__ == "__main__":
    main()
