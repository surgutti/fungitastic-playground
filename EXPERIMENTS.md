# Recommended segmentation experiments

This file assumes a single RTX 5070 Ti-class GPU and 64 GB system RAM. Start with the 512 px experiments, then move to 640 px only after the smoke tests and overfit checks pass.

## 0. Setup

```bash
uv sync
mkdir -p data
```

## 1. Baseline dataset already supported by the original README

```bash
uv run scripts/download.py --subset m --metadata --masks --images --size 300 --save_path ./data
uv run scripts/prepare_dataset.py
```

Run the old baseline once to keep a reference:

```bash
uv run python -m scripts.train_model_gpu src/config/encdecnet_segmenter.py --precision bf16-mixed --no_wandb
```

## 2. Strong 512 px dataset

```bash
uv run scripts/download.py --subset m --metadata --masks --images --size 500 --save_path ./data
uv run python -m scripts.prepare_dataset_configurable --subset m --image_size 500 --target_size 512 --resize_mode letterbox
uv run python -m scripts.dataset_stats --segmentation_root data/FungiTastic/SegmentationDataset-Mini-500p-512px
```

## 3. Smoke tests

Run every new config for two train/validation batches before long training:

```bash
uv run python -m scripts.train_model_gpu src/config/resnet50_unet_512.py --fast_dev_run 2 --no_wandb
uv run python -m scripts.train_model_gpu src/config/resnet101_unet_512.py --fast_dev_run 2 --no_wandb
uv run python -m scripts.train_model_gpu src/config/wide_resnet50_2_unet_512.py --fast_dev_run 2 --no_wandb
```

## 4. Overfit sanity check

This should nearly memorize a few batches. If it does not, there is likely a data, loss, or mask alignment problem.

```bash
uv run python -m scripts.train_model_gpu src/config/resnet50_unet_512.py --overfit_batches 8 --max_epochs 30 --no_wandb
```

## 5. Main experiments

Recommended order:

```bash
uv run python -m scripts.train_model_gpu src/config/resnet50_unet_512.py --precision bf16-mixed
uv run python -m scripts.train_model_gpu src/config/resnet101_unet_512.py --precision bf16-mixed --accumulate_grad_batches 2
uv run python -m scripts.train_model_gpu src/config/wide_resnet50_2_unet_512.py --precision bf16-mixed --accumulate_grad_batches 3
```

Use `--no_wandb` for local-only runs.

## 6. High-resolution experiment

Prepare the 640 px dataset from 720 px images:

```bash
uv run scripts/download.py --subset m --metadata --masks --images --size 720 --save_path ./data
uv run python -m scripts.prepare_dataset_configurable --subset m --image_size 720 --target_size 640 --resize_mode letterbox
uv run python -m scripts.train_model_gpu src/config/resnet101_unet_640.py --precision bf16-mixed --accumulate_grad_batches 4
```

If you hit CUDA OOM, first lower the config batch size from 2 to 1, then retry with the same `--accumulate_grad_batches 4`.

## 7. Test-time augmentation

After a run finishes, evaluate the best checkpoint. Replace the checkpoint path with the actual path printed by Lightning.

```bash
uv run python -m scripts.evaluate_tta src/config/resnet50_unet_512.py logs/<run-name>/last.ckpt --split val --scales 1.0 --hflip
uv run python -m scripts.evaluate_tta src/config/resnet50_unet_512.py logs/<run-name>/last.ckpt --split val --scales 0.75,1.0,1.25 --hflip
```

Use multi-scale TTA only for final reporting, not while iterating.

## 8. What to compare

Track these as separate runs:

1. Original EncDecNet / 300 px / old trainer.
2. Big EncDecNet / 300 px / advanced loss.
3. ResNet50 U-Net / 512 px.
4. ResNet101 U-Net / 512 px.
5. Wide ResNet50-2 U-Net / 512 px.
6. Best 512 px model with TTA.
7. ResNet101 U-Net / 640 px.
8. Best 640 px model with TTA.

Primary metric: `val/mean_iou` without background. Secondary metrics: `val/iou_class_1..5`, especially pores and ring.
