#!/usr/bin/env bash
set -euo pipefail

# Selected from the W&B CSV export around 2026-06-10:
# best   : advanced_wide_resnet50_2_unet_512_20260610_140127, val/mean_iou.max ~= 0.6975
# middle : weighted_augmented_deeplabv3_mobilenet_v3_large_segmenter_20260430_132604, val/mean_iou.max ~= 0.6419
# weak   : encdecnet_segmenter_20260430_011129, val/mean_iou.max ~= 0.1326
#
# The script finds a checkpoint in each logs/<run>/ directory and calls
# scripts.compare_checkpoint_predictions on the same validation filenames.

find_ckpt() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo "Missing log directory: $dir" >&2
    return 1
  fi

  # Prefer non-last checkpoints because ModelCheckpoint's top-k files are usually
  # better than last.ckpt. Fall back to last.ckpt when it is the only checkpoint.
  local ckpt
  ckpt="$(find "$dir" -maxdepth 1 -type f -name '*.ckpt' ! -name 'last.ckpt' | sort | tail -n 1 || true)"
  if [[ -n "$ckpt" ]]; then
    echo "$ckpt"
    return 0
  fi

  ckpt="$(find "$dir" -maxdepth 1 -type f -name 'last.ckpt' | sort | tail -n 1 || true)"
  if [[ -n "$ckpt" ]]; then
    echo "$ckpt"
    return 0
  fi

  echo "No checkpoint found in $dir" >&2
  return 1
}

BEST_RUN="advanced_wide_resnet50_2_unet_512_20260610_140127"
MID_RUN="weighted_augmented_deeplabv3_mobilenet_v3_large_segmenter_20260430_132604"
WEAK_RUN="encdecnet_segmenter_20260430_011129"

BEST_CKPT="$(find_ckpt "logs/${BEST_RUN}")"
MID_CKPT="$(find_ckpt "logs/${MID_RUN}")"
WEAK_CKPT="$(find_ckpt "logs/${WEAK_RUN}")"

echo "Best checkpoint:   ${BEST_CKPT}"
echo "Middle checkpoint: ${MID_CKPT}"
echo "Weak checkpoint:   ${WEAK_CKPT}"

uv run python -m scripts.compare_checkpoint_predictions \
  --model "best_wide_resnet=src/config/wide_resnet50_2_unet_512.py:${BEST_CKPT}" \
  --model "middle_deeplab_mobilenet=src/config/deeplabv3_mobilenet_v3_large_segmenter.py:${MID_CKPT}" \
  --model "weak_encdecnet=src/config/encdecnet_segmenter.py:${WEAK_CKPT}" \
  --split val \
  --prefer_class 4 \
  --prefer_class 5 \
  --num_samples 12 \
  --output_dir reports/prediction_comparison_best_mid_worst
