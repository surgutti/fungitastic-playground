#!/usr/bin/env bash
set -euo pipefail

# Three-column comparison for report figures:
#
# from scratch : custom EncDecNet trained from scratch on the old 8-class setup
# baseline     : ready-made torchvision DeepLabV3 MobileNetV3 segmentation model
# hybrid       : pretrained Wide-ResNet50-2 encoder + custom U-Net/ASPP decoder/losses
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

FROM_SCRATCH_RUN="encdecnet_segmenter_20260430_011129"
BASELINE_RUN="weighted_augmented_deeplabv3_mobilenet_v3_large_segmenter_20260430_132604"
HYBRID_RUN="advanced_wide_resnet50_2_unet_512_20260610_140127"

FROM_SCRATCH_CKPT="$(find_ckpt "logs/${FROM_SCRATCH_RUN}")"
BASELINE_CKPT="$(find_ckpt "logs/${BASELINE_RUN}")"
HYBRID_CKPT="$(find_ckpt "logs/${HYBRID_RUN}")"

echo "from scratch checkpoint: ${FROM_SCRATCH_CKPT}"
echo "baseline checkpoint:     ${BASELINE_CKPT}"
echo "hybrid checkpoint:       ${HYBRID_CKPT}"

uv run python -m scripts.compare_checkpoint_predictions \
  --model "from scratch=src/config/legacy_encdecnet_8class_segmenter.py:${FROM_SCRATCH_CKPT}" \
  --model "baseline=src/config/deeplabv3_mobilenet_v3_large_segmenter.py:${BASELINE_CKPT}" \
  --model "hybrid=src/config/wide_resnet50_2_unet_512.py:${HYBRID_CKPT}" \
  --split val \
  --prefer_class 4 \
  --prefer_class 5 \
  --num_samples 12 \
  --output_dir reports/prediction_comparison_from_scratch_baseline_hybrid
