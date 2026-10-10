#!/bin/bash
# Stage 2: caption -> MIDI training on MidiCaps / LMD-full, starting from the best
# stage-1 checkpoint. Requires outputs/stage2_cache (run/caption_cache.sh).
# Usage: run/stage2.sh {no_bpe|bpe}. Settings of the reported runs (2 GPUs).
source "$(dirname "$0")/common.sh"
TOK=${1:?no_bpe or bpe}
STAGE1=${STAGE1:-$RUNS/$TOK/stage1/velocity/onset-10ms_duration-10ms_velocity-32bin/checkpoints/final}

torchrun --nproc_per_node=2 scripts/finetune_smollm2.py \
  --model "$STAGE1" \
  --captions dataset/lmd_full_captions.jsonl --caption-text \
  --caption-cache-dir outputs/stage2_cache --caption-expansion-factor 3 \
  --max-text-tokens 256 \
  --out-dir "$RUNS/$TOK" \
  --tokenizer "$(tokenizer_for "$TOK")" \
  --batch-size 4 --window 2048 --target-tokens 500000 \
  --lr 1e-3 --weight-decay 0.1 \
  --lr-scheduler-type constant_with_warmup --warmup-steps 200 \
  --max-steps 150000 --early-stopping-patience 15 --early-stopping-threshold 0.001 \
  --save-steps 200 --eval-steps 200 --workers 3
