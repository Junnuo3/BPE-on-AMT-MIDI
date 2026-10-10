#!/bin/bash
# Stage 1: continued pretraining of SmolLM2-135M on MusicPile text + GigaMIDI MIDI.
# Usage: run/stage1.sh {no_bpe|bpe}. Settings of the reported runs (2 GPUs).
source "$(dirname "$0")/common.sh"
TOK=${1:?no_bpe or bpe}

torchrun --nproc_per_node=2 scripts/finetune_smollm2.py \
  --stage1 \
  --stage1-text-dir dataset/musicpile \
  --stage1-midi-dir dataset/gigamidi/midi \
  --stage1-text-ratio 0.55 \
  --out-dir "$RUNS/$TOK" \
  --tokenizer "$(tokenizer_for "$TOK")" \
  --batch-size 4 --window 2048 --target-tokens 500000 \
  --lr 3e-3 --weight-decay 0.1 \
  --lr-scheduler-type constant_with_warmup --warmup-steps 200 \
  --max-steps 150000 --early-stopping-patience 15 --early-stopping-threshold 0.001 \
  --save-steps 200 --eval-steps 200 --workers 3
