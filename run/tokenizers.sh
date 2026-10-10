#!/bin/bash
# Rebuild both tokenizers. The released files are already in tokenizers/;
# double_tokenizer.py skips existing outputs unless --force is given.
source "$(dirname "$0")/common.sh"

# 1. Scan LMD-full + GigaMIDI, build the base-token mapping and the serialized corpus.
python scripts/train_bpe.py --corpus-only --dataset dataset/lmd_full dataset/gigamidi/midi
# 2. BPE: 16,384 merges learned on 100,000 randomly sampled 100 s chunks (seed 42).
python scripts/train_bpe.py --merges 16384 --train-sample 100000
# 3. No-BPE baseline: the same alphabet with 0 merges.
python scripts/build_baseline_tokenizer.py
# 4. Add AMT control tokens and sentinels to both.
python scripts/double_tokenizer.py --all
