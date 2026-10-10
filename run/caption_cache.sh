#!/bin/bash
# Build the stage-2 cache of anticipation variants re-captioned by Qwen2.5-Omni-7B.
# The reported runs used passes k = 1, 3, 5 (span) and 7, 9 (instrument).
# GPU required; several GPU-days for the full training split. Resumable and
# shardable with --shard-id/--num-shards. Both tokenizers share the cache: it
# stores base-token text, which each tokenizer encodes at training time.
source "$(dirname "$0")/common.sh"

python scripts/build_caption_cache.py --ant-dir "$TOKDIR" --out-dir outputs/stage2_cache \
  --k-values 1,3,5,7,9 "$@"
