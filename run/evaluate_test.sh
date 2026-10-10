#!/bin/bash
# Generate and evaluate on the 896-song test split with the configuration selected
# on validation (override with T, P, K). Seed 0, batch size 256.
# Usage: run/evaluate_test.sh {no_bpe|bpe}
#   CKPT=checkpoints/bpe/stage2 run/evaluate_test.sh bpe   # released checkpoint
source "$(dirname "$0")/common.sh"
TOK=${1:?no_bpe or bpe}
CKPT=${CKPT:-$RUNS/$TOK/stage2/velocity/onset-10ms_duration-10ms_velocity-32bin/checkpoints/final}
case "$TOK" in
  bpe)    T=${T:-1.0} P=${P:-1.0} K=${K:-40} ;;
  no_bpe) T=${T:-1.0} P=${P:-1.0} K=${K:-0} ;;
esac
tag="t${T}_p${P}_k${K}_r1.0"

python scripts/generate_eval_set.py --checkpoint "$CKPT" --split test \
  --temperature "$T" --top-p "$P" --top-k "$K" --repetition-penalty 1.0 \
  --seed 0 --batch-size 256 --out-dir "generations/${TOK}_test/$tag"
python scripts/run_evaluation.py --generations-dir "generations/${TOK}_test/$tag" \
  --model-name "${TOK}_test_$tag" --split test \
  --metrics fad,clap,chroma,groove,groove_bar
