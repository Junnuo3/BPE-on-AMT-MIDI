#!/bin/bash
# Decoding-parameter search on the validation split (10,693 captions), one sample
# per caption, seed 0, batch size 256. Step 1: T x top-p; step 2: the two step-1
# temperatures used in the reported search x top-p x top-k.
# Usage: run/decoding_sweep.sh {no_bpe|bpe}
#   CKPT=checkpoints/bpe/stage2 run/decoding_sweep.sh bpe   # released checkpoint
# Leave MIDI_SOUNDFONT unset to render with pretty_midi's TimGM6mb.sf2, as reported.
source "$(dirname "$0")/common.sh"
TOK=${1:?no_bpe or bpe}
CKPT=${CKPT:-$RUNS/$TOK/stage2/velocity/onset-10ms_duration-10ms_velocity-32bin/checkpoints/final}
case "$TOK" in bpe) STEP2_T="0.9 1.0" ;; no_bpe) STEP2_T="1.0 1.1" ;; esac

configs=()
for t in 0.9 1.0 1.1 1.2 1.3 1.4; do for p in 0.95 0.98 1.0; do configs+=("$t:$p:0"); done; done
for t in $STEP2_T; do for p in 0.95 0.98 1.0; do for k in 40 100; do configs+=("$t:$p:$k"); done; done; done

for c in "${configs[@]}"; do
  IFS=: read -r t p k <<< "$c"
  tag="t${t}_p${p}_k${k}_r1.0"
  python scripts/generate_eval_set.py --checkpoint "$CKPT" --split valid \
    --temperature "$t" --top-p "$p" --top-k "$k" --repetition-penalty 1.0 \
    --seed 0 --batch-size 256 --out-dir "generations/${TOK}_valid/$tag"
  python scripts/run_evaluation.py --generations-dir "generations/${TOK}_valid/$tag" \
    --model-name "${TOK}_valid_$tag" --split valid \
    --metrics fad,clap,chroma,groove,groove_bar
done

python scripts/aggregate_hp_search.py --prefix "${TOK}_valid_" --rank \
  --csv-out "results/decoding_sweep_valid_${TOK}_rerun.csv"
