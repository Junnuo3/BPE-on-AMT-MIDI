# Shared settings for the run/ scripts. Run every script from the repository root.
set -euo pipefail
TOKDIR=tokenizers/anticipation/velocity/onset-10ms_duration-10ms_velocity-32bin
RUNS=${RUNS:-outputs/runs}

tokenizer_for() {   # no_bpe -> 0 merges, bpe -> 16,384 merges
  case "$1" in
    no_bpe) echo "$TOKDIR/tokenizers/amt_compound_bpe_vocab11293.json" ;;
    bpe)    echo "$TOKDIR/tokenizers/amt_compound_bpe_vocab27677.json" ;;
    *) echo "usage: $0 {no_bpe|bpe}" >&2; exit 1 ;;
  esac
}
