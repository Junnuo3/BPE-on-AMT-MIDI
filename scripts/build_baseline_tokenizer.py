"""
Build the baseline (no-BPE) tokenizer from the existing mapping.

Usage:
    python bpe/scripts/build_baseline_tokenizer.py
"""

import argparse
import json
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BPE_ROOT    = os.path.dirname(_SCRIPTS_DIR)
_DEFAULT_OUT = os.path.join(_BPE_ROOT, "tokenizers", "vocab_sweep",
                            "q_onset-10ms_duration-10ms_velocity-128bin")

sys.path.insert(0, os.path.join(_SCRIPTS_DIR, "..", "src"))

from bpe_utils import load_mapping
from bpe_train import SPECIAL_TOKENS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=_DEFAULT_OUT,
                   help="Outputs root (must contain mappings/). Default: bpe/outputs")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing tokenizer file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    map_path = os.path.join(args.out, "mappings", "amt_base_token_char_mapping.json")
    if not os.path.exists(map_path):
        print(f"ERROR: mapping not found at {map_path}")
        print("Run train_bpe.py (without --augment) at least once to build the mapping.")
        sys.exit(1)

    b2c, _ = load_mapping(map_path)
    initial_alphabet = list(b2c.values())
    B = len(initial_alphabet)
    S = len(SPECIAL_TOKENS)
    vocab_size = B + S
    print(f"Base alphabet B={B}, special tokens S={S}, vocab={vocab_size} (0 merges)")

    tok_dir  = os.path.join(args.out, f"vocab-{vocab_size}")
    out_path = os.path.join(tok_dir, f"amt_compound_bpe_vocab{vocab_size}.json")
    os.makedirs(tok_dir, exist_ok=True)

    if os.path.exists(out_path) and not args.force:
        print(f"Tokenizer already exists: {out_path}")
        print("Use --force to overwrite.")
        sys.exit(0)

    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Train on a tiny single-line corpus that contains every alphabet char once.
    # With vocab_size == B+S, the trainer learns zero merges.
    seed_corpus = [" ".join(initial_alphabet)]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=initial_alphabet,
        show_progress=False,
    )
    tokenizer.train_from_iterator(iter(seed_corpus), trainer=trainer)

    actual = tokenizer.get_vocab_size()
    assert actual == vocab_size, f"unexpected vocab size {actual} != {vocab_size}"

    tokenizer.save(out_path)
    with open(out_path, encoding="utf-8") as f:
        data = json.load(f)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)

    print(f"Saved → {out_path}  (vocab={actual}, merges=0)")


if __name__ == "__main__":
    main()
