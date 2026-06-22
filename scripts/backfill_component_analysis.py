"""
Backfill component_analysis_size{N}.csv for already-analyzed vocab dirs.

Usage:
    python bpe/scripts/backfill_component_analysis.py
"""

import sys
import os
import argparse
import csv
import json
import re
from collections import Counter

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BPE_ROOT    = os.path.dirname(_SCRIPTS_DIR)
_DEFAULT_OUT = os.path.join(_BPE_ROOT, "outputs")

sys.path.insert(0, os.path.join(_SCRIPTS_DIR, "..", "src"))

from tokenizers import Tokenizer
from bpe_utils import load_mapping

SPECIAL_TOKENS = {"[UNK]", "[PAD]", "[BOS]", "[EOS]"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=_DEFAULT_OUT)
    p.add_argument("--vocab-size", type=int, default=None,
                   help="Process a single vocab size instead of all.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing component_analysis_size*.csv files.")
    return p.parse_args()


def get_category(amt_token: str) -> str:
    m = re.match(r"<(\w+)-", amt_token)
    return m.group(1) if m else "unknown"


def load_freq_from_csv(analysis_dir: str) -> dict[str, int]:
    """Return {space-joined AMT tokens: corpus_count} from token_frequency.csv."""
    path = os.path.join(analysis_dir, "token_frequency.csv")
    freq: dict[str, int] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            freq[row["original_tokens"]] = int(row["corpus_count"])
    return freq


def compute_component_data(
    id_to_amt: dict[int, list[str]], freq: Counter
) -> dict[int, tuple[Counter, Counter]]:
    by_size: dict[int, list[int]] = {}
    for tok_id, amt_tokens in id_to_amt.items():
        by_size.setdefault(len(amt_tokens), []).append(tok_id)

    result: dict[int, tuple[Counter, Counter]] = {}
    for size in sorted(by_size):
        combo_vocab: Counter = Counter()
        combo_corpus: Counter = Counter()
        for tok_id in by_size[size]:
            cats = tuple(sorted(get_category(t) for t in id_to_amt[tok_id]))
            combo_vocab[cats] += 1
            combo_corpus[cats] += freq.get(tok_id, 0)
        result[size] = (combo_vocab, combo_corpus)
    return result


def run(analysis_dir: str, tok_path: str, mapping_path: str, force: bool) -> None:
    existing = [f for f in os.listdir(analysis_dir) if f.startswith("component_analysis_size")]
    summary_has_it = False
    summary_path = os.path.join(analysis_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary_has_it = "component_analysis" in json.load(f)

    if existing and summary_has_it and not force:
        print(f"  already done — skip (use --force to overwrite)")
        return

    tokenizer = Tokenizer.from_file(tok_path)
    _, char_to_base = load_mapping(mapping_path)

    id_to_amt: dict[int, list[str]] = {}
    for tok_str, tok_id in tokenizer.get_vocab().items():
        if tok_str not in SPECIAL_TOKENS and len(tok_str) > 1:
            id_to_amt[tok_id] = [char_to_base.get(c, f"<UNK:{ord(c):#x}>") for c in tok_str]

    repr_to_count = load_freq_from_csv(analysis_dir)
    freq: Counter = Counter()
    for tok_id, amt_tokens in id_to_amt.items():
        key = " ".join(amt_tokens)
        if key in repr_to_count:
            freq[tok_id] = repr_to_count[key]

    component_data = compute_component_data(id_to_amt, freq)

    # Write CSVs
    for size, (combo_vocab, combo_corpus) in sorted(component_data.items()):
        ordered = sorted(combo_vocab, key=lambda k: combo_corpus[k], reverse=True)
        out_path = os.path.join(analysis_dir, f"component_analysis_size{size}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["categories", "vocab_count", "corpus_count"])
            writer.writeheader()
            for cats in ordered:
                writer.writerow({
                    "categories": "+".join(cats),
                    "vocab_count": combo_vocab[cats],
                    "corpus_count": combo_corpus[cats],
                })
        print(f"  component_analysis_size{size}.csv")

    # Patch summary.json
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        summary["component_analysis"] = {
            str(size): [
                {
                    "categories": "+".join(cats),
                    "vocab_count": combo_vocab[cats],
                    "vocab_proportion": round(
                        combo_vocab[cats] / sum(combo_vocab.values()), 4
                    ) if combo_vocab else 0.0,
                    "corpus_count": combo_corpus[cats],
                }
                for cats in sorted(combo_vocab, key=lambda k: combo_corpus[k], reverse=True)[:10]
            ]
            for size, (combo_vocab, combo_corpus) in sorted(component_data.items())
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  summary.json (patched)")


def main() -> None:
    args = parse_args()

    mapping_path = os.path.join(args.out, "mappings", "amt_base_token_char_mapping.json")
    if not os.path.exists(mapping_path):
        print(f"ERROR: mapping not found at {mapping_path}")
        sys.exit(1)

    analysis_root = os.path.join(args.out, "analysis")

    if args.vocab_size:
        dirs = [f"vocab{args.vocab_size}"]
    else:
        dirs = sorted(d for d in os.listdir(analysis_root)
                      if os.path.isdir(os.path.join(analysis_root, d)))

    for vocab_dir in dirs:
        analysis_dir = os.path.join(analysis_root, vocab_dir)
        freq_csv = os.path.join(analysis_dir, "token_frequency.csv")
        if not os.path.exists(freq_csv):
            print(f"{vocab_dir}: no token_frequency.csv — skip")
            continue

        label = re.search(r"\d+", vocab_dir)
        if not label:
            print(f"{vocab_dir}: can't parse vocab size — skip")
            continue
        tok_path = os.path.join(args.out, "tokenizers", f"amt_compound_bpe_vocab{label.group()}.json")
        if not os.path.exists(tok_path):
            print(f"{vocab_dir}: tokenizer not found at {tok_path} — skip")
            continue

        print(f"{vocab_dir}")
        run(analysis_dir, tok_path, mapping_path, args.force)


if __name__ == "__main__":
    main()
