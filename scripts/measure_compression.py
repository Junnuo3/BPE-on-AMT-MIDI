"""
Measure how BPE changes sequence length on a set of Lakh MIDI files.

For each tokenizer, every song is split into 100 s chunks (as in training) and
each chunk is encoded autoregressively (no control tokens). Reported per
tokenizer:
  tokens_per_note        MIDI tokens / notes, over all chunks
  window_sec_median      median musical time covered by one training window
                         (the first `--window - 1` MIDI tokens of a chunk; the
                         rest of a longer chunk is not used in training)
  frac_chunks_fit        fraction of chunks that fit entirely in one window
  frac_tokens_in_window  fraction of all MIDI tokens that fall inside those windows

Usage:
    python scripts/measure_compression.py --ids assets/evaluation_set_lakh_ids.txt \
        --out results/compression.csv
"""

import argparse
import csv
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from bpe_utils import load_mapping  # noqa: E402
from midi_to_amt import midi_to_amt  # noqa: E402

_TOK_DIR = os.path.join(_REPO_ROOT, "tokenizers", "anticipation", "velocity",
                        "onset-10ms_duration-10ms_velocity-32bin")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", default=os.path.join(_REPO_ROOT, "assets", "evaluation_set_lakh_ids.txt"),
                   help="File with one Lakh md5 per line.")
    p.add_argument("--dataset", default=os.path.join(_REPO_ROOT, "dataset", "lmd_full"),
                   help="LMD-full root (<dataset>/<md5[0]>/<md5>.mid).")
    p.add_argument("--tokenizers", nargs="+", default=[
        os.path.join(_TOK_DIR, "tokenizers", "amt_compound_bpe_vocab11293.json"),
        os.path.join(_TOK_DIR, "tokenizers", "amt_compound_bpe_vocab27677.json"),
    ])
    p.add_argument("--mapping", default=os.path.join(_TOK_DIR, "mappings", "amt_base_token_char_mapping.json"))
    p.add_argument("--window", type=int, default=2048,
                   help="Training window; one position is taken by the mode token.")
    p.add_argument("--out", default=None, help="Optional CSV output path.")
    return p.parse_args()


def main() -> None:
    from tokenizers import Tokenizer

    args = parse_args()
    b2c, _ = load_mapping(args.mapping)
    with open(args.ids) as f:
        md5s = [line.strip() for line in f if line.strip()]

    # Serialize once; each chunk is (note words, note onsets in seconds, chunk end in seconds).
    chunks = []
    for md5 in md5s:
        path = os.path.join(args.dataset, md5[0], f"{md5}.mid")
        for notes in midi_to_amt(path, onset_ms=10.0, dur_ms=10.0, vel_bins=32):
            try:
                words = ["".join(b2c[t] for t in note) for note in notes]
            except KeyError:
                continue
            onsets = [int(n[0][6:-1]) / 100.0 for n in notes]
            ends   = [o + int(n[1][10:-1]) / 100.0 for o, n in zip(onsets, notes)]
            chunks.append((words, onsets, max(ends)))

    budget = args.window - 1
    rows = []
    for tok_path in args.tokenizers:
        tok = Tokenizer.from_file(tok_path)
        n_notes = n_tokens = n_fit = n_in_window = 0
        window_sec = []
        for words, onsets, end in chunks:
            enc = tok.encode(" ".join(words))
            n_notes  += len(words)
            n_tokens += len(enc.ids)
            n_in_window += min(len(enc.ids), budget)
            if len(enc.ids) <= budget:
                n_fit += 1
                window_sec.append(end)
            else:
                # The window ends inside note `last`; it covers music up to that note's onset.
                last = enc.word_ids[budget - 1]
                window_sec.append(onsets[last])
        rows.append({
            "tokenizer":         os.path.basename(tok_path),
            "songs":             len(md5s),
            "chunks":            len(chunks),
            "notes":             n_notes,
            "tokens":            n_tokens,
            "tokens_per_note":   round(n_tokens / n_notes, 3),
            "window_sec_median": round(float(np.median(window_sec)), 1),
            "frac_chunks_fit":   round(n_fit / len(chunks), 3),
            "frac_tokens_in_window": round(n_in_window / n_tokens, 3),
        })

    for r in rows:
        print(r)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
