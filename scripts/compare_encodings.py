"""
Compare baseline vs anticipation-aware BPE encoding.

Usage:
    python bpe/scripts/compare_encodings.py [--limit-files 500]
"""

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)
_BPE_ROOT   = os.path.join(_REPO_ROOT, "bpe")

sys.path.insert(0, os.path.join(_BPE_ROOT, "src"))

from tqdm import tqdm
from tokenizers import Tokenizer

from bpe_utils import (
    find_midi_files,
    load_mapping_doubled,
    is_mapping_doubled,
    serialize_midi_anticipation,
)
from midi_to_amt import midi_to_amt

_DEFAULT_DATASET  = os.path.join(_REPO_ROOT, "dataset", "lmd_matched")
_DEFAULT_ANT_OUT  = os.path.join(_BPE_ROOT, "tokenizers", "anticipation")
_DEFAULT_BASE_OUT = os.path.join(_BPE_ROOT, "tokenizers", "vocab_sweep",
                                 "q_onset-10ms_duration-10ms_velocity-32bin")
_WINDOW           = 4096


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",       default=_DEFAULT_DATASET)
    p.add_argument("--baseline-out",  default=_DEFAULT_BASE_OUT)
    p.add_argument("--ant-out",       default=_DEFAULT_ANT_OUT)
    p.add_argument("--limit-files",   type=int, default=None)
    p.add_argument("--window",        type=int, default=_WINDOW)
    p.add_argument("--out-json",      default=os.path.join(_SCRIPT_DIR, "comparison_results.json"))
    return p.parse_args()


def _find_tokenizer(tok_dir: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(tok_dir, "*.json")))
    return hits[-1] if hits else None


def main() -> None:
    args = parse_args()
    window = args.window

    print(f"Scanning {args.dataset} …")
    midi_files = find_midi_files(args.dataset)
    if args.limit_files:
        midi_files = midi_files[: args.limit_files]
    print(f"  {len(midi_files):,} files")

    ant_tok_path = _find_tokenizer(os.path.join(args.ant_out, "tokenizers"))
    ant_map_path = os.path.join(args.ant_out, "mappings", "amt_base_token_char_mapping.json")
    if ant_tok_path is None or not os.path.exists(ant_map_path):
        print(f"ERROR: anticipation tokenizer or mapping missing in {args.ant_out}.")
        print("Run:  python bpe/scripts/double_tokenizer.py")
        sys.exit(1)
    if not is_mapping_doubled(ant_map_path):
        print(f"ERROR: mapping at {ant_map_path} is not doubled.")
        sys.exit(1)

    print(f"Anticipation tokenizer: {os.path.basename(ant_tok_path)}")
    ant_tokenizer = Tokenizer.from_file(ant_tok_path)
    b2c, c2b, ctrl_b2c, ctrl_c2b = load_mapping_doubled(ant_map_path)

    ant_vocab     = ant_tokenizer.get_vocab()
    SEPARATOR_ID  = ant_vocab.get("[SEPARATOR]", -1)
    AUTOREGRESS_ID = ant_vocab.get("[AUTOREGRESS]", -1)
    ANTICIPATE_ID  = ant_vocab.get("[ANTICIPATE]", -1)
    ctrl_id_set   = set()
    for c in ctrl_b2c.values():
        tid = ant_vocab.get(c)
        if tid is not None:
            ctrl_id_set.add(tid)

    # Load anticipation train_config for quant params
    ant_cfg_path = os.path.join(args.ant_out, "train_config.json")
    if os.path.exists(ant_cfg_path):
        with open(ant_cfg_path) as f:
            ant_cfg = json.load(f)
        onset_ms = ant_cfg.get("onset_ms", 10.0)
        dur_ms   = ant_cfg.get("dur_ms",   10.0)
        vel_bins = ant_cfg.get("vel_bins",  128)
    else:
        onset_ms, dur_ms, vel_bins = 10.0, 10.0, 128

    print(f"  quant: onset_ms={onset_ms}, dur_ms={dur_ms}, vel_bins={vel_bins}")

    baseline_total_tokens   = 0   # 5 × number of notes
    treatment_total_tokens  = 0   # BPE token count
    treatment_ctrl_tokens   = 0   # BPE ctrl token count
    n_files_valid           = 0

    # For "notes per window" we simulate packing.
    baseline_packed_tokens  = 0   # tokens that fit in complete windows
    treatment_packed_tokens = 0
    baseline_notes_packed   = 0
    treatment_words_packed  = 0   # AMT base-token groups (5-char words)

    # Window-level [ANTICIPATE] fraction
    treatment_n_windows     = 0
    treatment_n_anticipate  = 0

    rng = np.random.default_rng(0)

    for path in tqdm(midi_files, desc="comparing", unit="file"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks = midi_to_amt(path, onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins)
        if not chunks:
            continue

        # baseline: 5 tokens per note, summed across all chunks
        n_notes = sum(len(chunk) for chunk in chunks)
        baseline_tok = n_notes * 5   # each note = 5 tokens

        # treatment: anticipation BPE, summed across all chunks
        file_rng = np.random.default_rng(abs(hash(path)) % (2 ** 31))
        texts = serialize_midi_anticipation(path, b2c, ctrl_b2c, k=1, rng=file_rng,
                                            onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins)
        if not texts:
            continue

        treatment_tok = 0
        n_ctrl_toks   = 0
        enc_ids_all: list[int] = []
        for text in texts:
            enc = ant_tokenizer.encode(text)
            treatment_tok += len(enc.ids)
            n_ctrl_toks   += sum(1 for i in enc.ids if i in ctrl_id_set)
            enc_ids_all.extend(enc.ids)

        baseline_total_tokens  += baseline_tok
        treatment_total_tokens += treatment_tok
        treatment_ctrl_tokens  += n_ctrl_toks
        n_files_valid += 1

        # Simulate 4096-token packing
        baseline_packed_tokens  += (baseline_tok // window) * window
        baseline_notes_packed   += (baseline_tok // window) * window // 5

        treatment_packed_tokens += (treatment_tok // window) * window
        # Count whitespace-delimited words that are 5-char event/ctrl groups (across all chunks)
        words_5 = sum(1 for w in " ".join(texts).split() if len(w) == 5)
        treatment_words_packed  += (treatment_tok // window) * window * words_5 // max(1, treatment_tok)

        # [ANTICIPATE] window fraction — evaluate per-chunk to avoid cross-chunk artifacts
        ids_arr = np.array(enc_ids_all, dtype=np.int32)
        n_win   = len(ids_arr) // window
        if n_win > 0:
            windows = ids_arr[:n_win * window].reshape(n_win, window)
            has_ctrl = np.isin(windows, np.array(sorted(ctrl_id_set), dtype=np.int32)).any(axis=1)
            treatment_n_windows    += n_win
            treatment_n_anticipate += int(has_ctrl.sum())

    if n_files_valid == 0:
        print("ERROR: no valid files processed.")
        sys.exit(1)

    compression = baseline_total_tokens / max(1, treatment_total_tokens)

    baseline_notes_per_window = (
        baseline_notes_packed / max(1, baseline_packed_tokens // window)
        if baseline_packed_tokens > 0 else 0
    )
    treatment_words_per_window = (
        treatment_words_packed / max(1, treatment_n_windows)
        if treatment_n_windows > 0 else 0
    )
    anticipate_fraction = (
        treatment_n_anticipate / max(1, treatment_n_windows)
        if treatment_n_windows > 0 else 0
    )

    results = {
        "n_files":                    n_files_valid,
        "window_size":                window,
        "baseline": {
            "description":            "raw 5-token-per-note, no BPE",
            "total_tokens":           baseline_total_tokens,
            "avg_tokens_per_file":    baseline_total_tokens / n_files_valid,
            "notes_per_window_approx": round(baseline_notes_per_window, 1),
        },
        "treatment": {
            "description":            "anticipation-aware BPE",
            "tokenizer":              os.path.basename(ant_tok_path),
            "total_tokens":           treatment_total_tokens,
            "avg_tokens_per_file":    treatment_total_tokens / n_files_valid,
            "ctrl_token_fraction":    treatment_ctrl_tokens / max(1, treatment_total_tokens),
            "notes_per_window_approx": round(treatment_words_per_window, 1),
            "anticipate_window_fraction": round(anticipate_fraction, 4),
            "n_windows":              treatment_n_windows,
            "n_anticipate_windows":   treatment_n_anticipate,
        },
        "compression_factor":         round(compression, 4),
        "reduction_ratio":            round(1.0 / compression, 4) if compression else 0,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Stage 6 Comparison  ({n_files_valid:,} files, window={window})")
    print(f"{'='*60}")
    print(f"\nBaseline (raw 5-token/note, no BPE):")
    print(f"  Total tokens:           {baseline_total_tokens:>15,}")
    print(f"  Avg tokens/file:        {baseline_total_tokens / n_files_valid:>15.1f}")
    print(f"  Notes per {window}-tok window: {baseline_notes_per_window:>15.1f}")
    print(f"\nTreatment (anticipation-aware BPE):")
    print(f"  Total tokens:           {treatment_total_tokens:>15,}")
    print(f"  Avg tokens/file:        {treatment_total_tokens / n_files_valid:>15.1f}")
    print(f"  Control token fraction: {treatment_ctrl_tokens / max(1, treatment_total_tokens):>15.4f}")
    print(f"  [ANTICIPATE] windows:   {anticipate_fraction:>15.4f} ({treatment_n_anticipate:,}/{treatment_n_windows:,})")
    print(f"\nCompression factor:     {compression:>17.4f}×")
    print(f"Reduction ratio:        {1.0/compression if compression else 0:>17.4f}")
    print(f"\nResults → {args.out_json}")


if __name__ == "__main__":
    main()
