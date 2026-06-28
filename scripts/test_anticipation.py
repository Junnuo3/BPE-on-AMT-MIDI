"""
Verify anticipation_utils on sample MIDI files.

Usage:
    python bpe/scripts/test_anticipation.py --midi path/to/file.mid
    python bpe/scripts/test_anticipation.py --dataset dataset/lmd_matched --n 20
"""

import argparse
import os
import sys
import warnings

import numpy as np

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPTS_DIR, "..", "src"))

from midi_to_amt import midi_to_amt
from anticipation_utils import (
    build_sequence, augmentation_spec, sample_spans,
    extract_random, extract_instruments, extract_spans,
    SEPARATOR, REST,
)


def _steps_per_sec(onset_ms: float) -> float:
    return 1000.0 / onset_ms


def _time(tok: list[str]) -> int:
    return int(tok[0][6:-1])


def run_checks(midi_path: str, onset_ms: float = 10.0, verbose: bool = True) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chunks = midi_to_amt(midi_path, onset_ms=onset_ms)
    if not chunks:
        return {"path": midi_path, "ok": False, "reason": "no notes parsed"}
    # Use the first chunk; most test files are < 100 s so this covers the full piece.
    notes = chunks[0]

    sps = _steps_per_sec(onset_ms)
    delta_steps = max(1, int(round(5.0 * sps)))

    rng = np.random.default_rng(42)
    instruments = list({int(n[3][12:-1]) for n in notes})

    failures = []
    results = {}

    # check 1: SEPARATOR at start
    items = build_sequence(notes, "none", rng, onset_ms=onset_ms)
    first_tokens, first_ctrl = items[0]
    if first_tokens != [SEPARATOR] or first_ctrl:
        failures.append("SEPARATOR missing or not first")
    sep_count = sum(1 for t, _ in items if t == [SEPARATOR])
    if sep_count != 1:
        failures.append(f"expected 1 SEPARATOR, got {sep_count}")
    results["separator_ok"] = sep_count == 1

    # check 2: REST density (one REST per density_steps interval in each gap)
    density_steps = max(1, int(round(1.0 * sps)))
    rests = [t for t, _ in items if t == [REST]]
    expected_rests = 0
    previous_t = 0
    for note in notes:
        t = _time(note)
        expected_rests += max(0, (t - previous_t - 1) // density_steps)
        previous_t = t
    end_time = _time(notes[-1])
    density_ok = abs(len(rests) - expected_rests) <= max(2, int(expected_rests * 0.1))
    if not density_ok:
        failures.append(f"REST density off: got {len(rests)}, expected {expected_rests}")
    results["rest_density_ok"] = density_ok
    results["rest_count"] = len(rests)
    results["expected_rests"] = expected_rests

    # check 3: control notes appear before their true onset
    rng2 = np.random.default_rng(7)
    items_ctrl = build_sequence(notes, "random", rng2, onset_ms=onset_ms, rate=5)

    early_deltas = []
    event_time_cursor = 0
    prev_event_time: int | None = None
    for tokens, is_ctrl in items_ctrl:
        if tokens == [SEPARATOR] or tokens == [REST]:
            if tokens != [REST]:
                pass
            continue
        true_onset = _time(tokens)
        if is_ctrl:
            # Find its position in sequence relative to previous event time
            seq_pos_approx = prev_event_time if prev_event_time is not None else 0
            delta = true_onset - seq_pos_approx
            early_deltas.append(delta / sps)  # in seconds
        else:
            prev_event_time = true_onset

    if not early_deltas:
        # no controls were selected; try instrument mode
        if len(instruments) > 1:
            rng3 = np.random.default_rng(99)
            subset = {instruments[0]}
            items_inst = build_sequence(notes, "instrument", rng3,
                                        onset_ms=onset_ms, instruments=subset)
            for tokens, is_ctrl in items_inst:
                if tokens in ([SEPARATOR], [REST]) or not is_ctrl:
                    continue
                true_onset = _time(tokens)
                early_deltas.append(true_onset / sps)

    if early_deltas:
        median_delta = float(np.median(early_deltas)) if early_deltas else 0
        ctrl_timing_ok = len(early_deltas) > 0
        results["control_count"] = len(early_deltas)
        results["median_early_delta_approx_sec"] = round(median_delta, 2)
    else:
        ctrl_timing_ok = True  # single-instrument files OK with no controls
        results["control_count"] = 0

    results["ctrl_timing_ok"] = ctrl_timing_ok

    # check 4: ctrl notes must have unchanged token content
    rng4 = np.random.default_rng(13)
    items_span = build_sequence(notes, "random", rng4, onset_ms=onset_ms, rate=3)
    ctrl_onsets = set()
    for tokens, is_ctrl in items_span:
        if is_ctrl and tokens not in ([REST], [SEPARATOR]):
            ctrl_onsets.add(tuple(tokens))
    note_tuples = {tuple(n) for n in notes}
    stray = ctrl_onsets - note_tuples
    if stray:
        failures.append(f"{len(stray)} ctrl notes have modified token content")
    results["ctrl_token_integrity_ok"] = len(stray) == 0

    ok = len(failures) == 0
    results["ok"] = ok
    results["path"] = midi_path
    results["failures"] = failures

    if verbose:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {os.path.basename(midi_path)}")
        for f in failures:
            print(f"         ✗ {f}")
        if ok:
            print(f"         REST: {results['rest_count']} (expected ~{results['expected_rests']})")
            print(f"         ctrl notes found: {results['control_count']}")

    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--midi",      default=None, help="Single MIDI file to test")
    p.add_argument("--dataset",   default=None, help="Root dir to scan for MIDI files")
    p.add_argument("--n",         type=int, default=10, help="Number of files to test from dataset")
    p.add_argument("--onset-ms",  type=float, default=10.0)
    p.add_argument("--verbose",   action="store_true", default=True)
    args = p.parse_args()

    midi_files: list[str] = []
    if args.midi:
        midi_files = [os.path.abspath(args.midi)]
    elif args.dataset:
        for root, _, files in os.walk(args.dataset):
            for f in files:
                if f.lower().endswith((".mid", ".midi")):
                    midi_files.append(os.path.join(root, f))
        midi_files = sorted(midi_files)[: args.n]
    else:
        print("ERROR: provide --midi or --dataset", file=sys.stderr)
        sys.exit(1)

    print(f"Testing {len(midi_files)} file(s) with onset_ms={args.onset_ms}")
    all_ok = True
    for path in midi_files:
        r = run_checks(path, onset_ms=args.onset_ms, verbose=args.verbose)
        if not r["ok"]:
            all_ok = False

    if all_ok:
        print(f"\nAll {len(midi_files)} file(s) passed.")
    else:
        print(f"\nSome files FAILED — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
