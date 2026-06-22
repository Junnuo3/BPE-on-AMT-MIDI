"""
Sweep onset_ms, dur_ms, and vel_bins to see how quantization affects BPE compression.

Usage:
    python bpe/scripts/run_quant_experiments.py [--factor onset|duration|velocity] [--dry-run]
"""

import argparse
import csv
import json
import os
import subprocess
import sys

_SCRIPTS_DIR       = os.path.dirname(os.path.abspath(__file__))
_BPE_ROOT          = os.path.dirname(_SCRIPTS_DIR)
_REPO_ROOT         = os.path.dirname(_BPE_ROOT)
_DEFAULT_DATASET   = os.path.join(_REPO_ROOT, "dataset", "lmd_matched")
_DEFAULT_QUANT_OUT        = os.path.join(_BPE_ROOT, "tokenizers", "quantization_sweep", "merges-8192")
_DEFAULT_QUANT_OUT_NO_ONS = os.path.join(_BPE_ROOT, "tokenizers", "merge_constraints", "no_onset_merge", "merges-8192")

sys.path.insert(0, os.path.join(_SCRIPTS_DIR, "..", "src"))
from bpe_utils import derive_corpus_from_fine, derive_no_onset_corpus, corpus_is_ready

MERGES = 8192  # BPE merges above B+S for every experiment

# One factor varied at a time, others fixed at defaults (10ms onset, 10ms dur, 128 bins).
# finest_first=True: smaller value = finer (onset/duration). False: larger = finer (velocity bins).
EXPERIMENTS: dict[str, dict] = {
    "onset": {
        "arg":          "--onset-ms",
        "values":       [1, 2, 5, 10, 20],   # ms
        "fixed":        {"--dur-ms": 10, "--vel-bins": 128},
        "finest_first": True,
    },
    "duration": {
        "arg":          "--dur-ms",
        "values":       [1, 2, 5, 10, 20],   # ms
        "fixed":        {"--onset-ms": 10, "--vel-bins": 128},
        "finest_first": True,
    },
    "velocity": {
        "arg":          "--vel-bins",
        "values":       [8, 16, 32, 64, 128],
        "fixed":        {"--onset-ms": 10, "--dur-ms": 10},
        "finest_first": False,  # more bins = finer
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=_DEFAULT_DATASET,
                   help="Path to MIDI dataset")
    p.add_argument("--out", default=None,
                   help="Root output dir for quantization experiments "
                        "(default: quantization/ or quantization_no_onset/ depending on mode)")
    p.add_argument("--factor", choices=list(EXPERIMENTS.keys()), default=None,
                   help="Run only this factor (default: all)")
    p.add_argument("--onset-standalone", action="store_true",
                   help="Run experiments with onset kept as standalone tokens")
    p.add_argument("--baseline-quant-out", default=_DEFAULT_QUANT_OUT,
                   help="Baseline quantization output dir to reuse corpora from "
                        "(only used with --onset-standalone)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing them")
    args = p.parse_args()
    if args.out is None:
        args.out = _DEFAULT_QUANT_OUT_NO_ONS if args.onset_standalone else _DEFAULT_QUANT_OUT
    return args


def config_out_dir(quant_out: str, factor: str, value, spec: dict) -> str:
    """Output dir for a (factor, value) config, with all quant params encoded in the path."""
    fixed = spec["fixed"]
    all_q = {k.lstrip("-").replace("-", "_"): float(v) for k, v in fixed.items()}
    key = {"onset": "onset_ms", "duration": "dur_ms", "velocity": "vel_bins"}[factor]
    all_q[key] = float(value)
    onset = all_q.get("onset_ms", 10.0)
    dur   = all_q.get("dur_ms",   10.0)
    vel   = int(all_q.get("vel_bins", 128))
    folder = f"onset-{onset:g}ms_duration-{dur:g}ms_velocity-{vel}bin"
    return os.path.join(quant_out, factor, folder)


def is_done(config_out: str) -> bool:
    """True if eval results already exist for this config."""
    path = os.path.join(config_out, "results", "length_reduction_results.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            return bool(json.load(f))
    except Exception:
        return False


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print("  $", " ".join(str(x) for x in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def load_result(config_out: str) -> dict | None:
    path = os.path.join(config_out, "results", "length_reduction_results.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data[0] if data else None
    except Exception:
        return None


def run_factor(factor: str, spec: dict, args: argparse.Namespace) -> list[dict]:
    sweep_arg    = spec["arg"]
    fixed        = spec["fixed"]
    values       = spec["values"]
    finest_first = spec["finest_first"]

    # process finest first — it needs a full MIDI scan; coarser configs are derived from it
    fine_val = min(values) if finest_first else max(values)
    ordered  = [fine_val] + [v for v in values if v != fine_val]

    train_script = os.path.join(_SCRIPTS_DIR, "train_bpe.py")
    eval_script  = os.path.join(_SCRIPTS_DIR, "evaluate_bpe.py")

    rows: list[dict] = []

    for value in ordered:
        config_out = config_out_dir(args.out, factor, value, spec)
        os.makedirs(config_out, exist_ok=True)

        quant_args: list[str] = [sweep_arg, str(value)]
        for k, v in fixed.items():
            quant_args += [k, str(v)]

        label = f"{factor}={value}"
        print(f"\n── {label} ──")

        if is_done(config_out):
            print("  [skip] already complete")
        else:
            fine_out = config_out_dir(args.out, factor, fine_val, spec)

            # no-onset fine config: derive from baseline rather than a full scan
            if args.onset_standalone and value == fine_val and not corpus_is_ready(config_out):
                baseline_fine_out = config_out_dir(args.baseline_quant_out, factor, fine_val, spec)
                if corpus_is_ready(baseline_fine_out):
                    print(f"  Deriving no-onset corpus from baseline {factor}={fine_val} …")
                    if not args.dry_run:
                        result = derive_no_onset_corpus(baseline_fine_out, config_out)
                        if result is None:
                            print("  WARNING: no-onset derivation failed — falling back to full scan")
                else:
                    print(f"  Baseline corpus ({factor}={fine_val}) not ready; will do full scan")

            # coarser configs: remap the fine corpus instead of scanning MIDI again
            if value != fine_val and not corpus_is_ready(config_out):
                if corpus_is_ready(fine_out):
                    print(f"  Deriving corpus from {factor}={fine_val} …")
                    if not args.dry_run:
                        result = derive_corpus_from_fine(
                            fine_out_dir=fine_out,
                            coarse_out_dir=config_out,
                            factor=factor,
                            fine_value=float(fine_val),
                            coarse_value=float(value),
                        )
                        if result is None:
                            print("  WARNING: derivation failed — falling back to full scan")
                else:
                    print(f"  Fine corpus ({factor}={fine_val}) not ready; "
                          "will do full scan instead")

            onset_flag = ["--onset-standalone"] if args.onset_standalone else []

            run_cmd([sys.executable, train_script,
                     "--dataset", args.dataset,
                     "--out",     config_out,
                     "--merges",  str(MERGES),
                     ] + quant_args + onset_flag, args.dry_run)

            run_cmd([sys.executable, eval_script,
                     "--dataset", args.dataset,
                     "--out",     config_out,
                     ] + quant_args + onset_flag, args.dry_run)

        result = load_result(config_out)
        if result is not None:
            all_quant = {k.lstrip("-").replace("-", "_"): v for k, v in fixed.items()}
            all_quant[sweep_arg.lstrip("-").replace("-", "_")] = value
            rows.append({
                "factor":                factor,
                "sweep_value":           value,
                "onset_ms":              all_quant.get("onset_ms",  value if factor == "onset"    else 10),
                "dur_ms":                all_quant.get("dur_ms",    value if factor == "duration" else 10),
                "vel_bins":              all_quant.get("vel_bins",  value if factor == "velocity" else 128),
                "vocab_size":            result.get("vocab_size"),
                "num_files":             result.get("num_files"),
                "total_original_tokens": result.get("total_original_tokens"),
                "total_bpe_tokens":      result.get("total_bpe_tokens"),
                "reduction_ratio":       result.get("reduction_ratio"),
                "compression_factor":    result.get("compression_factor"),
            })

    return rows


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    factors = {args.factor: EXPERIMENTS[args.factor]} if args.factor else EXPERIMENTS

    summary_rows: list[dict] = []
    for factor, spec in factors.items():
        summary_rows.extend(run_factor(factor, spec, args))

    if summary_rows:
        json_path = os.path.join(args.out, "summary.json")
        csv_path  = os.path.join(args.out, "summary.csv")
        with open(json_path, "w") as f:
            json.dump(summary_rows, f, indent=2)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary → {json_path}")
        print(f"Summary → {csv_path}")
    elif args.dry_run:
        print("\n(dry-run: no results to summarize yet)")

    print("\nDone.")


if __name__ == "__main__":
    main()
