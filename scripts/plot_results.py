"""
Plot BPE compression results.

Usage:
    python bpe/scripts/plot_results.py
    python bpe/scripts/plot_results.py --quant
"""

import argparse
import json
import math
import os
import re

import matplotlib.pyplot as plt

_SCRIPTS_DIR      = os.path.dirname(os.path.abspath(__file__))
_BPE_ROOT         = os.path.dirname(_SCRIPTS_DIR)
_DEFAULT_OUT      = os.path.join(_BPE_ROOT, "tokenizers", "vocab_sweep",
                                  "q_onset-10ms_duration-10ms_velocity-128bin")
_DEFAULT_QUANT    = os.path.join(_BPE_ROOT, "tokenizers", "quantization_sweep", "merges-8192")
_DEFAULT_NO_ONSET = os.path.join(_BPE_ROOT, "tokenizers", "merge_constraints",
                                  "no_onset_merge", "merges-8192")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out",          default=_DEFAULT_OUT,
                   help="vocab_sweep root for the vocab-size plot "
                        f"(default: {os.path.relpath(_DEFAULT_OUT, _BPE_ROOT)})")
    p.add_argument("--quant",        action="store_true",
                   help="Plot quantization sweep results instead of vocab-size curve")
    p.add_argument("--quant-dir",    default=_DEFAULT_QUANT,
                   help="quantization_sweep merges root (used with --quant)")
    p.add_argument("--no-onset-dir", default=_DEFAULT_NO_ONSET,
                   help="no_onset_merge merges root (used with --quant)")
    return p.parse_args()



def plot_vocab_vs_compression(out_dir: str) -> None:
    results_path = os.path.join(out_dir, "results", "length_reduction_results.json")
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found. Run evaluate_bpe.py first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    results = [r for r in results if not math.isnan(r.get("reduction_ratio", float("nan")))]
    results.sort(key=lambda r: r["vocab_size"])
    if not results:
        print("No valid results to plot.")
        return

    vocab_sizes         = [r["vocab_size"]         for r in results]
    compression_factors = [r["compression_factor"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(vocab_sizes, compression_factors, "o-", color="coral", linewidth=2, markersize=7)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="no compression")
    ax.set_xlabel("BPE Vocab Size")
    ax.set_ylabel("Compression Factor  (Original / BPE tokens)")
    ax.set_title(
        "BPE on AMT 5-Token MIDI\nCompression Factor (higher = more compression)",
        fontsize=11,
    )
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    from collections import Counter
    raw = [f"{round(v/1000)}k" for v in vocab_sizes]
    counts = Counter(raw)
    labels = [f"{v/1000:.1f}k" if counts[raw[i]] > 1 else raw[i]
              for i, v in enumerate(vocab_sizes)]
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(vocab_sizes))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "vocab_vs_length_reduction.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {plot_path}")


_FACTOR_META = {
    "onset":    {"label": "Onset quantization step (ms)",    "unit": "ms",   "coarser": "larger"},
    "duration": {"label": "Duration quantization step (ms)", "unit": "ms",   "coarser": "larger"},
    "velocity": {"label": "Velocity bins",                   "unit": "bins", "coarser": "fewer"},
}

# velocity: fewer bins = coarser, so invert the x-axis to keep fine→coarse left-to-right
_INVERT_X = {"onset": False, "duration": False, "velocity": True}



def _load_summary(path: str) -> list[dict] | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return [r for r in data if not math.isnan(r.get("compression_factor", float("nan")))]


_CONFIG_DIR_RE = re.compile(
    r"onset-(\d+(?:\.\d+)?)ms_duration-(\d+(?:\.\d+)?)ms_velocity-(\d+)bin"
)


def _parse_config_dir(factor: str, config_dir: str) -> float | None:
    """Extract the sweep value for this factor from a descriptive config dir name."""
    m = _CONFIG_DIR_RE.match(config_dir)
    if m is None:
        return None
    onset, dur, vel = float(m.group(1)), float(m.group(2)), int(m.group(3))
    return {"onset": onset, "duration": dur, "velocity": float(vel)}[factor]


def _load_quant_results(quant_out: str) -> list[dict]:
    """Scan per-config result files directly — works even when summary.json is partial."""
    rows = []
    for factor in _FACTOR_META:
        factor_dir = os.path.join(quant_out, factor)
        if not os.path.isdir(factor_dir):
            continue
        for config_dir in sorted(os.listdir(factor_dir)):
            result_path = os.path.join(factor_dir, config_dir, "results",
                                       "length_reduction_results.json")
            if not os.path.exists(result_path):
                continue
            sweep_value = _parse_config_dir(factor, config_dir)
            if sweep_value is None:
                continue
            try:
                with open(result_path) as f:
                    data = json.load(f)
                if not data:
                    continue
                r = data[0]
                if math.isnan(r.get("compression_factor", float("nan"))):
                    continue
                rows.append({**r, "factor": factor, "sweep_value": sweep_value})
            except Exception:
                continue
    return rows


def _knee_index(xs: list[float], ys: list[float]) -> int:
    """Index of the point with maximum distance from the line connecting endpoints."""
    if len(xs) < 3:
        return -1
    x0, y0 = xs[0],  ys[0]
    x1, y1 = xs[-1], ys[-1]
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy)
    if norm == 0:
        return -1
    dists = [abs(dy * (xi - x0) - dx * (yi - y0)) / norm
             for xi, yi in zip(xs, ys)]
    return int(max(range(len(dists)), key=lambda i: dists[i]))


def plot_quant_sweep(baseline_dir: str, no_onset_dir: str, plots_root: str) -> None:

    # read per-config result files directly (works even when summary.json is partial)
    baseline = _load_quant_results(baseline_dir) or None
    no_onset = _load_quant_results(no_onset_dir) or None

    if baseline is None and no_onset is None:
        print("ERROR: no quantization results found. Run run_quant_experiments.py first.")
        print(f"  Expected under: {baseline_dir}")
        return

    factors = list(_FACTOR_META.keys())

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.suptitle("BPE Compression vs. Quantization Level", fontsize=13)

    all_series: dict[str, dict] = {}  # baseline xs/ys per factor, reused in leverage chart

    for ax, factor in zip(axes, factors):
        meta   = _FACTOR_META[factor]
        invert = _INVERT_X[factor]

        for rows, color, style, series_label in [
            (baseline,  "coral",      "-",  "baseline"),
            (no_onset,  "steelblue",  "--", "no-onset"),
        ]:
            if rows is None:
                continue
            subset = sorted(
                [r for r in rows if r.get("factor") == factor],
                key=lambda r: r["sweep_value"],
            )
            if not subset:
                continue

            xs = [r["sweep_value"]        for r in subset]
            ys = [r["compression_factor"] for r in subset]

            ax.plot(xs, ys, "o" + style, color=color, linewidth=2,
                    markersize=7, label=series_label)

            if series_label == "baseline":
                all_series[factor] = {"xs": xs, "ys": ys}

            # skip knee annotation if the line is basically flat
            y_range = max(ys) - min(ys)
            if y_range > 0.01:
                ki = _knee_index(xs, ys)
                if ki >= 0:
                    ax.axvline(xs[ki], color=color, linestyle=":", linewidth=1.2, alpha=0.7)
                    ax.annotate(
                        f"knee\n{xs[ki]}{meta['unit']}",
                        xy=(xs[ki], ys[ki]),
                        xytext=(8, -18), textcoords="offset points",
                        fontsize=7, color=color,
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
                    )

        if invert:
            ax.invert_xaxis()  # velocity: fine (more bins) on the left

        ax.set_xlabel(meta["label"])
        ax.set_ylabel("Compression Factor" if factor == "onset" else "")
        ax.set_title(factor.capitalize())
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plots_dir = os.path.join(plots_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    sweep_path = os.path.join(plots_dir, "quant_vs_compression.png")
    plt.savefig(sweep_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sweep plot  → {sweep_path}")

    # leverage bar chart
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    lever_factors, lever_vals, lever_colors = [], [], []
    for factor in factors:
        if factor not in all_series:
            continue
        ys = all_series[factor]["ys"]
        lever_factors.append(factor.capitalize())
        lever_vals.append(max(ys) - min(ys))
        lever_colors.append(
            "coral" if factor == "velocity" else
            "steelblue" if factor == "duration" else "gray"
        )

    bars = ax2.bar(lever_factors, lever_vals, color=lever_colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, lever_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax2.set_ylabel("Compression range  (max − min)")
    ax2.set_title("Which factor has the most leverage on compression?")
    ax2.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    lever_path = os.path.join(plots_dir, "quant_leverage.png")
    plt.savefig(lever_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Leverage    → {lever_path}")

    # summary table
    print()
    print(f"{'Factor':<10} {'Range':>8}  {'@ finest':>10}  {'@ coarsest':>12}")
    print("-" * 46)
    for factor in factors:
        if factor not in all_series:
            continue
        subset = sorted([r for r in (baseline or []) if r.get("factor") == factor],
                        key=lambda r: r["sweep_value"])
        ys_raw = [r["compression_factor"] for r in subset]
        if not ys_raw:
            continue
        lever = max(ys_raw) - min(ys_raw)
        if _INVERT_X.get(factor):
            finest, coarse = ys_raw[-1], ys_raw[0]
        else:
            finest, coarse = ys_raw[0],  ys_raw[-1]
        print(f"{factor:<10} {lever:>8.4f}  {finest:>10.4f}  {coarse:>12.4f}")


def main() -> None:
    args = parse_args()
    if args.quant:
        plot_quant_sweep(args.quant_dir, args.no_onset_dir, args.out)
    else:
        plot_vocab_vs_compression(args.out)


if __name__ == "__main__":
    main()
