"""
Plot BPE vocab size vs. compression factor.

Usage:
    python bpe/scripts/plot_results.py --out outputs
"""

import argparse
import json
import math
import os

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results_path = os.path.join(args.out, "results", "length_reduction_results.json")
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
    ax.set_xticks(vocab_sizes)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plots_dir = os.path.join(args.out, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "vocab_vs_length_reduction.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {plot_path}")


if __name__ == "__main__":
    main()
