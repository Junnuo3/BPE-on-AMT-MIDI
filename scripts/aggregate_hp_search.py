"""
Collect evaluation/<model_name>/summary.json for a decoding sweep into one
table (default sort: FAD ascending).

--rank ranks every config on each --rank-metrics column (default FAD, CLAP
[higher is better], chroma_js, groove_bar_js; 1 = best) and sorts by the sum of
ranks. pareto marks configs that no other config matches or beats on every
ranked metric. groove_js (5 s windows) and skip_frac (pieces with no pitched
notes or shorter than one window) are reported but not ranked.

Usage:
    python scripts/aggregate_hp_search.py --prefix bpe_valid_ --rank --csv-out results/sweep_bpe.csv
"""

import argparse
import glob
import json
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_SCRIPT_DIR)
_EVAL_ROOT  = os.path.join(_REPO_ROOT, "evaluation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", required=True,
                   help="Only model_name dirs under evaluation/ starting with this are included.")
    p.add_argument("--csv-out", default=None, help="Optional path to also write the table as CSV.")
    p.add_argument("--rank", action="store_true",
                   help="Rank on --rank-metrics (rank sum + Pareto flag) and sort by it.")
    p.add_argument("--rank-metrics", default="fad,clap,chroma_js,groove_bar_js",
                   help="Comma list of columns to rank on (clap = higher is better, others lower).")
    return p.parse_args()


def _get(d: dict, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def main() -> None:
    args = parse_args()

    rows = []
    for summary_path in sorted(glob.glob(os.path.join(_EVAL_ROOT, f"{args.prefix}*", "summary.json"))):
        model_name = os.path.basename(os.path.dirname(summary_path))
        with open(summary_path) as f:
            s = json.load(f)
        metrics = s.get("metrics", {})

        gen_chroma = _get(metrics, "chroma", "generated", default={})
        gen_groove = _get(metrics, "groove", "generated", default={})
        n_pieces   = gen_chroma.get("n_pieces") or gen_groove.get("n_pieces")
        n_skipped  = gen_chroma.get("n_skipped", gen_groove.get("n_skipped"))

        rows.append({
            "model_name":     model_name,
            "fad":            _get(metrics, "fad", "fad"),
            "chroma_js":      _get(metrics, "chroma", "chroma_js_distance"),
            "groove_js":      _get(metrics, "groove", "groove_js_distance"),
            "clap":           _get(metrics, "clap", "clap_cosine_mean"),
            "groove_bar_js":  _get(metrics, "groove_bar", "groove_bar_js_distance"),
            "n_pieces":       n_pieces,
            "n_skipped":      n_skipped,
            "skip_frac":      round(n_skipped / n_pieces, 3) if n_pieces else None,
            "groove_skip_frac": (round(gen_groove["n_skipped"] / n_pieces, 3)
                                 if n_pieces and "n_skipped" in gen_groove else None),
        })

    if not rows:
        print(f"No results found under {_EVAL_ROOT}/{args.prefix}*/summary.json")
        return

    headers = ["model_name", "fad", "clap", "chroma_js", "groove_js", "groove_bar_js", "n_pieces", "n_skipped",
               "skip_frac", "groove_skip_frac"]
    if args.rank:
        # (metric, higher_is_better)
        higher_better = {"clap"}
        sel = [(m, m in higher_better) for m in args.rank_metrics.split(",")]
        missing = [r["model_name"] for r in rows if any(r[m] is None for m, _ in sel)]
        if missing:
            raise SystemExit(f"--rank needs all of {args.rank_metrics}; missing in: {missing}")
        for m, hib in sel:
            for i, r in enumerate(sorted(rows, key=lambda r: -r[m] if hib else r[m])):
                r[f"rank_{m}"] = i + 1
        for r in rows:
            r["rank_sum"] = sum(r[f"rank_{m}"] for m, _ in sel)
            better = lambda o, m, hib: o[m] > r[m] if hib else o[m] < r[m]
            worse = lambda o, m, hib: o[m] < r[m] if hib else o[m] > r[m]
            r["pareto"] = not any(
                all(not worse(o, m, hib) for m, hib in sel) and any(better(o, m, hib) for m, hib in sel)
                for o in rows if o is not r)
        rows.sort(key=lambda r: (r["rank_sum"], r["rank_fad"]))
        headers += [f"rank_{m}" for m, _ in sel] + ["rank_sum", "pareto"]
    else:
        rows.sort(key=lambda r: (r["fad"] is None, r["fad"]))
    widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in headers}
    print("  ".join(h.ljust(widths[h]) for h in headers))
    for r in rows:
        print("  ".join(str(r[h]).ljust(widths[h]) for h in headers))

    if args.csv_out:
        import csv
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv_out}")


if __name__ == "__main__":
    main()
