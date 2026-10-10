"""
Chroma JS Distance: Jensen-Shannon distance (base-2 log, bounded in [0, 1])
between chroma-entropy histograms of a generated set and the real test set.

Unlike groove (src/metrics/groove.py), the histogram is built from every
valid window-level (chunk) entropy value pooled across all pieces, not from
one mean-of-windows value per piece -- every chunk gets equal weight rather
than every piece. Per-piece H_piece (mean over that piece's windows) is still
computed and reported as a diagnostic (mean_h_piece/std_h_piece), following
MIDI-LLM's evaluation structure, but no
longer feeds the histogram/JS-distance computation.

Computed directly from decoded MIDI via src/chroma_groove.py -- unlike
clap.py/fad.py this metric never touches synthesized audio, so
run_evaluation.py skips the FluidSynth stage entirely when only chroma/groove
are requested (see Metric.needs_audio).

Ground-truth (real test set) per-piece scores don't depend on the model being
evaluated, so -- mirroring fad.py's shared VGGish-embedding cache -- they're
cached once per metric-parameter combination under
evaluation/_ground_truth/chroma/ and reused across every model_name.
"""

from __future__ import annotations

import json
import os

import numpy as np

from chroma_groove import (
    CHROMA_MAX_ENTROPY, append_csv_row, chroma_bin_edges, chroma_entropy_piece,
    js_distance, load_csv_rows,
)

from .base import EvalContext, Metric, register

_CSV_FIELDS = ["id", "h_piece", "window_entropies", "n_valid_windows", "n_total_windows", "reason"]


def _config_tag(window_sec: float, include_drums: bool, discard_partial_window: bool) -> str:
    return f"w{window_sec:g}_drums{int(include_drums)}_partial{int(not discard_partial_window)}"


def _score_set(items, path_attr: str, csv_path: str, *,
                window_sec: float, include_drums: bool, discard_partial_window: bool) -> dict:
    """items: GeneratedSample or TestEntry objects. Resumable: ids already
    present in csv_path are reused; only missing ones are (re)computed and
    appended. Returns rows for `items` only -- the shared ground-truth CSV also
    holds other evaluation sets' rows (e.g. test and valid), which must not
    leak into this set's statistics."""
    done = {row["id"]: row for row in load_csv_rows(csv_path)}
    wanted = []

    for item in items:
        item_id = getattr(item, "sample_id", None) or getattr(item, "md5")
        wanted.append(item_id)
        if item_id in done:
            continue
        mid_path = getattr(item, path_attr)

        row = {"id": item_id}
        try:
            import pretty_midi
            pm = pretty_midi.PrettyMIDI(mid_path)
        except Exception as e:
            row.update(h_piece="nan", window_entropies="[]", n_valid_windows=0, n_total_windows=0,
                       reason=f"parse_error: {e}")
        else:
            result = chroma_entropy_piece(
                pm, window_sec=window_sec, include_drums=include_drums,
                discard_partial_window=discard_partial_window,
            )
            row.update(h_piece=result.h_piece, window_entropies=json.dumps(result.window_entropies or []),
                       n_valid_windows=result.n_valid_windows,
                       n_total_windows=result.n_total_windows, reason=result.reason or "")

        append_csv_row(csv_path, _CSV_FIELDS, row)
        done[item_id] = row

    return {item_id: done[item_id] for item_id in wanted}


def _valid_values(rows: dict) -> list[float]:
    """Per-piece H_piece values (diagnostic only -- not used for the histogram)."""
    return [float(r["h_piece"]) for r in rows.values() if r["reason"] == ""]


def _pooled_window_values(rows: dict) -> list[float]:
    """Every valid window-level entropy value, pooled across all pieces --
    this is what the chroma histogram/JS-distance is built from."""
    pooled = []
    for r in rows.values():
        if r["reason"] == "":
            pooled.extend(json.loads(r["window_entropies"]))
    return pooled


def _stats(rows: dict, values: list[float], pooled: list[float]) -> dict:
    return {
        "n_pieces": len(rows),
        "n_valid": len(values),
        "n_skipped": len(rows) - len(values),
        "n_pooled_windows": len(pooled),
        "mean_h_piece": float(np.mean(values)) if values else float("nan"),
        "std_h_piece": float(np.std(values)) if values else float("nan"),
        "failure_reasons": sorted({r["reason"] for r in rows.values() if r["reason"]}),
    }


@register("chroma")
class ChromaMetric(Metric):
    needs_audio = False

    def __init__(
        self, window_sec: float = 5.0, include_drums: bool = False,
        discard_partial_window: bool = True, n_bins: int = 20,
        ground_truth_cache_dir: str | None = None,
    ):
        self.window_sec = window_sec
        self.include_drums = include_drums
        self.discard_partial_window = discard_partial_window
        self.n_bins = n_bins
        self.ground_truth_cache_dir = ground_truth_cache_dir

    def run(self, ctx: EvalContext) -> dict:
        out_dir = os.path.join(ctx.output_root, self.name)
        os.makedirs(out_dir, exist_ok=True)

        tag = _config_tag(self.window_sec, self.include_drums, self.discard_partial_window)
        gen_csv = os.path.join(out_dir, "per_piece.csv")
        gt_dir = self.ground_truth_cache_dir or out_dir
        os.makedirs(gt_dir, exist_ok=True)
        gt_csv = os.path.join(gt_dir, f"ground_truth_per_piece_{tag}.csv")

        common = dict(window_sec=self.window_sec, include_drums=self.include_drums,
                      discard_partial_window=self.discard_partial_window)
        gen_rows = _score_set(ctx.generated, "mid_path", gen_csv, **common)
        gt_rows = _score_set(ctx.ground_truth, "midi_path", gt_csv, **common)

        gen_values, gt_values = _valid_values(gen_rows), _valid_values(gt_rows)
        gen_pooled, gt_pooled = _pooled_window_values(gen_rows), _pooled_window_values(gt_rows)
        bin_edges = chroma_bin_edges(self.n_bins)
        distance, distance_reason = js_distance(gen_pooled, gt_pooled, bin_edges)

        summary = {
            "params": {
                "window_sec": self.window_sec,
                "include_drums": self.include_drums,
                "discard_partial_window": self.discard_partial_window,
                "n_bins": self.n_bins,
                "bin_edges": bin_edges.tolist(),
                "theoretical_range": [0.0, CHROMA_MAX_ENTROPY],
                "histogram_unit": "pooled window-level (chunk) entropy values, not per-piece H_piece",
            },
            "generated": _stats(gen_rows, gen_values, gen_pooled),
            "ground_truth": _stats(gt_rows, gt_values, gt_pooled),
            "chroma_js_distance": distance,
            "chroma_js_distance_reason": distance_reason,
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary
