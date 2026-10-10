"""
Groove JS Distance: Jensen-Shannon distance (base-2 log, bounded in [0, 1])
between piece-level rhythmic groove-similarity histograms of a generated set
and the real songs of the same split, following MIDI-LLM's evaluation structure.

Computed directly from decoded MIDI via src/chroma_groove.py -- unlike
clap.py/fad.py this metric never touches synthesized audio, so
run_evaluation.py skips the FluidSynth stage entirely when only chroma/groove
are requested (see Metric.needs_audio).

Ground-truth (real test set) per-piece scores don't depend on the model being
evaluated, so -- mirroring fad.py's shared VGGish-embedding cache -- they're
cached once per metric-parameter combination under
evaluation/_ground_truth/groove/ and reused across every model_name.
"""

from __future__ import annotations

import json
import os

import numpy as np

from chroma_groove import (
    append_csv_row, groove_bin_edges, groove_similarity_piece, js_distance, load_csv_rows,
)

from .base import EvalContext, Metric, register

_CSV_FIELDS = ["id", "gs_piece", "onset_density", "n_valid_windows", "n_total_windows", "reason"]


def _config_tag(window_sec: float, grid_ms: float, include_drums: bool,
                 discard_partial_window: bool) -> str:
    return (f"w{window_sec:g}_g{grid_ms:g}_drums{int(include_drums)}"
            f"_partial{int(not discard_partial_window)}")


def _score_set(items, path_attr: str, csv_path: str, *,
                window_sec: float, grid_ms: float, include_drums: bool,
                discard_partial_window: bool) -> dict:
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
            row.update(gs_piece="nan", onset_density="nan", n_valid_windows=0,
                       n_total_windows=0, reason=f"parse_error: {e}")
        else:
            result = groove_similarity_piece(
                pm, window_sec=window_sec, grid_ms=grid_ms, include_drums=include_drums,
                discard_partial_window=discard_partial_window,
            )
            row.update(gs_piece=result.gs_piece, onset_density=result.onset_density,
                       n_valid_windows=result.n_valid_windows,
                       n_total_windows=result.n_total_windows, reason=result.reason or "")

        append_csv_row(csv_path, _CSV_FIELDS, row)
        done[item_id] = row

    return {item_id: done[item_id] for item_id in wanted}


def _valid_values(rows: dict) -> list[float]:
    return [float(r["gs_piece"]) for r in rows.values() if r["reason"] == ""]


def _stats(rows: dict, values: list[float]) -> dict:
    densities = [float(r["onset_density"]) for r in rows.values() if r["reason"] == ""]
    return {
        "n_pieces": len(rows),
        "n_valid": len(values),
        "n_skipped": len(rows) - len(values),
        "mean_gs_piece": float(np.mean(values)) if values else float("nan"),
        "std_gs_piece": float(np.std(values)) if values else float("nan"),
        "mean_onset_density": float(np.mean(densities)) if densities else float("nan"),
        "std_onset_density": float(np.std(densities)) if densities else float("nan"),
        "failure_reasons": sorted({r["reason"] for r in rows.values() if r["reason"]}),
    }


@register("groove")
class GrooveMetric(Metric):
    needs_audio = False

    def __init__(
        self, window_sec: float = 5.0, grid_ms: float = 50.0, include_drums: bool = True,
        discard_partial_window: bool = True, n_bins: int = 20,
        ground_truth_cache_dir: str | None = None,
    ):
        self.window_sec = window_sec
        self.grid_ms = grid_ms
        self.include_drums = include_drums
        self.discard_partial_window = discard_partial_window
        self.n_bins = n_bins
        self.ground_truth_cache_dir = ground_truth_cache_dir

    def run(self, ctx: EvalContext) -> dict:
        out_dir = os.path.join(ctx.output_root, self.name)
        os.makedirs(out_dir, exist_ok=True)

        tag = _config_tag(self.window_sec, self.grid_ms, self.include_drums,
                           self.discard_partial_window)
        gen_csv = os.path.join(out_dir, "per_piece.csv")
        gt_dir = self.ground_truth_cache_dir or out_dir
        os.makedirs(gt_dir, exist_ok=True)
        gt_csv = os.path.join(gt_dir, f"ground_truth_per_piece_{tag}.csv")

        common = dict(window_sec=self.window_sec, grid_ms=self.grid_ms,
                      include_drums=self.include_drums,
                      discard_partial_window=self.discard_partial_window)
        gen_rows = _score_set(ctx.generated, "mid_path", gen_csv, **common)
        gt_rows = _score_set(ctx.ground_truth, "midi_path", gt_csv, **common)

        gen_values, gt_values = _valid_values(gen_rows), _valid_values(gt_rows)
        bin_edges = groove_bin_edges(self.n_bins)
        distance, distance_reason = js_distance(gen_values, gt_values, bin_edges)

        summary = {
            "params": {
                "window_sec": self.window_sec,
                "grid_ms": self.grid_ms,
                "include_drums": self.include_drums,
                "discard_partial_window": self.discard_partial_window,
                "n_bins": self.n_bins,
                "bin_edges": bin_edges.tolist(),
                "theoretical_range": [0.0, 1.0],
            },
            "generated": _stats(gen_rows, gen_values),
            "ground_truth": _stats(gt_rows, gt_values),
            "groove_js_distance": distance,
            "groove_js_distance_reason": distance_reason,
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary
