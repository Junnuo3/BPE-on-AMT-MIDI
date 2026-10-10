"""
Bar-based Groove JS Distance: same structure as groove.py (per-piece groove
similarity -> 20-bin histogram over [0, 1] -> base-2 Jensen-Shannon distance
between the generated and real sets), but each piece's groove similarity is
computed over bars detected by a downbeat tracker (src/downbeats.py, Beat
This!) on its rendered audio instead of fixed 5 s windows -- see
chroma_groove.groove_similarity_piece_bars. groove.py is unchanged.

Needs audio (rendered generations; real pieces' audio for any ground-truth row
not cached yet). Detected downbeats are stored in each per-piece CSV row, and
ground-truth rows are cached once per parameter set under
evaluation/_ground_truth/groove_bar/ (filled for a whole split by
scripts/build_ground_truth_cache.py so concurrent eval jobs only read it).
"""

from __future__ import annotations

import json
import os

import numpy as np

from chroma_groove import (
    append_csv_row, groove_bin_edges, groove_similarity_piece_bars, js_distance, load_csv_rows,
)
from downbeats import CHECKPOINT, DownbeatTracker

from .base import EvalContext, Metric, register
from .groove import _stats, _valid_values

_CSV_FIELDS = ["id", "gs_piece", "onset_density", "n_valid_windows", "n_total_windows", "reason",
               "downbeats"]


def _config_tag(positions_per_bar: int, include_drums: bool, checkpoint: str = CHECKPOINT) -> str:
    return f"q{positions_per_bar}_drums{int(include_drums)}_bt-{checkpoint}"


def _score_set(items, midi_attr: str, csv_path: str, tracker: DownbeatTracker, *,
               positions_per_bar: int, include_drums: bool) -> dict:
    """Resumable like groove._score_set; returns rows for `items` only."""
    done = {row["id"]: row for row in load_csv_rows(csv_path)}
    wanted = []
    for k, item in enumerate(items):
        item_id = getattr(item, "sample_id", None) or getattr(item, "md5")
        wanted.append(item_id)
        if item_id in done:
            continue
        wav_path = getattr(item, "wav_path", None)
        row = {"id": item_id, "downbeats": "[]"}
        try:
            import pretty_midi
            pm = pretty_midi.PrettyMIDI(getattr(item, midi_attr))
        except Exception as e:
            row.update(gs_piece="nan", onset_density="nan", n_valid_windows=0,
                       n_total_windows=0, reason=f"parse_error: {e}")
        else:
            if wav_path is None:
                if midi_attr == "mid_path" or not any(inst.notes for inst in pm.instruments):
                    # a generation that didn't render (no notes / synth failure) is a
                    # skipped piece, like groove.py's unscoreable pieces
                    row.update(gs_piece="nan", onset_density="nan", n_valid_windows=0,
                               n_total_windows=0, reason="no rendered audio")
                    append_csv_row(csv_path, _CSV_FIELDS, row)
                    done[item_id] = row
                    continue
                raise RuntimeError(
                    f"groove_bar: no rendered audio for {item_id} and no cached row in {csv_path} "
                    f"(for ground truth, run scripts/build_ground_truth_cache.py first)")
            downbeats = tracker(wav_path)
            result = groove_similarity_piece_bars(pm, downbeats, positions_per_bar=positions_per_bar,
                                                  include_drums=include_drums)
            row.update(gs_piece=result.gs_piece, onset_density=result.onset_density,
                       n_valid_windows=result.n_valid_windows, n_total_windows=result.n_total_windows,
                       reason=result.reason or "",
                       downbeats=json.dumps([round(float(d), 3) for d in downbeats]))
        append_csv_row(csv_path, _CSV_FIELDS, row)
        done[item_id] = row
        if (k + 1) % 1000 == 0:
            print(f"  groove_bar: {k + 1}/{len(items)} scored")
    return {item_id: done[item_id] for item_id in wanted}


@register("groove_bar")
class GrooveBarMetric(Metric):
    needs_audio = True

    def __init__(self, positions_per_bar: int = 64, include_drums: bool = True, n_bins: int = 20,
                 ground_truth_cache_dir: str | None = None, device: str | None = None):
        self.positions_per_bar = positions_per_bar
        self.include_drums = include_drums
        self.n_bins = n_bins
        self.ground_truth_cache_dir = ground_truth_cache_dir
        self.tracker = DownbeatTracker(device=device)

    def run(self, ctx: EvalContext) -> dict:
        out_dir = os.path.join(ctx.output_root, self.name)
        os.makedirs(out_dir, exist_ok=True)
        tag = _config_tag(self.positions_per_bar, self.include_drums)
        gen_csv = os.path.join(out_dir, "per_piece.csv")
        gt_dir = self.ground_truth_cache_dir or out_dir
        os.makedirs(gt_dir, exist_ok=True)
        gt_csv = os.path.join(gt_dir, f"ground_truth_per_piece_{tag}.csv")

        common = dict(positions_per_bar=self.positions_per_bar, include_drums=self.include_drums)
        gen_rows = _score_set(ctx.generated, "mid_path", gen_csv, self.tracker, **common)
        gt_rows = _score_set(ctx.ground_truth, "midi_path", gt_csv, self.tracker, **common)

        gen_values, gt_values = _valid_values(gen_rows), _valid_values(gt_rows)
        bin_edges = groove_bin_edges(self.n_bins)
        distance, distance_reason = js_distance(gen_values, gt_values, bin_edges)
        summary = {
            "params": {
                "unit": "detected bar",
                "positions_per_bar": self.positions_per_bar,
                "include_drums": self.include_drums,
                "partial_bars": "discarded (pickup and after last downbeat)",
                "downbeat_tracker": f"beat_this {CHECKPOINT}, dbn=False",
                "n_bins": self.n_bins,
                "bin_edges": bin_edges.tolist(),
                "theoretical_range": [0.0, 1.0],
            },
            "generated": _stats(gen_rows, gen_values),
            "ground_truth": _stats(gt_rows, gt_values),
            "groove_bar_js_distance": distance,
            "groove_bar_js_distance_reason": distance_reason,
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary
