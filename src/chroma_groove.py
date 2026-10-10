"""
Time-windowed chroma-entropy and rhythmic groove-similarity metrics, computed
directly from decoded MIDI (independent of tokenization), following MIDI-LLM's
evaluation structure:

  1. one chroma-entropy score (H_piece) and one groove-similarity score
     (GS_piece) per piece, plus (for chroma) the raw per-window entropy
     values that H_piece was averaged from;
  2. dataset-level histograms: groove pools one GS_piece value per piece (so
     every piece gets equal weight); chroma pools every valid window-level
     entropy value across all pieces (every chunk gets equal weight -- see
     src/metrics/chroma.py);
  3. Jensen-Shannon distance (base-2 log, so bounded in [0, 1]) between a
     generated set's histogram and the real test set's histogram, using
     identical fixed bin edges for both.

MidiCaps/Lakh MIDI files don't have reliable beat/bar annotations, so windows
are fixed-duration (seconds), not bar-based. This is a practical replacement
for bar-based segmentation, not an exact musical equivalent: it does not
capture bar-relative rhythmic structure, and absolute-time windows are not
tempo-invariant (the same rhythmic pattern played at a different tempo falls
into different-sized musical spans).

pretty_midi resolves MIDI ticks to seconds internally, including tempo-change
handling -- every Note.start/Note.end read from a file already accounts for
the tempo map, so nothing here needs a separate tick->second conversion step.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass

import numpy as np

CHROMA_MAX_ENTROPY = math.log2(12)


@dataclass(frozen=True)
class NoteOnset:
    time: float
    pitch: int
    is_drum: bool


@dataclass
class ChromaResult:
    h_piece: float             # NaN if invalid (see `reason`); mean(window_entropies)
    n_valid_windows: int
    n_total_windows: int
    reason: str | None = None       # set iff h_piece is NaN
    window_entropies: list[float] = None  # per-window entropies h_piece was averaged from; [] if invalid


@dataclass
class GrooveResult:
    gs_piece: float             # NaN if invalid (see `reason`)
    onset_density: float        # mean fraction of set grid bins across valid windows; NaN if no valid windows
    n_valid_windows: int
    n_total_windows: int
    reason: str | None = None   # set iff gs_piece is NaN


def note_onsets(pm, *, include_drums: bool) -> list[NoteOnset]:
    """Onsets from every instrument, sorted by time. `pm` is a
    pretty_midi.PrettyMIDI (duck-typed here so tests can pass lightweight
    stand-ins without importing pretty_midi)."""
    onsets = []
    for inst in pm.instruments:
        if inst.is_drum and not include_drums:
            continue
        for note in inst.notes:
            onsets.append(NoteOnset(time=note.start, pitch=note.pitch, is_drum=inst.is_drum))
    onsets.sort(key=lambda o: o.time)
    return onsets


def _num_windows(total_duration: float, window_sec: float, *, discard_partial: bool) -> int:
    if total_duration <= 0 or window_sec <= 0:
        return 0
    full = int(total_duration // window_sec)
    remainder = total_duration - full * window_sec
    if remainder > 1e-9 and not discard_partial:
        full += 1
    return full


def _window_index(t: float, window_sec: float) -> int:
    return int(t // window_sec)


def pitch_class_entropy(pitches) -> float:
    """Shannon entropy in bits of the 12-dim onset-count pitch-class histogram
    h_c = n_c / sum_j(n_j), including only terms with h_c > 0. NaN if `pitches`
    is empty."""
    counts = np.zeros(12)
    for p in pitches:
        counts[p % 12] += 1
    total = counts.sum()
    if total == 0:
        return float("nan")
    probs = counts[counts > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def chroma_entropy_piece(
    pm, *, window_sec: float = 5.0, include_drums: bool = False,
    discard_partial_window: bool = True,
) -> ChromaResult:
    """H_piece = mean over non-empty windows of the window's pitch-class entropy."""
    total_duration = pm.get_end_time()
    n_windows = _num_windows(total_duration, window_sec, discard_partial=discard_partial_window)
    if n_windows == 0:
        return ChromaResult(float("nan"), 0, 0, "piece shorter than one window", [])

    onsets = note_onsets(pm, include_drums=include_drums)
    window_pitches: dict[int, list[int]] = {}
    for o in onsets:
        w = _window_index(o.time, window_sec)
        if 0 <= w < n_windows:
            window_pitches.setdefault(w, []).append(o.pitch)

    window_entropies = [pitch_class_entropy(ps) for ps in window_pitches.values() if ps]
    if not window_entropies:
        return ChromaResult(float("nan"), 0, n_windows, "no pitched-note onsets in any window", [])

    return ChromaResult(float(np.mean(window_entropies)), len(window_entropies), n_windows,
                         None, window_entropies)


def _grid_vector(onset_times, window_start: float, window_sec: float, grid_sec: float) -> np.ndarray:
    n_bins = max(1, round(window_sec / grid_sec))
    grid = np.zeros(n_bins, dtype=bool)
    for t in onset_times:
        idx = int((t - window_start) // grid_sec)
        idx = min(max(idx, 0), n_bins - 1)
        grid[idx] = True  # multiple onsets in the same bin still map to a single 1
    return grid


def groove_similarity_piece(
    pm, *, window_sec: float = 5.0, grid_ms: float = 50.0, include_drums: bool = True,
    discard_partial_window: bool = True,
) -> GrooveResult:
    """GS_piece = mean over unordered pairs of valid windows of
    1 - mean(XOR(grid_a, grid_b)). Requires >= 2 valid (non-empty) windows."""
    grid_sec = grid_ms / 1000.0
    total_duration = pm.get_end_time()
    n_windows = _num_windows(total_duration, window_sec, discard_partial=discard_partial_window)
    if n_windows == 0:
        return GrooveResult(float("nan"), float("nan"), 0, 0, "piece shorter than one window")

    onsets = note_onsets(pm, include_drums=include_drums)
    window_times: dict[int, list[float]] = {}
    for o in onsets:
        w = _window_index(o.time, window_sec)
        if 0 <= w < n_windows:
            window_times.setdefault(w, []).append(o.time)

    valid_windows = sorted(window_times)
    if len(valid_windows) < 2:
        density = float("nan")
        if len(valid_windows) == 1:
            w = valid_windows[0]
            grid = _grid_vector(window_times[w], w * window_sec, window_sec, grid_sec)
            density = float(grid.mean())
        return GrooveResult(float("nan"), density, len(valid_windows), n_windows,
                             "fewer than 2 valid (non-empty) windows")

    grids = {
        w: _grid_vector(window_times[w], w * window_sec, window_sec, grid_sec)
        for w in valid_windows
    }

    pair_scores = [
        1.0 - float(np.mean(grids[valid_windows[i]] ^ grids[valid_windows[j]]))
        for i in range(len(valid_windows))
        for j in range(i + 1, len(valid_windows))
    ]
    density = float(np.mean([g.mean() for g in grids.values()]))
    return GrooveResult(float(np.mean(pair_scores)), density, len(valid_windows), n_windows)


def groove_similarity_piece_bars(
    pm, downbeats, *, positions_per_bar: int = 64, include_drums: bool = True,
) -> GrooveResult:
    """Bar-based variant of groove_similarity_piece: identical XOR pairwise
    similarity, but over detected bars instead of fixed-length time windows.

    `downbeats` (seconds, ascending) come from a beat/downbeat tracker run on the
    rendered audio (src/downbeats.py). Bar k spans [downbeats[k],
    downbeats[k+1]); the pickup before the first downbeat and the open-ended span
    after the last one are partial bars and are discarded (as the fixed-window
    version discards its partial final window). Each bar becomes a binary vector of
    `positions_per_bar` equal subdivisions of that bar's own duration (Jazz
    Transformer's grooving pattern, Q = 64), so bars of different tempo stay
    comparable. GS_piece = mean over unordered pairs of non-empty bars of
    1 - mean(XOR); needs >= 2 non-empty bars. GrooveResult's *_windows fields
    count bars here."""
    downbeats = np.asarray(downbeats, dtype=float)
    n_bars = max(len(downbeats) - 1, 0)
    if n_bars == 0:
        return GrooveResult(float("nan"), float("nan"), 0, 0, "fewer than 2 detected downbeats")

    times = np.array([o.time for o in note_onsets(pm, include_drums=include_drums)])
    grids = np.zeros((n_bars, positions_per_bar), dtype=bool)
    if len(times):
        bar = np.searchsorted(downbeats, times, side="right") - 1
        inside = (bar >= 0) & (bar < n_bars)
        bar, t = bar[inside], times[inside]
        start, length = downbeats[bar], downbeats[bar + 1] - downbeats[bar]
        # +1e-6: an onset exactly on a subdivision (common in quantized MIDI) must not
        # float-round down into the previous position
        pos = np.clip(np.floor((t - start) / length * positions_per_bar + 1e-6).astype(int),
                      0, positions_per_bar - 1)
        grids[bar, pos] = True  # multiple onsets at one position still map to a single 1

    valid = grids[grids.any(axis=1)]
    if len(valid) < 2:
        density = float(valid.mean()) if len(valid) else float("nan")
        return GrooveResult(float("nan"), density, len(valid), n_bars,
                            "fewer than 2 valid (non-empty) bars")
    xor = (valid[:, None, :] ^ valid[None, :, :]).mean(axis=2)
    iu = np.triu_indices(len(valid), k=1)
    return GrooveResult(float(np.mean(1.0 - xor[iu])), float(valid.mean()), len(valid), n_bars)


def histogram_probs(values, bin_edges: np.ndarray) -> np.ndarray:
    """Normalized histogram (probabilities) of `values` over fixed `bin_edges`.
    All-zero if `values` is empty (no valid pieces) -- callers must check this
    before treating the result as a probability distribution."""
    counts, _ = np.histogram(np.asarray(list(values), dtype=float), bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return counts.astype(float)
    return counts / total


def js_distance(gen_values, real_values, bin_edges: np.ndarray) -> tuple[float, str | None]:
    """Jensen-Shannon distance (base-2 log, bounded in [0, 1]) between the
    dataset-level histograms of `gen_values` and `real_values`, built with
    identical `bin_edges`. Returns (nan, reason) if either set has no valid
    piece-level values to build a histogram from."""
    p = histogram_probs(gen_values, bin_edges)
    q = histogram_probs(real_values, bin_edges)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan"), "no valid piece-level scores in generated and/or real set"
    from scipy.spatial.distance import jensenshannon
    return float(jensenshannon(p, q, base=2.0)), None


def chroma_bin_edges(n_bins: int = 20) -> np.ndarray:
    return np.linspace(0.0, CHROMA_MAX_ENTROPY, n_bins + 1)


def groove_bin_edges(n_bins: int = 20) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


# --- small resumable-CSV helpers shared by metrics/chroma.py and metrics/groove.py ---

def load_csv_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_csv_row(path: str, fieldnames: list[str], row: dict) -> None:
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
