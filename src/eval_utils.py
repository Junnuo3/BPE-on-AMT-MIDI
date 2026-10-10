"""
Shared synthesis + path helpers for scripts/run_evaluation.py. Metric
implementations (src/metrics/) only ever see wav paths -- this module is
the one place that knows about MIDI-to-audio rendering and where results live
on disk, so a new metric never has to duplicate synthesis/caching logic.
"""

from __future__ import annotations

import os

_SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SRC_DIR)

EVALUATION_ROOT = os.path.join(_REPO_ROOT, "evaluation")

# Ground-truth test-set audio is model-independent, so it's synthesized once
# and shared across every model_name's evaluation run instead of being
# re-rendered per model.
GROUND_TRUTH_CACHE_DIR = os.path.join(EVALUATION_ROOT, "_ground_truth")


def model_output_dir(model_name: str, metric_name: str | None = None) -> str:
    path = os.path.join(EVALUATION_ROOT, model_name)
    if metric_name is not None:
        path = os.path.join(path, metric_name)
    os.makedirs(path, exist_ok=True)
    return path


def derive_model_name(generations_dir: str, run_dir: str | None) -> str:
    """Best-effort human-readable model name from a run dir, e.g.
    outputs/runs/bpe/stage2/velocity/onset-10ms_duration-10ms_velocity-32bin
    -> bpe_stage2_velocity_onset-10ms_duration-10ms_velocity-32bin
    """
    base = run_dir or generations_dir
    base = os.path.abspath(base)
    runs_root = os.path.join(_REPO_ROOT, "outputs", "runs")
    if base.startswith(runs_root + os.sep):
        rel = os.path.relpath(base, runs_root)
    else:
        rel = os.path.basename(base.rstrip(os.sep))
    return rel.replace(os.sep, "_")


def wav_path_for(mid_path: str) -> str:
    """Rendered-wav path for a generated <tag>/<md5>/sample_XXXX.mid. hp_search
    wavs were renamed to <tag>/<md5>/<tag>_<md5>.wav so clips stay
    distinguishable once copied out of their dir; use that name when present,
    else the sibling sample_XXXX.wav.
    """
    md5_dir = os.path.dirname(mid_path)
    tag = os.path.basename(os.path.dirname(md5_dir))
    renamed = os.path.join(md5_dir, f"{tag}_{os.path.basename(md5_dir)}.wav")
    if os.path.basename(mid_path) == "sample_0000.mid" and os.path.exists(renamed):
        return renamed
    return os.path.splitext(mid_path)[0] + ".wav"


def synth_midi_cached(
    mid_path: str, wav_path: str, *,
    sf2_path: str | None, sample_rate: int, use_fluid: bool,
) -> bool:
    """Render mid_path -> wav_path via src/audio_utils.py, skipping work
    already done (resumability). Returns False (and leaves no wav behind) if
    the MIDI has no notes or synthesis raises, so callers can skip that sample
    instead of crashing the whole evaluation run.
    """
    if os.path.exists(wav_path):
        return True

    import pretty_midi

    from audio_utils import synth_and_write

    try:
        pm = pretty_midi.PrettyMIDI(mid_path)
    except Exception as e:
        print(f"  WARNING: could not parse {mid_path}: {e}")
        return False
    if not any(inst.notes for inst in pm.instruments):
        print(f"  WARNING: {mid_path} has no notes, skipping.")
        return False

    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    # Per-process suffix: concurrent evaluation runs (e.g. a SLURM array sweep)
    # can race to synthesize the same shared ground-truth wav_path at once. A
    # shared ".tmp" name let one process's os.replace() consume/clear the
    # other's tmp file out from under it (FileNotFoundError on rename).
    tmp_path = f"{wav_path}.tmp.{os.getpid()}"
    try:
        synth_and_write(pm, tmp_path, sf2_path, sample_rate, use_fluid)
    except Exception as e:
        print(f"  WARNING: synthesis failed for {mid_path}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False
    if os.path.exists(wav_path):
        # Another concurrent process already finished this file; discard ours.
        os.remove(tmp_path)
        return True
    os.replace(tmp_path, wav_path)  # atomic: a crash mid-render never leaves a fake "done" wav
    return True
