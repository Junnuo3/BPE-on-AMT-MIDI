"""Shared interface + registry for evaluation metrics.

Each metric receives an EvalContext (generated + ground-truth samples, already
synthesized to wav by scripts/run_evaluation.py) and is responsible for its
own per-item resumable caching under ctx.output_dir/<metric.name>/ -- the
orchestrator doesn't know or care how a metric checkpoints its progress.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval_data import GeneratedSample, TestEntry

METRICS: dict[str, type["Metric"]] = {}


def register(name: str):
    def _wrap(cls: type["Metric"]) -> type["Metric"]:
        cls.name = name
        METRICS[name] = cls
        return cls
    return _wrap


@dataclass
class EvalContext:
    model_name: str
    generated: list["GeneratedSample"]   # .wav_path set for samples that synthesized OK
    ground_truth: list["TestEntry"]      # .wav_path set for entries that synthesized OK
    output_root: str                     # evaluation/<model_name>/
    sample_rate: int
    device: str | None = None


class Metric:
    """Subclasses set `name` via @register(...) and implement run()."""

    name: str = ""
    # Whether this metric needs ctx.generated/ctx.ground_truth's .wav_path
    # (i.e. FluidSynth-rendered audio). MIDI-only metrics (chroma, groove) set
    # this False so scripts/run_evaluation.py can skip synthesis entirely
    # when no requested metric needs it.
    needs_audio: bool = True

    def run(self, ctx: EvalContext) -> dict:  # pragma: no cover - interface
        raise NotImplementedError
