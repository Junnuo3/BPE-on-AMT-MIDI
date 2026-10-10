"""
Beat/downbeat tracking for the bar-based groove metric (metrics/groove_bar.py).

Uses Beat This! (Foscarin et al., ISMIR 2024; `pip install beat_this`, which
needs torchaudio) on rendered audio, checkpoint "final0", without the optional
DBN post-processing (the authors' recommended setting -- it avoids imposing a
fixed meter/tempo prior). The checkpoint downloads once to the torch hub cache.
MidiCaps/Lakh have no reliable beat/bar annotations and decoded generations carry none (the tokens are time-based), so bars are
detected from audio, for real and generated pieces alike.
"""

from __future__ import annotations

import numpy as np

CHECKPOINT = "final0"


class DownbeatTracker:
    def __init__(self, device: str | None = None, checkpoint: str = CHECKPOINT):
        self.device = device or "cpu"
        self.checkpoint = checkpoint
        self._f2b = None

    def __call__(self, wav_path: str) -> np.ndarray:
        """Downbeat times (seconds, ascending) for an audio file."""
        if self._f2b is None:
            from beat_this.inference import File2Beats

            self._f2b = File2Beats(checkpoint_path=self.checkpoint, device=self.device, dbn=False)
        _beats, downbeats = self._f2b(wav_path)
        return np.sort(np.asarray(downbeats, dtype=float))
