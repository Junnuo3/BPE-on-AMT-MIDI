"""
Render a pretty_midi.PrettyMIDI to a WAV file, via FluidSynth (with a
soundfont) or a dependency-free numpy sine-wave fallback.

Shared by scripts/roundtrip.py and scripts/generate_smollm2.py.
"""

import os
import wave

import numpy as np
import pretty_midi

SOUNDFONT_SEARCH_PATHS = [
    # env var override
    os.environ.get("MIDI_SOUNDFONT", ""),
    # home directory (typical manual download location)
    os.path.expanduser("~/FluidR3_GM.sf2"),
    os.path.expanduser("~/FluidR3_GM2-2.sf2"),
    # common Linux system locations
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/FluidR3_GM2-2.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM2-2.sf2",
]


def find_soundfont() -> str | None:
    for path in SOUNDFONT_SEARCH_PATHS:
        if path and os.path.exists(path):
            return path
    return None


def try_fluidsynth_import() -> bool:
    try:
        import fluidsynth as _fs  # noqa: F401
        return True
    except Exception:
        return False


def synth_fluidsynth(pm: pretty_midi.PrettyMIDI,
                     sf2_path: str | None, fs: int) -> np.ndarray:
    return pm.fluidsynth(fs=fs, sf2_path=sf2_path)


def synth_numpy(pm: pretty_midi.PrettyMIDI, fs: int) -> np.ndarray:
    """Simple additive sine synth — no system deps needed."""
    end_time = max(
        (note.end for inst in pm.instruments for note in inst.notes),
        default=0.0,
    )
    if end_time == 0.0:
        return np.zeros(fs, dtype=np.float64)

    n_samples = int(np.ceil(end_time * fs)) + fs  # 1-s tail
    audio = np.zeros(n_samples, dtype=np.float64)

    for inst in pm.instruments:
        for note in inst.notes:
            freq        = 440.0 * 2 ** ((note.pitch - 69) / 12.0)
            # clamp to 0: a negative i0 would make audio[i0:i1] a Python
            # negative-index slice (wraps from the end) instead of an
            # out-of-range one, producing a length that silently disagrees
            # with n = i1 - i0 and breaking the += below.
            i0          = max(0, int(note.start * fs))
            i1          = min(int(note.end * fs), n_samples)
            n           = i1 - i0
            if n <= 0:
                continue
            t           = np.arange(n) / fs
            env         = np.ones(n)
            atk         = min(int(0.01 * fs), n // 4)
            rel         = min(int(0.05 * fs), n // 4)
            if atk > 0:
                env[:atk]  = np.linspace(0.0, 1.0, atk)
            if rel > 0:
                env[-rel:] *= np.linspace(1.0, 0.0, rel)
            audio[i0:i1] += (np.sin(2 * np.pi * freq * t)
                             * env * (note.velocity / 127.0) * 0.3)

    peak = np.abs(audio).max()
    if peak > 0:
        audio *= 0.9 / peak
    return audio


def write_wav(path: str, audio: np.ndarray, fs: int) -> None:
    audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_i16.tobytes())


def synth_and_write(pm: pretty_midi.PrettyMIDI, out_path: str,
                    sf2_path: str | None, fs: int, use_fluid: bool) -> None:
    if use_fluid:
        try:
            audio = synth_fluidsynth(pm, sf2_path, fs)
        except ValueError:
            # pretty_midi's PrettyMIDI.fluidsynth() sums each instrument's
            # rendered array into one fixed-length buffer sized from the
            # piece's overall end time; an instrument whose own render comes
            # out a different length (seen e.g. on some no_bpe-decoded MIDI --
            # empty/near-empty instruments render length 0) raises a broadcast
            # ValueError and would otherwise lose this sample entirely. Fall
            # back to the numpy synth so callers still get *some* audio.
            audio = synth_numpy(pm, fs)
    else:
        audio = synth_numpy(pm, fs)
    write_wav(out_path, audio, fs)
