"""
Encode→decode a MIDI through BPE and write WAVs for listening.

Usage:
    python bpe/scripts/roundtrip.py --midi foo.mid
"""

import argparse
import glob
import json
import os
import shutil
import sys
import wave

import numpy as np

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BPE_ROOT    = os.path.dirname(_SCRIPTS_DIR)

sys.path.insert(0, os.path.join(_BPE_ROOT, "src"))

import pretty_midi
from tokenizers import Tokenizer

from bpe_utils   import (
    serialize_midi, serialize_midi_anticipation,
    load_mapping, load_mapping_doubled, is_mapping_doubled,
    decode_bpe_ids,
)
from midi_to_amt import amt_to_midi
from anticipation_utils import SEPARATOR, REST

_DEFAULT_QUANT_OUT        = os.path.join(_BPE_ROOT, "tokenizers", "quantization_sweep", "merges-8192")
_DEFAULT_ANTICIPATION_OUT = os.path.join(_BPE_ROOT, "tokenizers", "anticipation")
_UNIT = {"onset": "ms", "duration": "ms", "velocity": "bins"}



_SOUNDFONT_SEARCH_PATHS = [
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


def _find_soundfont() -> str | None:
    for path in _SOUNDFONT_SEARCH_PATHS:
        if path and os.path.exists(path):
            return path
    return None


def _try_fluidsynth_import() -> bool:
    try:
        import fluidsynth as _fs  # noqa: F401
        return True
    except Exception:
        return False


def _synth_fluidsynth(pm: pretty_midi.PrettyMIDI,
                      sf2_path: str | None, fs: int) -> np.ndarray:
    return pm.fluidsynth(fs=fs, sf2_path=sf2_path)


def _synth_numpy(pm: pretty_midi.PrettyMIDI, fs: int) -> np.ndarray:
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
            i0          = int(note.start * fs)
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


def _write_wav(path: str, audio: np.ndarray, fs: int) -> None:
    audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_i16.tobytes())


def _synth_and_write(pm: pretty_midi.PrettyMIDI, out_path: str,
                     sf2_path: str | None, fs: int, use_fluid: bool) -> None:
    if use_fluid:
        audio = _synth_fluidsynth(pm, sf2_path, fs)
    else:
        audio = _synth_numpy(pm, fs)
    _write_wav(out_path, audio, fs)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BPE roundtrip: encode + decode a MIDI at each quantization level, output WAV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--midi",         required=True,  help="Input MIDI file")
    p.add_argument("--factor",       choices=["onset", "duration", "velocity"], default=None)
    p.add_argument("--value",        type=float,     default=None)
    p.add_argument("--soundfont",    default=None,
                   help="Path to .sf2 soundfont. Auto-discovered from ~/FluidR3_GM.sf2, "
                        "/usr/share/sounds/sf2/, or $MIDI_SOUNDFONT. "
                        "Recommended: Fluid R3 GM (musical-artifacts.com/artifacts/738).")
    p.add_argument("--sample-rate",  type=int,       default=44100)
    p.add_argument("--clip",         type=float, default=10.0,
                   help="Clip all outputs to this many seconds (default: 10)")
    p.add_argument("--numpy-synth",  action="store_true",
                   help="Force numpy sine synthesizer even if FluidSynth is available")
    p.add_argument("--quant-out",    default=_DEFAULT_QUANT_OUT)
    p.add_argument("--out-dir",      default=None)
    p.add_argument("--anticipation", action="store_true",
                   help="Use anticipation-trained tokenizer from tokenizers/anticipation/. "
                        "Control tokens are stripped; notes reconstruct from true onset.")
    p.add_argument("--anticipation-out", default=_DEFAULT_ANTICIPATION_OUT,
                   help="Path to anticipation tokenizers dir (default: tokenizers/anticipation/)")
    return p.parse_args()


def _trim_midi(midi_path: str, max_sec: float) -> str:
    """Copy midi_path, dropping notes that start at or after max_sec, to a temp file."""
    import tempfile
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = pretty_midi.PrettyMIDI(midi_path)
    for inst in pm.instruments:
        inst.notes = [n for n in inst.notes if n.start < max_sec]
        for n in inst.notes:
            n.end = min(n.end, max_sec)
    fd, tmp = tempfile.mkstemp(suffix=".mid")
    os.close(fd)
    pm.write(tmp)
    return tmp


def _find_tokenizer(exp_dir: str) -> str | None:
    hits = glob.glob(os.path.join(exp_dir, "tokenizers", "*.json"))
    return hits[0] if hits else None


def _value_label(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)


def _roundtrip_one(midi_path, exp_dir, factor, value, out_dir,
                   sf2_path, fs, use_fluid) -> None:
    """Baseline (non-anticipation) roundtrip for one quantization config."""
    config_path = os.path.join(exp_dir, "train_config.json")
    tok_path    = _find_tokenizer(exp_dir)
    map_path    = os.path.join(exp_dir, "mappings", "amt_base_token_char_mapping.json")
    label       = f"{factor}={_value_label(value)}{_UNIT[factor]}"

    missing = None
    if not os.path.exists(config_path):
        missing = "train_config.json"
    elif tok_path is None:
        missing = "tokenizer"
    elif not os.path.exists(map_path):
        missing = "mapping"
    if missing:
        print(f"  [skip] {label}: {missing} not found")
        return

    with open(config_path) as f:
        cfg = json.load(f)
    onset_ms = cfg["onset_ms"]
    dur_ms   = cfg["dur_ms"]
    vel_bins = cfg["vel_bins"]

    b2c, c2b  = load_mapping(map_path)
    tokenizer = Tokenizer.from_file(tok_path)

    texts = serialize_midi(midi_path, b2c,
                           onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins)
    if not texts:
        print(f"  [fail]  {label}: MIDI could not be serialized with this vocab")
        return
    text = texts[0]  # roundtrip uses a short clip, so one chunk

    enc   = tokenizer.encode(text)
    notes = decode_bpe_ids(enc, tokenizer, c2b)
    pm    = amt_to_midi(notes, onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins)

    if not pm.instruments:
        print(f"  [empty] {label}: no notes decoded")
        return

    out_path = os.path.join(out_dir, f"{factor}_{_value_label(value)}{_UNIT[factor]}.wav")
    _synth_and_write(pm, out_path, sf2_path, fs, use_fluid)

    n_notes = sum(len(i.notes) for i in pm.instruments)
    n_orig  = len(text.split())
    print(f"  {label:<22s}  {n_orig:>6} notes → {n_notes:>6} notes   {out_path}")


def _roundtrip_anticipation(midi_path: str, ant_out_dir: str,
                             out_dir: str, sf2_path: str | None,
                             fs: int, use_fluid: bool) -> None:
    """Anticipation roundtrip: encode with pass k=1, strip ctrl tokens, reconstruct from true onset."""
    config_path = os.path.join(ant_out_dir, "train_config.json")
    tok_path    = _find_tokenizer(ant_out_dir)
    map_path    = os.path.join(ant_out_dir, "mappings", "amt_base_token_char_mapping.json")

    for path, name in [(config_path, "train_config.json"),
                       (map_path, "mapping")]:
        if not os.path.exists(path):
            print(f"  [skip] anticipation: {name} not found at {path}")
            return
    if tok_path is None:
        print(f"  [skip] anticipation: no tokenizer found in {ant_out_dir}/tokenizers/")
        return
    if not is_mapping_doubled(map_path):
        print(f"  [skip] anticipation: mapping at {map_path} is not doubled (run double_tokenizer.py)")
        return

    with open(config_path) as f:
        cfg = json.load(f)
    onset_ms = cfg["onset_ms"]
    dur_ms   = cfg["dur_ms"]
    vel_bins = cfg["vel_bins"]

    b2c, c2b, ctrl_b2c, ctrl_c2b = load_mapping_doubled(map_path)
    tokenizer = Tokenizer.from_file(tok_path)

    import numpy as _np
    rng = _np.random.default_rng(0)
    # k=1: span-augmented (more interesting than k=0 which has no ctrl tokens)
    texts = serialize_midi_anticipation(midi_path, b2c, ctrl_b2c, k=1, rng=rng,
                                        onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins)
    if not texts:
        print(f"  [fail]  anticipation: MIDI could not be serialized")
        return
    text = texts[0]  # roundtrip uses a short clip, so one chunk

    enc   = tokenizer.encode(text)
    notes = decode_bpe_ids(enc, tokenizer, c2b,
                           ctrl_char_to_base_token=ctrl_c2b,
                           strip_control=True)
    pm = amt_to_midi(notes, onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins)

    if not pm.instruments:
        print(f"  [empty] anticipation: no notes decoded after stripping ctrl tokens")
        return

    out_path = os.path.join(out_dir, "anticipation_roundtrip.wav")
    _synth_and_write(pm, out_path, sf2_path, fs, use_fluid)

    n_notes = sum(len(i.notes) for i in pm.instruments)
    n_words = len(text.split())
    n_ctrl  = sum(1 for w in text.split()
                  if w not in (SEPARATOR, REST) and len(w) == 5
                  and any(c in ctrl_b2c.values() for c in w))
    print(f"  anticipation              "
          f"{n_words:>6} words ({n_ctrl} ctrl) → {n_notes:>6} notes   {out_path}")


def main() -> None:
    args = parse_args()

    midi_path = os.path.abspath(args.midi)
    if not os.path.exists(midi_path):
        print(f"ERROR: {midi_path} not found", file=sys.stderr)
        sys.exit(1)

    midi_stem = os.path.splitext(os.path.basename(midi_path))[0]
    out_dir   = args.out_dir or os.path.join(
        os.path.dirname(args.quant_out), "roundtrip", midi_stem,
    )
    os.makedirs(out_dir, exist_ok=True)

    if args.numpy_synth:
        use_fluid = False
        print("  Synthesizer: numpy sine (--numpy-synth; no timbre differentiation)")
    elif not _try_fluidsynth_import():
        print(
            "ERROR: FluidSynth is not available.\n"
            "  Install the system library:  conda install -c conda-forge fluidsynth\n"
            "                           or: sudo apt-get install -y fluidsynth libfluidsynth-dev\n"
            "  Install the Python binding:  pip install pyfluidsynth\n"
            "  Then download Fluid R3 GM:   https://musical-artifacts.com/artifacts/738\n"
            "  Or use --numpy-synth for a basic sinewave fallback (no timbre).",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        use_fluid = True

    sf2_path = args.soundfont or _find_soundfont()
    fs       = args.sample_rate

    if use_fluid:
        if sf2_path:
            print(f"  Synthesizer: FluidSynth  soundfont={sf2_path}")
        else:
            print(
                "  Synthesizer: FluidSynth  (no soundfont found — using pretty_midi's "
                "bundled TimGM6mb.sf2; for better timbre download Fluid R3 GM from "
                "musical-artifacts.com/artifacts/738 and pass --soundfont or set $MIDI_SOUNDFONT)"
            )

    clip_sec  = args.clip
    trim_path = _trim_midi(midi_path, clip_sec)
    print(f"  (all outputs clipped to {clip_sec}s)")

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pm_orig = pretty_midi.PrettyMIDI(trim_path)
        orig_out = os.path.join(out_dir, "original.wav")
        _synth_and_write(pm_orig, orig_out, sf2_path, fs, use_fluid)
        print(f"  original               → {orig_out}")

        if args.anticipation:
            print(f"\n── anticipation ──")
            _roundtrip_anticipation(trim_path, args.anticipation_out,
                                    out_dir, sf2_path, fs, use_fluid)
        else:
            factors = (["onset", "duration", "velocity"]
                       if args.factor is None else [args.factor])

            for factor in factors:
                factor_dir = os.path.join(args.quant_out, factor)
                if not os.path.isdir(factor_dir):
                    print(f"\n[skip] no dir for factor '{factor}'")
                    continue
                print(f"\n── {factor} ──")
                for name in sorted(os.listdir(factor_dir)):
                    exp_dir = os.path.join(factor_dir, name)
                    if not os.path.isdir(exp_dir):
                        continue
                    try:
                        value = float(name)
                    except ValueError:
                        continue
                    if args.value is not None and value != args.value:
                        continue
                    _roundtrip_one(trim_path, exp_dir, factor, value, out_dir,
                                   sf2_path, fs, use_fluid)
    finally:
        os.unlink(trim_path)

    print(f"\nOutput dir: {out_dir}")


if __name__ == "__main__":
    main()
