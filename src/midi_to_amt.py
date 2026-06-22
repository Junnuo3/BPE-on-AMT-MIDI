# Parse MIDI → AMT 5-token compound notes (time, duration, pitch, instrument, velocity).
# onset_ms / dur_ms: quantization step in ms. vel_bins < 128 re-quantizes velocity linearly.
# Drums map to instrument 128.

import pretty_midi
from collections import defaultdict

_MAX_ONSET_SEC = 100   # clip onsets beyond 100 s
_MAX_DUR_SEC   = 10    # clip durations beyond 10 s


def midi_to_amt(
    midi_path: str,
    onset_ms: float = 10.0,
    dur_ms:   float = 10.0,
    vel_bins: int   = 128,
) -> list[list[str]]:
    """Return AMT compound notes sorted by (onset, pitch, instrument)."""
    onset_scale = 1000.0 / onset_ms   # steps per second
    dur_scale   = 1000.0 / dur_ms

    max_onset = round(_MAX_ONSET_SEC * onset_scale)
    max_dur   = round(_MAX_DUR_SEC   * dur_scale)

    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return []

    raw: list[tuple[int, int, int, int, int]] = []
    for inst in pm.instruments:
        iid = 128 if inst.is_drum else inst.program
        for note in inst.notes:
            t = min(round(note.start * onset_scale), max_onset - 1)
            d = max(1, min(round((note.end - note.start) * dur_scale), max_dur))
            v = round(note.velocity * (vel_bins - 1) / 127) if vel_bins < 128 else note.velocity
            raw.append((t, d, note.pitch, iid, v))

    raw.sort(key=lambda n: (n[0], n[2], n[3]))

    return [
        [f"<time-{t}>", f"<duration-{d}>", f"<pitch-{p}>",
         f"<instrument-{i}>", f"<velocity-{v}>"]
        for t, d, p, i, v in raw
    ]


def amt_to_midi(
    notes: list[list[str]],
    onset_ms: float = 10.0,
    dur_ms:   float = 10.0,
    vel_bins: int   = 128,
) -> pretty_midi.PrettyMIDI:
    """Reconstruct a PrettyMIDI object from AMT compound note token lists. Malformed notes are skipped."""
    onset_scale = onset_ms / 1000.0
    dur_scale   = dur_ms   / 1000.0

    by_instrument: dict[int, list[tuple[float, float, int, int]]] = defaultdict(list)

    for note_tokens in notes:
        if len(note_tokens) != 5:
            continue
        try:
            t = int(note_tokens[0][6:-1])    # <time-T>
            d = int(note_tokens[1][10:-1])   # <duration-D>
            p = int(note_tokens[2][7:-1])    # <pitch-P>
            i = int(note_tokens[3][12:-1])   # <instrument-I>
            v = int(note_tokens[4][10:-1])   # <velocity-V>
        except (ValueError, IndexError):
            continue

        start = t * onset_scale
        end   = start + d * dur_scale

        if vel_bins < 128:
            v = round(v * 127 / (vel_bins - 1))
        v = max(1, min(127, v))
        p = max(0, min(127, p))

        by_instrument[i].append((start, end, p, v))

    pm = pretty_midi.PrettyMIDI()
    for iid, note_list in sorted(by_instrument.items()):
        if iid == 128:
            inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
        else:
            inst = pretty_midi.Instrument(program=min(iid, 127),
                                          name=f"Program {iid}")
        for start, end, pitch, vel in note_list:
            inst.notes.append(pretty_midi.Note(
                velocity=vel, pitch=pitch, start=start, end=end,
            ))
        pm.instruments.append(inst)

    return pm
