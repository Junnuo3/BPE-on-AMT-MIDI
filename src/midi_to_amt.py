# Parse MIDI → AMT 5-token compound notes (time, duration, pitch, instrument, velocity).
# Time/duration quantized at 100 bins/sec (10 ms). Drums map to instrument 128.

import pretty_midi

TIME_RESOLUTION = 100
MAX_TIME = 10_000
MAX_DUR  = 1_000


def midi_to_amt(midi_path: str) -> list[list[str]]:
    """Return AMT compound notes sorted by (onset, pitch, instrument)."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return []

    raw: list[tuple[int, int, int, int, int]] = []
    for inst in pm.instruments:
        iid = 128 if inst.is_drum else inst.program
        for note in inst.notes:
            t = min(round(TIME_RESOLUTION * note.start), MAX_TIME - 1)
            d = max(1, min(round(TIME_RESOLUTION * (note.end - note.start)), MAX_DUR))
            raw.append((t, d, note.pitch, iid, note.velocity))

    raw.sort(key=lambda n: (n[0], n[2], n[3]))

    return [
        [f"<time-{t}>", f"<duration-{d}>", f"<pitch-{p}>",
         f"<instrument-{i}>", f"<velocity-{v}>"]
        for t, d, p, i, v in raw
    ]
