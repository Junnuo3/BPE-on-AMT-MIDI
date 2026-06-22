# Anticipation interleaving for AMT 5-token compound notes.
# Ports anticipation-main's control extraction onto the compound note representation.
#
# [REST], [SEPARATOR], [AUTOREGRESS], [ANTICIPATE] are standalone BPE words (never merged).
#
# Main entry points: build_sequence() for a full sequence, augmentation_spec() for the 10-pass schedule.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REST        = "[REST]"
SEPARATOR   = "[SEPARATOR]"
AUTOREGRESS = "[AUTOREGRESS]"
ANTICIPATE  = "[ANTICIPATE]"

SPECIAL_TOKENS = [REST, SEPARATOR, AUTOREGRESS, ANTICIPATE]

ANTICIPATION_RATES = 10   # matches anticipation-main/anticipation/tokenize.py


def _time(note: list[str]) -> int:
    """Extract the quantized onset step from a 5-token note list."""
    return int(note[0][len("<time-"):-1])


def _instrument(note: list[str]) -> int:
    """Extract the instrument id from a 5-token note list."""
    return int(note[3][len("<instrument-"):-1])



def extract_random(
    notes: list[list[str]],
    rate: int,
    rng: np.random.Generator,
) -> tuple[list[list[str]], list[list[str]]]:
    """Randomly label notes as controls with probability rate / ANTICIPATION_RATES."""
    events:   list[list[str]] = []
    controls: list[list[str]] = []
    p = rate / float(ANTICIPATION_RATES)
    for note in notes:
        (controls if rng.random() < p else events).append(note)
    return events, controls


def extract_instruments(
    notes: list[list[str]],
    instruments: set[int],
) -> tuple[list[list[str]], list[list[str]]]:
    """Notes whose instrument id is in `instruments` become controls."""
    events:   list[list[str]] = []
    controls: list[list[str]] = []
    for note in notes:
        (controls if _instrument(note) in instruments else events).append(note)
    return events, controls


def extract_spans(
    notes: list[list[str]],
    spans: list[tuple[int, int]],
) -> tuple[list[list[str]], list[list[str]]]:
    """Notes whose onset falls within any (start, end) span become controls."""
    events:   list[list[str]] = []
    controls: list[list[str]] = []
    for note in notes:
        t = _time(note)
        in_span = any(s <= t <= e for s, e in spans)
        (controls if in_span else events).append(note)
    return events, controls


def sample_spans(
    notes: list[list[str]],
    rate: float,
    delta_steps: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Generate random time spans using exponential-gap intervals.

    State machine: start out-of-span → wait Exp(rate) steps → open a delta_steps span → repeat.
    Returns (start, end) step pairs rather than partitioning inline, so they can be tested separately.
    """
    spans: list[tuple[int, int]] = []
    in_span = True          # initial state per AMT (immediately flipped at t=0)
    end_span = 0
    next_span = 0
    current_start: int | None = None

    for note in notes:
        t = _time(note)
        if in_span and t >= end_span:
            if current_start is not None:
                spans.append((current_start, end_span))
                current_start = None
            in_span = False
            next_span = t + int(rng.exponential(1.0 / rate))
        if (not in_span) and t >= next_span:
            in_span = True
            current_start = t
            end_span = t + delta_steps

    if in_span and current_start is not None:
        spans.append((current_start, end_span))

    return spans



@dataclass
class _Slot:
    """Internal working type carrying onset time alongside a note or REST sentinel."""
    time: int
    tokens: list[str]   # 5-token note list, or ["[REST]"] for a REST sentinel
    is_control: bool = False

    @property
    def is_rest(self) -> bool:
        return self.tokens == [REST]


def pad_with_rest(
    events: list[list[str]],
    end_time: int,
    density_steps: int,
) -> list[list[str]]:
    """Insert [REST] sentinels every density_steps into the event stream.

    Call this before interleave_anticipation — REST belongs in events, not controls.
    """
    out: list[list[str]] = []
    previous = 0
    for note in events:
        t = _time(note)
        while t > previous + density_steps:
            previous += density_steps
            out.append([REST])
        out.append(note)
        previous = t
    while end_time > previous + density_steps:
        previous += density_steps
        out.append([REST])
    return out


# REST slots don't carry a time field, so recover it from the density cadence.
def _to_slots(
    padded: list[list[str]],
    density_steps: int,
    is_control: bool = False,
) -> list[_Slot]:
    """Convert a padded note list to _Slot objects, reconstructing REST times from the density cadence."""
    slots: list[_Slot] = []
    previous = 0
    for tokens in padded:
        if tokens == [REST]:
            previous += density_steps
            slots.append(_Slot(time=previous, tokens=tokens, is_control=is_control))
        else:
            t = _time(tokens)
            previous = t
            slots.append(_Slot(time=t, tokens=tokens, is_control=is_control))
    return slots



def interleave_anticipation(
    events: list[list[str]],
    controls: list[list[str]],
    delta_steps: int,
    density_steps: int,
) -> list[_Slot]:
    """Splice controls delta_steps before their true onset in the event stream.

    Control notes keep their original <time-T> — only their sequence position changes.
    `events` must already be padded with REST sentinels.
    """
    if not controls:
        return _to_slots(events, density_steps)

    event_slots  = _to_slots(events, density_steps)
    control_slots = [_Slot(time=_time(n), tokens=n, is_control=True) for n in controls]

    out: list[_Slot] = []
    remaining = list(control_slots)
    event_time = 0

    for slot in event_slots:
        while remaining and event_time >= remaining[0].time - delta_steps:
            out.append(remaining.pop(0))
        event_time = slot.time
        out.append(slot)

    # controls whose window extends past the last event
    out.extend(remaining)

    return out



def add_separator(items: list[tuple[list[str], bool]]) -> list[tuple[list[str], bool]]:
    """Prepend a [SEPARATOR] sentinel.  One per file, at the beginning."""
    return [([SEPARATOR], False)] + items



def augmentation_spec(
    k: int,
    notes: list[list[str]],
    instruments: list[int],
    delta_steps: int,
    rng: np.random.Generator,
) -> tuple[str, dict]:
    """Return the augmentation (mode, kwargs) for pass k in the 10-pass schedule.

    k%10 == 0 → no control (pure autoregressive)
    k%10 == 1 → span-based control
    k%10 2..5 → random-fraction control
    k%10 6..9 → instrument-based control
    """
    m = k % 10
    if m == 0:
        return "none", {}
    if m == 1:
        spans = sample_spans(notes, rate=0.05, delta_steps=delta_steps, rng=rng)
        return "span", {"spans": spans}
    if m < 6:
        rate = int(rng.integers(1, ANTICIPATION_RATES))
        return "random", {"rate": rate}
    # instrument mode
    if len(instruments) > 1:
        u = 1 + int(rng.integers(0, len(instruments) - 1))
        subset = set(int(x) for x in rng.choice(instruments, size=u, replace=False))
        return "instrument", {"instruments": subset}
    return "none", {}



def build_sequence(
    notes: list[list[str]],
    mode: str,
    rng: np.random.Generator,
    *,
    onset_ms: float,
    delta_sec: float = 5.0,
    rest_density_sec: float = 1.0,
    **mode_kwargs,
) -> list[tuple[list[str], bool]]:
    """Build a full anticipation sequence: extract controls → pad REST → interleave → prepend SEPARATOR.

    Returns (token_list, is_control) pairs. The caller joins event notes to 5 event PUA chars,
    control notes to 5 ctrl PUA chars, and emits REST/SEPARATOR as literal strings.
    mode: "none" | "random" | "instrument" | "span"
    """
    if not notes:
        return [([SEPARATOR], False)]

    onset_steps_per_sec = 1000.0 / onset_ms
    delta_steps   = max(1, int(round(delta_sec * onset_steps_per_sec)))
    density_steps = max(1, int(round(rest_density_sec * onset_steps_per_sec)))
    end_time = _time(notes[-1])

    if mode == "none":
        events, controls = notes, []
    elif mode == "random":
        events, controls = extract_random(notes, mode_kwargs["rate"], rng)
    elif mode == "instrument":
        events, controls = extract_instruments(notes, mode_kwargs["instruments"])
    elif mode == "span":
        events, controls = extract_spans(notes, mode_kwargs["spans"])
    else:
        raise ValueError(f"unknown anticipation mode: {mode!r}")

    padded = pad_with_rest(events, end_time, density_steps)
    slots = interleave_anticipation(padded, controls, delta_steps, density_steps)

    items: list[tuple[list[str], bool]] = [([SEPARATOR], False)]
    for slot in slots:
        items.append((slot.tokens, slot.is_control))

    return items
