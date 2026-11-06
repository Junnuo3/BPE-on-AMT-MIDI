"""Note-level decoding for the released checkpoints.

A note is five AMT base tokens (onset, duration, pitch, instrument, velocity); each
base token is one private-use character, and one BPE token covers a contiguous run of
fields within a note. Generation is constrained by the note grammar, and generated
tokens are decoded with the strict decoder; both are what the reported numbers use.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import pretty_midi

MAPPING_PATH = os.path.join(os.path.dirname(__file__), "assets", "amt_base_token_char_mapping.json")
ONSET_MS, DUR_MS, VEL_BINS = 10.0, 10.0, 32


def load_mapping(path: str = MAPPING_PATH) -> tuple[dict[str, str], dict[str, str]]:
    """char -> base token, for event notes and for control (anticipated) notes."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["char_to_base_token"], data["ctrl_char_to_base_token"]

FIELDS = ("time", "duration", "pitch", "instrument", "velocity")
SENTINELS = {"[REST]", "[SEPARATOR]", "[AUTOREGRESS]", "[ANTICIPATE]",
             "[UNK]", "[PAD]", "[BOS]", "[EOS]"}


def field_index(base_token: str) -> int | None:
    """'<pitch-60>' -> 2; None for anything that is not a note field."""
    if not (base_token.startswith("<") and base_token.endswith(">") and "-" in base_token):
        return None
    name = base_token[1:base_token.index("-")]
    return FIELDS.index(name) if name in FIELDS else None


def decode_notes(
    token_strings: list[str],
    char_to_base_token: dict[str, str],
    ctrl_char_to_base_token: dict[str, str] | None = None,
    strip_control: bool = True,
    stats: dict | None = None,
) -> list[list[str]]:
    """Decode generated token strings into complete 5-field notes.

    Any note whose fields arrive out of order, mix control and event chars, or are
    interrupted by a sentinel/text token is dropped and counted in stats
    ["malformed_notes"]; decoding resumes at the next time field, so one bad token
    can't shift every later note. stats also gets "notes", "control_notes",
    "skipped_chars" (chars that could not start or continue a note) and
    "truncated_tail_notes" (a final note cut off by the generation length limit).
    """
    ctrl_map = ctrl_char_to_base_token or {}
    st = {"notes": 0, "control_notes": 0, "malformed_notes": 0, "skipped_chars": 0,
          "truncated_tail_notes": 0}
    result: list[list[str]] = []
    buf: list[str] = []
    buf_ctrl = False

    def abandon() -> None:
        nonlocal buf
        if buf:
            st["malformed_notes"] += 1
        buf = []

    for tok in token_strings:
        if tok in SENTINELS:
            abandon()
            continue
        for c in tok:
            is_ctrl = c in ctrl_map
            base = ctrl_map[c] if is_ctrl else char_to_base_token.get(c)
            f = field_index(base) if base is not None else None
            if f is None:
                abandon()
                st["skipped_chars"] += 1
                continue
            if f != len(buf) or (buf and is_ctrl != buf_ctrl):
                abandon()
                if f != 0:
                    st["skipped_chars"] += 1
                    continue
            buf.append(base)
            buf_ctrl = is_ctrl
            if len(buf) == 5:
                if buf_ctrl:
                    st["control_notes"] += 1
                if not (buf_ctrl and strip_control):
                    result.append(buf)
                    st["notes"] += 1
                buf = []
    if buf:  # a note cut off by the end of generation, not a malformed one
        st["truncated_tail_notes"] += 1
    if stats is not None:
        for k, v in st.items():
            stats[k] = stats.get(k, 0) + v
    return result


def token_field_runs(vocab: dict[str, int], char_to_base_token: dict[str, str]) -> dict[int, tuple[int, int]]:
    """token id -> (first field, n fields) for every event-note token with a contiguous
    field run; tokens with control chars, unknown chars or broken runs are absent."""
    runs = {}
    for tok, i in vocab.items():
        if not tok or tok in SENTINELS:
            continue
        fields = []
        for c in tok:
            base = char_to_base_token.get(c)
            f = field_index(base) if base is not None else None
            if f is None:
                break
            fields.append(f)
        else:
            if fields == list(range(fields[0], fields[0] + len(fields))) and fields[-1] < 5:
                runs[i] = (fields[0], len(fields))
    return runs


class MidiGrammarLogitsProcessor:
    """transformers LogitsProcessor: mask every token that can't continue the
    current note. Tracks each row's position within its note (0-4) from the
    tokens generated since the first call. HF applies custom processors before
    temperature/top-k/top-p, so sampling only ever sees permitted tokens.
    """

    def __init__(self, hf_tokenizer, char_to_base_token: dict[str, str], *,
                 end_id: int | None, n_scores: int | None = None) -> None:
        import torch

        vocab = hf_tokenizer.get_vocab()
        size = max(n_scores or 0, max(vocab.values()) + 1)
        allowed = torch.zeros(5, size, dtype=torch.bool)
        advance = torch.zeros(size, dtype=torch.long)
        for i, (start, n) in token_field_runs(vocab, char_to_base_token).items():
            allowed[start, i] = True
            advance[i] = n
        rest = vocab.get("[REST]")
        if rest is not None:
            allowed[0, rest] = True
        if end_id is not None:
            allowed[0, end_id] = True
        self.allowed = allowed
        self.advance = advance
        self._start_len: int | None = None
        self._pos = None

    def __call__(self, input_ids, scores):
        if self._start_len is None or self._pos is None or self._pos.shape[0] != input_ids.shape[0]:
            self._start_len = input_ids.shape[1]
            self._pos = input_ids.new_zeros(input_ids.shape[0])
            self.allowed = self.allowed.to(scores.device)
            self.advance = self.advance.to(scores.device)
        elif input_ids.shape[1] > self._start_len:
            self._pos = (self._pos + self.advance[input_ids[:, -1]]) % 5
        mask = self.allowed[self._pos][:, :scores.shape[1]]
        return scores.masked_fill(~mask, float("-inf"))


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
