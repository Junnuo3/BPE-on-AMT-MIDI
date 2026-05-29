# Mapping, corpus serialization, and caching for BPE training.
# Cache layout (outputs/cache/):
#   filelist.txt / filelist_meta.json  — scanned MIDI paths
#   vocab_scan.json                    — incremental vocab scan checkpoint
#   corpus.txt / corpus_meta.json      — serialized Unicode strings

import hashlib
import json
import os
import warnings
from typing import Iterable, Iterator

from tqdm import tqdm

from midi_to_amt import midi_to_amt

_PUA_RANGES: list[tuple[int, int]] = [
    (0xE000,  0xF8FF),
    (0xF0000, 0xFFFFF),
    (0x100000, 0x10FFFF),
]


def _index_to_pua_char(idx: int) -> str:
    for start, end in _PUA_RANGES:
        size = end - start + 1
        if idx < size:
            return chr(start + idx)
        idx -= size
    raise ValueError(f"Index {idx} exceeds available PUA code points")


_CORPUS_FILE   = "corpus.txt"
_CORPUS_META   = "corpus_meta.json"
_FILELIST_FILE = "filelist.txt"
_FILELIST_META = "filelist_meta.json"
_VOCAB_CKPT    = "vocab_scan.json"

VOCAB_CKPT_INTERVAL   = 500
CORPUS_FLUSH_INTERVAL = 200


def find_midi_files(dataset_dir: str) -> list[str]:
    paths: list[str] = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith((".mid", ".midi")):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def compute_corpus_fingerprint(midi_files: list[str], time_resolution: int) -> str:
    h = hashlib.sha1()
    h.update(f"time_resolution={time_resolution}\n".encode())
    for path in midi_files:
        try:
            mtime = f"{os.path.getmtime(path):.3f}"
        except OSError:
            mtime = "missing"
        h.update(f"{path}:{mtime}\n".encode())
    return h.hexdigest()


def save_filelist_cache(
    midi_files: list[str], cache_dir: str,
    limit_files: int | None, time_resolution: int,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, _FILELIST_FILE), "w") as f:
        f.write("\n".join(midi_files))
    with open(os.path.join(cache_dir, _FILELIST_META), "w") as f:
        json.dump({"num_files": len(midi_files),
                   "limit_files": limit_files,
                   "time_resolution": time_resolution}, f, indent=2)


def load_filelist_cache(
    cache_dir: str, limit_files: int | None, time_resolution: int,
) -> list[str] | None:
    fl   = os.path.join(cache_dir, _FILELIST_FILE)
    meta = os.path.join(cache_dir, _FILELIST_META)
    if not (os.path.exists(fl) and os.path.exists(meta)):
        return None
    with open(meta) as f:
        m = json.load(f)
    if m.get("limit_files") != limit_files or m.get("time_resolution") != time_resolution:
        return None
    with open(fl) as f:
        return [line for line in (ln.strip() for ln in f) if line]


def _save_vocab_ckpt(path: str, fingerprint: str, files_processed: int,
                     observed: set[str]) -> None:
    # Write directly without tmp rename — os.replace can silently no-op on overlay fs
    with open(path, "w") as f:
        json.dump({"fingerprint": fingerprint,
                   "files_processed": files_processed,
                   "observed_tokens": sorted(observed)}, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def _load_vocab_ckpt(path: str, fingerprint: str) -> tuple[int, set[str]] | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        ckpt = json.load(f)
    if ckpt.get("fingerprint") != fingerprint:
        return None
    return ckpt["files_processed"], set(ckpt["observed_tokens"])


def build_mapping_resumable(
    midi_files: list[str],
    fingerprint: str,
    cache_dir: str,
    checkpoint_interval: int = VOCAB_CKPT_INTERVAL,
) -> tuple[dict[str, str], dict[str, str]]:
    """Scan all MIDI files to find every AMT base token, map each to a PUA char."""
    os.makedirs(cache_dir, exist_ok=True)
    ckpt_path = os.path.join(cache_dir, _VOCAB_CKPT)

    start_idx: int  = 0
    observed: set[str] = set()

    result = _load_vocab_ckpt(ckpt_path, fingerprint)
    if result is not None:
        start_idx, observed = result
        tqdm.write(f"  Resuming vocab scan from {start_idx:,} / {len(midi_files):,}")

    remaining = midi_files[start_idx:]
    with tqdm(remaining, initial=start_idx, total=len(midi_files),
              desc="  vocab scan", unit="file", leave=False) as bar:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for rel_idx, path in enumerate(bar):
                for note in midi_to_amt(path):
                    observed.update(note)
                abs_idx = start_idx + rel_idx + 1
                if abs_idx % checkpoint_interval == 0:
                    _save_vocab_ckpt(ckpt_path, fingerprint, abs_idx, observed)

    _save_vocab_ckpt(ckpt_path, fingerprint, len(midi_files), observed)

    sorted_tokens = sorted(observed)
    b2c = {tok: _index_to_pua_char(i) for i, tok in enumerate(sorted_tokens)}
    c2b = {v: k for k, v in b2c.items()}
    return b2c, c2b


def save_mapping(
    base_token_to_char: dict[str, str],
    char_to_base_token: dict[str, str],
    path: str,
    fingerprint: str = "",
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint,
                   "base_token_to_char": base_token_to_char,
                   "char_to_base_token": char_to_base_token},
                  f, ensure_ascii=True, indent=2)


def load_mapping(path: str) -> tuple[dict[str, str], dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["base_token_to_char"], data["char_to_base_token"]


def load_mapping_if_fresh(
    path: str, fingerprint: str,
) -> tuple[dict[str, str], dict[str, str]] | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("fingerprint") != fingerprint:
        return None
    return data["base_token_to_char"], data["char_to_base_token"]


def _write_corpus_meta(
    meta_path: str, fingerprint: str, files_processed: int,
    lines_written: int, num_files_total: int, status: str,
) -> None:
    with open(meta_path, "w") as f:
        json.dump({"fingerprint": fingerprint, "status": status,
                   "files_processed": files_processed,
                   "num_files_scanned": num_files_total,
                   "num_files_cached": lines_written}, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def serialize_midi_resumable(
    midi_files: list[str],
    base_token_to_char: dict[str, str],
    fingerprint: str,
    cache_dir: str,
    flush_interval: int = CORPUS_FLUSH_INTERVAL,
) -> list[str]:
    """Serialize MIDI files to Unicode strings, writing to disk incrementally."""
    os.makedirs(cache_dir, exist_ok=True)
    corpus_path = os.path.join(cache_dir, _CORPUS_FILE)
    meta_path   = os.path.join(cache_dir, _CORPUS_META)

    start_idx: int   = 0
    corpus: list[str] = []

    if os.path.exists(meta_path) and os.path.exists(corpus_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if (meta.get("fingerprint") == fingerprint
                and meta.get("status") == "in_progress"):
            start_idx = meta["files_processed"]
            with open(corpus_path, encoding="utf-8") as f:
                corpus = [ln.rstrip("\n") for ln in f if ln.strip()]
            tqdm.write(f"  Resuming corpus from {start_idx:,} / {len(midi_files):,} "
                       f"({len(corpus):,} lines cached)")

    if start_idx == 0:
        open(corpus_path, "w", encoding="utf-8").close()

    _write_corpus_meta(meta_path, fingerprint, start_idx,
                       len(corpus), len(midi_files), "in_progress")

    remaining = midi_files[start_idx:]
    with open(corpus_path, "a", encoding="utf-8") as f_out:
        with tqdm(remaining, initial=start_idx, total=len(midi_files),
                  desc="  serialising", unit="file", leave=False) as bar:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for rel_idx, path in enumerate(bar):
                    text = serialize_midi(path, base_token_to_char)
                    if text is not None:
                        f_out.write(text + "\n")
                        corpus.append(text)
                    abs_idx = start_idx + rel_idx + 1
                    if abs_idx % flush_interval == 0:
                        f_out.flush()
                        _write_corpus_meta(meta_path, fingerprint, abs_idx,
                                           len(corpus), len(midi_files), "in_progress")

    _write_corpus_meta(meta_path, fingerprint, len(midi_files),
                       len(corpus), len(midi_files), "complete")
    return corpus


def load_corpus_cache(fingerprint: str, cache_dir: str) -> list[str] | None:
    corpus_path = os.path.join(cache_dir, _CORPUS_FILE)
    meta_path   = os.path.join(cache_dir, _CORPUS_META)
    if not (os.path.exists(corpus_path) and os.path.exists(meta_path)):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("fingerprint") != fingerprint:
        return None
    if meta.get("status", "complete") != "complete":
        return None
    with open(corpus_path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def load_corpus_unconditional(cache_dir: str) -> list[str] | None:
    """Load corpus.txt without fingerprint check — only used when filelist cache is trusted."""
    corpus_path = os.path.join(cache_dir, _CORPUS_FILE)
    meta_path   = os.path.join(cache_dir, _CORPUS_META)
    if not (os.path.exists(corpus_path) and os.path.exists(meta_path)):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("status", "complete") != "complete":
        return None
    with open(corpus_path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def serialize_midi(
    midi_path: str,
    base_token_to_char: dict[str, str],
) -> str | None:
    """Serialize one MIDI file as space-separated 5-char note words. None if invalid."""
    notes = midi_to_amt(midi_path)
    if not notes:
        return None
    words: list[str] = []
    for note in notes:
        try:
            word = "".join(base_token_to_char[tok] for tok in note)
        except KeyError:
            return None
        words.append(word)
    return " ".join(words)


def serialize_midi_iter(
    midi_files: list[str],
    base_token_to_char: dict[str, str],
) -> Iterator[str]:
    for path in midi_files:
        text = serialize_midi(path, base_token_to_char)
        if text is not None:
            yield text


def decode_bpe_ids(
    token_ids: list[int],
    tokenizer,
    char_to_base_token: dict[str, str],
) -> list[list[str]]:
    """Decode BPE token IDs back to AMT compound notes."""
    text: str = tokenizer.decode(token_ids, skip_special_tokens=True)
    return [
        [char_to_base_token.get(c, f"<UNK:{ord(c):#x}>") for c in word]
        for word in text.split()
    ]
