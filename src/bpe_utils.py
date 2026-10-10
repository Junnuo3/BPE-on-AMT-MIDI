# Mapping, corpus serialization, and caching for BPE training.
# Cache layout (<tokenizer dir>/cache/):
#   filelist.txt / filelist_meta.json  — scanned MIDI paths
#   vocab_scan.json                    — incremental vocab scan checkpoint
#   corpus.txt / corpus_meta.json      — serialized Unicode strings

import hashlib
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Iterator

import numpy as np
from tqdm import tqdm

from midi_to_amt import midi_to_amt
from anticipation_utils import build_sequence, augmentation_spec, REST, SEPARATOR


def _default_workers() -> int:
    """CPUs allocated to this SLURM task, or all available CPUs."""
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        return int(slurm)
    return os.cpu_count() or 1


class CorpusFile:
    """Streams corpus.txt line-by-line without loading into RAM.

    Supports len() (reads count from meta.json) and iter() (opens the file each call).
    """

    def __init__(self, corpus_path: str, meta_path: str) -> None:
        self._corpus_path = corpus_path
        self._meta_path   = meta_path
        self._len: int | None = None

    def __len__(self) -> int:
        if self._len is None:
            with open(self._meta_path) as f:
                self._len = json.load(f).get("num_files_cached", 0)
        return self._len

    def __iter__(self) -> Iterator[str]:
        with open(self._corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    yield line


def _worker_scan(args: tuple) -> frozenset:
    """Scan one MIDI file and return its unique AMT token strings."""
    path, onset_ms, dur_ms, vel_bins = args
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        chunks = midi_to_amt(path, onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins)
    return frozenset(tok for chunk in chunks for note in chunk for tok in note)


_SER_B2C:    dict[str, str] = {}
_SER_PARAMS: tuple           = (10.0, 10.0, 128, False)


def _init_serialize_worker(b2c_items: list, params: tuple) -> None:
    global _SER_B2C, _SER_PARAMS
    _SER_B2C    = dict(b2c_items)
    _SER_PARAMS = params


def _worker_serialize(path: str) -> "list[str]":
    onset_ms, dur_ms, vel_bins, onset_standalone = _SER_PARAMS
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        return serialize_midi(path, _SER_B2C,
                              onset_ms=onset_ms, dur_ms=dur_ms,
                              vel_bins=vel_bins,
                              onset_standalone=onset_standalone)


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


def load_lakh_evaluation_ids(path: str) -> set[str]:
    """Load MIDI-LLM's 896-song evaluation set (one Lakh md5 per line; copied
    from assets/evaluation_set_lakh_ids.txt in slSeanWU/MIDI-LLM).
    """
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def split_caption_matched_files_lakh_ef(
    midi_files: list[str], caption_map: dict, eval_ids: set[str],
) -> tuple[list[str], list[str], list[str]]:
    """Split caption-matched files by Lakh md5 prefix, following MIDI-LLM's
    stage-2 split (and AMT's LAKH_VALID=['e'] / LAKH_TEST=['f'] convention):

      md5 in eval_ids (MIDI-LLM's 896-song list, all 'f') -> test
      md5 starting with 'e'                                -> validation
      other 'f' md5s                                       -> unused
      everything else                                      -> train

    Only the first path per md5 is kept, so duplicated files count once.
    """
    train: list[str] = []
    valid: list[str] = []
    test:  list[str] = []
    seen: set[str] = set()
    for f in midi_files:
        md5 = os.path.splitext(os.path.basename(f))[0]
        if md5 not in caption_map or md5 in seen:
            continue
        seen.add(md5)
        if md5 in eval_ids:
            test.append(f)
        elif md5.startswith("e"):
            valid.append(f)
        elif md5.startswith("f"):
            continue  # rest of the held-out 'f' bucket: unused, not train
        else:
            train.append(f)
    return train, valid, test


def stable_seed(md5: str, *parts: int) -> int:
    """Deterministic seed from an md5 hex string plus integer parts (e.g. chunk_idx, k).

    Unlike Python's built-in hash() on strings, which is randomized per-process via
    PYTHONHASHSEED and only stable *within* one run, this is reproducible across
    separate invocations of a script -- required for a resumable offline cache build
    where re-running must reproduce the same augmentation draws.
    """
    h = hashlib.sha256(md5.encode())
    for p in parts:
        h.update(str(p).encode())
    return int(h.hexdigest(), 16) % (2 ** 31)


def compute_corpus_fingerprint(
    midi_files: list[str],
    onset_ms: float,
    dur_ms:   float,
    vel_bins: int,
    onset_standalone: bool = False,
    extra: str = "",
) -> str:
    h = hashlib.sha1()
    h.update(f"onset_ms={onset_ms}\ndur_ms={dur_ms}\nvel_bins={vel_bins}\n".encode())
    if onset_standalone:
        h.update(b"onset_standalone=True\n")
    if extra:
        h.update(f"{extra}\n".encode())
    for path in midi_files:
        try:
            mtime = f"{os.path.getmtime(path):.3f}"
        except OSError:
            mtime = "missing"
        h.update(f"{path}:{mtime}\n".encode())
    return h.hexdigest()


def save_filelist_cache(
    midi_files: list[str], cache_dir: str,
    limit_files: int | None,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, _FILELIST_FILE), "w") as f:
        f.write("\n".join(midi_files))
    with open(os.path.join(cache_dir, _FILELIST_META), "w") as f:
        json.dump({"num_files": len(midi_files),
                   "limit_files": limit_files}, f, indent=2)


def load_filelist_cache(
    cache_dir: str, limit_files: int | None,
) -> list[str] | None:
    fl   = os.path.join(cache_dir, _FILELIST_FILE)
    meta = os.path.join(cache_dir, _FILELIST_META)
    if not (os.path.exists(fl) and os.path.exists(meta)):
        return None
    with open(meta) as f:
        m = json.load(f)
    if m.get("limit_files") != limit_files:
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
    onset_ms: float = 10.0,
    dur_ms:   float = 10.0,
    vel_bins: int   = 128,
    n_workers: int  = 1,
    checkpoint_interval: int = VOCAB_CKPT_INTERVAL,
) -> tuple[dict[str, str], dict[str, str]]:
    """Scan all MIDI files to find every AMT base token, map each to a PUA char."""
    os.makedirs(cache_dir, exist_ok=True)
    ckpt_path = os.path.join(cache_dir, _VOCAB_CKPT)

    start_idx: int     = 0
    observed: set[str] = set()

    result = _load_vocab_ckpt(ckpt_path, fingerprint)
    if result is not None:
        start_idx, observed = result
        tqdm.write(f"  Resuming vocab scan from {start_idx:,} / {len(midi_files):,}")

    remaining = midi_files[start_idx:]

    if n_workers > 1:
        batch_size = n_workers * 64
        scan_args  = [(p, onset_ms, dur_ms, vel_bins) for p in remaining]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            with tqdm(total=len(midi_files), initial=start_idx,
                      desc="  vocab scan", unit="file", leave=False) as bar:
                for batch_start in range(0, len(scan_args), batch_size):
                    batch = scan_args[batch_start:batch_start + batch_size]
                    for tokens in executor.map(_worker_scan, batch, chunksize=16):
                        observed.update(tokens)
                    abs_idx = start_idx + batch_start + len(batch)
                    bar.update(len(batch))
                    _save_vocab_ckpt(ckpt_path, fingerprint, abs_idx, observed)
    else:
        with tqdm(remaining, initial=start_idx, total=len(midi_files),
                  desc="  vocab scan", unit="file", leave=False) as bar:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for rel_idx, path in enumerate(bar):
                    for chunk in midi_to_amt(path, onset_ms=onset_ms,
                                             dur_ms=dur_ms, vel_bins=vel_bins):
                        for note in chunk:
                            observed.update(note)
                    abs_idx = start_idx + rel_idx + 1
                    if abs_idx % checkpoint_interval == 0:
                        _save_vocab_ckpt(ckpt_path, fingerprint, abs_idx, observed)

    _save_vocab_ckpt(ckpt_path, fingerprint, len(midi_files), observed)

    sorted_tokens = sorted(observed)
    b2c = {tok: _index_to_pua_char(i) for i, tok in enumerate(sorted_tokens)}
    c2b = {v: k for k, v in b2c.items()}
    return b2c, c2b


def build_ctrl_mapping(
    b2c: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Build the ctrl PUA mapping from an existing event mapping.

    Ctrl chars are assigned PUA(N+i) immediately after the event chars PUA(i).
    """
    N = len(b2c)
    sorted_tokens = sorted(b2c.keys())
    ctrl_b2c: dict[str, str] = {
        tok: _index_to_pua_char(N + i) for i, tok in enumerate(sorted_tokens)
    }
    ctrl_c2b: dict[str, str] = {v: k for k, v in ctrl_b2c.items()}
    return ctrl_b2c, ctrl_c2b


def save_mapping(
    base_token_to_char: dict[str, str],
    char_to_base_token: dict[str, str],
    path: str,
    fingerprint: str = "",
    ctrl_b2c: dict[str, str] | None = None,
    ctrl_c2b: dict[str, str] | None = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data: dict = {
        "fingerprint": fingerprint,
        "base_token_to_char": base_token_to_char,
        "char_to_base_token": char_to_base_token,
    }
    if ctrl_b2c is not None:
        data["base_token_to_ctrl_char"] = ctrl_b2c
    if ctrl_c2b is not None:
        data["ctrl_char_to_base_token"] = ctrl_c2b
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def load_mapping(path: str) -> tuple[dict[str, str], dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["base_token_to_char"], data["char_to_base_token"]


def load_mapping_doubled(
    path: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    """Load a doubled (event + control) mapping.  Raises KeyError if ctrl side absent."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return (
        data["base_token_to_char"],
        data["char_to_base_token"],
        data["base_token_to_ctrl_char"],
        data["ctrl_char_to_base_token"],
    )


def is_mapping_doubled(path: str) -> bool:
    """Return True if the mapping file contains the doubled (ctrl) side."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return "base_token_to_ctrl_char" in data
    except Exception:
        return False


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
    onset_ms: float = 10.0,
    dur_ms:   float = 10.0,
    vel_bins: int   = 128,
    onset_standalone: bool = False,
    n_workers: int  = 1,
    flush_interval: int = CORPUS_FLUSH_INTERVAL,
) -> "CorpusFile":
    """Serialize MIDI files to Unicode strings, writing to disk incrementally.

    Returns a CorpusFile handle — the corpus is never accumulated in RAM.
    """
    os.makedirs(cache_dir, exist_ok=True)
    corpus_path = os.path.join(cache_dir, _CORPUS_FILE)
    meta_path   = os.path.join(cache_dir, _CORPUS_META)

    start_idx: int    = 0
    lines_written: int = 0

    if os.path.exists(meta_path) and os.path.exists(corpus_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if (meta.get("fingerprint") == fingerprint
                and meta.get("status") == "in_progress"):
            start_idx    = meta["files_processed"]
            lines_written = meta.get("num_files_cached", 0)
            tqdm.write(f"  Resuming corpus from {start_idx:,} / {len(midi_files):,} "
                       f"({lines_written:,} lines cached)")

    if start_idx == 0:
        open(corpus_path, "w", encoding="utf-8").close()

    _write_corpus_meta(meta_path, fingerprint, start_idx,
                       lines_written, len(midi_files), "in_progress")

    remaining = midi_files[start_idx:]

    with open(corpus_path, "a", encoding="utf-8") as f_out:
        if n_workers > 1:
            batch_size = n_workers * 32
            params     = (onset_ms, dur_ms, vel_bins, onset_standalone)
            b2c_items  = list(base_token_to_char.items())
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_serialize_worker,
                initargs=(b2c_items, params),
            ) as executor:
                with tqdm(total=len(midi_files), initial=start_idx,
                          desc="  serialising", unit="file", leave=False) as bar:
                    for batch_start in range(0, len(remaining), batch_size):
                        batch = remaining[batch_start:batch_start + batch_size]
                        for texts in executor.map(_worker_serialize, batch, chunksize=16):
                            for text in texts:
                                f_out.write(text + "\n")
                                lines_written += 1
                        abs_idx = start_idx + batch_start + len(batch)
                        bar.update(len(batch))
                        f_out.flush()
                        _write_corpus_meta(meta_path, fingerprint, abs_idx,
                                           lines_written, len(midi_files), "in_progress")
        else:
            with tqdm(remaining, initial=start_idx, total=len(midi_files),
                      desc="  serialising", unit="file", leave=False) as bar:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for rel_idx, path in enumerate(bar):
                        texts = serialize_midi(path, base_token_to_char,
                                              onset_ms=onset_ms, dur_ms=dur_ms,
                                              vel_bins=vel_bins,
                                              onset_standalone=onset_standalone)
                        for text in texts:
                            f_out.write(text + "\n")
                            lines_written += 1
                        abs_idx = start_idx + rel_idx + 1
                        if abs_idx % flush_interval == 0:
                            f_out.flush()
                            _write_corpus_meta(meta_path, fingerprint, abs_idx,
                                               lines_written, len(midi_files), "in_progress")

    _write_corpus_meta(meta_path, fingerprint, len(midi_files),
                       lines_written, len(midi_files), "complete")
    return CorpusFile(corpus_path, meta_path)


def load_corpus_cache(fingerprint: str, cache_dir: str) -> "CorpusFile | None":
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
    return CorpusFile(corpus_path, meta_path)


def load_corpus_unconditional(cache_dir: str) -> "CorpusFile | None":
    """Return a CorpusFile handle without fingerprint check — used when filelist cache is trusted."""
    corpus_path = os.path.join(cache_dir, _CORPUS_FILE)
    meta_path   = os.path.join(cache_dir, _CORPUS_META)
    if not (os.path.exists(corpus_path) and os.path.exists(meta_path)):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("status", "complete") != "complete":
        return None
    return CorpusFile(corpus_path, meta_path)


def serialize_midi(
    midi_path: str,
    base_token_to_char: dict[str, str],
    onset_ms: float = 10.0,
    dur_ms:   float = 10.0,
    vel_bins: int   = 128,
    onset_standalone: bool = False,
) -> list[str]:
    """Serialize a MIDI to a list of space-separated note-word strings, one per 100-second chunk.

    Returns an empty list if the file is invalid or any token is out-of-vocabulary.
    With onset_standalone=True each note becomes two words (onset + 4-char body),
    preventing BPE from merging across the onset boundary.
    """
    chunks = midi_to_amt(midi_path, onset_ms=onset_ms,
                         dur_ms=dur_ms, vel_bins=vel_bins)
    if not chunks:
        return []
    result: list[str] = []
    for notes in chunks:
        words: list[str] = []
        for note in notes:
            try:
                chars = [base_token_to_char[tok] for tok in note]
            except KeyError:
                return []
            if onset_standalone:
                words.append(chars[0])
                words.append("".join(chars[1:]))
            else:
                words.append("".join(chars))
        if words:
            result.append(" ".join(words))
    return result


def items_to_text(
    items: list[tuple[list[str], bool]],
    event_b2c: dict[str, str],
    ctrl_b2c:  dict[str, str],
) -> str | None:
    """Join build_sequence's (tokens, is_control) items into a serialized chunk string.

    Event notes → joined event PUA chars; ctrl notes → joined ctrl PUA chars;
    [SEPARATOR] and [REST] are emitted as literal strings. Returns None if any
    token is out-of-vocabulary or the joined string would be empty.
    """
    words: list[str] = []
    for tokens, is_ctrl in items:
        if tokens == [REST]:
            words.append(REST)
        elif tokens == [SEPARATOR]:
            words.append(SEPARATOR)
        else:
            b2c = ctrl_b2c if is_ctrl else event_b2c
            try:
                words.append("".join(b2c[t] for t in tokens))
            except KeyError:
                return None
    return " ".join(words) if words else None


def serialize_midi_anticipation(
    midi_path: str,
    event_b2c: dict[str, str],
    ctrl_b2c:  dict[str, str],
    k: int,
    rng: "np.random.Generator",
    onset_ms: float = 10.0,
    dur_ms:   float = 10.0,
    vel_bins: int   = 128,
) -> list[str]:
    """Serialize a MIDI with anticipation augmentation pass k, one string per 100-second chunk.

    Returns an empty list if the file is invalid or any token is out-of-vocabulary.
    Event notes → joined event PUA chars; ctrl notes → joined ctrl PUA chars;
    [SEPARATOR] and [REST] are emitted as literal strings.
    """
    chunks = midi_to_amt(midi_path, onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins)
    if not chunks:
        return []

    onset_steps_per_sec = 1000.0 / onset_ms
    delta_steps = max(1, int(round(5.0 * onset_steps_per_sec)))

    result: list[str] = []
    for notes in chunks:
        instruments = list({int(n[3][12:-1]) for n in notes})
        mode, kwargs = augmentation_spec(k, notes, instruments, delta_steps, rng)
        items = build_sequence(notes, mode, rng, onset_ms=onset_ms, **kwargs)

        text = items_to_text(items, event_b2c, ctrl_b2c)
        if text is not None:
            result.append(text)
    return result


def decode_generated_ids(
    token_strings: list[str],
    char_to_base_token: dict[str, str],
    ctrl_char_to_base_token: dict[str, str] | None = None,
    strip_control: bool = True,
) -> list[list[str]]:
    """Decode a raw model.generate() token stream back to AMT compound notes.

    generate() produces a flat id sequence with no word_ids, so note boundaries
    can't be read off an Encoding. Every note word is exactly
    5 PUA chars (Whitespace pre-tokenizer never merges across the space), so chars
    are regrouped every 5 instead. Sentinels ([REST]/[SEPARATOR]/mode tokens) are
    always skipped and any partial note buffered before one (a malformed generation)
    is discarded.
    """
    SENTINEL_STRINGS = {"[REST]", "[SEPARATOR]", "[AUTOREGRESS]", "[ANTICIPATE]",
                        "[UNK]", "[PAD]", "[BOS]", "[EOS]"}

    result: list[list[str]] = []
    buf: list[str] = []
    buf_is_ctrl = False

    for tok_str in token_strings:
        if tok_str in SENTINEL_STRINGS:
            buf, buf_is_ctrl = [], False  # discard any partial note
            continue
        for c in tok_str:
            if ctrl_char_to_base_token and c in ctrl_char_to_base_token:
                buf.append(ctrl_char_to_base_token[c])
                buf_is_ctrl = True
            else:
                buf.append(char_to_base_token.get(c, f"<UNK:{ord(c):#x}>"))
            if len(buf) == 5:
                if not (buf_is_ctrl and strip_control):
                    result.append(buf)
                buf, buf_is_ctrl = [], False

    return result
