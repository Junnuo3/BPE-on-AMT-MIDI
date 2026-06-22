# Mapping, corpus serialization, and caching for BPE training.
# Cache layout (outputs/cache/):
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
        notes = midi_to_amt(path, onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins)
    return frozenset(tok for note in notes for tok in note)


_SER_B2C:    dict[str, str] = {}
_SER_PARAMS: tuple           = (10.0, 10.0, 128, False)


def _init_serialize_worker(b2c_items: list, params: tuple) -> None:
    global _SER_B2C, _SER_PARAMS
    _SER_B2C    = dict(b2c_items)
    _SER_PARAMS = params


def _worker_serialize(path: str) -> "str | None":
    onset_ms, dur_ms, vel_bins, onset_standalone = _SER_PARAMS
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        return serialize_midi(path, _SER_B2C,
                              onset_ms=onset_ms, dur_ms=dur_ms,
                              vel_bins=vel_bins,
                              onset_standalone=onset_standalone)



_ANT_EVENT_B2C: dict[str, str] = {}
_ANT_CTRL_B2C:  dict[str, str] = {}
_ANT_PARAMS:    tuple           = (10.0, 10.0, 128)   # onset_ms, dur_ms, vel_bins


def _init_anticipation_worker(
    event_b2c_items: list,
    ctrl_b2c_items:  list,
    params: tuple,
) -> None:
    global _ANT_EVENT_B2C, _ANT_CTRL_B2C, _ANT_PARAMS
    _ANT_EVENT_B2C = dict(event_b2c_items)
    _ANT_CTRL_B2C  = dict(ctrl_b2c_items)
    _ANT_PARAMS    = params


def _worker_serialize_anticipation(args: tuple) -> "str | None":
    """Worker: serialize one (path, k) pair for anticipation corpus."""
    path, k = args
    onset_ms, dur_ms, vel_bins = _ANT_PARAMS
    # Deterministic per-file-per-pass seed so re-runs reproduce the same corpus.
    seed = abs(hash(path)) % (2 ** 31) ^ (k * 1_000_003)
    rng  = np.random.default_rng(seed)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        return serialize_midi_anticipation(
            path, _ANT_EVENT_B2C, _ANT_CTRL_B2C, k, rng,
            onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins,
        )


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
                    for note in midi_to_amt(path, onset_ms=onset_ms,
                                            dur_ms=dur_ms, vel_bins=vel_bins):
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
                        for text in executor.map(_worker_serialize, batch, chunksize=16):
                            if text is not None:
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
                        text = serialize_midi(path, base_token_to_char,
                                             onset_ms=onset_ms, dur_ms=dur_ms,
                                             vel_bins=vel_bins,
                                             onset_standalone=onset_standalone)
                        if text is not None:
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


def stream_anticipation_corpus(
    tasks: list[tuple[str, int]],
    event_b2c: dict[str, str],
    ctrl_b2c:  dict[str, str],
    onset_ms: float = 10.0,
    dur_ms:   float = 10.0,
    vel_bins: int   = 128,
    n_workers: int  = 1,
) -> Iterator[str]:
    """Yield anticipation corpus strings without disk I/O.

    tasks: list of (path, pass_k) pairs. Yields one string per valid (file, pass).
    """
    if n_workers > 1:
        params = (onset_ms, dur_ms, vel_bins)
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_anticipation_worker,
            initargs=(list(event_b2c.items()), list(ctrl_b2c.items()), params),
        ) as executor:
            for text in executor.map(_worker_serialize_anticipation, tasks, chunksize=16):
                if text is not None:
                    yield text
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for path, k in tasks:
                seed = abs(hash(path)) % (2 ** 31) ^ (k * 1_000_003)
                rng  = np.random.default_rng(seed)
                text = serialize_midi_anticipation(
                    path, event_b2c, ctrl_b2c, k, rng,
                    onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins,
                )
                if text is not None:
                    yield text


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
) -> str | None:
    """Serialize a MIDI to space-separated 5-char note words. Returns None if invalid.

    With onset_standalone=True each note becomes two words (onset + 4-char body),
    preventing BPE from merging across the onset boundary.
    """
    notes = midi_to_amt(midi_path, onset_ms=onset_ms,
                        dur_ms=dur_ms, vel_bins=vel_bins)
    if not notes:
        return None
    words: list[str] = []
    for note in notes:
        try:
            chars = [base_token_to_char[tok] for tok in note]
        except KeyError:
            return None
        if onset_standalone:
            words.append(chars[0])
            words.append("".join(chars[1:]))
        else:
            words.append("".join(chars))
    return " ".join(words)


def serialize_midi_anticipation(
    midi_path: str,
    event_b2c: dict[str, str],
    ctrl_b2c:  dict[str, str],
    k: int,
    rng: "np.random.Generator",
    onset_ms: float = 10.0,
    dur_ms:   float = 10.0,
    vel_bins: int   = 128,
) -> str | None:
    """Serialize a MIDI with anticipation augmentation pass k. Returns None if invalid.

    Event notes → joined event PUA chars; ctrl notes → joined ctrl PUA chars;
    [SEPARATOR] and [REST] are emitted as literal strings.
    """
    notes = midi_to_amt(midi_path, onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins)
    if not notes:
        return None

    instruments = list({int(n[3][12:-1]) for n in notes})
    onset_steps_per_sec = 1000.0 / onset_ms
    delta_steps = max(1, int(round(5.0 * onset_steps_per_sec)))

    mode, kwargs = augmentation_spec(k, notes, instruments, delta_steps, rng)
    items = build_sequence(notes, mode, rng, onset_ms=onset_ms, **kwargs)

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


def serialize_midi_iter(
    midi_files: list[str],
    base_token_to_char: dict[str, str],
    onset_ms: float = 10.0,
    dur_ms:   float = 10.0,
    vel_bins: int   = 128,
    onset_standalone: bool = False,
) -> Iterator[str]:
    for path in midi_files:
        text = serialize_midi(path, base_token_to_char,
                              onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins,
                              onset_standalone=onset_standalone)
        if text is not None:
            yield text


def derive_corpus_from_fine(
    fine_out_dir: str,
    coarse_out_dir: str,
    factor: str,
    fine_value: float,
    coarse_value: float,
) -> tuple[dict[str, str], dict[str, str]] | None:
    """Remap a fine corpus to a coarser quantization without re-reading any MIDI files.

    Streams corpus.txt char-by-char and rewrites PUA chars using the new token mapping.
    factor: "onset" | "duration" | "velocity" — which field to remap.
    Returns (b2c, c2b) for the derived config, or None if the fine cache is missing.
    """
    fine_map_path    = os.path.join(fine_out_dir, "mappings", "amt_base_token_char_mapping.json")
    fine_cache_dir   = os.path.join(fine_out_dir, "cache")
    fine_corpus_path = os.path.join(fine_cache_dir, _CORPUS_FILE)
    fine_meta_path   = os.path.join(fine_cache_dir, _CORPUS_META)

    for p in (fine_map_path, fine_corpus_path, fine_meta_path):
        if not os.path.exists(p):
            return None
    with open(fine_meta_path) as f:
        if json.load(f).get("status") != "complete":
            return None

    _, fine_c2b = load_mapping(fine_map_path)

    # remap fine base token → coarse base token
    def _remap(tok: str) -> str:
        if factor == "onset" and tok.startswith("<time-"):
            T = int(tok[6:-1])
            return f"<time-{max(0, round(T * fine_value / coarse_value))}>"
        if factor == "duration" and tok.startswith("<duration-"):
            D = int(tok[10:-1])
            return f"<duration-{max(1, round(D * fine_value / coarse_value))}>"
        if factor == "velocity" and tok.startswith("<velocity-"):
            V = int(tok[10:-1])
            return f"<velocity-{round(V * (coarse_value - 1) / (fine_value - 1))}>"
        return tok

    fine_to_coarse_tok: dict[str, str] = {fc: _remap(ft) for fc, ft in fine_c2b.items()}
    new_observed = set(fine_to_coarse_tok.values())

    sorted_toks = sorted(new_observed)
    new_b2c: dict[str, str] = {
        tok: _index_to_pua_char(i) for i, tok in enumerate(sorted_toks)
    }
    new_c2b: dict[str, str] = {v: k for k, v in new_b2c.items()}

    fine_to_coarse_char: dict[str, str] = {
        fc: new_b2c[ct] for fc, ct in fine_to_coarse_tok.items()
    }

    coarse_cache_dir = os.path.join(coarse_out_dir, "cache")
    coarse_map_dir   = os.path.join(coarse_out_dir, "mappings")
    os.makedirs(coarse_cache_dir, exist_ok=True)
    os.makedirs(coarse_map_dir,   exist_ok=True)

    coarse_corpus_path = os.path.join(coarse_cache_dir, _CORPUS_FILE)
    lines_written = 0
    with open(fine_corpus_path, encoding="utf-8") as f_in, \
         open(coarse_corpus_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.rstrip("\n")
            if not line:
                continue
            new_line = "".join(
                fine_to_coarse_char.get(c, c) if c != " " else " "
                for c in line
            )
            f_out.write(new_line + "\n")
            lines_written += 1
        f_out.flush()
        os.fsync(f_out.fileno())

    _write_corpus_meta(
        os.path.join(coarse_cache_dir, _CORPUS_META),
        fingerprint="derived",
        files_processed=lines_written,
        lines_written=lines_written,
        num_files_total=lines_written,
        status="complete",
    )

    import shutil
    for fname in (_FILELIST_FILE, _FILELIST_META):
        src = os.path.join(fine_cache_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(coarse_cache_dir, fname))

    coarse_map_path = os.path.join(coarse_map_dir, "amt_base_token_char_mapping.json")
    save_mapping(new_b2c, new_c2b, coarse_map_path, fingerprint="derived")

    tqdm.write(f"  Derived: {lines_written:,} corpus strings, "
               f"{len(new_b2c):,} coarse tokens  [{factor} {fine_value}→{coarse_value}]")
    return new_b2c, new_c2b


def derive_no_onset_corpus(
    baseline_out_dir: str,
    no_onset_out_dir: str,
) -> tuple[dict[str, str], dict[str, str]] | None:
    """Split each 5-char note word into "onset body" so BPE can't merge across onset.

    Copies the mapping and filelist from baseline (identical). Returns (b2c, c2b),
    or None if the baseline corpus isn't ready.
    """
    import shutil

    baseline_map_path    = os.path.join(baseline_out_dir, "mappings", "amt_base_token_char_mapping.json")
    baseline_cache_dir   = os.path.join(baseline_out_dir, "cache")
    baseline_corpus_path = os.path.join(baseline_cache_dir, _CORPUS_FILE)
    baseline_meta_path   = os.path.join(baseline_cache_dir, _CORPUS_META)

    for p in (baseline_map_path, baseline_corpus_path, baseline_meta_path):
        if not os.path.exists(p):
            return None
    with open(baseline_meta_path) as f:
        if json.load(f).get("status") != "complete":
            return None

    b2c, c2b = load_mapping(baseline_map_path)

    no_onset_cache_dir = os.path.join(no_onset_out_dir, "cache")
    no_onset_map_dir   = os.path.join(no_onset_out_dir, "mappings")
    os.makedirs(no_onset_cache_dir, exist_ok=True)
    os.makedirs(no_onset_map_dir,   exist_ok=True)

    no_onset_corpus_path = os.path.join(no_onset_cache_dir, _CORPUS_FILE)
    lines_written = 0
    with open(baseline_corpus_path, encoding="utf-8") as f_in, \
         open(no_onset_corpus_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.rstrip("\n")
            if not line:
                continue
            new_words: list[str] = []
            for word in line.split():
                new_words.append(word[0])   # onset char (1)
                new_words.append(word[1:])  # dur+pitch+inst+vel (4)
            f_out.write(" ".join(new_words) + "\n")
            lines_written += 1
        f_out.flush()
        os.fsync(f_out.fileno())

    _write_corpus_meta(
        os.path.join(no_onset_cache_dir, _CORPUS_META),
        fingerprint="derived_no_onset",
        files_processed=lines_written,
        lines_written=lines_written,
        num_files_total=lines_written,
        status="complete",
    )

    for fname in (_FILELIST_FILE, _FILELIST_META):
        src = os.path.join(baseline_cache_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(no_onset_cache_dir, fname))

    no_onset_map_path = os.path.join(no_onset_map_dir, "amt_base_token_char_mapping.json")
    shutil.copy2(baseline_map_path, no_onset_map_path)

    tqdm.write(f"  Derived no-onset corpus: {lines_written:,} strings from baseline")
    return b2c, c2b


def corpus_is_ready(out_dir: str) -> bool:
    """True if corpus.txt exists and is marked complete in out_dir/cache/."""
    meta   = os.path.join(out_dir, "cache", _CORPUS_META)
    corpus = os.path.join(out_dir, "cache", _CORPUS_FILE)
    if not (os.path.exists(meta) and os.path.exists(corpus)):
        return False
    try:
        with open(meta) as f:
            return json.load(f).get("status") == "complete"
    except Exception:
        return False


def decode_bpe_ids(
    encoding,
    tokenizer,
    char_to_base_token: dict[str, str],
    ctrl_char_to_base_token: dict[str, str] | None = None,
    strip_control: bool = True,
) -> list[list[str]]:
    """Decode a BPE Encoding back to AMT compound notes.

    Uses word_ids + tokens instead of tokenizer.decode() to preserve note boundaries
    (decode() concatenates PUA chars without spaces, losing all note splits).
    In anticipation mode, ctrl notes are stripped by default (strip_control=True).
    Sentinels ([REST], [SEPARATOR]) are always skipped.
    """
    SENTINEL_STRINGS = {"[REST]", "[SEPARATOR]", "[AUTOREGRESS]", "[ANTICIPATE]"}

    words: dict[int, list[str]] = {}
    word_is_ctrl: dict[int, bool] = {}

    for tok_str, word_id in zip(encoding.tokens, encoding.word_ids):
        if word_id is None:
            continue
        if tok_str in SENTINEL_STRINGS:
            continue
        entry = words.setdefault(word_id, [])
        is_ctrl_word = word_is_ctrl.get(word_id, False)

        for c in tok_str:
            if ctrl_char_to_base_token and c in ctrl_char_to_base_token:
                entry.append(ctrl_char_to_base_token[c])
                word_is_ctrl[word_id] = True
            else:
                entry.append(char_to_base_token.get(c, f"<UNK:{ord(c):#x}>"))

    result: list[list[str]] = []
    for wid in sorted(words):
        if word_is_ctrl.get(wid, False) and strip_control:
            continue
        result.append(words[wid])
    return result
