"""
Train SmolLM2 on AMT note tokens (with or without BPE), following MIDI-LLM's
two-stage recipe:

  --stage1                     continued pretraining on MusicPile text + GigaMIDI MIDI
  --captions ... --caption-text  stage 2: caption -> MIDI on MidiCaps / Lakh MIDI

MIDI examples use AMT-style anticipation augmentation; control tokens are
conditioning context and are masked out of the loss. See run/stage1.sh and
run/stage2.sh for the exact arguments of the reported runs.
"""

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

try:
    from torch.utils.data import Dataset as _Dataset
    from torch.utils.data import IterableDataset as _IterableDataset
except ImportError:
    _Dataset = object          # allows import without torch (linting)
    _IterableDataset = object

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)

# Used only when --tokenizer is not given and nothing is found under --ant-dir.
_DEFAULT_TOKENIZER = os.path.join(
    _REPO_ROOT, "tokenizers", "anticipation", "velocity",
    "onset-10ms_duration-10ms_velocity-32bin",
    "tokenizers", "amt_compound_bpe_vocab27677.json",
)

_DEFAULT_ANT_DIR = os.path.join(_REPO_ROOT, "tokenizers", "anticipation",
                                "velocity", "onset-10ms_duration-10ms_velocity-32bin")

# Written inside a checkpoint dir after it is fully saved; absence means the
# save was interrupted and the checkpoint must not be resumed from.
_CKPT_DONE_MARKER = ".done"

sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))



def _cleanup_incomplete_checkpoints(ckpt_dir: str, verbose: bool = True) -> None:
    """Delete checkpoint dirs that were interrupted mid-save.

    A checkpoint is considered complete when it contains _CKPT_DONE_MARKER.
    For backwards compatibility, a dir with trainer_state.json but no marker
    gets the marker backfilled rather than deleted.
    """
    import shutil
    if not os.path.isdir(ckpt_dir):
        return
    for name in sorted(os.listdir(ckpt_dir)):
        if not name.startswith("checkpoint-"):
            continue
        try:
            int(name[len("checkpoint-"):])
        except ValueError:
            continue
        path = os.path.join(ckpt_dir, name)
        if not os.path.isdir(path):
            continue
        done_path  = os.path.join(path, _CKPT_DONE_MARKER)
        state_path = os.path.join(path, "trainer_state.json")
        if os.path.exists(done_path):
            continue  # fully complete
        if os.path.exists(state_path):
            # Legacy checkpoint saved before this feature — treat as complete.
            open(done_path, "w").close()
        else:
            if verbose:
                print(f"Removing incomplete checkpoint (interrupted mid-save): {path}")
            shutil.rmtree(path)


def _auto_tokenizer(corpus_path: str | None, ant_dir: str) -> str | None:
    """Infer tokenizer path from corpus sibling or ant_dir/tokenizers/."""
    if corpus_path is not None:
        lm_dir  = os.path.dirname(corpus_path)
        out_dir = os.path.dirname(lm_dir)
        hits    = sorted(glob.glob(os.path.join(out_dir, "tokenizers", "*.json")))
        if hits:
            return hits[-1]
    hits = sorted(glob.glob(os.path.join(ant_dir, "tokenizers", "*.json")))
    if hits:
        return hits[-1]
    return _DEFAULT_TOKENIZER if os.path.exists(_DEFAULT_TOKENIZER) else None


def _auto_mapping(tokenizer_path: str) -> str | None:
    """Infer doubled mapping from tokenizer path (looks in tokenizers/../mappings/)."""
    tok_dir = os.path.dirname(tokenizer_path)
    ant_dir = os.path.dirname(tok_dir)
    path = os.path.join(ant_dir, "mappings", "amt_base_token_char_mapping.json")
    return path if os.path.exists(path) else None


def _auto_train_cfg(tokenizer_path: str) -> dict:
    """Load train_config.json from the parent of the tokenizers/ dir."""
    tok_dir  = os.path.dirname(tokenizer_path)
    ant_dir  = os.path.dirname(tok_dir)
    cfg_path = os.path.join(ant_dir, "train_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def _auto_wandb_step_offset(model_path: str | None) -> int | None:
    """If --model points at a checkpoint dir this script itself wrote (e.g. a
    stage-1 checkpoint being handed to stage 2), return that run's final
    global_step from the sibling run_dir/final_metrics.json two levels up
    (checkpoints/<name> -> checkpoints -> run_dir). Returns None if there's no
    such file (--model is a plain HF hub id, or a checkpoint dir from a run
    that hasn't finished/written final_metrics.json yet).

    Used so a stage-2 run sharing --run-name with its stage-1 run can continue
    the same W&B step timeline instead of restarting at 0 (see
    --wandb-step-offset help).
    """
    if not model_path or not os.path.isdir(model_path):
        return None
    run_dir      = os.path.dirname(os.path.dirname(os.path.normpath(model_path)))
    metrics_path = os.path.join(run_dir, "final_metrics.json")
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path) as f:
        return json.load(f).get("step")



class AMTChunkDataset:
    """Static dataset wrapping a pre-built chunks.npy (baseline mode)."""

    def __init__(self, chunks: np.ndarray, id_map: np.ndarray) -> None:
        self.chunks = chunks
        self.id_map = id_map

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int):
        import torch
        ids    = torch.tensor(self.id_map[self.chunks[idx].astype(np.int64)], dtype=torch.long)
        labels = ids.clone()
        return {"input_ids": ids, "labels": labels}


class AMTStreamDataset(_IterableDataset):
    """On-the-fly anticipation augmentation dataset.

    Each file gets one "natural" draw per epoch (with probability ant_prob it
    receives anticipation augmentation, strategy sampled uniformly from
    span/instrument; otherwise it is autoregressive), plus expansion_factor - 1
    additional forced-augmented draws, so the effective per-epoch size is
    ~expansion_factor x the file count. expansion_factor=1 (default) reproduces
    the original one-draw-per-file-per-epoch behavior. Epochs count
    original-dataset passes.
    """

    def __init__(
        self,
        midi_files: list[str],
        bpe_tokenizer,          # tokenizers.Tokenizer (doubled)
        b2c: dict,
        ctrl_b2c: dict,
        ctrl_ids: set[int],     # AMT-BPE IDs that are control-variant tokens
        autoregress_id: int,
        anticipate_id: int,
        id_map: np.ndarray,     # amt_bpe_id → smollm2_extended_id
        pad_id: int,            # extended-vocab id used to pad short chunks (e.g. hf eos id)
        window_size: int = 4096,
        ant_prob: float = 0.9,  # probability of anticipation augmentation per file
        expansion_factor: int = 1,  # 1 natural draw + (factor - 1) forced-augmented draws
        onset_ms: float = 10.0,
        dur_ms: float = 10.0,
        vel_bins: int = 32,
    ) -> None:
        self.midi_files       = midi_files
        self.bpe_tokenizer    = bpe_tokenizer
        self.b2c              = b2c
        self.ctrl_b2c         = ctrl_b2c
        self.ctrl_ids         = np.array(sorted(ctrl_ids), dtype=np.int64)
        self.autoregress_id   = autoregress_id
        self.anticipate_id    = anticipate_id
        self.id_map           = id_map
        self.pad_id           = pad_id
        self.window_size      = window_size
        self.ant_prob         = ant_prob
        self.expansion_factor = expansion_factor
        self.onset_ms         = onset_ms
        self.dur_ms           = dur_ms
        self.vel_bins         = vel_bins

    def _iter_shard(self, tasks):
        """Return only the tasks for this DataLoader worker, sharded by DDP rank first."""
        import torch
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            tasks = tasks[dist.get_rank() :: dist.get_world_size()]
        info = torch.utils.data.get_worker_info()
        if info is None:
            return tasks
        return tasks[info.id :: info.num_workers]

    def __iter__(self):
        import torch
        from bpe_utils import serialize_midi_anticipation

        n         = len(self.midi_files)
        rng_epoch = np.random.default_rng()  # fresh each epoch

        # Draw per-file augmentation decisions for this epoch: one "natural" draw
        # per file (k=0 → autoregressive; k∈[1,9] → anticipation strategy
        # span/instrument, chosen with probability ant_prob), plus
        # expansion_factor - 1 additional forced-augmented draws (k∈[1,9]) per
        # file. draw_idx distinguishes repeat draws of the same file so each gets
        # an independently seeded rng below (two draws can land on the same k by
        # chance, and would otherwise replay identical control extraction).
        use_ant   = rng_epoch.random(n) < self.ant_prob
        k_vals    = np.where(use_ant, rng_epoch.integers(1, 10, size=n), 0)
        task_list = [(self.midi_files[i], int(k_vals[i]), 0) for i in range(n)]
        for draw_idx in range(1, self.expansion_factor):
            extra_k = rng_epoch.integers(1, 10, size=n)
            task_list.extend(
                (self.midi_files[i], int(extra_k[i]), draw_idx) for i in range(n)
            )

        perm  = rng_epoch.permutation(len(task_list))
        tasks = [task_list[i] for i in perm]
        tasks = list(self._iter_shard(tasks))

        for path, k, draw_idx in tasks:
            seed = abs(hash((path, draw_idx))) % (2 ** 31)
            rng  = np.random.default_rng(seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                chunk_texts = serialize_midi_anticipation(
                    path, self.b2c, self.ctrl_b2c, k, rng,
                    onset_ms=self.onset_ms,
                    dur_ms=self.dur_ms,
                    vel_bins=self.vel_bins,
                )
            if not chunk_texts:
                continue

            # Each chunk is a separate ~100-second segment whose first event is at t=0
            # (guaranteed by midi_to_amt normalization). Process chunks independently so
            # no window ever crosses a chunk or file boundary.
            for chunk_text in chunk_texts:
                ids = self.bpe_tokenizer.encode(chunk_text).ids
                if not ids:
                    continue

                # Reserve slot 0 for the mode-prefix token; use up to window_size-1
                # content tokens, right-padding short chunks instead of discarding
                # them (GigaMIDI in particular is loop-heavy -- most chunks are
                # shorter than one window). Padding is at the end only, so under
                # causal attention it can never influence any real token's
                # prediction; its own label positions are masked with -100.
                n_content = min(len(ids), self.window_size - 1)
                window    = np.array(ids[:n_content], dtype=np.int64)

                ctrl_mask  = np.isin(window, self.ctrl_ids)
                has_ctrl   = bool(ctrl_mask.any())
                prefix_id  = self.anticipate_id if has_ctrl else self.autoregress_id
                real       = np.empty(1 + n_content, dtype=np.int64)
                real[0]    = prefix_id
                real[1:]   = window

                mapped_real = self.id_map[real]
                pad_len     = self.window_size - len(mapped_real)
                if pad_len > 0:
                    mapped = np.concatenate(
                        [mapped_real, np.full(pad_len, self.pad_id, dtype=np.int64)]
                    )
                else:
                    mapped = mapped_real

                input_ids = torch.tensor(mapped, dtype=torch.long)
                # Control notes are conditioning context, not a generation target --
                # mask them out of the loss (only the mode token, padding rule below,
                # and target/REST/SEPARATOR tokens stay scored).
                labels_np = mapped.copy()
                labels_np[1:1 + n_content][ctrl_mask] = -100
                if pad_len > 0:
                    labels_np[-pad_len:] = -100
                labels = torch.tensor(labels_np, dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}



def load_caption_map(jsonl_path: str) -> dict:
    """Load md5 → caption string from a MidiCaps-style JSONL file."""
    caption_map = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            md5     = entry.get("md5")
            caption = entry.get("caption")
            if md5 and caption:
                caption_map[md5] = caption
    return caption_map


def assemble_captioned_example(
    caption: str,
    chunk_text: str,
    *,
    hf_tokenizer,
    bpe_tokenizer,
    system_prompt: str,
    max_text_tokens: int,
    window_size: int,
    ctrl_ids: np.ndarray,
    autoregress_id: int,
    anticipate_id: int,
    id_map: np.ndarray,
    pad_id: int,
):
    """Assemble one training example: [caption text][mode token][MIDI tokens][pad].

    Shared by AMTCaptionStreamDataset (on-the-fly captions/chunks) and
    AMTCachedCaptionStreamDataset (precomputed captions/chunks from
    scripts/build_caption_cache.py) so both paths tokenize/pad/mask
    identically. Returns None if the caption's text budget leaves no room for
    MIDI content, or the chunk tokenizes to nothing.
    """
    import torch

    full_text = system_prompt + caption + " "
    text_ids  = hf_tokenizer.encode(full_text, add_special_tokens=False)
    text_ids  = text_ids[:max_text_tokens]
    text_len  = len(text_ids)

    midi_budget = window_size - text_len - 1  # -1 for the mode token
    if midi_budget < 1:
        return None

    amt_ids = bpe_tokenizer.encode(chunk_text).ids
    if not amt_ids:
        return None

    # Right-pad short MIDI content instead of discarding the chunk (see
    # AMTStreamDataset for the same pattern/rationale). Padding only ever
    # sits after real content, so it can't influence any real token's
    # prediction under causal attention.
    n_midi     = min(len(amt_ids), midi_budget)
    amt_window = np.array(amt_ids[:n_midi], dtype=np.int64)
    ctrl_mask  = np.isin(amt_window, ctrl_ids)
    has_ctrl   = bool(ctrl_mask.any())
    mode_bpe   = anticipate_id if has_ctrl else autoregress_id

    # Map AMT-BPE IDs to the extended SmolLM2 vocab
    mode_mapped = int(id_map[mode_bpe])
    midi_mapped = id_map[amt_window]

    # Assemble: [text (base vocab)] [mode token] [MIDI (extended vocab)] [pad]
    pad_len = midi_budget - n_midi
    seq = np.empty(window_size, dtype=np.int64)
    seq[:text_len]                           = text_ids
    seq[text_len]                            = mode_mapped
    seq[text_len + 1:text_len + 1 + n_midi]  = midi_mapped
    if pad_len > 0:
        seq[text_len + 1 + n_midi:] = pad_id

    input_ids = torch.tensor(seq, dtype=torch.long)
    # Control notes are conditioning context the model is given, not something it
    # should be scored on generating -- only the caption text and the target
    # (non-control) MIDI span incur loss.
    labels_np = seq.copy()
    labels_np[text_len + 1:text_len + 1 + n_midi][ctrl_mask] = -100
    if pad_len > 0:
        labels_np[-pad_len:] = -100
    labels = torch.tensor(labels_np, dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels}


class AMTCaptionStreamDataset(_IterableDataset):
    """On-the-fly text-captioned MIDI dataset.

    Each example has the format:
        <HF text tokens>  [AUTOREGRESS / ANTICIPATE]  <AMT-BPE MIDI tokens>

    The text prefix is tokenized with the SmolLM2 HF tokenizer (base vocab range).
    MIDI tokens are produced by AMT-BPE and mapped to the extended vocabulary.
    The mode token ([AUTOREGRESS] or [ANTICIPATE]) sits between the two halves.

    Files without a matching caption in caption_map are silently skipped.
    Causal-LM loss is computed over the caption text and the target (non-control)
    MIDI span; control-note tokens are conditioning context and are masked out
    of labels (see assemble_captioned_example).
    """

    _DEFAULT_SYSTEM_PROMPT = (
        "You are a world-class composer. Please compose some music "
        "according to the following description: "
    )

    def __init__(
        self,
        midi_files: list,
        caption_map: dict,
        hf_tokenizer,
        bpe_tokenizer,
        b2c: dict,
        ctrl_b2c: dict,
        ctrl_ids: set,
        autoregress_id: int,
        anticipate_id: int,
        id_map: np.ndarray,
        pad_id: int,            # extended-vocab id used to pad short MIDI tails
        window_size: int = 2048,
        ant_prob: float = 0.9,
        onset_ms: float = 10.0,
        dur_ms: float = 10.0,
        vel_bins: int = 32,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        max_text_tokens: int = 256,
    ) -> None:
        self.midi_files      = midi_files
        self.caption_map     = caption_map
        self.hf_tokenizer    = hf_tokenizer
        self.bpe_tokenizer   = bpe_tokenizer
        self.b2c             = b2c
        self.ctrl_b2c        = ctrl_b2c
        self.ctrl_ids        = np.array(sorted(ctrl_ids), dtype=np.int64)
        self.autoregress_id  = autoregress_id
        self.anticipate_id   = anticipate_id
        self.id_map          = id_map
        self.pad_id          = pad_id
        self.window_size     = window_size
        self.ant_prob        = ant_prob
        self.onset_ms        = onset_ms
        self.dur_ms          = dur_ms
        self.vel_bins        = vel_bins
        self.system_prompt   = system_prompt
        self.max_text_tokens = max_text_tokens

    def _iter_shard(self, tasks):
        """Return only the tasks for this DataLoader worker, sharded by DDP rank first."""
        import torch
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            tasks = tasks[dist.get_rank() :: dist.get_world_size()]
        info = torch.utils.data.get_worker_info()
        if info is None:
            return tasks
        return tasks[info.id :: info.num_workers]

    def __iter__(self):
        from bpe_utils import serialize_midi_anticipation

        n         = len(self.midi_files)
        rng_epoch = np.random.default_rng()

        use_ant = rng_epoch.random(n) < self.ant_prob
        k_vals  = np.where(use_ant, rng_epoch.integers(1, 10, size=n), 0)

        perm  = rng_epoch.permutation(n)
        tasks = [(self.midi_files[i], int(k_vals[i])) for i in perm]
        tasks = list(self._iter_shard(tasks))

        for path, k in tasks:
            # Match by md5 (filename without extension)
            md5     = os.path.splitext(os.path.basename(path))[0]
            caption = self.caption_map.get(md5)
            if caption is None:
                continue

            seed = abs(hash(path)) % (2 ** 31)
            rng  = np.random.default_rng(seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                chunk_texts = serialize_midi_anticipation(
                    path, self.b2c, self.ctrl_b2c, k, rng,
                    onset_ms=self.onset_ms,
                    dur_ms=self.dur_ms,
                    vel_bins=self.vel_bins,
                )
            if not chunk_texts:
                continue

            for chunk_text in chunk_texts:
                ex = assemble_captioned_example(
                    caption, chunk_text,
                    hf_tokenizer=self.hf_tokenizer, bpe_tokenizer=self.bpe_tokenizer,
                    system_prompt=self.system_prompt, max_text_tokens=self.max_text_tokens,
                    window_size=self.window_size, ctrl_ids=self.ctrl_ids,
                    autoregress_id=self.autoregress_id, anticipate_id=self.anticipate_id,
                    id_map=self.id_map, pad_id=self.pad_id,
                )
                if ex is not None:
                    yield ex


class AMTCachedCaptionStreamDataset(_IterableDataset):
    """Stage-2 dataset with precomputed, re-captioned anticipation variants.

    Original (k=0) examples are untouched: static MidiCaps caption + full song,
    computed on the fly exactly as AMTCaptionStreamDataset. Only the augmented
    branch changes -- instead of drawing a fresh anticipation pass and reusing
    the file's own caption, per chunk this emits (expansion_factor - 1)
    additional draws, each sampling one precomputed (chunk_text, caption) pair
    from cache_dir/<md5>.jsonl (built by scripts/build_caption_cache.py).
    Net effective size per epoch is ~expansion_factor x the original file count
    (MIDI-LLM reports ~3x token expansion from this augmentation).

    A caption-matched training file with no cache entry (no <md5>.done marker)
    is a hard error, not a skip: the cache build is a required, ordered
    prerequisite, so a gap here means the cache is incomplete and the run
    should stop rather than silently train on less augmentation than
    --caption-expansion-factor configures.
    """

    _DEFAULT_SYSTEM_PROMPT = AMTCaptionStreamDataset._DEFAULT_SYSTEM_PROMPT

    def __init__(
        self,
        midi_files: list,
        caption_map: dict,
        cache_dir: str,
        hf_tokenizer,
        bpe_tokenizer,
        b2c: dict,
        ctrl_b2c: dict,
        ctrl_ids: set,
        autoregress_id: int,
        anticipate_id: int,
        id_map: np.ndarray,
        pad_id: int,
        window_size: int = 2048,
        expansion_factor: int = 3,
        onset_ms: float = 10.0,
        dur_ms: float = 10.0,
        vel_bins: int = 32,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        max_text_tokens: int = 256,
    ) -> None:
        self.midi_files       = midi_files
        self.caption_map      = caption_map
        self.cache_dir        = cache_dir
        self.hf_tokenizer     = hf_tokenizer
        self.bpe_tokenizer    = bpe_tokenizer
        self.b2c              = b2c
        self.ctrl_b2c         = ctrl_b2c
        self.ctrl_ids         = np.array(sorted(ctrl_ids), dtype=np.int64)
        self.autoregress_id   = autoregress_id
        self.anticipate_id    = anticipate_id
        self.id_map           = id_map
        self.pad_id           = pad_id
        self.window_size      = window_size
        self.expansion_factor = expansion_factor
        self.onset_ms         = onset_ms
        self.dur_ms           = dur_ms
        self.vel_bins         = vel_bins
        self.system_prompt    = system_prompt
        self.max_text_tokens  = max_text_tokens
        # Running totals for the realized-expansion-ratio log line (labels != -100
        # is the actual trained-on token count, i.e. what the ~3x claim is about).
        self._original_tokens  = 0
        self._augmented_tokens = 0
        self._files_seen       = 0

    def _iter_shard(self, tasks):
        """Return only the tasks for this DataLoader worker, sharded by DDP rank first."""
        import torch
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            tasks = tasks[dist.get_rank() :: dist.get_world_size()]
        info = torch.utils.data.get_worker_info()
        if info is None:
            return tasks
        return tasks[info.id :: info.num_workers]

    def _load_cache(self, md5: str) -> dict[int, list[tuple[str, str]]]:
        """chunk_idx -> [(chunk_text, caption), ...] of cached augmented variants."""
        done_path = os.path.join(self.cache_dir, f"{md5}.done")
        if not os.path.exists(done_path):
            raise RuntimeError(
                f"Missing/incomplete caption cache for md5={md5} (expected {done_path}). "
                f"Run scripts/build_caption_cache.py to completion for the full "
                f"caption-matched training set before training with --caption-cache-dir."
            )
        by_chunk: dict[int, list[tuple[str, str]]] = {}
        with open(os.path.join(self.cache_dir, f"{md5}.jsonl")) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                by_chunk.setdefault(rec["chunk_idx"], []).append((rec["chunk_text"], rec["caption"]))
        return by_chunk

    def __iter__(self):
        from bpe_utils import serialize_midi_anticipation

        n         = len(self.midi_files)
        rng_epoch = np.random.default_rng()
        perm      = rng_epoch.permutation(n)
        tasks     = [self.midi_files[i] for i in perm]
        tasks     = list(self._iter_shard(tasks))

        for path in tasks:
            md5     = os.path.splitext(os.path.basename(path))[0]
            caption = self.caption_map.get(md5)
            if caption is None:
                continue

            seed = abs(hash(path)) % (2 ** 31)
            rng  = np.random.default_rng(seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                original_chunk_texts = serialize_midi_anticipation(
                    path, self.b2c, self.ctrl_b2c, 0, rng,
                    onset_ms=self.onset_ms, dur_ms=self.dur_ms, vel_bins=self.vel_bins,
                )
            if not original_chunk_texts:
                continue

            cached_by_chunk = self._load_cache(md5)

            for chunk_idx, orig_text in enumerate(original_chunk_texts):
                ex = assemble_captioned_example(
                    caption, orig_text,
                    hf_tokenizer=self.hf_tokenizer, bpe_tokenizer=self.bpe_tokenizer,
                    system_prompt=self.system_prompt, max_text_tokens=self.max_text_tokens,
                    window_size=self.window_size, ctrl_ids=self.ctrl_ids,
                    autoregress_id=self.autoregress_id, anticipate_id=self.anticipate_id,
                    id_map=self.id_map, pad_id=self.pad_id,
                )
                if ex is not None:
                    self._original_tokens += int((ex["labels"] != -100).sum())
                    yield ex

                variants = cached_by_chunk.get(chunk_idx, [])
                if not variants:
                    continue
                for _ in range(self.expansion_factor - 1):
                    v_idx = int(rng_epoch.integers(len(variants)))
                    cached_text, cached_caption = variants[v_idx]
                    ex = assemble_captioned_example(
                        cached_caption, cached_text,
                        hf_tokenizer=self.hf_tokenizer, bpe_tokenizer=self.bpe_tokenizer,
                        system_prompt=self.system_prompt, max_text_tokens=self.max_text_tokens,
                        window_size=self.window_size, ctrl_ids=self.ctrl_ids,
                        autoregress_id=self.autoregress_id, anticipate_id=self.anticipate_id,
                        id_map=self.id_map, pad_id=self.pad_id,
                    )
                    if ex is not None:
                        self._augmented_tokens += int((ex["labels"] != -100).sum())
                        yield ex

            self._files_seen += 1
            if self._files_seen % 200 == 0:
                total = self._original_tokens + self._augmented_tokens
                ratio = total / self._original_tokens if self._original_tokens else float("nan")
                print(f"  [caption cache] files={self._files_seen:,} "
                      f"realized token expansion ratio={ratio:.2f} "
                      f"(target ~{self.expansion_factor})")


class MusicPileTextStreamDataset(_IterableDataset):
    """On-the-fly text packing for MusicPile continued-pretraining data (MIDI-LLM
    stage 1's "music-adjacent text" domain).

    Concatenates tokenized `text` rows (EOS-separated) into a running buffer and
    yields fixed window_size slices as the buffer fills -- standard continued-
    pretraining packing. Shard order (not row order within a shard) is shuffled
    per epoch; full row-level shuffling would require materializing the whole
    corpus in memory. Base-vocab-range ids only; labels = input_ids (full
    causal-LM loss, no id_map involved -- this is plain HF-tokenizer text, not
    AMT-BPE).
    """

    def __init__(
        self,
        parquet_files: list[str],
        hf_tokenizer,
        window_size: int = 2048,
    ) -> None:
        self.parquet_files = parquet_files
        self.hf_tokenizer  = hf_tokenizer
        self.window_size   = window_size

    def _iter_shard(self, tasks):
        """Return only the tasks for this DataLoader worker, sharded by DDP rank first."""
        import torch
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            tasks = tasks[dist.get_rank() :: dist.get_world_size()]
        info = torch.utils.data.get_worker_info()
        if info is None:
            return tasks
        return tasks[info.id :: info.num_workers]

    def __iter__(self):
        import torch
        import pyarrow.parquet as pq

        n         = len(self.parquet_files)
        rng_epoch = np.random.default_rng()
        perm      = rng_epoch.permutation(n)
        files     = [self.parquet_files[i] for i in perm]
        files     = self._iter_shard(files)

        eos_id = self.hf_tokenizer.eos_token_id
        buffer: list[int] = []

        for path in files:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(columns=["text"], batch_size=256):
                for text in batch.column("text").to_pylist():
                    if not text:
                        continue
                    ids = self.hf_tokenizer.encode(text, add_special_tokens=False)
                    buffer.extend(ids)
                    buffer.append(eos_id)
                    while len(buffer) >= self.window_size:
                        window = torch.tensor(buffer[: self.window_size], dtype=torch.long)
                        buffer = buffer[self.window_size:]
                        yield {"input_ids": window, "labels": window.clone()}


def build_musicpile_eval_examples(
    parquet_files: list[str],
    hf_tokenizer,
    window_size: int,
    max_examples: int = 50,
) -> list[dict]:
    """Deterministic, fixed set of packed MusicPile text windows for held-out stage-1
    eval loss -- same EOS-separated concatenation packing as
    MusicPileTextStreamDataset, but reading shards in a fixed order (not shuffled)
    and capped at max_examples, so the eval set (and hence eval loss) is
    reproducible across checkpoints, matching build_fixed_eval_examples' role for
    the MIDI side.
    """
    import pyarrow.parquet as pq
    import torch

    eos_id = hf_tokenizer.eos_token_id
    buffer: list[int] = []
    examples: list[dict] = []

    for path in parquet_files:
        if len(examples) >= max_examples:
            break
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["text"], batch_size=256):
            if len(examples) >= max_examples:
                break
            for text in batch.column("text").to_pylist():
                if not text:
                    continue
                ids = hf_tokenizer.encode(text, add_special_tokens=False)
                buffer.extend(ids)
                buffer.append(eos_id)
                while len(buffer) >= window_size and len(examples) < max_examples:
                    window = torch.tensor(buffer[:window_size], dtype=torch.long)
                    buffer = buffer[window_size:]
                    examples.append({"input_ids": window, "labels": window.clone()})

    return examples


class Stage1MixedStreamDataset(_IterableDataset):
    """Mixes MIDI-LLM stage 1's two continued-pretraining streams: standalone MIDI
    (an AMTStreamDataset with on-the-fly span/instrument anticipation augmentation,
    see --stage1-ant-prob/--stage1-expansion-factor) and music-adjacent text (a
    MusicPileTextStreamDataset). Each yielded example is drawn from one stream or
    the other at a fixed text_ratio, matching the paper's Table 1 token mix
    (1.69B text : 1.38B MIDI tokens ~= 0.55 text_ratio) -- two separate document
    streams mixed at the example level, not fused into one sequence like stage 2's
    caption+MIDI fusion.

    Whichever sub-stream is smaller is transparently restarted (fresh shuffle/
    fresh augmentation draws) when it exhausts, rather than ending the mix --
    otherwise the realized text:MIDI ratio would depend on which corpus happens
    to be smaller (and --stage1-expansion-factor directly changes the MIDI
    corpus's relative size, so that dependency would silently shift the ratio
    whenever expansion_factor changes). Restarting decouples the ratio from
    relative corpus size: text_ratio governs every draw, indefinitely, and
    --max-steps is what actually ends training.
    """

    def __init__(
        self,
        text_dataset: MusicPileTextStreamDataset,
        midi_dataset: AMTStreamDataset,
        text_ratio: float = 0.55,
    ) -> None:
        self.text_dataset = text_dataset
        self.midi_dataset = midi_dataset
        self.text_ratio   = text_ratio

    def __iter__(self):
        rng       = np.random.default_rng()
        text_iter = iter(self.text_dataset)
        midi_iter = iter(self.midi_dataset)
        while True:
            use_text = rng.random() < self.text_ratio
            try:
                yield next(text_iter if use_text else midi_iter)
            except StopIteration:
                # Restart the exhausted stream (fresh shuffle for text, fresh
                # augmentation draws for MIDI) instead of ending the mix. If the
                # freshly restarted stream is itself immediately empty, this
                # StopIteration propagates and correctly ends iteration -- that's
                # a genuinely empty dataset, not a mid-epoch exhaustion.
                if use_text:
                    text_iter = iter(self.text_dataset)
                    yield next(text_iter)
                else:
                    midi_iter = iter(self.midi_dataset)
                    yield next(midi_iter)


class ListDataset(_Dataset):
    """Fixed, order-preserving map-style dataset: one pre-tokenized window per example.

    Used for evaluation, where examples must be identical across every call to
    Trainer.evaluate() (unlike the training streams, which re-sample each epoch).

    Each example is a dict with "input_ids", "labels" (may differ from input_ids:
    captioned examples carry -100 over the text span, see build_fixed_eval_examples)
    and "base_token_count" (BPE-compression-normalized token count, used only by
    compute_comparable_loss below, not by Trainer's own eval loop).
    """

    def __init__(self, examples: list) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        return {"input_ids": ex["input_ids"], "labels": ex["labels"]}


def _bpe_base_token_count(ids, id_to_token) -> int:
    """Number of pre-merge AMT base tokens represented by a span of AMT-BPE ids.

    Per-token loss is not comparable across tokenizers because a BPE token can
    stand for a variable number of base tokens. A non-sentinel token's
    string is a concatenation of PUA base chars (BPE never merges across the
    per-note space), so its string length equals the base tokens it replaces;
    sentinels ([REST]/[SEPARATOR]/[AUTOREGRESS]/[ANTICIPATE]) are each exactly one
    base unit. Summing this over a window gives a per-sequence compression ratio
    (this count / len(ids)) rather than one dataset-wide average.
    """
    from bpe_train import ANTICIPATION_SENTINELS

    sentinels = set(ANTICIPATION_SENTINELS)
    total = 0
    for i in ids:
        tok = id_to_token(int(i))
        total += 1 if tok in sentinels else len(tok)
    return total


def compute_comparable_loss(model, examples: list[dict], batch_size: int, device) -> dict:
    """Corpus-level nats-per-base-token loss over `examples`. Diagnostic only.

    Raw eval loss is not comparable (a) across tokenizers, since tokens carry a
    variable amount of information, or (b) across MIDI-only vs. captioned runs,
    since text tokens have no MIDI equivalent. Each example's
    `labels` already has -100 over any masked text span and `base_token_count`
    already accounts for BPE merges (see build_fixed_eval_examples), so summing
    nats and dividing by total base tokens yields one number comparable across all
    run variants. Never feeds training or checkpoint selection.
    """
    import torch
    import torch.nn.functional as F

    was_training = model.training
    model.eval()
    nats_sum = 0.0
    ntok_sum = 0
    base_sum = sum(ex["base_token_count"] for ex in examples)

    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch     = examples[i : i + batch_size]
            input_ids = torch.stack([ex["input_ids"] for ex in batch]).to(device)
            labels    = torch.stack([ex["labels"]    for ex in batch]).to(device)

            logits       = model(input_ids=input_ids).logits
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()

            nats_sum += F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            ).item()
            ntok_sum += int((shift_labels != -100).sum().item())

    if was_training:
        model.train()

    return {
        "nats_per_bpe_token":  nats_sum / ntok_sum if ntok_sum else float("nan"),
        "nats_per_base_token": nats_sum / base_sum if base_sum else float("nan"),
    }


def build_fixed_eval_examples(
    files: list[str],
    *,
    force_autoregressive: bool,
    bpe_tokenizer,
    b2c: dict, ctrl_b2c: dict, ctrl_ids: np.ndarray,
    autoregress_id: int, anticipate_id: int,
    id_map: np.ndarray,
    pad_id: int,             # extended-vocab id used to pad short MIDI tails
    window_size: int,
    onset_ms: float, dur_ms: float, vel_bins: int,
    caption_map: dict | None = None,
    hf_tokenizer=None,
    system_prompt: str | None = None,
    max_text_tokens: int = 256,
) -> list:
    """Build one deterministic token window per file (first chunk long enough).

    force_autoregressive=True  -> every file uses k=0 ([AUTOREGRESS], no randomness).
    force_autoregressive=False -> every file uses a fixed anticipation pass k in
    [1, 9], derived from the file's md5 (not epoch-random), so the augmentation
    strategy and the control notes it extracts are identical on every call —
    required for eval loss to be reproducible/comparable across checkpoints.
    """
    import torch
    from bpe_utils import serialize_midi_anticipation

    ctrl_ids = np.array(sorted(ctrl_ids), dtype=np.int64)

    examples: list = []
    for path in files:
        md5     = os.path.splitext(os.path.basename(path))[0]
        md5_int = int(md5, 16)
        seed    = md5_int % (2 ** 31)
        rng     = np.random.default_rng(seed)
        k       = 0 if force_autoregressive else 1 + (md5_int % 9)

        caption = None
        if caption_map is not None:
            caption = caption_map.get(md5)
            if caption is None:
                continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunk_texts = serialize_midi_anticipation(
                path, b2c, ctrl_b2c, k, rng,
                onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins,
            )
        if not chunk_texts:
            continue

        if caption is not None:
            full_text = system_prompt + caption + " "
            text_ids  = hf_tokenizer.encode(full_text, add_special_tokens=False)[:max_text_tokens]
            text_len  = len(text_ids)
            midi_budget = window_size - text_len - 1  # -1 for the mode token
            if midi_budget < 1:
                continue
        else:
            text_ids    = None
            midi_budget = window_size - 1  # -1 for the mode token (see text branch above)

        for chunk_text in chunk_texts:
            amt_ids = bpe_tokenizer.encode(chunk_text).ids
            if not amt_ids:
                continue

            # Right-pad short MIDI content instead of discarding the window (see
            # AMTStreamDataset for the same pattern/rationale) -- otherwise a
            # held-out slice of mostly-short files (e.g. GigaMIDI loops) can end
            # up contributing zero eval examples.
            n_midi    = min(len(amt_ids), midi_budget)
            window    = np.array(amt_ids[:n_midi], dtype=np.int64)
            pad_len   = midi_budget - n_midi
            ctrl_mask = np.isin(window, ctrl_ids)
            has_ctrl  = bool(ctrl_mask.any())
            mode_bpe  = anticipate_id if has_ctrl else autoregress_id

            if text_ids is not None:
                mode_mapped = int(id_map[mode_bpe])
                midi_mapped = id_map[window]
                seq = np.empty(window_size, dtype=np.int64)
                seq[:text_len]                          = text_ids
                seq[text_len]                           = mode_mapped
                seq[text_len + 1:text_len + 1 + n_midi] = midi_mapped
                if pad_len > 0:
                    seq[text_len + 1 + n_midi:] = pad_id
                # Eval loss is MIDI-only: caption tokens stay in input_ids as
                # context but are masked from labels, as are control notes and
                # padding. The mode token and target MIDI span are scored.
                labels = seq.copy()
                labels[:text_len] = -100
                labels[text_len + 1:text_len + 1 + n_midi][ctrl_mask] = -100
                if pad_len > 0:
                    labels[-pad_len:] = -100
                midi_ids_for_base = window[~ctrl_mask]
            else:
                real        = np.empty(1 + n_midi, dtype=np.int64)
                real[0]     = mode_bpe
                real[1:]    = window
                mapped_real = id_map[real]
                if pad_len > 0:
                    seq = np.concatenate(
                        [mapped_real, np.full(pad_len, pad_id, dtype=np.int64)]
                    )
                else:
                    seq = mapped_real
                labels = seq.copy()
                labels[1:1 + n_midi][ctrl_mask] = -100
                if pad_len > 0:
                    labels[-pad_len:] = -100
                midi_ids_for_base = window[~ctrl_mask]

            # Denominator for nats-per-base-token: base tokens in the scored
            # (non-control) MIDI span, plus the mode token when it is scored.
            # The causal shift never scores position 0, so in the non-captioned
            # branch (mode token at position 0) it is not counted.
            base_token_count = _bpe_base_token_count(midi_ids_for_base, bpe_tokenizer.id_to_token)
            if text_ids is not None:
                base_token_count += 1  # mode token at position text_len, always scored

            examples.append({
                "input_ids":        torch.tensor(seq, dtype=torch.long),
                "labels":           torch.tensor(labels, dtype=torch.long),
                "base_token_count": base_token_count,
            })
            break  # one window per file, so eval-set size == #held-out files

    return examples


def build_model_and_tokenizer(
    smollm2_model_id: str,
    amt_tok_path: str,
    verbose: bool = True,
) -> tuple:
    """Load SmolLM2 and extend its vocab with AMT-BPE tokens.

    Always loads the base model from smollm2_model_id; Trainer handles overlaying
    checkpoint weights when resuming via resume_from_checkpoint.

    Returns (model, hf_tokenizer, id_offset, id_map, amt_tokenizer).
    id_map[amt_bpe_id] = smollm2_extended_id.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tokenizers import Tokenizer as HFTokenizer

    if verbose:
        print(f"Loading SmolLM2: {smollm2_model_id}")

    hf_tokenizer = AutoTokenizer.from_pretrained(smollm2_model_id)
    model        = AutoModelForCausalLM.from_pretrained(smollm2_model_id)

    amt_tokenizer     = HFTokenizer.from_file(amt_tok_path)
    amt_vocab         = amt_tokenizer.get_vocab()
    amt_tokens_sorted = sorted(amt_vocab, key=amt_vocab.__getitem__)

    id_offset = len(hf_tokenizer)
    if verbose:
        print(f"  SmolLM2 vocab size: {id_offset:,}")
    n_added = hf_tokenizer.add_tokens(amt_tokens_sorted)
    model.resize_token_embeddings(len(hf_tokenizer))
    if verbose:
        print(f"  AMT-BPE tokens added: {n_added:,}  (offset {id_offset:,})")

    id_map = np.array(
        [hf_tokenizer.convert_tokens_to_ids(tok) for tok in amt_tokens_sorted],
        dtype=np.int64,
    )

    if verbose:
        print(f"  Extended vocab size: {len(hf_tokenizer):,}")
    return model, hf_tokenizer, id_offset, id_map, amt_tokenizer



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus",      default=None,
                   help="Pre-built chunks.npy (baseline mode). "
                        "Omit for on-the-fly streaming (treatment mode).")
    p.add_argument("--ant-dir",    default=_DEFAULT_ANT_DIR,
                   help="Anticipation output directory containing tokenizers/, mappings/, "
                        "train_config.json (default: anticipation/velocity/onset-10ms_duration-10ms_velocity-32bin).")
    p.add_argument("--tokenizer",   default=None,
                   help="AMT-BPE tokenizer JSON. Overrides --ant-dir auto-detect.")
    p.add_argument("--mapping",     default=None,
                   help="Doubled mapping JSON (streaming mode). "
                        "Auto-detected from tokenizer dir sibling if omitted.")
    p.add_argument("--dataset",     default=os.path.join(_REPO_ROOT, "dataset", "lmd_full"),
                   help="MIDI dataset directory (streaming mode).")
    p.add_argument("--ant-prob", type=float, default=0.9,
                   help="Per-file probability of anticipation augmentation (streaming mode). "
                        "Complement uses autoregressive mode. Epochs count original-file passes.")
    p.add_argument("--captions", default=None,
                   help="MidiCaps JSONL file (e.g. dataset/lmd_full_captions.jsonl). "
                        "Filters the training file list to only MIDI files that have a "
                        "matching caption entry (by md5). Use --caption-text to also "
                        "prepend the caption to each training example.")
    p.add_argument("--eval-ids", default=os.path.join(_REPO_ROOT, "assets", "evaluation_set_lakh_ids.txt"),
                   help="MIDI-LLM's 896-song evaluation md5 list (test split; only used "
                        "when --captions is given).")
    p.add_argument("--caption-text", action="store_true", default=False,
                   help="Prepend the MidiCaps caption to each training example as a text "
                        "prefix: <system prompt + caption> [AUTOREGRESS/ANTICIPATE] <MIDI>. "
                        "Requires --captions. Without this flag, --captions only filters "
                        "the file list; the caption text is not used during training.")
    p.add_argument("--system-prompt", default=None,
                   help="Text prefix prepended to each caption in captioned mode. "
                        "Defaults to the MIDI-LLM system prompt when --captions is set.")
    p.add_argument("--max-text-tokens", type=int, default=256,
                   help="Maximum HF-tokenizer tokens reserved for the text prefix "
                        "(captioned mode only). The remainder of the window is filled with "
                        "MIDI tokens.")
    p.add_argument("--caption-cache-dir", default=None,
                   help="Directory of precomputed captioned-anticipation variants "
                        "(scripts/build_caption_cache.py output). Requires "
                        "--captions --caption-text. When set, replaces AMTCaptionStreamDataset's "
                        "on-the-fly augmented (k>=1) branch with AMTCachedCaptionStreamDataset: "
                        "original (k=0) examples stay on the fly with the static MidiCaps "
                        "caption; augmented examples are sampled from this cache. Every "
                        "caption-matched training file must have a completed cache entry "
                        "(<md5>.done) -- a missing one is a hard error, not a skip.")
    p.add_argument("--caption-expansion-factor", type=int, default=3,
                   help="Per chunk, emit this many draws when --caption-cache-dir is set: "
                        "1 original (k=0) draw plus (factor - 1) cached-augmented draws. "
                        "Default 3 matches MIDI-LLM's reported ~3x stage-2 token expansion.")
    p.add_argument("--stage1",      action="store_true", default=False,
                   help="MIDI-LLM stage-1 continued pretraining: mix music-adjacent text "
                        "(MusicPile) and standalone MIDI (GigaMIDI, on-the-fly span/instrument "
                        "anticipation augmentation, see --stage1-ant-prob/--stage1-expansion-factor) "
                        "instead of the captioned/MIDI-only streaming modes. Mutually exclusive "
                        "with --captions.")
    p.add_argument("--stage1-ant-prob", type=float, default=0.9,
                   help="Stage1 MIDI stream: probability a file's 'natural' per-epoch draw is "
                        "anticipation-augmented (span/instrument) rather than autoregressive. "
                        "Same semantics as --ant-prob, applied to the GigaMIDI stream.")
    p.add_argument("--stage1-expansion-factor", type=int, default=3,
                   help="Stage1 MIDI stream: emit this many draws per file per epoch -- "
                        "1 natural draw (see --stage1-ant-prob) plus (factor - 1) additional "
                        "forced-augmented draws, each a fresh on-the-fly span/instrument pass. "
                        "Default 3 mirrors --caption-expansion-factor's MIDI-LLM ~3x expansion, "
                        "applied here on-the-fly instead of from a cache since stage1 has no "
                        "captions to precompute.")
    p.add_argument("--stage1-text-dir", default=os.path.join(_REPO_ROOT, "dataset", "musicpile"),
                   help="Directory of MusicPile parquet shards (stage1 mode; searched "
                        "recursively for data/*.parquet).")
    p.add_argument("--stage1-midi-dir", default=os.path.join(_REPO_ROOT, "dataset", "gigamidi", "midi"),
                   help="Directory of extracted standalone GigaMIDI .mid files (stage1 mode; "
                        "see scripts/extract_gigamidi_midi.py).")
    p.add_argument("--stage1-text-ratio", type=float, default=0.55,
                   help="Probability of drawing a text (vs. MIDI) example in stage1 mode. "
                        "Default ~= 1.69B/(1.69B+1.38B) from MIDI-LLM's Table 1 token mix.")
    p.add_argument("--stage1-eval-files", type=int, default=300,
                   help="Number of held-out GigaMIDI files and MusicPile windows "
                        "used for stage-1 eval loss.")
    p.add_argument("--window",      type=int, default=2048,
                   help="Sequence window size in tokens.")
    p.add_argument("--model",       default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--out-dir",     default=os.path.join(_REPO_ROOT, "outputs", "runs", "smollm2"),
                   help="Base output dir. The run is placed in "
                        "<out-dir>/{stage1,stage2}/<feature>/<bins>/.")
    p.add_argument("--max-steps",   type=int,   default=25000,
                   help="Total gradient updates to run (primary training duration control). "
                        "Set to 0 to rely on --epochs instead.")
    p.add_argument("--epochs",      type=int,   default=10,
                   help="Max data passes (safety ceiling; training stops at --max-steps first).")
    p.add_argument("--early-stopping-patience", type=int, default=0,
                   help="If > 0, stop training once the canonical eval metric "
                        "(eval_autoregressive_loss, == metric_for_best_model) has not "
                        "improved for this many consecutive evaluations. Requires eval "
                        "to be active (always in --stage1 mode; needs --captions in "
                        "stage 2). Use this to train to convergence instead of a fixed "
                        "--max-steps -- keep --max-steps as a large safety ceiling. The "
                        "patience counter survives a resume from checkpoint.")
    p.add_argument("--early-stopping-threshold", type=float, default=0.0,
                   help="Minimum absolute decrease in eval_autoregressive_loss (nats) "
                        "for an evaluation to count as an improvement under "
                        "--early-stopping-patience.")
    p.add_argument("--batch-size",  type=int,   default=4)
    p.add_argument("--grad-accum",  type=int,   default=None,
                   help="Gradient accumulation steps. Auto-computed from --target-tokens if omitted.")
    p.add_argument("--target-tokens", type=int, default=500_000,
                   help="Target effective batch size in tokens "
                        "(seq_len × per-GPU-batch × num-GPUs × accum). "
                        "Used to auto-compute --grad-accum when not explicitly set.")
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="AdamW weight decay (default 0.0, HF's default).")
    p.add_argument("--warmup-steps", type=int,  default=200)
    p.add_argument("--lr-scheduler-type", default="constant_with_warmup",
                   help="HF TrainingArguments lr_scheduler_type. Default is a flat LR "
                        "after --warmup-steps' linear ramp-up (no decay over training) -- "
                        "pass 'cosine' for the previous cosine-decay behavior, or "
                        "'constant' for no warmup either (flat from step 0).")
    p.add_argument("--save-steps",   type=int,  default=200)
    p.add_argument("--eval-steps",   type=int,  default=None,
                   help="Evaluation interval in gradient steps (default: --save-steps). "
                        "Eval runs in --stage1 mode and when --captions is given.")
    p.add_argument("--eval-limit",   type=int,  default=None,
                   help="Cap the number of validation files used to build the fixed "
                        "eval sets (debugging/smoke-testing only; default: use all).")
    p.add_argument("--log-steps",    type=int,  default=10,
                   help="Trainer logging interval in gradient steps.")
    p.add_argument("--fp16",         action="store_true", default=False)
    p.add_argument("--no-fp16",      dest="fp16", action="store_false")
    p.add_argument("--bf16",         action="store_true", default=True,
                   help="BFloat16 mixed precision (default; no grad scaler needed).")
    p.add_argument("--no-bf16",      dest="bf16", action="store_false")
    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--workers",      type=int,  default=0,
                   help="DataLoader worker processes (streaming mode).")
    p.add_argument("--wandb",        action="store_true", default=False,
                   help="Enable Weights & Biases logging.")
    p.add_argument("--wandb-project", default="llm-midi-bpe",
                   help="W&B project name.")
    p.add_argument("--run-name",     default=None,
                   help="W&B run name and id (default: derived from the run dir). Reusing "
                        "a stage-1 run name in stage 2 continues the same W&B run "
                        "(see --wandb-step-offset).")
    p.add_argument("--wandb-group",  default=None,
                   help="W&B run group (e.g. 'lr_sweep'), for grouping related diagnostic "
                        "runs together in the W&B UI.")
    p.add_argument("--wandb-tags",   default=None,
                   help="Comma-separated W&B tags (e.g. 'stage1,smollm,bpe,lr-sweep').")
    p.add_argument("--wandb-step-offset", type=int, default=None,
                   help="Added to Trainer's global_step before every W&B log call. W&B "
                        "requires an explicit-step run's steps to be monotonically "
                        "increasing, so without this, a stage-2 run sharing --run-name with "
                        "its stage-1 run (both start counting from step 0) would have its "
                        "logs silently dropped once step 0 <= stage 1's last logged step. "
                        "Default: auto-detected from --model, if it points at a checkpoint "
                        "dir with a sibling final_metrics.json (i.e. a stage-1 checkpoint "
                        "produced by this script) -- that run's final step is used. Pass "
                        "explicitly to override, or 0 to disable.")
    return p.parse_args()



def main() -> None:
    args = parse_args()

    try:
        import torch
        from transformers import TrainingArguments, Trainer, TrainerCallback
        from transformers import EarlyStoppingCallback
        from transformers import default_data_collator
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as e:
        print(f"ERROR: required package not found: {e}")
        sys.exit(1)

    class SafeCheckpointCallback(TrainerCallback):
        """Write _CKPT_DONE_MARKER after each checkpoint is fully flushed to disk.

        on_save fires after both model weights and trainer_state.json are written,
        so the marker's presence is a reliable signal that the checkpoint is intact.
        """
        def on_save(self, args, state, control, **kwargs):
            if args.local_rank in (-1, 0):
                ckpt_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                open(os.path.join(ckpt_path, _CKPT_DONE_MARKER), "w").close()

    class MetricsLogCallback(TrainerCallback):
        """Merge every metrics dict logged for a given step (train loss, and one
        eval_*_loss dict per named eval dataset — HF fires a separate on_log call
        for each) into a single line in <run_dir>/metrics.jsonl, so the full
        training + validation-loss history survives independent of trainer_state.json
        inside whichever 2 checkpoints save_total_limit happens to have kept.

        Also owns all W&B logging (Trainer's own report_to is left off "wandb" --
        see TrainingArguments below -- specifically so this is the only place a
        step number is chosen): each flush is one wandb.log(..., step=global_step
        + wandb_step_offset) call. The offset is what lets a stage-2 run sharing
        --run-name with its stage-1 run continue the same step timeline instead
        of restarting at 0, which W&B would silently drop (explicit-step runs
        require monotonically increasing steps) -- see --wandb-step-offset.

        Drops throughput/timing noise (runtime, samples/steps per second, flops)
        that HF logs alongside every eval call — only loss/optimization signals
        are kept.
        """
        _DROP_SUBSTRINGS = ("runtime", "per_second", "total_flos")

        def __init__(self, path: str, wandb_module=None, step_offset: int = 0) -> None:
            self.path         = path
            self.wandb_module = wandb_module   # None disables W&B logging
            self.step_offset  = step_offset
            self._pending: dict | None = None
            self._pending_step: int | None = None

        def _flush(self) -> None:
            if self._pending:
                with open(self.path, "a") as f:
                    f.write(json.dumps(self._pending) + "\n")
                if self.wandb_module is not None:
                    self.wandb_module.log(self._pending, step=self._pending_step + self.step_offset)
            self._pending = None
            self._pending_step = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            if args.local_rank not in (-1, 0) or not logs:
                return
            clean = {k: v for k, v in logs.items()
                     if not any(s in k for s in self._DROP_SUBSTRINGS)}
            if not clean:
                return
            if self._pending_step is not None and state.global_step != self._pending_step:
                self._flush()
            if self._pending is None:
                self._pending = {"step": state.global_step, "epoch": state.epoch}
                self._pending_step = state.global_step
            self._pending.update(clean)

        def on_train_end(self, args, state, control, **kwargs):
            if args.local_rank in (-1, 0):
                self._flush()

    class ComparableLossCallback(TrainerCallback):
        """Logs a BPE-compression-normalized, MIDI-only nats-per-base-token loss on
        the canonical autoregressive eval set alongside Trainer's own eval_* metrics,
        so it reaches wandb and metrics.jsonl like every other reported number.

        It is reporting only: it never affects the training loss or which
        checkpoint is selected as best.

        `trainer` is assigned after construction (this callback is built before the
        Trainer that will own it); on_evaluate fires after Trainer has already
        logged its own eval metrics, so trainer.log() here appends to the same
        step's line in metrics.jsonl instead of creating a new one.
        """
        def __init__(self, examples: list, batch_size: int) -> None:
            self.examples   = examples
            self.batch_size = batch_size
            self.trainer    = None  # set post-construction

        def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
            if args.local_rank not in (-1, 0) or not metrics or "eval_autoregressive_loss" not in metrics:
                return
            device = next(model.parameters()).device
            result = compute_comparable_loss(model, self.examples, self.batch_size, device)
            self.trainer.log({"eval_autoregressive_comparable_loss": result["nats_per_base_token"]})

    torch.manual_seed(args.seed)

    # is_main / world_size from env vars set by torchrun (Trainer initializes dist internally).
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main    = (local_rank == 0)

    # Auto-compute grad_accum so effective batch ≈ target_tokens.
    if args.grad_accum is None:
        args.grad_accum = max(1, round(args.target_tokens / (world_size * args.batch_size * args.window)))
    eff_tokens = world_size * args.batch_size * args.grad_accum * args.window
    if is_main:
        print(f"GPUs: {world_size}  batch/GPU: {args.batch_size}  "
              f"grad_accum: {args.grad_accum}  seq_len: {args.window}  "
              f"→ effective batch: {eff_tokens:,} tokens  (target: {args.target_tokens:,})")

    streaming = args.corpus is None

    # tokenizer
    tokenizer_path = args.tokenizer or _auto_tokenizer(args.corpus, args.ant_dir)
    if tokenizer_path is None:
        print("ERROR: --tokenizer not given and could not auto-detect one.")
        sys.exit(1)
    if is_main:
        print(f"AMT-BPE tokenizer: {os.path.basename(tokenizer_path)}")

    # quantization params
    cfg      = _auto_train_cfg(tokenizer_path)
    onset_ms = cfg.get("onset_ms", 10.0)
    dur_ms   = cfg.get("dur_ms",   10.0)
    vel_bins = cfg.get("vel_bins", 32)
    if is_main:
        print(f"Quantization: onset={onset_ms}ms  dur={dur_ms}ms  vel={vel_bins}bins")

    captioned = streaming and args.captions is not None and args.caption_text
    stage1    = streaming and args.stage1
    if stage1 and args.captions is not None:
        print("ERROR: --stage1 and --captions are mutually exclusive.")
        sys.exit(1)
    if args.caption_cache_dir is not None and not captioned:
        print("ERROR: --caption-cache-dir requires --captions and --caption-text.")
        sys.exit(1)

    # run dir mirrors the anticipation layout; captioned/stage1 runs get an extra prefix
    if streaming:
        ant      = os.path.realpath(args.ant_dir)
        bins_str = os.path.basename(ant)
        feature  = os.path.basename(os.path.dirname(ant))
    else:
        bins_str = f"onset-{onset_ms:g}ms_duration-{dur_ms:g}ms_velocity-{vel_bins}bin"
        feature  = "velocity"
    subdirs  = (["stage1", feature, bins_str] if stage1 else
                ["stage2", feature, bins_str] if captioned else [feature, bins_str])
    run_dir  = os.path.join(args.out_dir, *subdirs)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    if is_main:
        print(f"Run dir: {run_dir}")

    # model — always build from base; Trainer overlays checkpoint weights when resuming
    model, hf_tokenizer, id_offset, id_map, amt_tokenizer = build_model_and_tokenizer(
        args.model, tokenizer_path, verbose=is_main,
    )

    # Reused to right-pad any MIDI/text window shorter than window_size instead of
    # discarding it (see AMTStreamDataset/AMTCaptionStreamDataset/build_fixed_eval_examples).
    # A base-vocab-range token, so it's valid whether or not the vocab has been
    # AMT-extended and needs no id_map lookup.
    pad_id = hf_tokenizer.eos_token_id

    # Save config for inference (rank 0 only; written before training so it exists even if job dies)
    if is_main:
        with open(os.path.join(run_dir, "amt_config.json"), "w") as f:
            json.dump({
                "smollm2_model":    args.model,
                "amt_tokenizer":    tokenizer_path,
                "id_offset":        id_offset,
                "window_size":      args.window,
                "streaming_mode":   streaming,
                "caption_filter":   args.captions is not None,
                "caption_text":     captioned,
                "captions_file":    args.captions,
                "max_text_tokens":  args.max_text_tokens if captioned else None,
                "stage1":           stage1,
                "stage1_text_ratio": args.stage1_text_ratio if stage1 else None,
                "stage1_eval_files": args.stage1_eval_files if stage1 else None,
                "stage1_ant_prob":  args.stage1_ant_prob if stage1 else None,
                "stage1_expansion_factor": args.stage1_expansion_factor if stage1 else None,
            }, f, indent=2)

    # dataset
    eval_datasets = None  # only built in streaming mode when --captions gives a valid split
    if streaming:
        from bpe_utils import load_mapping_doubled, load_filelist_cache, find_midi_files

        mapping_path = args.mapping or _auto_mapping(tokenizer_path)
        if mapping_path is None:
            print("ERROR: --mapping not given and could not auto-detect from tokenizer dir.")
            sys.exit(1)
        if is_main:
            print(f"Mapping: {mapping_path}")

        b2c, c2b, ctrl_b2c, ctrl_c2b = load_mapping_doubled(mapping_path)

        amt_vocab  = amt_tokenizer.get_vocab()
        ctrl_chars = set(ctrl_b2c.values())
        ctrl_ids   = {amt_vocab[c] for c in ctrl_chars if c in amt_vocab}
        if is_main:
            print(f"  {len(ctrl_ids):,} ctrl token IDs")

        autoregress_id = amt_vocab.get("[AUTOREGRESS]")
        anticipate_id  = amt_vocab.get("[ANTICIPATE]")
        if autoregress_id is None or anticipate_id is None:
            print("ERROR: tokenizer is missing [AUTOREGRESS]/[ANTICIPATE] tokens. "
                  "Run scripts/double_tokenizer.py first.")
            sys.exit(1)

        caption_map = None
        valid_files: list[str] = []
        test_files:  list[str] = []

        if stage1:
            # Stage-1 MIDI comes from extracted, stage-2-deduped GigaMIDI files
            # (see scripts/extract_gigamidi_midi.py) -- no cache/fingerprint
            # reuse from the lmd_full-oriented cache_dir below, since this is a
            # different corpus entirely.
            midi_files = find_midi_files(args.stage1_midi_dir)
            if is_main:
                print(f"Stage-1 standalone MIDI files: {len(midi_files):,} "
                      f"(from {args.stage1_midi_dir})")
            # Held-out slice: the first N files in md5 order (deterministic and
            # unrelated to content).
            n_valid     = min(args.stage1_eval_files, len(midi_files))
            valid_files = midi_files[:n_valid]
            midi_files  = midi_files[n_valid:]
            if is_main:
                print(f"  train={len(midi_files):,}  valid={len(valid_files):,}")
        else:
            cache_dir  = os.path.join(os.path.dirname(os.path.dirname(tokenizer_path)), "cache")
            midi_files = load_filelist_cache(cache_dir, limit_files=None)
            if midi_files is None:
                if is_main:
                    print(f"No file list cache found; scanning {args.dataset} …")
                midi_files = find_midi_files(args.dataset)
            if is_main:
                print(f"MIDI files (full): {len(midi_files):,}")

            # MIDI-LLM's stage-2 split of caption-matched Lakh files: md5 prefix 'e'
            # is validation, the published 896-song list (all 'f') is test, the
            # rest of 'f' is unused (see split_caption_matched_files_lakh_ef).
            if args.captions is not None:
                from bpe_utils import load_lakh_evaluation_ids, split_caption_matched_files_lakh_ef

                caption_map = load_caption_map(args.captions)
                eval_ids    = load_lakh_evaluation_ids(args.eval_ids)
                train_files, valid_files, test_files = split_caption_matched_files_lakh_ef(
                    midi_files, caption_map, eval_ids,
                )
                midi_files = train_files
                if is_main:
                    print(f"Captions loaded: {len(caption_map):,} entries from {args.captions}")
                    print(f"MIDI files (caption-matched, Lakh e/f split): "
                          f"train={len(train_files):,}  valid={len(valid_files):,}  test={len(test_files):,}")
                    # The split depends only on --captions/--dataset/--eval-ids, so it
                    # is written once next to the captions file.
                    splits_dir = os.path.join(os.path.dirname(os.path.abspath(args.captions)), "splits_lakh_ef")
                    if not os.path.exists(os.path.join(splits_dir, "train.txt")):
                        os.makedirs(splits_dir, exist_ok=True)
                        for name, files in (("train", train_files), ("valid", valid_files), ("test", test_files)):
                            with open(os.path.join(splits_dir, f"{name}.txt"), "w") as f:
                                f.write("\n".join(files))
                        print(f"Splits written: {splits_dir}")
                    else:
                        print(f"Splits already cached: {splits_dir}")

        if stage1:
            text_files = sorted(glob.glob(os.path.join(args.stage1_text_dir, "**", "*.parquet"), recursive=True))
            if not text_files:
                print(f"ERROR: no parquet files found under {args.stage1_text_dir}")
                sys.exit(1)
            # Reserve one shard (the last, for a deterministic split) as held-out
            # text eval -- otherwise stage1 has zero visibility into whether the
            # text stream is training sanely, only the MIDI side gets eval loss.
            text_valid_files = text_files[-1:] if len(text_files) > 1 else []
            text_files       = text_files[:-1] if len(text_files) > 1 else text_files
            if is_main:
                print(f"Stage-1 text shards: {len(text_files):,} train, "
                      f"{len(text_valid_files):,} held out (from {args.stage1_text_dir})")
            text_dataset = MusicPileTextStreamDataset(
                parquet_files = text_files,
                hf_tokenizer  = hf_tokenizer,
                window_size   = args.window,
            )
            midi_dataset = AMTStreamDataset(
                midi_files       = midi_files,
                bpe_tokenizer    = amt_tokenizer,
                b2c              = b2c,
                ctrl_b2c         = ctrl_b2c,
                ctrl_ids         = ctrl_ids,
                autoregress_id   = autoregress_id,
                anticipate_id    = anticipate_id,
                id_map           = id_map,
                pad_id           = pad_id,
                window_size      = args.window,
                ant_prob         = args.stage1_ant_prob,
                expansion_factor = args.stage1_expansion_factor,
                onset_ms         = onset_ms,
                dur_ms           = dur_ms,
                vel_bins         = vel_bins,
            )
            dataset = Stage1MixedStreamDataset(
                text_dataset = text_dataset,
                midi_dataset = midi_dataset,
                text_ratio   = args.stage1_text_ratio,
            )
            if is_main:
                print(f"Stage-1 text_ratio: {args.stage1_text_ratio}")
                print(f"Stage-1 MIDI ant_prob={args.stage1_ant_prob} "
                      f"expansion_factor={args.stage1_expansion_factor} "
                      f"(~{args.stage1_expansion_factor}x on-the-fly span/instrument augmentation)")

            text_eval_examples = (
                build_musicpile_eval_examples(
                    text_valid_files, hf_tokenizer, args.window,
                    max_examples=args.stage1_eval_files,
                )
                if text_valid_files else []
            )
            if is_main:
                print(f"Stage-1 text validation examples: {len(text_eval_examples):,}")
        elif captioned:
            system_prompt = args.system_prompt or AMTCaptionStreamDataset._DEFAULT_SYSTEM_PROMPT
            if args.caption_cache_dir is not None:
                dataset = AMTCachedCaptionStreamDataset(
                    midi_files       = midi_files,
                    caption_map      = caption_map,
                    cache_dir        = args.caption_cache_dir,
                    hf_tokenizer     = hf_tokenizer,
                    bpe_tokenizer    = amt_tokenizer,
                    b2c              = b2c,
                    ctrl_b2c         = ctrl_b2c,
                    ctrl_ids         = ctrl_ids,
                    autoregress_id   = autoregress_id,
                    anticipate_id    = anticipate_id,
                    id_map           = id_map,
                    pad_id           = pad_id,
                    window_size      = args.window,
                    expansion_factor = args.caption_expansion_factor,
                    onset_ms         = onset_ms,
                    dur_ms           = dur_ms,
                    vel_bins         = vel_bins,
                    system_prompt    = system_prompt,
                    max_text_tokens  = args.max_text_tokens,
                )
                if is_main:
                    print(f"  Caption cache: {args.caption_cache_dir} "
                          f"(expansion factor {args.caption_expansion_factor})")
            else:
                dataset = AMTCaptionStreamDataset(
                    midi_files      = midi_files,
                    caption_map     = caption_map,
                    hf_tokenizer    = hf_tokenizer,
                    bpe_tokenizer   = amt_tokenizer,
                    b2c             = b2c,
                    ctrl_b2c        = ctrl_b2c,
                    ctrl_ids        = ctrl_ids,
                    autoregress_id  = autoregress_id,
                    anticipate_id   = anticipate_id,
                    id_map          = id_map,
                    pad_id          = pad_id,
                    window_size     = args.window,
                    ant_prob        = args.ant_prob,
                    onset_ms        = onset_ms,
                    dur_ms          = dur_ms,
                    vel_bins        = vel_bins,
                    system_prompt   = system_prompt,
                    max_text_tokens = args.max_text_tokens,
                )
            if is_main:
                print(f"  System prompt ({args.max_text_tokens} token budget): {system_prompt!r}")
                print(f"Estimated chunks/epoch: {len(midi_files) * 5_000 // args.window:,}")
        else:
            dataset = AMTStreamDataset(
                midi_files     = midi_files,
                bpe_tokenizer  = amt_tokenizer,
                b2c            = b2c,
                ctrl_b2c       = ctrl_b2c,
                ctrl_ids       = ctrl_ids,
                autoregress_id = autoregress_id,
                anticipate_id  = anticipate_id,
                id_map         = id_map,
                pad_id         = pad_id,
                window_size    = args.window,
                ant_prob       = args.ant_prob,
                onset_ms       = onset_ms,
                dur_ms         = dur_ms,
                vel_bins       = vel_bins,
            )
            if is_main:
                print(f"Estimated chunks/epoch: {len(midi_files) * 5_000 // args.window:,}")

        # Fixed validation sets: one deterministic window per held-out file, so eval
        # loss is reproducible across checkpoints (unlike the training streams above,
        # which re-sample augmentation strategy/controls fresh every epoch). Reports
        # autoregressive (canonical, no randomness), anticipation, and combined loss,
        # per the AMT/MIDI-LLM convention.
        if args.eval_limit is not None:
            valid_files = valid_files[: args.eval_limit]

        if valid_files:
            eval_kwargs = dict(
                bpe_tokenizer   = amt_tokenizer,
                b2c             = b2c,
                ctrl_b2c        = ctrl_b2c,
                ctrl_ids        = ctrl_ids,
                autoregress_id  = autoregress_id,
                anticipate_id   = anticipate_id,
                id_map          = id_map,
                pad_id          = pad_id,
                window_size     = args.window,
                onset_ms        = onset_ms,
                dur_ms          = dur_ms,
                vel_bins        = vel_bins,
                caption_map     = caption_map if captioned else None,
                hf_tokenizer    = hf_tokenizer if captioned else None,
                system_prompt   = system_prompt if captioned else None,
                max_text_tokens = args.max_text_tokens,
            )
            ar_examples  = build_fixed_eval_examples(valid_files, force_autoregressive=True,  **eval_kwargs)
            ant_examples = build_fixed_eval_examples(valid_files, force_autoregressive=False, **eval_kwargs)
            # metric_for_best_model needs eval_autoregressive_loss, so eval is
            # only enabled when the autoregressive set is non-empty.
            if ar_examples:
                eval_datasets = {
                    "autoregressive": ListDataset(ar_examples),
                    "anticipation":   ListDataset(ant_examples),
                    "combined":       ListDataset(ar_examples + ant_examples),
                }
                # metric_for_best_model requires eval_autoregressive_loss to exist,
                # so only add the text eval set alongside a non-empty MIDI one --
                # never as the sole eval_datasets entry (see the WARNING branch
                # below for why an empty ar_examples must leave eval inactive).
                if stage1 and text_eval_examples:
                    eval_datasets["text"] = ListDataset(text_eval_examples)
            elif is_main:
                print("WARNING: no held-out file produced an eval example "
                      "(too short / --eval-limit too small) -- skipping eval this run.")
            if is_main and ar_examples:
                print(f"Validation examples: autoregressive={len(ar_examples):,}  "
                      f"anticipation={len(ant_examples):,}  "
                      f"(from {len(valid_files):,} held-out files)")

    else:
        if is_main:
            print(f"Loading corpus: {args.corpus}")
        chunks = np.load(args.corpus, mmap_mode="r")
        if is_main:
            print(f"  Shape: {chunks.shape}  dtype: {chunks.dtype}")
        dataset = AMTChunkDataset(chunks, id_map)

    # W&B: initialize before Trainer so we can pass the full config dict.
    # Trainer detects an active wandb run and skips re-initialization.
    wandb_module      = None
    wandb_step_offset = 0
    if is_main and args.wandb:
        import re
        import wandb as wandb_module
        # Deterministic run id so a resumed job reconnects to the same W&B run.
        run_id = args.run_name or re.sub(r"[^A-Za-z0-9_-]", "_", run_dir)

        wandb_step_offset = (
            args.wandb_step_offset if args.wandb_step_offset is not None
            else (_auto_wandb_step_offset(args.model) or 0)
        )
        if wandb_step_offset:
            print(f"W&B step offset: {wandb_step_offset:,} (continuing run '{run_id}')")

        wandb_module.init(
            project=args.wandb_project,
            name=args.run_name,
            id=run_id,
            resume="allow",
            group=args.wandb_group,
            tags=args.wandb_tags.split(",") if args.wandb_tags else None,
            config={
                "model":            args.model,
                "window":           args.window,
                "batch_size":       args.batch_size,
                "grad_accum":       args.grad_accum,
                "world_size":       world_size,
                "eff_tokens":       eff_tokens,
                "lr":               args.lr,
                "weight_decay":     args.weight_decay,
                "warmup_steps":     args.warmup_steps,
                "max_steps":        args.max_steps,
                "epochs":           args.epochs,
                "streaming":        streaming,
                "ant_prob":         args.ant_prob if streaming else None,
                "captioned":        captioned,
                "max_text_tokens":  args.max_text_tokens if captioned else None,
                "caption_cache_dir": args.caption_cache_dir if captioned else None,
                "caption_expansion_factor": args.caption_expansion_factor if captioned and args.caption_cache_dir else None,
                "stage1":           stage1,
                "stage1_text_ratio": args.stage1_text_ratio if stage1 else None,
                "stage1_eval_files": args.stage1_eval_files if stage1 else None,
                "stage1_ant_prob":  args.stage1_ant_prob if stage1 else None,
                "stage1_expansion_factor": args.stage1_expansion_factor if stage1 else None,
                "wandb_step_offset": wandb_step_offset,
            },
        )

    # load_best_model_at_end requires save_steps to be a round multiple of eval_steps,
    # so lock them together whenever eval is active (eval fires only on save points).
    eval_active = eval_datasets is not None
    eval_steps  = (args.eval_steps or args.save_steps) if eval_active else None
    save_steps  = eval_steps if eval_active else args.save_steps

    training_args = TrainingArguments(
        output_dir                  = ckpt_dir,
        max_steps                   = args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        weight_decay                = args.weight_decay,
        warmup_steps                = args.warmup_steps,
        lr_scheduler_type           = args.lr_scheduler_type,
        # bf16 takes priority; fp16 only activates when bf16 is off
        bf16                        = args.bf16,
        fp16                        = args.fp16 and not args.bf16,
        max_grad_norm               = 1.0,
        save_strategy                = "steps",
        save_steps                  = save_steps,
        # With load_best_model_at_end, Trainer always protects the best checkpoint
        # from rotation even under save_total_limit, so history isn't lost.
        save_total_limit            = 2,
        logging_steps               = args.log_steps,
        eval_strategy               = "steps" if eval_active else "no",
        eval_steps                  = eval_steps,
        load_best_model_at_end      = eval_active,
        # Autoregressive loss is canonical (no augmentation randomness) per the
        # AMT/MIDI-LLM convention — anticipation/combined are diagnostic only.
        metric_for_best_model       = "autoregressive_loss" if eval_active else None,
        greater_is_better           = False if eval_active else None,
        seed                        = args.seed,
        dataloader_num_workers      = args.workers,
        # W&B logging goes through MetricsLogCallback (which applies the step
        # offset), so Trainer's own W&B integration stays off.
        report_to                   = "none",
        run_name                    = args.run_name,
        # labels is in the CausalLM forward signature, but set False to be safe
        remove_unused_columns        = False,
        ddp_find_unused_parameters   = False,
        # Streaming datasets use fresh shuffles each epoch, so replaying N batches
        # on resume would process different data anyway and is extremely slow.
        ignore_data_skip             = streaming,
        # Restores the early-stopping patience counter when resuming from a checkpoint.
        restore_callback_states_from_checkpoint = True,
    )

    metrics_callback = (
        MetricsLogCallback(
            os.path.join(run_dir, "metrics.jsonl"),
            wandb_module=wandb_module,
            step_offset=wandb_step_offset,
        ) if is_main else None
    )
    callbacks = [SafeCheckpointCallback()]
    if metrics_callback is not None:
        callbacks.append(metrics_callback)

    # Train-to-convergence: let eval_autoregressive_loss plateauing end the run
    # rather than a fixed --max-steps (which stays as a safety ceiling). Needs the
    # same eval machinery load_best_model_at_end already relies on, so it is only
    # wired in when eval is active.
    if args.early_stopping_patience > 0:
        if eval_active:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience   = args.early_stopping_patience,
                early_stopping_threshold  = args.early_stopping_threshold,
            ))
            if is_main:
                print(f"Early stopping ENABLED: metric=eval_autoregressive_loss  "
                      f"patience={args.early_stopping_patience} evals "
                      f"(= {args.early_stopping_patience * (eval_steps or 0)} steps)  "
                      f"threshold={args.early_stopping_threshold}")
        elif is_main:
            print("WARNING: --early-stopping-patience set but eval is inactive "
                  "(no stage1 eval / no --captions) -- early stopping disabled.")

    comparable_loss_callback = None
    if is_main and eval_datasets is not None:
        comparable_loss_callback = ComparableLossCallback(ar_examples, args.batch_size)
        callbacks.append(comparable_loss_callback)

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = dataset,
        eval_dataset  = eval_datasets,
        data_collator = default_data_collator,
        processing_class = hf_tokenizer,  # saved alongside each checkpoint
        callbacks     = callbacks,
    )
    if comparable_loss_callback is not None:
        comparable_loss_callback.trainer = trainer

    # Remove any checkpoint dirs that were interrupted mid-save before detecting
    # which checkpoint to resume from, so only fully-written checkpoints are considered.
    if is_main:
        _cleanup_incomplete_checkpoints(ckpt_dir, verbose=True)

    # Auto-detect a previous Trainer checkpoint (checkpoint-{step} dirs with trainer_state.json)
    resume_ckpt = get_last_checkpoint(ckpt_dir)
    if is_main:
        if stage1:
            mode_label = "streaming stage1 (music-adjacent text + standalone MIDI)"
        elif captioned:
            mode_label = "streaming captioned (text + MIDI)"
        elif streaming:
            mode_label = "streaming (on-the-fly augmentation)"
        else:
            mode_label = "static corpus"
        print(f"\nFinetuning [{mode_label}]  "
              f"max_steps={args.max_steps}  batch={args.batch_size}  "
              f"grad_accum={args.grad_accum}  lr={args.lr}  wd={args.weight_decay}")
        if resume_ckpt:
            print(f"Resuming from checkpoint: {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    def _last_metric(key: str):
        """Most recent value of `key` in trainer.state.log_history (train loss lives
        under the plain "loss" key; eval losses under "eval_<name>_loss")."""
        for entry in reversed(trainer.state.log_history):
            if key in entry:
                return entry[key]
        return None

    def _fmt(x) -> str:
        return f"{x:.4f}" if x is not None else "n/a"

    train_loss    = _last_metric("loss")
    final_metrics = {"step": trainer.state.global_step, "train_loss": train_loss}

    # Final validation numbers. These are from the last evaluation; the saved
    # model (checkpoints/final) is the best checkpoint, whose step and loss are
    # in the last checkpoint's trainer_state.json (best_model_checkpoint,
    # best_metric). Autoregressive loss is the canonical number
    # (AMT/MIDI-LLM convention) since it involves no augmentation randomness;
    # anticipation and combined are supplementary diagnostics. If training
    # stopped on an eval step, the logged values are reused.
    if eval_datasets is not None:
        if eval_steps and trainer.state.global_step % eval_steps == 0:
            eval_metrics = {
                k: _last_metric(k) for k in
                ("eval_autoregressive_loss", "eval_anticipation_loss", "eval_combined_loss")
            }
        else:
            eval_metrics = trainer.evaluate()
            # trainer.train()'s on_train_end already flushed metrics_callback's
            # buffer; this standalone post-training eval needs its own flush.
            if metrics_callback is not None:
                metrics_callback._flush()
        final_metrics.update({
            "eval_autoregressive_loss": eval_metrics.get("eval_autoregressive_loss"),
            "eval_anticipation_loss":   eval_metrics.get("eval_anticipation_loss"),
            "eval_combined_loss":       eval_metrics.get("eval_combined_loss"),
            # MIDI-only nats per base token on the autoregressive set: the
            # number to compare across tokenizers.
            "eval_autoregressive_comparable_loss": _last_metric("eval_autoregressive_comparable_loss"),
            "canonical_metric":         "eval_autoregressive_loss",
        })

    if is_main:
        summary = f"Final loss — train: {_fmt(train_loss)}"
        if eval_datasets is not None:
            summary += (
                f"  validation autoregressive (canonical): {_fmt(final_metrics['eval_autoregressive_loss'])}"
                f"  anticipation: {_fmt(final_metrics['eval_anticipation_loss'])}"
                f"  combined: {_fmt(final_metrics['eval_combined_loss'])}"
                f"  autoregressive comparable (nats/base-token): {_fmt(final_metrics['eval_autoregressive_comparable_loss'])}"
            )
        print(f"\n{summary}")
        with open(os.path.join(run_dir, "final_metrics.json"), "w") as f:
            json.dump(final_metrics, f, indent=2)

    # Save the final model on rank 0
    if is_main:
        final_dir = os.path.join(ckpt_dir, "final")
        trainer.save_model(final_dir)
        hf_tokenizer.save_pretrained(final_dir)
        print(f"\nFinal model → {final_dir}")

    if is_main and args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
