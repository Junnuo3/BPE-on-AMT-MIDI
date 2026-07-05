"""
Finetune SmolLM2 on AMT-BPE symbolic music with anticipation augmentation.
Uses HuggingFace Trainer for distributed training, checkpointing, and logging.

Usage:
    python bpe/scripts/finetune_smollm2.py --ant-dir bpe/tokenizers/anticipation/velocity/onset-10ms_duration-10ms_velocity-32bin
"""

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

try:
    from torch.utils.data import IterableDataset as _IterableDataset
except ImportError:
    _IterableDataset = object  # allows import without torch (linting)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BPE_ROOT   = os.path.dirname(_SCRIPT_DIR)   # repo root
_REPO_ROOT  = os.path.dirname(_BPE_ROOT)     # one level up (only used for --dataset default)

# Default: BPE tokenizer trained with vel_bins=32, onset=10ms, dur=10ms, vocab=19485.
_DEFAULT_TOKENIZER = os.path.join(
    _BPE_ROOT, "tokenizers", "quantization_sweep", "merges-8192",
    "velocity", "onset-10ms_duration-10ms_velocity-32bin",
    "tokenizers", "amt_compound_bpe_vocab19485.json",
)

# Default anticipation dir for vel=32
_DEFAULT_ANT_DIR = os.path.join(_BPE_ROOT, "tokenizers", "anticipation",
                                "velocity", "onset-10ms_duration-10ms_velocity-32bin")

# Written inside a checkpoint dir after it is fully saved; absence means the
# save was interrupted and the checkpoint must not be resumed from.
_CKPT_DONE_MARKER = ".done"

sys.path.insert(0, os.path.join(_BPE_ROOT, "src"))



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

    Each file is processed once per epoch. With probability ant_prob it receives
    anticipation augmentation (strategy sampled uniformly from span/random/instrument);
    otherwise it is autoregressive. Epochs count original-dataset passes.
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
        window_size: int = 4096,
        ant_prob: float = 0.9,  # probability of anticipation augmentation per file
        onset_ms: float = 10.0,
        dur_ms: float = 10.0,
        vel_bins: int = 32,
    ) -> None:
        self.midi_files     = midi_files
        self.bpe_tokenizer  = bpe_tokenizer
        self.b2c            = b2c
        self.ctrl_b2c       = ctrl_b2c
        self.ctrl_ids       = np.array(sorted(ctrl_ids), dtype=np.int64)
        self.autoregress_id = autoregress_id
        self.anticipate_id  = anticipate_id
        self.id_map         = id_map
        self.window_size    = window_size
        self.ant_prob       = ant_prob
        self.onset_ms       = onset_ms
        self.dur_ms         = dur_ms
        self.vel_bins       = vel_bins

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

        # Draw per-file augmentation decisions for this epoch.
        # k=0 → autoregressive; k∈[1,9] → anticipation strategy (span/random/instrument)
        use_ant = rng_epoch.random(n) < self.ant_prob
        k_vals  = np.where(use_ant, rng_epoch.integers(1, 10, size=n), 0)

        perm  = rng_epoch.permutation(n)
        tasks = [(self.midi_files[i], int(k_vals[i])) for i in perm]
        tasks = list(self._iter_shard(tasks))

        for path, k in tasks:
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

            # Each chunk is a separate ~100-second segment whose first event is at t=0
            # (guaranteed by midi_to_amt normalization). Process chunks independently so
            # no window ever crosses a chunk or file boundary.
            for chunk_text in chunk_texts:
                ids = self.bpe_tokenizer.encode(chunk_text).ids
                if len(ids) < self.window_size:
                    continue

                window = np.array(ids[: self.window_size], dtype=np.int64)

                has_ctrl   = bool(np.isin(window, self.ctrl_ids).any())
                prefix_id  = self.anticipate_id if has_ctrl else self.autoregress_id
                prefixed   = np.empty(self.window_size, dtype=np.int64)
                prefixed[0]  = prefix_id
                prefixed[1:] = window[: self.window_size - 1]

                mapped = torch.tensor(self.id_map[prefixed], dtype=torch.long)
                yield {"input_ids": mapped, "labels": mapped.clone()}



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


class AMTCaptionStreamDataset(_IterableDataset):
    """On-the-fly text-captioned MIDI dataset.

    Each example has the format:
        <HF text tokens>  [AUTOREGRESS / ANTICIPATE]  <AMT-BPE MIDI tokens>

    The text prefix is tokenized with the SmolLM2 HF tokenizer (base vocab range).
    MIDI tokens are produced by AMT-BPE and mapped to the extended vocabulary.
    The mode token ([AUTOREGRESS] or [ANTICIPATE]) sits between the two halves.

    Files without a matching caption in caption_map are silently skipped.
    Full causal-LM loss is computed over the entire window (text + MIDI).
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
        import torch
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

            # Tokenize text prefix with the HF tokenizer (base vocab, no mapping needed)
            full_text = self.system_prompt + caption + " "
            text_ids  = self.hf_tokenizer.encode(full_text, add_special_tokens=False)
            text_ids  = text_ids[: self.max_text_tokens]
            text_len  = len(text_ids)

            midi_budget = self.window_size - text_len - 1  # -1 for the mode token
            if midi_budget < 1:
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
                amt_ids = self.bpe_tokenizer.encode(chunk_text).ids
                if len(amt_ids) < midi_budget:
                    continue

                amt_window = np.array(amt_ids[:midi_budget], dtype=np.int64)
                has_ctrl   = bool(np.isin(amt_window, self.ctrl_ids).any())
                mode_bpe   = self.anticipate_id if has_ctrl else self.autoregress_id

                # Map AMT-BPE IDs to the extended SmolLM2 vocab
                mode_mapped = int(self.id_map[mode_bpe])
                midi_mapped = self.id_map[amt_window]

                # Assemble: [text (base vocab)] [mode token] [MIDI (extended vocab)]
                seq = np.empty(self.window_size, dtype=np.int64)
                seq[:text_len]     = text_ids
                seq[text_len]      = mode_mapped
                seq[text_len + 1:] = midi_mapped

                mapped = torch.tensor(seq, dtype=torch.long)
                yield {"input_ids": mapped, "labels": mapped.clone()}


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
    p.add_argument("--dataset",     default=os.path.join(_REPO_ROOT, "dataset", "lmd_matched"),
                   help="MIDI dataset directory (streaming mode).")
    p.add_argument("--ant-prob", type=float, default=0.9,
                   help="Per-file probability of anticipation augmentation (streaming mode). "
                        "Complement uses autoregressive mode. Epochs count original-file passes.")
    p.add_argument("--captions", default=None,
                   help="MidiCaps JSONL file (e.g. dataset/lmd_matched_captions.jsonl). "
                        "Filters the training file list to only MIDI files that have a "
                        "matching caption entry (by md5). Use --caption-text to also "
                        "prepend the caption to each training example.")
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
    p.add_argument("--window",      type=int, default=2048,
                   help="Sequence window size in tokens.")
    p.add_argument("--model",       default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--out-dir",     default=os.path.join(_BPE_ROOT, "outputs", "runs", "smollm2"),
                   help="Base output dir. Run is placed in <out-dir>/<feature>/<bins>/ "
                        "mirroring the anticipation layout.")
    p.add_argument("--max-steps",   type=int,   default=25000,
                   help="Total gradient updates to run (primary training duration control). "
                        "Set to 0 to rely on --epochs instead.")
    p.add_argument("--epochs",      type=int,   default=10,
                   help="Max data passes (safety ceiling; training stops at --max-steps first).")
    p.add_argument("--batch-size",  type=int,   default=4)
    p.add_argument("--grad-accum",  type=int,   default=None,
                   help="Gradient accumulation steps. Auto-computed from --target-tokens if omitted.")
    p.add_argument("--target-tokens", type=int, default=500_000,
                   help="Target effective batch size in tokens "
                        "(seq_len × per-GPU-batch × num-GPUs × accum). "
                        "Used to auto-compute --grad-accum when not explicitly set.")
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int,  default=200)
    p.add_argument("--save-steps",   type=int,  default=200)
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
    p.add_argument("--run-name",     default=None,
                   help="W&B run name (default: auto).")
    return p.parse_args()



def main() -> None:
    args = parse_args()

    try:
        import torch
        from transformers import TrainingArguments, Trainer, TrainerCallback
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

    # run dir mirrors the anticipation layout; captioned runs get an extra prefix
    if streaming:
        ant      = os.path.realpath(args.ant_dir)
        bins_str = os.path.basename(ant)
        feature  = os.path.basename(os.path.dirname(ant))
    else:
        bins_str = f"onset-{onset_ms:g}ms_duration-{dur_ms:g}ms_velocity-{vel_bins}bin"
        feature  = "velocity"
    subdirs  = (["captioned", feature, bins_str] if captioned else [feature, bins_str])
    run_dir  = os.path.join(args.out_dir, *subdirs)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    if is_main:
        print(f"Run dir: {run_dir}")

    # model — always build from base; Trainer overlays checkpoint weights when resuming
    model, hf_tokenizer, id_offset, id_map, amt_tokenizer = build_model_and_tokenizer(
        args.model, tokenizer_path, verbose=is_main,
    )

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
            }, f, indent=2)

    # dataset
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
                  "Run bpe/scripts/double_tokenizer.py first.")
            sys.exit(1)

        cache_dir  = os.path.join(os.path.dirname(os.path.dirname(tokenizer_path)), "cache")
        midi_files = load_filelist_cache(cache_dir, limit_files=None)
        if midi_files is None:
            if is_main:
                print(f"No file list cache found; scanning {args.dataset} …")
            midi_files = find_midi_files(args.dataset)
        if is_main:
            print(f"MIDI files (full): {len(midi_files):,}")

        # When --captions is given, restrict both variants to caption-matched files so
        # any later comparison reflects conditioning, not a different data subset.
        caption_map = None
        if args.captions is not None:
            caption_map = load_caption_map(args.captions)
            captioned_md5s = set(caption_map.keys())
            midi_files = [f for f in midi_files
                          if os.path.splitext(os.path.basename(f))[0] in captioned_md5s]
            if is_main:
                print(f"Captions loaded: {len(caption_map):,} entries from {args.captions}")
                print(f"MIDI files (caption-matched): {len(midi_files):,}")

        if captioned:
            system_prompt = args.system_prompt or AMTCaptionStreamDataset._DEFAULT_SYSTEM_PROMPT
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
                window_size    = args.window,
                ant_prob       = args.ant_prob,
                onset_ms       = onset_ms,
                dur_ms         = dur_ms,
                vel_bins       = vel_bins,
            )
            if is_main:
                print(f"Estimated chunks/epoch: {len(midi_files) * 5_000 // args.window:,}")

    else:
        if is_main:
            print(f"Loading corpus: {args.corpus}")
        chunks = np.load(args.corpus, mmap_mode="r")
        if is_main:
            print(f"  Shape: {chunks.shape}  dtype: {chunks.dtype}")
        dataset = AMTChunkDataset(chunks, id_map)

    # W&B: initialize before Trainer so we can pass the full config dict.
    # Trainer detects an active wandb run and skips re-initialization.
    if is_main and args.wandb:
        import wandb
        wandb.init(
            project="llm-midi-bpe",
            name=args.run_name,
            config={
                "model":            args.model,
                "window":           args.window,
                "batch_size":       args.batch_size,
                "grad_accum":       args.grad_accum,
                "world_size":       world_size,
                "eff_tokens":       eff_tokens,
                "lr":               args.lr,
                "warmup_steps":     args.warmup_steps,
                "max_steps":        args.max_steps,
                "epochs":           args.epochs,
                "streaming":        streaming,
                "ant_prob":         args.ant_prob if streaming else None,
                "captioned":        captioned,
                "max_text_tokens":  args.max_text_tokens if captioned else None,
            },
        )

    training_args = TrainingArguments(
        output_dir                  = ckpt_dir,
        max_steps                   = args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        warmup_steps                = args.warmup_steps,
        lr_scheduler_type           = "cosine",
        # bf16 takes priority; fp16 only activates when bf16 is off
        bf16                        = args.bf16,
        fp16                        = args.fp16 and not args.bf16,
        max_grad_norm               = 1.0,
        save_steps                  = args.save_steps,
        save_total_limit            = 2,      # keep the 2 most recent checkpoints
        logging_steps               = args.log_steps,
        seed                        = args.seed,
        dataloader_num_workers      = args.workers,
        report_to                   = "wandb" if args.wandb else "none",
        run_name                    = args.run_name,
        # labels is in the CausalLM forward signature, but set False to be safe
        remove_unused_columns        = False,
        ddp_find_unused_parameters   = False,
        # Streaming datasets use fresh shuffles each epoch, so replaying N batches
        # on resume would process different data anyway and is extremely slow.
        ignore_data_skip             = streaming,
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = dataset,
        data_collator = default_data_collator,
        processing_class = hf_tokenizer,  # saved alongside each checkpoint
        callbacks     = [SafeCheckpointCallback()],
    )

    # Remove any checkpoint dirs that were interrupted mid-save before detecting
    # which checkpoint to resume from, so only fully-written checkpoints are considered.
    if is_main:
        _cleanup_incomplete_checkpoints(ckpt_dir, verbose=True)

    # Auto-detect a previous Trainer checkpoint (checkpoint-{step} dirs with trainer_state.json)
    resume_ckpt = get_last_checkpoint(ckpt_dir)
    if is_main:
        if captioned:
            mode_label = "streaming captioned (text + MIDI)"
        elif streaming:
            mode_label = "streaming (on-the-fly augmentation)"
        else:
            mode_label = "static corpus"
        print(f"\nFinetuning [{mode_label}]  "
              f"max_steps={args.max_steps}  batch={args.batch_size}  "
              f"grad_accum={args.grad_accum}  lr={args.lr}")
        if resume_ckpt:
            print(f"Resuming from checkpoint: {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)

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
