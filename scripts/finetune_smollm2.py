"""
Finetune SmolLM2 on AMT-BPE symbolic music with anticipation augmentation.

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
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kw):  # silent fallback if tqdm not installed
        return it

try:
    from torch.utils.data import IterableDataset as _IterableDataset
except ImportError:
    _IterableDataset = object  # allows import without torch (linting)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)

# Default: BPE tokenizer trained with vel_bins=32, onset=10ms, dur=10ms, vocab=19485.
_DEFAULT_TOKENIZER = os.path.join(
    _REPO_ROOT, "bpe", "tokenizers", "quantization_sweep", "merges-8192",
    "velocity", "onset-10ms_duration-10ms_velocity-32bin",
    "tokenizers", "amt_compound_bpe_vocab19485.json",
)

# Default anticipation dir for vel=32
_DEFAULT_ANT_DIR = os.path.join(_REPO_ROOT, "bpe", "tokenizers", "anticipation",
                                "velocity", "onset-10ms_duration-10ms_velocity-32bin")

sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))



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


def _find_resume_checkpoint(ckpt_dir: str):
    """Return (path, global_step, epoch) of the latest resumable checkpoint, or (None, 0, 0)."""
    state_files = glob.glob(os.path.join(ckpt_dir, "*/training_state.json"))
    if not state_files:
        return None, 0, 0
    best_path, best_step, best_epoch = None, -1, 0
    for sf in state_files:
        with open(sf) as f:
            s = json.load(f)
        if s.get("global_step", 0) > best_step:
            best_step  = s["global_step"]
            best_epoch = s.get("epoch", 0)
            best_path  = os.path.dirname(sf)
    return best_path, best_step, best_epoch



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



def build_model_and_tokenizer(
    smollm2_model_id: str,
    amt_tok_path: str,
    verbose: bool = True,
    resume_from: str | None = None,
) -> tuple:
    """Load SmolLM2 and extend its vocab with AMT-BPE tokens.

    When resume_from is set, loads model+tokenizer from that checkpoint directory
    (vocab already extended) instead of from HuggingFace.

    Returns (model, hf_tokenizer, id_offset, id_map, amt_tokenizer).
    id_map[amt_bpe_id] = smollm2_extended_id.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tokenizers import Tokenizer as HFTokenizer

    src = resume_from if resume_from else smollm2_model_id
    if verbose:
        label = "Resuming model from" if resume_from else "Loading SmolLM2"
        print(f"{label}: {src}")

    hf_tokenizer = AutoTokenizer.from_pretrained(src)
    model        = AutoModelForCausalLM.from_pretrained(src)

    amt_tokenizer     = HFTokenizer.from_file(amt_tok_path)
    amt_vocab         = amt_tokenizer.get_vocab()
    amt_tokens_sorted = sorted(amt_vocab, key=amt_vocab.__getitem__)

    if resume_from:
        # Tokenizer already has AMT tokens; just build the id_map.
        id_offset = len(hf_tokenizer) - len(amt_tokens_sorted)
    else:
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
    p.add_argument("--window",      type=int, default=2048,
                   help="Sequence window size in tokens.")
    p.add_argument("--model",       default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--out-dir",     default=os.path.join(_REPO_ROOT, "bpe", "outputs", "runs", "smollm2"),
                   help="Base output dir. Run is placed in <out-dir>/<feature>/<bins>/ "
                        "mirroring the anticipation layout.")
    p.add_argument("--max-steps",   type=int,   default=25000,
                   help="Total gradient updates to run (primary training duration control).")
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
        import torch.distributed as dist
        from torch.utils.data import DataLoader
        from transformers import get_cosine_schedule_with_warmup
    except ImportError as e:
        print(f"ERROR: required package not found: {e}")
        sys.exit(1)

    torch.manual_seed(args.seed)

    # DDP initialization — torchrun sets LOCAL_RANK when launched with multi-GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    is_main = (local_rank == 0)

    # Auto-compute grad_accum so effective batch ≈ target_tokens.
    world_size = dist.get_world_size() if dist.is_initialized() else 1
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

    # run dir mirrors the anticipation layout
    if streaming:
        ant      = os.path.realpath(args.ant_dir)
        bins_str = os.path.basename(ant)            # e.g. onset-10ms_duration-10ms_velocity-32bin
        feature  = os.path.basename(os.path.dirname(ant))   # e.g. velocity
    else:
        bins_str = f"onset-{onset_ms:g}ms_duration-{dur_ms:g}ms_velocity-{vel_bins}bin"
        feature  = "velocity"
    run_dir  = os.path.join(args.out_dir, feature, bins_str)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    if is_main:
        print(f"Run dir: {run_dir}")

    resume_ckpt, resume_step, resume_epoch = _find_resume_checkpoint(ckpt_dir)
    if is_main and resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}  "
              f"(step={resume_step}, epoch={resume_epoch})")

    # model
    model, hf_tokenizer, id_offset, id_map, amt_tokenizer = build_model_and_tokenizer(
        args.model, tokenizer_path, verbose=is_main,
        resume_from=resume_ckpt,
    )

    # Save config for inference (rank 0 only)
    if is_main:
        with open(os.path.join(run_dir, "amt_config.json"), "w") as f:
            json.dump({
                "smollm2_model":   args.model,
                "amt_tokenizer":   tokenizer_path,
                "id_offset":       id_offset,
                "window_size":     args.window,
                "streaming_mode":  streaming,
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

        # ctrl token IDs in AMT-BPE vocab
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
            print(f"MIDI files: {len(midi_files):,}")

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
        sampler    = None  # _iter_shard handles DDP rank splitting for IterableDataset
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )
        # rough estimate: ~5000 BPE tokens per file per epoch on average
        n_chunks_est = len(midi_files) * 5_000 // args.window
        total_steps  = n_chunks_est // (args.batch_size * world_size) * args.epochs // args.grad_accum
        if is_main:
            print(f"Estimated chunks/epoch: {n_chunks_est:,}  "
                  f"estimated total steps: {total_steps:,}")

    else:
        if is_main:
            print(f"Loading corpus: {args.corpus}")
        chunks = np.load(args.corpus, mmap_mode="r")
        if is_main:
            print(f"  Shape: {chunks.shape}  dtype: {chunks.dtype}")
        n_chunks = len(chunks)
        dataset = AMTChunkDataset(chunks, id_map)
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True) \
                  if dist.is_initialized() else None
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size,
            sampler=sampler, shuffle=(sampler is None),
            num_workers=0, pin_memory=torch.cuda.is_available(),
        )
        total_steps = n_chunks // (args.batch_size * world_size) * args.epochs // args.grad_accum

    # optimizer
    has_cuda  = torch.cuda.is_available()
    device    = torch.device(f"cuda:{local_rank}" if has_cuda else "cpu")
    use_bf16  = args.bf16 and has_cuda
    use_fp16  = args.fp16 and not use_bf16 and has_cuda
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    if is_main:
        print(f"Device: {device}  (world_size={world_size})")
    model.to(device)

    # Wrap with DDP when torchrun launches multi-GPU
    if dist.is_initialized():
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if dist.is_initialized() else model

    max_steps = args.max_steps if args.max_steps > 0 else total_steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)  # bf16 never needs scaling

    if resume_ckpt:
        opt_path = os.path.join(resume_ckpt, "optimizer.pt")
        sch_path = os.path.join(resume_ckpt, "scheduler.pt")
        if os.path.exists(opt_path) and os.path.exists(sch_path):
            optimizer.load_state_dict(
                torch.load(opt_path, map_location=device, weights_only=True))
            scheduler.load_state_dict(
                torch.load(sch_path, map_location=device, weights_only=True))
            if is_main:
                print(f"Restored optimizer and scheduler states.")
        else:
            # Advance scheduler to match resume_step; optimizer moments are reset.
            for _ in range(resume_step):
                scheduler.step()
            if is_main:
                print(f"Advanced scheduler to step {resume_step} (no saved optimizer state).")

    # W&B (rank 0 only)
    if is_main and args.wandb:
        import wandb
        wandb.init(
            project="llm-midi-bpe",
            name=args.run_name,
            config={
                "model":          args.model,
                "window":         args.window,
                "batch_size":     args.batch_size,
                "grad_accum":     args.grad_accum,
                "world_size":     world_size,
                "eff_tokens":     eff_tokens,
                "lr":             args.lr,
                "warmup_steps":   args.warmup_steps,
                "max_steps":      max_steps,
                "epochs":         args.epochs,
                "streaming":      streaming,
                "ant_prob":       args.ant_prob if streaming else None,
            },
        )

    # training — step-based, single progress bar
    if is_main:
        mode_label = "streaming (on-the-fly augmentation)" if streaming else "static corpus"
        print(f"\nFinetuning [{mode_label}]  max_steps={max_steps}  "
              f"batch={args.batch_size}  grad_accum={args.grad_accum}  lr={args.lr}")

    global_step  = resume_step
    epoch        = resume_epoch
    last_ckpt    = None  # path of the previous rolling checkpoint (deleted on next save)
    pbar = _tqdm(total=max_steps, initial=resume_step, desc="Training", unit="step",
                 dynamic_ncols=True, disable=not is_main)

    while global_step < max_steps and epoch < args.epochs:
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        run_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(use_bf16 or use_fp16)):
                outputs = model(input_ids=input_ids, labels=labels)
                loss    = outputs.loss / args.grad_accum

            scaler.scale(loss).backward()
            run_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg    = run_loss / (step + 1)
                lr_now = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_now:.2e}")
                pbar.update(1)

                if is_main and args.wandb:
                    import wandb
                    wandb.log({
                        "train/loss":        avg,
                        "train/lr":          lr_now,
                        "train/data_pass":   epoch + 1,
                        "train/global_step": global_step,
                    }, step=global_step)

                if is_main and global_step % args.save_steps == 0:
                    ckpt = os.path.join(ckpt_dir, f"step_{global_step}")
                    raw_model.save_pretrained(ckpt)
                    hf_tokenizer.save_pretrained(ckpt)
                    torch.save(optimizer.state_dict(), os.path.join(ckpt, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(ckpt, "scheduler.pt"))
                    with open(os.path.join(ckpt, "training_state.json"), "w") as f:
                        json.dump({"global_step": global_step, "epoch": epoch}, f)
                    pbar.write(f"  Checkpoint → {ckpt}")
                    if last_ckpt is not None and os.path.isdir(last_ckpt):
                        import shutil
                        shutil.rmtree(last_ckpt)
                    last_ckpt = ckpt

                if global_step >= max_steps:
                    break

        epoch += 1
        if is_main:
            avg_pass = run_loss / max(1, step + 1)
            pbar.write(f"  [data pass {epoch}] avg loss: {avg_pass:.4f}")
            ckpt = os.path.join(ckpt_dir, f"pass_{epoch}")
            raw_model.save_pretrained(ckpt)
            hf_tokenizer.save_pretrained(ckpt)
            torch.save(optimizer.state_dict(), os.path.join(ckpt, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(ckpt, "scheduler.pt"))
            with open(os.path.join(ckpt, "training_state.json"), "w") as f:
                json.dump({"global_step": global_step, "epoch": epoch}, f)
            if last_ckpt is not None and os.path.isdir(last_ckpt):
                import shutil
                shutil.rmtree(last_ckpt)
            last_ckpt = ckpt

    pbar.close()

    if is_main:
        final_dir = os.path.join(ckpt_dir, "final")
        raw_model.save_pretrained(final_dir)
        hf_tokenizer.save_pretrained(final_dir)
        print(f"\nFinal model → {final_dir}")

    if is_main and args.wandb:
        import wandb
        wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
