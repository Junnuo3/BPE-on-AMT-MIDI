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

    10-pass schedule: pass 0 → [AUTOREGRESS], passes 1-9 → [ANTICIPATE] with control extraction.
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
        augment_passes: int = 10,
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
        self.augment_passes = augment_passes
        self.onset_ms       = onset_ms
        self.dur_ms         = dur_ms
        self.vel_bins       = vel_bins

    def _iter_shard(self, tasks):
        """Return only the tasks for this DataLoader worker."""
        import torch
        info = torch.utils.data.get_worker_info()
        if info is None:
            return tasks
        return tasks[info.id :: info.num_workers]

    def __iter__(self):
        import torch
        from bpe_utils import serialize_midi_anticipation

        tasks = [(p, k)
                 for p in self.midi_files
                 for k in range(self.augment_passes)]

        # Shuffle each epoch; per-file seeds stay deterministic for reproducibility.
        rng_order = np.random.default_rng()
        idx_order = rng_order.permutation(len(tasks)).tolist()
        tasks     = [tasks[i] for i in idx_order]
        tasks     = list(self._iter_shard(tasks))

        buffer: list[int] = []

        for path, k in tasks:
            seed = abs(hash(path)) % (2 ** 31) ^ (k * 1_000_003)
            rng  = np.random.default_rng(seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                text = serialize_midi_anticipation(
                    path, self.b2c, self.ctrl_b2c, k, rng,
                    onset_ms=self.onset_ms,
                    dur_ms=self.dur_ms,
                    vel_bins=self.vel_bins,
                )
            if text is None:
                continue

            ids = self.bpe_tokenizer.encode(text).ids
            buffer.extend(ids)

            while len(buffer) >= self.window_size:
                window = np.array(buffer[: self.window_size], dtype=np.int64)
                buffer = buffer[self.window_size :]

                has_ctrl   = bool(np.isin(window, self.ctrl_ids).any())
                prefix_id  = self.anticipate_id if has_ctrl else self.autoregress_id
                prefixed   = np.empty(self.window_size, dtype=np.int64)
                prefixed[0]  = prefix_id
                prefixed[1:] = window[: self.window_size - 1]

                mapped = torch.tensor(self.id_map[prefixed], dtype=torch.long)
                yield {"input_ids": mapped, "labels": mapped.clone()}



def build_model_and_tokenizer(smollm2_model_id: str, amt_tok_path: str) -> tuple:
    """Load SmolLM2 and extend its vocab with AMT-BPE tokens.

    Returns (model, hf_tokenizer, id_offset, id_map, amt_tokenizer).
    id_map[amt_bpe_id] = smollm2_extended_id.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tokenizers import Tokenizer as HFTokenizer

    print(f"Loading SmolLM2: {smollm2_model_id}")
    hf_tokenizer = AutoTokenizer.from_pretrained(smollm2_model_id)
    model        = AutoModelForCausalLM.from_pretrained(smollm2_model_id)

    id_offset = len(hf_tokenizer)
    print(f"  SmolLM2 vocab size: {id_offset:,}")

    amt_tokenizer     = HFTokenizer.from_file(amt_tok_path)
    amt_vocab         = amt_tokenizer.get_vocab()
    amt_tokens_sorted = sorted(amt_vocab, key=amt_vocab.__getitem__)

    n_added = hf_tokenizer.add_tokens(amt_tokens_sorted)
    model.resize_token_embeddings(len(hf_tokenizer))

    id_map = np.array(
        [hf_tokenizer.convert_tokens_to_ids(tok) for tok in amt_tokens_sorted],
        dtype=np.int64,
    )

    print(f"  AMT-BPE tokens added: {n_added:,}  (offset {id_offset:,})")
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
    p.add_argument("--augment-passes", type=int, default=10,
                   help="Augmentation passes per MIDI file per epoch (streaming mode).")
    p.add_argument("--window",      type=int, default=4096,
                   help="Sequence window size in tokens.")
    p.add_argument("--model",       default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--out-dir",     default=os.path.join(_REPO_ROOT, "bpe", "outputs", "runs", "smollm2"),
                   help="Base output dir. Run is placed in <out-dir>/<feature>/<bins>/ "
                        "mirroring the anticipation layout.")
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--batch-size",  type=int,   default=4)
    p.add_argument("--grad-accum",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int,  default=200)
    p.add_argument("--save-steps",   type=int,  default=500)
    p.add_argument("--fp16",         action="store_true", default=False)
    p.add_argument("--no-fp16",      dest="fp16", action="store_false")
    p.add_argument("--bf16",         action="store_true", default=True,
                   help="BFloat16 mixed precision (default; no grad scaler needed).")
    p.add_argument("--no-bf16",      dest="bf16", action="store_false")
    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--workers",      type=int,  default=0,
                   help="DataLoader worker processes (streaming mode).")
    return p.parse_args()



def main() -> None:
    args = parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup
    except ImportError as e:
        print(f"ERROR: required package not found: {e}")
        sys.exit(1)

    torch.manual_seed(args.seed)

    streaming = args.corpus is None

    # tokenizer
    tokenizer_path = args.tokenizer or _auto_tokenizer(args.corpus, args.ant_dir)
    if tokenizer_path is None:
        print("ERROR: --tokenizer not given and could not auto-detect one.")
        sys.exit(1)
    print(f"AMT-BPE tokenizer: {os.path.basename(tokenizer_path)}")

    # quantization params
    cfg      = _auto_train_cfg(tokenizer_path)
    onset_ms = cfg.get("onset_ms", 10.0)
    dur_ms   = cfg.get("dur_ms",   10.0)
    vel_bins = cfg.get("vel_bins", 32)
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
    print(f"Run dir: {run_dir}")

    # model
    model, hf_tokenizer, id_offset, id_map, amt_tokenizer = build_model_and_tokenizer(
        args.model, tokenizer_path,
    )

    # Save config for inference
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
        print(f"Mapping: {mapping_path}")

        b2c, c2b, ctrl_b2c, ctrl_c2b = load_mapping_doubled(mapping_path)

        # ctrl token IDs in AMT-BPE vocab
        amt_vocab  = amt_tokenizer.get_vocab()
        ctrl_chars = set(ctrl_b2c.values())
        ctrl_ids   = {amt_vocab[c] for c in ctrl_chars if c in amt_vocab}
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
            print(f"No file list cache found; scanning {args.dataset} …")
            midi_files = find_midi_files(args.dataset)
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
            augment_passes = args.augment_passes,
            onset_ms       = onset_ms,
            dur_ms         = dur_ms,
            vel_bins       = vel_bins,
        )
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )
        # rough estimate: ~5000 BPE tokens per file per pass on average
        n_chunks_est = len(midi_files) * args.augment_passes * 5_000 // args.window
        total_steps  = n_chunks_est // args.batch_size * args.epochs // args.grad_accum
        print(f"Estimated chunks/epoch: {n_chunks_est:,}  "
              f"estimated total steps: {total_steps:,}")

    else:
        print(f"Loading corpus: {args.corpus}")
        chunks = np.load(args.corpus, mmap_mode="r")
        print(f"  Shape: {chunks.shape}  dtype: {chunks.dtype}")
        n_chunks = len(chunks)
        dataset    = AMTChunkDataset(chunks, id_map)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=torch.cuda.is_available(),
        )
        total_steps = n_chunks // args.batch_size * args.epochs // args.grad_accum

    # optimizer
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_cuda = torch.cuda.is_available()
    # bf16 takes priority; fp16 is a fallback; bf16 never needs a grad scaler
    use_bf16 = args.bf16 and has_cuda
    use_fp16 = args.fp16 and not use_bf16 and has_cuda
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Device: {device}")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)  # bf16 never needs scaling

    # training
    mode_label = "streaming (on-the-fly augmentation)" if streaming else "static corpus"
    print(f"\nFinetuning [{mode_label}]  epochs={args.epochs}  "
          f"batch={args.batch_size}  grad_accum={args.grad_accum}")

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = _tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}",
                     unit="batch", dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(use_bf16 or use_fp16)):
                outputs = model(input_ids=input_ids, labels=labels)
                loss    = outputs.loss / args.grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg = epoch_loss / max(1, step + 1)
                lr_now = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_now:.2e}", step=global_step)

                if global_step % args.save_steps == 0:
                    ckpt = os.path.join(ckpt_dir, f"step_{global_step}")
                    model.save_pretrained(ckpt)
                    hf_tokenizer.save_pretrained(ckpt)
                    pbar.write(f"  Checkpoint → {ckpt}")

        pbar.close()
        avg = epoch_loss / max(1, step + 1)
        print(f"Epoch {epoch+1}/{args.epochs} — avg loss: {avg:.4f}")

        ckpt = os.path.join(ckpt_dir, f"epoch_{epoch+1}")
        model.save_pretrained(ckpt)
        hf_tokenizer.save_pretrained(ckpt)
        print(f"  Epoch checkpoint → {ckpt}")

    final_dir = os.path.join(ckpt_dir, "final")
    model.save_pretrained(final_dir)
    hf_tokenizer.save_pretrained(final_dir)
    print(f"\nFinal model → {final_dir}")


if __name__ == "__main__":
    main()
