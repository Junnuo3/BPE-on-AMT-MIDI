"""
Generate one MIDI per caption for the test split (MIDI-LLM's 896 songs) or the
validation split (`--split valid`), in the layout scripts/run_evaluation.py reads.
The prompt is the system prompt plus the caption only.

Output layout (resumable -- an entry with a `.done` marker is skipped):
    <out-dir>/<md5>/prompt.txt
    <out-dir>/<md5>/sample_0000.mid
    <out-dir>/<md5>/sample_0001.mid   (if --num-samples > 1)
    <out-dir>/<md5>/.done

Usage:
    python scripts/generate_eval_set.py --checkpoint checkpoints/bpe/stage2 --split test \
        --temperature 1.0 --top-p 1.0 --top-k 40 --seed 0 --batch-size 256 \
        --out-dir generations/bpe_test

Shardable with --shard-id/--num-shards; batches and seeds do not depend on sharding.
"""

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_SCRIPT_DIR)

sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, _SCRIPT_DIR)

from eval_data import load_lakh_split  # noqa: E402
from generate_smollm2 import build_prompt, load_checkpoint_for_generation, tokens_to_midi  # noqa: E402

_DEFAULT_DATASET  = os.path.join(_REPO_ROOT, "dataset", "lmd_full")
_DEFAULT_CAPTIONS = os.path.join(_REPO_ROOT, "dataset", "lmd_full_captions.jsonl")
_DEFAULT_EVAL_IDS = os.path.join(_REPO_ROOT, "assets", "evaluation_set_lakh_ids.txt")
_DONE_MARKER = ".done"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir",    default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--step",       default=None)
    p.add_argument("--out-dir",    default=None,
                   help="Default: <run_dir>/generations/<checkpoint-name>/eval_lakh_<split>/")

    p.add_argument("--dataset",    default=_DEFAULT_DATASET)
    p.add_argument("--captions",   default=_DEFAULT_CAPTIONS)
    p.add_argument("--eval-ids",   default=_DEFAULT_EVAL_IDS,
                   help="MIDI-LLM's 896-song evaluation md5 list (test split). Must match "
                        "the value used elsewhere (finetune_smollm2.py's default).")
    p.add_argument("--split",      choices=["test", "valid"], default="test",
                   help="Which held-out split to prompt with / score against. Use 'valid' "
                        "for decoding-hyperparameter selection so test stays untouched.")
    p.add_argument("--subset-size", type=int, default=None,
                   help="Fixed random subset of the split (e.g. 896 to size-match the "
                        "~10.7K-song valid split to test). Must match between "
                        "generate_eval_set.py and run_evaluation.py.")
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--limit",      type=int, default=None, help="Cap number of prompts (smoke tests).")
    p.add_argument("--shard-id",   type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)

    p.add_argument("--mode",       choices=["autoregress", "anticipate"], default="autoregress")
    p.add_argument("--system-prompt",   default=None)
    p.add_argument("--max-text-tokens", type=int, default=None)

    p.add_argument("--num-samples",    type=int,   default=1, help="Samples per prompt.")
    p.add_argument("--batch-size",     type=int,   default=1,
                   help="Prompts per generate() call (left-padded). Batch composition "
                        "and per-batch seeds are fixed by the full test list, not the shard.")
    p.add_argument("--max-new-tokens", type=int,   default=None)
    p.add_argument("--do-sample",      action="store_true", default=True)
    p.add_argument("--greedy",         dest="do_sample", action="store_false")
    p.add_argument("--temperature",    type=float, default=1.0)
    p.add_argument("--top-p",          type=float, default=0.98)
    p.add_argument("--top-k",          type=int,   default=0,
                   help="0 disables top-k filtering.")
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--seed",           type=int,   default=None,
                   help="Batch b is seeded with seed + b (see main()).")
    p.add_argument("--device",         default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    loaded = load_checkpoint_for_generation(args.run_dir, args.checkpoint, args.step, args.device)
    run_dir            = loaded["run_dir"]
    checkpoint_dir      = loaded["checkpoint_dir"]
    amt_cfg            = loaded["amt_cfg"]
    hf_tokenizer       = loaded["hf_tokenizer"]
    model              = loaded["model"]
    char_to_base_token = loaded["char_to_base_token"]
    ctrl_char_to_base_token = loaded["ctrl_char_to_base_token"]
    onset_ms, dur_ms, vel_bins = loaded["onset_ms"], loaded["dur_ms"], loaded["vel_bins"]
    window_size        = loaded["window_size"]
    pad_token_id       = loaded["pad_token_id"]
    device             = loaded["device"]

    if not amt_cfg.get("caption_text", False):
        print("WARNING: this checkpoint was not finetuned with --caption-text; generations "
              "will be unconditioned even though prompt.txt records the MidiCaps caption. "
              "CLAP scores against this run will reflect that mismatch, not a bug.")

    ckpt_name = os.path.basename(checkpoint_dir.rstrip("/"))
    out_dir   = args.out_dir or os.path.join(run_dir, "generations", ckpt_name, f"eval_lakh_{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    test_entries = load_lakh_split(
        args.dataset, args.captions, args.eval_ids, split=args.split,
        subset_size=args.subset_size, subset_seed=args.subset_seed,
    )
    print(f"Prompting with {len(test_entries):,} '{args.split}' split entries.")
    if args.limit is not None:
        test_entries = test_entries[:args.limit]

    # Batches are formed over the full split before sharding, sorted by prompt
    # length (ties by md5) to minimize left-padding. Batch b is seeded with
    # seed + b, so a prompt's random draw depends on seed and batch size but not
    # on sharding or resumption. Changing --batch-size changes every draw.
    prompts = {
        e.md5: build_prompt(
            hf_tokenizer, amt_cfg,
            mode=args.mode, caption=e.caption,
            system_prompt=args.system_prompt, max_text_tokens=args.max_text_tokens,
        )
        for e in test_entries
    }
    ordered = sorted(test_entries, key=lambda e: (len(prompts[e.md5]), e.md5))
    batches = [ordered[i:i + args.batch_size] for i in range(0, len(ordered), args.batch_size)]
    my_batches = [(b, batch) for b, batch in enumerate(batches) if b % args.num_shards == args.shard_id]
    n_mine = sum(len(batch) for _, batch in my_batches)
    print(f"Shard {args.shard_id}/{args.num_shards}: {n_mine:,} prompts in "
          f"{len(my_batches):,} batches (batch_size={args.batch_size})")

    sampling_kwargs = {}
    if args.do_sample:
        sampling_kwargs = {
            "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        }
    eos_id = hf_tokenizer.eos_token_id

    n_done, n_built = 0, 0
    for b, batch in my_batches:
        if all(os.path.exists(os.path.join(out_dir, e.md5, _DONE_MARKER)) for e in batch):
            n_done += len(batch)
            continue

        prompt_lens = [len(prompts[e.md5]) for e in batch]
        max_len     = max(prompt_lens)
        # Per-prompt generation length is identical to unbatched generation
        # (window_size - prompt_len); the batch runs to the longest one and each
        # row is truncated to its own length below.
        keep = [args.max_new_tokens or max(1, window_size - n) for n in prompt_lens]
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, e in enumerate(batch):
            ids = prompts[e.md5]
            input_ids[i, max_len - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, max_len - len(ids):] = 1

        if args.seed is not None:
            torch.manual_seed(args.seed + b)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=max(keep),
                do_sample=args.do_sample,
                repetition_penalty=args.repetition_penalty,
                num_return_sequences=args.num_samples,
                pad_token_id=pad_token_id,
                **sampling_kwargs,
            )
        output_ids = output_ids[:, max_len:].cpu()

        for i, e in enumerate(batch):
            entry_dir = os.path.join(out_dir, e.md5)
            os.makedirs(entry_dir, exist_ok=True)
            with open(os.path.join(entry_dir, "prompt.txt"), "w", encoding="utf-8") as f:
                f.write(e.caption)

            n_generated, hit_eos = [], []
            for s in range(args.num_samples):
                generated_ids = output_ids[i * args.num_samples + s, :keep[i]].tolist()
                # Rows that finished early are padded with eos by generate(); cut there.
                eos_hit = eos_id is not None and eos_id in generated_ids
                if eos_hit:
                    generated_ids = generated_ids[:generated_ids.index(eos_id)]
                n_generated.append(len(generated_ids))
                hit_eos.append(eos_hit)
                pm, _, _ = tokens_to_midi(
                    generated_ids, hf_tokenizer, char_to_base_token, ctrl_char_to_base_token,
                    onset_ms, dur_ms, vel_bins,
                )
                pm.write(os.path.join(entry_dir, f"sample_{s:04d}.mid"))

            with open(os.path.join(entry_dir, "generation_config.json"), "w") as f:
                json.dump({
                    "checkpoint": checkpoint_dir, "md5": e.md5, "mode": args.mode,
                    "num_samples": args.num_samples, "prompt_tokens": prompt_lens[i],
                    "max_new_tokens": keep[i], "do_sample": args.do_sample,
                    "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                    "repetition_penalty": args.repetition_penalty, "seed": args.seed,
                    "batch_size": args.batch_size, "batch_index": b,
                    "generated_tokens": n_generated, "hit_eos": hit_eos,
                }, f, indent=2)

            open(os.path.join(entry_dir, _DONE_MARKER), "w").close()
            n_built += 1
        print(f"  batch {b}: built={n_built:,} already-done={n_done:,} / {n_mine:,} in this shard",
              flush=True)

    print(f"Done. built={n_built:,} already-done={n_done:,} / {len(test_entries):,} in this shard")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
