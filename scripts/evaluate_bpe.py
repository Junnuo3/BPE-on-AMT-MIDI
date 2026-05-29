"""
Evaluate BPE compression on AMT MIDI sequences.

Usage:
    python bpe/scripts/evaluate_bpe.py --dataset dataset/lmd_matched --out outputs
Resumable — re-run to continue from the last checkpoint.
Use --force-reeval to start fresh.
"""

import sys
import os
import argparse
import json
import csv
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tqdm import tqdm
from tokenizers import Tokenizer

from bpe_utils import find_midi_files, load_mapping, serialize_midi, load_corpus_unconditional

EVAL_CKPT_INTERVAL = 1_000
EVAL_BATCH_SIZE    = 512
_JOINT_CKPT        = "eval_joint_checkpoint.json"


def _partial_path(results_dir: str, vocab_size: int) -> str:
    return os.path.join(results_dir, f"eval_partial_vocab{vocab_size}.json")


def _save_partial(results_dir: str, data: dict) -> None:
    path = _partial_path(results_dir, data["requested_vocab_size"])
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def _load_partial(results_dir: str, vocab_size: int) -> dict | None:
    path = _partial_path(results_dir, vocab_size)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _joint_ckpt_path(results_dir: str) -> str:
    return os.path.join(results_dir, _JOINT_CKPT)


def _save_joint_ckpt(results_dir: str, data: dict) -> None:
    path = _joint_ckpt_path(results_dir)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def _load_joint_ckpt(
    results_dir: str, active_sizes: list[int],
    source_type: str, source_len: int,
) -> dict | None:
    path = _joint_ckpt_path(results_dir)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        ckpt = json.load(f)
    if sorted(ckpt.get("tokenizer_vocab_sizes", [])) != sorted(active_sizes):
        print("  Joint checkpoint: tokenizer set changed — ignoring.")
        return None
    ckpt_source = ckpt.get("source_type", "midi")
    if ckpt_source != source_type:
        print(f"  Joint checkpoint: source changed ({ckpt_source!r} → {source_type!r}) — ignoring.")
        return None
    if ckpt.get("source_len") != source_len:
        print("  Joint checkpoint: source size changed — ignoring.")
        return None
    return ckpt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset/lmd_matched")
    p.add_argument("--out", default="outputs")
    p.add_argument("--limit-files", type=int, default=None)
    p.add_argument("--force-reeval", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = os.path.join(args.out, "results")
    os.makedirs(results_dir, exist_ok=True)

    mapping_path = os.path.join(args.out, "mappings", "amt_base_token_char_mapping.json")
    if not os.path.exists(mapping_path):
        print(f"ERROR: mapping not found at {mapping_path}. Run train_bpe.py first.")
        sys.exit(1)

    print(f"Loading mapping from {mapping_path} …")
    base_token_to_char, _ = load_mapping(mapping_path)
    print(f"  {len(base_token_to_char)} base tokens")

    # Prefer pre-serialized corpus; fall back to raw MIDI
    cache_dir = os.path.join(args.out, "cache")
    corpus: list[str] | None = None if args.limit_files else load_corpus_unconditional(cache_dir)

    if corpus is not None:
        source_items: list[str] = corpus
        source_type              = "corpus"
        print(f"Corpus cache loaded — {len(corpus):,} strings")
    else:
        print(f"Scanning {args.dataset} …")
        midi_files = find_midi_files(args.dataset)
        if args.limit_files:
            midi_files = midi_files[: args.limit_files]
        source_items = midi_files
        source_type  = "midi"
        print(f"  {len(midi_files):,} files")

    source_len = len(source_items)

    tok_dir   = os.path.join(args.out, "tokenizers")
    tok_paths = sorted(glob.glob(os.path.join(tok_dir, "amt_compound_bpe_vocab*.json")))
    if not tok_paths:
        print(f"ERROR: no tokenizers in {tok_dir}. Run train_bpe.py first.")
        sys.exit(1)

    tokenizer_meta: list[tuple[int, str]] = []
    for tok_path in tok_paths:
        basename = os.path.basename(tok_path)
        try:
            vs = int(basename.replace("amt_compound_bpe_vocab", "").replace(".json", ""))
        except ValueError:
            vs = -1
        tokenizer_meta.append((vs, tok_path))

    print(f"Found {len(tokenizer_meta)} tokenizer(s)")

    done_results: list[dict] = []
    active_meta:  list[tuple[int, str]] = []

    for vs, tok_path in tokenizer_meta:
        cached = None if args.force_reeval else _load_partial(results_dir, vs)
        if cached is not None and cached.get("status") == "complete":
            print(f"  [skip] vocab={vs} already evaluated")
            done_results.append(cached)
        else:
            active_meta.append((vs, tok_path))

    active_sizes = [vs for vs, _ in active_meta]

    if not active_meta:
        print("All tokenizers already evaluated.")
        all_results = done_results
    else:
        print(f"\nLoading {len(active_meta)} tokenizer(s) …")
        tokenizers: dict[int, Tokenizer] = {
            vs: Tokenizer.from_file(tok_path) for vs, tok_path in active_meta
        }

        totals: dict[int, dict] = {
            vs: {"num_files": 0, "total_original_tokens": 0, "total_bpe_tokens": 0}
            for vs in active_sizes
        }

        start_idx = 0
        if not args.force_reeval:
            ckpt = _load_joint_ckpt(results_dir, active_sizes, source_type, source_len)
            if ckpt is not None and ckpt.get("status") == "in_progress":
                start_idx = ckpt["files_processed"]
                for vs in active_sizes:
                    saved = ckpt["totals"].get(str(vs), {})
                    totals[vs]["num_files"]             = saved.get("num_files", 0)
                    totals[vs]["total_original_tokens"] = saved.get("total_original_tokens", 0)
                    totals[vs]["total_bpe_tokens"]      = saved.get("total_bpe_tokens", 0)
                print(f"Resuming from {start_idx:,} / {source_len:,}")

        def _ckpt_payload(files_processed: int) -> dict:
            return {
                "status": "in_progress",
                "source_type": source_type,
                "source_len": source_len,
                "tokenizer_vocab_sizes": active_sizes,
                "files_processed": files_processed,
                "totals": {str(vs): totals[vs] for vs in active_sizes},
            }

        _save_joint_ckpt(results_dir, _ckpt_payload(start_idx))

        remaining        = source_items[start_idx:]
        batch_texts:      list[str] = []
        batch_orig_lens:  list[int] = []
        next_abs_idx:     int       = start_idx

        def flush_batch() -> None:
            if not batch_texts:
                return
            for vs, tok in tokenizers.items():
                encodings = tok.encode_batch(batch_texts)
                t = totals[vs]
                t["num_files"]             += len(batch_texts)
                t["total_original_tokens"] += sum(batch_orig_lens)
                t["total_bpe_tokens"]      += sum(len(e.ids) for e in encodings)
            batch_texts.clear()
            batch_orig_lens.clear()

        with tqdm(remaining, initial=start_idx, total=source_len,
                  desc="files", unit="file") as bar:
            for rel_idx, item in enumerate(bar):
                text = item if source_type == "corpus" else serialize_midi(item, base_token_to_char)
                next_abs_idx = start_idx + rel_idx + 1
                if text is not None:
                    batch_texts.append(text)
                    batch_orig_lens.append(len(text.split()) * 5)

                if len(batch_texts) >= EVAL_BATCH_SIZE:
                    flush_batch()

                if next_abs_idx % EVAL_CKPT_INTERVAL == 0:
                    flush_batch()
                    _save_joint_ckpt(results_dir, _ckpt_payload(next_abs_idx))

        flush_batch()

        newly_done: list[dict] = []
        for vs in active_sizes:
            t          = totals[vs]
            total_orig = t["total_original_tokens"]
            total_bpe  = t["total_bpe_tokens"]
            metrics = {
                "requested_vocab_size": vs,
                "status":               "complete",
                "vocab_size":            tokenizers[vs].get_vocab_size(),
                "num_files":             t["num_files"],
                "total_original_tokens": total_orig,
                "total_bpe_tokens":      total_bpe,
                "reduction_ratio":       total_bpe  / total_orig if total_orig else float("nan"),
                "compression_factor":    total_orig / total_bpe  if total_bpe  else float("nan"),
            }
            _save_partial(results_dir, metrics)
            newly_done.append(metrics)

            print(f"\n  vocab={vs}")
            print(f"    files:           {metrics['num_files']:,}")
            print(f"    original tokens: {total_orig:,}")
            print(f"    BPE tokens:      {total_bpe:,}")
            print(f"    reduction ratio: {metrics['reduction_ratio']:.4f}")
            print(f"    compression:     {metrics['compression_factor']:.4f}x")

        _save_joint_ckpt(results_dir, {**_ckpt_payload(source_len), "status": "complete"})
        all_results = done_results + newly_done

    all_results.sort(key=lambda r: r["vocab_size"])

    json_path = os.path.join(results_dir, "length_reduction_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults (JSON) → {json_path}")

    csv_path = os.path.join(results_dir, "length_reduction_results.csv")
    if all_results:
        csv_fields = [k for k in all_results[0] if k not in ("status", "requested_vocab_size")]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
    print(f"Results (CSV)  → {csv_path}")

    print(f"\nDone. Evaluated: {len(all_results) - len(done_results)}, skipped: {len(done_results)}.")


if __name__ == "__main__":
    main()
