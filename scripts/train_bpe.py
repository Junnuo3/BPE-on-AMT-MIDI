"""
Train BPE tokenizers on AMT compound MIDI sequences.

Usage:
    python bpe/scripts/train_bpe.py --dataset dataset/lmd_matched --out outputs
Resumable — interrupt and re-run to continue where it stopped.
Use --force-rescan to ignore all caches.
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tqdm import tqdm

from bpe_utils import (
    find_midi_files,
    compute_corpus_fingerprint,
    build_mapping_resumable,
    save_mapping,
    load_mapping,
    load_mapping_if_fresh,
    serialize_midi_resumable,
    load_corpus_cache,
    load_corpus_unconditional,
    save_filelist_cache,
    load_filelist_cache,
    VOCAB_CKPT_INTERVAL,
    CORPUS_FLUSH_INTERVAL,
)
from bpe_train import train_bpe, SPECIAL_TOKENS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset/lmd_matched")
    p.add_argument("--out", default="outputs")
    p.add_argument("--time-resolution", type=int, default=100)
    p.add_argument("--vocab-sizes", type=int, nargs="+", default=None)
    p.add_argument("--limit-files", type=int, default=None)
    p.add_argument("--force-rescan", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tok_dir   = os.path.join(args.out, "tokenizers")
    map_dir   = os.path.join(args.out, "mappings")
    cache_dir = os.path.join(args.out, "cache")
    for d in (tok_dir, map_dir, cache_dir):
        os.makedirs(d, exist_ok=True)

    mapping_path = os.path.join(map_dir, "amt_base_token_char_mapping.json")

    midi_files = base_token_to_char = char_to_base_token = corpus = None

    # Fast path: all caches present → skip every MIDI-touching step
    if not args.force_rescan:
        cached_files = load_filelist_cache(cache_dir, args.limit_files, args.time_resolution)
        if cached_files is not None and os.path.exists(mapping_path):
            cached_corpus = load_corpus_unconditional(cache_dir)
            if cached_corpus is not None:
                try:
                    b2c, c2b       = load_mapping(mapping_path)
                    midi_files     = cached_files
                    base_token_to_char = b2c
                    char_to_base_token = c2b
                    corpus         = cached_corpus
                    print("All caches loaded — skipping file scan and serialisation.")
                    print(f"  {len(midi_files):,} files  |  "
                          f"{len(corpus):,} corpus strings  |  "
                          f"{len(base_token_to_char):,} base tokens")
                    print("  (Use --force-rescan if the dataset has changed.)")
                except Exception as exc:
                    print(f"  Cache load failed ({exc}); falling back to full scan.")
                    midi_files = None

    if midi_files is None:
        # Step 1: file list
        cached_files = (None if args.force_rescan
                        else load_filelist_cache(cache_dir, args.limit_files,
                                                 args.time_resolution))
        if cached_files is not None:
            midi_files = cached_files
            print(f"File list loaded from cache — {len(midi_files):,} files")
        else:
            print(f"Scanning {args.dataset} …")
            midi_files = find_midi_files(args.dataset)
            if args.limit_files:
                midi_files = midi_files[: args.limit_files]
            print(f"  {len(midi_files):,} files found")
            save_filelist_cache(midi_files, cache_dir,
                                args.limit_files, args.time_resolution)

        fingerprint = compute_corpus_fingerprint(midi_files, args.time_resolution)
        print(f"  Fingerprint: {fingerprint[:16]}…")

        # Step 2: vocab scan
        result = (None if args.force_rescan
                  else load_mapping_if_fresh(mapping_path, fingerprint))
        if result is not None:
            base_token_to_char, char_to_base_token = result
            print(f"  Mapping cache hit — {len(base_token_to_char):,} base tokens")
        else:
            print(f"  Vocab scan …  (checkpoint every {VOCAB_CKPT_INTERVAL:,} files)")
            base_token_to_char, char_to_base_token = build_mapping_resumable(
                midi_files, fingerprint, cache_dir
            )
            save_mapping(base_token_to_char, char_to_base_token,
                         mapping_path, fingerprint=fingerprint)
            print(f"  Mapping saved → {mapping_path} ({len(base_token_to_char):,} tokens)")

        # Step 3: corpus serialization
        corpus = (None if args.force_rescan
                  else load_corpus_cache(fingerprint, cache_dir))
        if corpus is not None:
            print(f"  Corpus cache hit — {len(corpus):,} strings")
        else:
            print(f"  Serialising …  (flushed every {CORPUS_FLUSH_INTERVAL:,} files)")
            corpus = serialize_midi_resumable(
                midi_files, base_token_to_char, fingerprint, cache_dir
            )
            print(f"  Corpus complete — {len(corpus):,} valid files")

    B = len(base_token_to_char)
    S = len(SPECIAL_TOKENS)
    print(f"\n  Base alphabet B={B},  special tokens S={S},  min vocab={B+S}")

    if args.vocab_sizes:
        raw_sizes = args.vocab_sizes
    else:
        raw_sizes = [B+S+512, B+S+1_024, B+S+2_048, B+S+4_096, B+S+8_192, B+S+16_384]

    vocab_sizes = sorted({vs for vs in raw_sizes if vs > B + S})
    if not vocab_sizes:
        print("ERROR: all requested vocab sizes are <= B + S.")
        sys.exit(1)

    print(f"  Vocab sizes: {vocab_sizes}")

    with open(os.path.join(args.out, "train_config.json"), "w") as f:
        json.dump({"base_alphabet_size": B, "num_special_tokens": S,
                   "vocab_sizes": vocab_sizes, "num_files": len(midi_files),
                   "num_corpus_strings": len(corpus),
                   "limit_files": args.limit_files,
                   "time_resolution": args.time_resolution}, f, indent=2)

    # Step 4: BPE training (skips existing tokenizer files)
    initial_alphabet = list(base_token_to_char.values())

    def make_iterator():
        return iter(corpus)

    print()
    skipped   = 0
    outer_bar = tqdm(vocab_sizes, desc="Tokenizers", unit="tokenizer", position=0)
    for vs in outer_bar:
        out_path = os.path.join(tok_dir, f"amt_compound_bpe_vocab{vs}.json")

        if os.path.exists(out_path):
            outer_bar.set_postfix_str(f"vocab={vs} SKIPPED")
            tqdm.write(f"  [skip] vocab={vs} already exists")
            skipped += 1
            continue

        outer_bar.set_postfix_str(f"vocab={vs} training…")
        tqdm.write(f"\n── vocab={vs} ──")
        tqdm.write(f"   corpus: {len(corpus):,}  target: {vs:,}")

        tokenizer = train_bpe(make_iterator(), vs, initial_alphabet, show_progress=True)
        tokenizer.save(out_path)

        # Re-dump with ensure_ascii so PUA chars appear as \uXXXX, not raw glyphs
        with open(out_path, encoding="utf-8") as _f:
            _data = json.load(_f)
        with open(out_path, "w", encoding="utf-8") as _f:
            json.dump(_data, _f, ensure_ascii=True, indent=2)

        actual = tokenizer.get_vocab_size()
        outer_bar.set_postfix_str(f"vocab={vs} done (actual={actual:,})")
        tqdm.write(f"   Saved → {out_path}  (actual: {actual:,})")

    outer_bar.close()
    print(f"\nDone. Trained: {len(vocab_sizes) - skipped}, skipped: {skipped}.")


if __name__ == "__main__":
    main()
