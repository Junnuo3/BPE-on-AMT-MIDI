"""
Train BPE tokenizers on AMT compound MIDI sequences.

Usage:
    python bpe/scripts/train_bpe.py
    python bpe/scripts/train_bpe.py --augment
"""

import sys
import os
import argparse
import json

_SCRIPTS_DIR     = os.path.dirname(os.path.abspath(__file__))
_BPE_ROOT        = os.path.dirname(_SCRIPTS_DIR)
_REPO_ROOT       = os.path.dirname(_BPE_ROOT)
_DEFAULT_DATASET = os.path.join(_REPO_ROOT, "dataset", "lmd_matched")
_DEFAULT_OUT          = os.path.join(_BPE_ROOT, "tokenizers", "vocab_sweep",
                                     "q_onset-10ms_duration-10ms_velocity-32bin")
_DEFAULT_OUT_NO_ONSET = os.path.join(_BPE_ROOT, "tokenizers", "merge_constraints",
                                     "no_onset_merge", "merges-8192")
_DEFAULT_ANT_ROOT     = os.path.join(_BPE_ROOT, "tokenizers", "anticipation")

sys.path.insert(0, os.path.join(_SCRIPTS_DIR, "..", "src"))

from tqdm import tqdm

from bpe_utils import (
    find_midi_files,
    compute_corpus_fingerprint,
    build_mapping_resumable,
    build_ctrl_mapping,
    save_mapping,
    load_mapping,
    load_mapping_if_fresh,
    is_mapping_doubled,
    load_mapping_doubled,
    serialize_midi_resumable,
    load_corpus_cache,
    load_corpus_unconditional,
    save_filelist_cache,
    load_filelist_cache,
    _default_workers,
    VOCAB_CKPT_INTERVAL,
    CORPUS_FLUSH_INTERVAL,
)
from bpe_train import train_bpe, double_tokenizer, SPECIAL_TOKENS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      default=_DEFAULT_DATASET)
    p.add_argument("--out",          default=None)
    p.add_argument("--onset-ms",     type=float, default=10.0,
                   help="Onset quantization step in ms (default 10 ms)")
    p.add_argument("--dur-ms",       type=float, default=10.0,
                   help="Duration quantization step in ms (default 10 ms)")
    p.add_argument("--vel-bins",     type=int, default=32,
                   help="Velocity quantization levels (default 32)")
    p.add_argument("--merges",       type=int, default=None,
                   help="Train to B+S+MERGES vocab size (alternative to --vocab-sizes)")
    p.add_argument("--vocab-sizes",  type=int, nargs="+", default=None)
    p.add_argument("--workers",      type=int, default=None,
                   help="Parallel workers for scan/serialise "
                        "(default: SLURM_CPUS_PER_TASK or all CPUs)")
    p.add_argument("--limit-files",       type=int, default=None)
    p.add_argument("--corpus-only",        action="store_true",
                   help="Build and cache the corpus then exit without BPE training. "
                        "Use this before a parallel vocab-size array job.")
    p.add_argument("--force-rescan",      action="store_true")
    p.add_argument("--onset-standalone",  action="store_true",
                   help="Keep onset tokens as standalone words; BPE only merges "
                        "duration/pitch/instrument/velocity 4-grams")
    p.add_argument("--augment",       action="store_true",
                   help="Anticipation mode: train BPE on the baseline corpus, then "
                        "double each tokenizer to add control-token counterparts. "
                        "Doubled tokenizers are saved under tokenizers/anticipation/<factor>/<config>/.")
    args = p.parse_args()

    # Default output dir depends on mode
    if args.out is None:
        if args.augment:
            onset = args.onset_ms; dur = args.dur_ms; vel = args.vel_bins
            config = f"onset-{onset:g}ms_duration-{dur:g}ms_velocity-{vel}bin"
            args.out = os.path.join(_DEFAULT_ANT_ROOT, "velocity", config)
        elif args.onset_standalone:
            args.out = _DEFAULT_OUT_NO_ONSET
        else:
            args.out = _DEFAULT_OUT
    return args


def main() -> None:
    args = parse_args()

    onset_ms         = args.onset_ms
    dur_ms           = args.dur_ms
    vel_bins         = args.vel_bins
    onset_standalone = args.onset_standalone
    augment          = args.augment
    n_workers        = args.workers if args.workers is not None else _default_workers()
    print(f"  Workers: {n_workers}")
    if augment:
        print("  Anticipation mode: train baseline BPE then double tokenizers")

    tok_dir   = os.path.join(args.out, "tokenizers")
    map_dir   = os.path.join(args.out, "mappings")
    cache_dir = os.path.join(args.out, "cache")
    for d in (tok_dir, map_dir, cache_dir):
        os.makedirs(d, exist_ok=True)

    mapping_path = os.path.join(map_dir, "amt_base_token_char_mapping.json")

    midi_files = base_token_to_char = char_to_base_token = corpus = None
    ctrl_b2c   = ctrl_c2b = None

    # fast path: all caches present, skip every MIDI-touching step
    if not args.force_rescan:
        cached_files = load_filelist_cache(cache_dir, args.limit_files)
        if cached_files is not None and os.path.exists(mapping_path):
            cached_corpus = load_corpus_unconditional(cache_dir)
            if cached_corpus is not None:
                try:
                    if augment and is_mapping_doubled(mapping_path):
                        b2c, c2b, cb2c, cc2b = load_mapping_doubled(mapping_path)
                        ctrl_b2c, ctrl_c2b   = cb2c, cc2b
                    else:
                        b2c, c2b = load_mapping(mapping_path)
                    midi_files         = cached_files
                    base_token_to_char = b2c
                    char_to_base_token = c2b
                    corpus             = cached_corpus
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
                        else load_filelist_cache(cache_dir, args.limit_files))
        if cached_files is not None:
            midi_files = cached_files
            print(f"File list loaded from cache — {len(midi_files):,} files")
        else:
            print(f"Scanning {args.dataset} …")
            midi_files = find_midi_files(args.dataset)
            if args.limit_files:
                midi_files = midi_files[: args.limit_files]
            print(f"  {len(midi_files):,} files found")
            save_filelist_cache(midi_files, cache_dir, args.limit_files)

        fingerprint = compute_corpus_fingerprint(
            midi_files, onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins,
            onset_standalone=onset_standalone,
        )
        print(f"  Fingerprint: {fingerprint[:16]}…")

        # Step 2: vocab scan
        result = (None if args.force_rescan
                  else load_mapping_if_fresh(mapping_path, fingerprint))
        if result is not None:
            base_token_to_char, char_to_base_token = result
            if augment and is_mapping_doubled(mapping_path):
                _, _, ctrl_b2c, ctrl_c2b = load_mapping_doubled(mapping_path)
            print(f"  Mapping cache hit — {len(base_token_to_char):,} base tokens")
        else:
            print(f"  Vocab scan …  (checkpoint every {VOCAB_CKPT_INTERVAL:,} files)")
            base_token_to_char, char_to_base_token = build_mapping_resumable(
                midi_files, fingerprint, cache_dir,
                onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins,
                n_workers=n_workers,
            )
            if augment:
                ctrl_b2c, ctrl_c2b = build_ctrl_mapping(base_token_to_char)
                save_mapping(base_token_to_char, char_to_base_token,
                             mapping_path, fingerprint=fingerprint,
                             ctrl_b2c=ctrl_b2c, ctrl_c2b=ctrl_c2b)
                print(f"  Doubled mapping saved → {mapping_path} "
                      f"({len(base_token_to_char):,} event + "
                      f"{len(ctrl_b2c):,} ctrl tokens)")
            else:
                save_mapping(base_token_to_char, char_to_base_token,
                             mapping_path, fingerprint=fingerprint)
                print(f"  Mapping saved → {mapping_path} "
                      f"({len(base_token_to_char):,} tokens)")

        # Step 3: serialize corpus
        corpus = (None if args.force_rescan
                  else load_corpus_cache(fingerprint, cache_dir))
        if corpus is not None:
            print(f"  Corpus cache hit — {len(corpus):,} strings")
        else:
            print(f"  Serialising …  (flushed every {CORPUS_FLUSH_INTERVAL:,} files)")
            corpus = serialize_midi_resumable(
                midi_files, base_token_to_char, fingerprint, cache_dir,
                onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins,
                onset_standalone=onset_standalone,
                n_workers=n_workers,
            )
            print(f"  Corpus complete — {len(corpus):,} valid files")

    if args.corpus_only:
        print(f"\nCorpus ready — {len(corpus):,} strings. Exiting (--corpus-only).")
        return

    if augment and ctrl_b2c is None:  # ctrl mapping may still be missing if we hit the cache fast path
        ctrl_b2c, ctrl_c2b = build_ctrl_mapping(base_token_to_char)

    # train on the baseline (event-only) corpus; augment mode doubles afterward
    initial_alphabet = list(base_token_to_char.values())
    special_toks = SPECIAL_TOKENS

    B = len(initial_alphabet)
    S = len(special_toks)
    print(f"\n  Base alphabet B={B},  special tokens S={S},  min vocab={B+S}")

    if args.merges is not None:
        raw_sizes = [B + S + args.merges]
    elif args.vocab_sizes:
        raw_sizes = args.vocab_sizes
    else:
        raw_sizes = [B+S+256, B+S+512, B+S+1_024, B+S+2_048, B+S+4_096,
                     B+S+8_192, B+S+16_384, B+S+32_768, B+S+65_536]

    vocab_sizes = sorted({vs for vs in raw_sizes if vs >= B + S})
    if not vocab_sizes:
        print("ERROR: all requested vocab sizes are < B + S.")
        sys.exit(1)

    print(f"  Vocab sizes: {vocab_sizes}")
    if augment:
        doubled_sizes = [2 * vs - S for vs in vocab_sizes]
        print(f"  Doubled vocab sizes (after doubling): {doubled_sizes}")

    with open(os.path.join(args.out, "train_config.json"), "w") as f:
        json.dump({"base_alphabet_size": B, "num_special_tokens": S,
                   "vocab_sizes": vocab_sizes, "num_files": len(midi_files),
                   "num_corpus_strings": len(corpus),
                   "limit_files": args.limit_files,
                   "onset_ms": onset_ms,
                   "dur_ms": dur_ms,
                   "vel_bins": vel_bins,
                   "onset_standalone": onset_standalone,
                   "augment": augment}, f, indent=2)

    # Step 4: BPE training (skips existing tokenizer files)
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

        tokenizer = train_bpe(make_iterator(), vs, initial_alphabet,
                              show_progress=True, special_tokens=special_toks)

        if augment:
            # add ctrl chars/merges and rename baseline specials to anticipation sentinels
            tokenizer = double_tokenizer(tokenizer, base_token_to_char, ctrl_b2c)

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
