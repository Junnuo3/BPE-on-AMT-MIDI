#!/usr/bin/env python3
"""
Extract standalone MIDI files from the downloaded GigaMIDI v2.0.0 parquet shards
(dataset/gigamidi/v2.0.0/{train,validation,test}/*.parquet) into plain <md5>.mid
files under dataset/gigamidi/midi/, for use as MIDI-LLM stage-1 continued-
pretraining data.

Excludes every md5 in the MidiCaps-captioned Lakh set
(dataset/lmd_full_captions.jsonl), so no stage-2 train, validation or test song
appears in stage 1; GigaMIDI contains Lakh, so this matters. Rows with fewer
than 20 notes are skipped. GigaMIDI md5s are unique across all v2.0.0 shards.

Usage:
    python scripts/extract_gigamidi_midi.py
"""

import argparse
import glob
import json
import os

import pyarrow.parquet as pq

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_SCRIPT_DIR)

_DEFAULT_GIGAMIDI_DIR   = os.path.join(_REPO_ROOT, "dataset", "gigamidi", "v2.0.0")
_DEFAULT_OUT_DIR        = os.path.join(_REPO_ROOT, "dataset", "gigamidi", "midi")
_DEFAULT_EXCLUSION_FILE = os.path.join(_REPO_ROOT, "dataset", "lmd_full_captions.jsonl")

_MIN_TOTAL_NOTES = 20


def load_md5_exclusion_set(jsonl_path: str) -> set[str]:
    """md5s already used in stage-2 SFT (MidiCaps-captioned LMD)."""
    exclude: set[str] = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            exclude.add(json.loads(line)["md5"])
    return exclude


def _category_column(schema_names: list[str]) -> str | None:
    """GigaMIDI's category column name is the full enum-description string
    (an artifact of the HF dataset card export), e.g.
    'instrument_category: drums-only: 0, all-instruments-with-drums: 1,no-drums: 2'
    -- resolve defensively instead of hardcoding it."""
    for name in schema_names:
        if name.startswith("instrument_category"):
            return name
    return None


def extract(
    gigamidi_dir: str,
    out_dir: str,
    exclude: set[str],
    categories: set[int] | None,
    batch_size: int = 2048,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(gigamidi_dir, "*", "*.parquet")))
    print(f"Found {len(shards)} parquet shards under {gigamidi_dir}")

    n_kept = n_skip_stage2 = n_skip_notes = n_skip_category = n_skip_exists = 0

    for shard_path in shards:
        pf = pq.ParquetFile(shard_path)
        cat_col = _category_column(pf.schema_arrow.names)
        columns = ["music", "md5", "total_notes"] + ([cat_col] if cat_col else [])

        for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
            rows = batch.to_pylist()
            for row in rows:
                md5 = row["md5"]

                if md5 in exclude:
                    n_skip_stage2 += 1
                    continue
                if (row.get("total_notes") or 0) < _MIN_TOTAL_NOTES:
                    n_skip_notes += 1
                    continue
                if categories is not None and cat_col is not None and row[cat_col] not in categories:
                    n_skip_category += 1
                    continue

                target = os.path.join(out_dir, f"{md5}.mid")
                if os.path.exists(target):
                    n_skip_exists += 1
                    continue

                with open(target, "wb") as f:
                    f.write(row["music"])
                n_kept += 1

        print(f"  {os.path.relpath(shard_path, gigamidi_dir)}: "
              f"kept={n_kept:,} skip_stage2={n_skip_stage2:,} "
              f"skip_low_notes={n_skip_notes:,} skip_category={n_skip_category:,} "
              f"skip_exists={n_skip_exists:,}")

    print(f"\nDone. kept={n_kept:,} skip_stage2={n_skip_stage2:,} "
          f"skip_low_notes={n_skip_notes:,} skip_category={n_skip_category:,} "
          f"skip_already_extracted={n_skip_exists:,}")
    print(f"Output: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gigamidi-dir", default=_DEFAULT_GIGAMIDI_DIR)
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    parser.add_argument("--exclusion-file", default=_DEFAULT_EXCLUSION_FILE,
                         help="JSONL with a 'md5' field per line (stage-2 files to exclude).")
    parser.add_argument("--categories", default=None,
                         help="Comma-separated instrument_category values to keep "
                              "(0=drums-only, 1=all-instruments-with-drums, 2=no-drums). "
                              "Default: keep all.")
    args = parser.parse_args()

    categories = None
    if args.categories:
        categories = {int(c) for c in args.categories.split(",")}

    print(f"Loading stage-2 exclusion set from {args.exclusion_file} ...")
    exclude = load_md5_exclusion_set(args.exclusion_file)
    print(f"  {len(exclude):,} md5s to exclude")

    extract(args.gigamidi_dir, args.out_dir, exclude, categories)


if __name__ == "__main__":
    main()
