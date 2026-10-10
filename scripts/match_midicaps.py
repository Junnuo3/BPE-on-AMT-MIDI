#!/usr/bin/env python3
"""
Match LMD-full MIDI files with MidiCaps text captions.

Downloads train.json from amaai-lab/MidiCaps via huggingface_hub and matches
by MD5 hex ID (filename stem) against dataset/lmd_full. MidiCaps captions
the full Lakh MIDI Dataset (its `location` field points into lmd_full/), so
matching against lmd_full recovers all 168,385 captions.

Output: dataset/lmd_full_captions.jsonl  (one JSON object per line,
  one row per unique MD5 — see dedup note below)
  {"md5": "<hash>", "midi_path": "...", "caption": "...", ...}

Some LMD MIDI files are physically duplicated under multiple directories.
MidiCaps' own train.json is already one entry per unique MD5, so we keep
exactly one physical path per MD5 here too rather than emitting a duplicate
row per duplicate path.

Resumable: if the output file already exists, already-processed MD5s are
skipped so the run picks up where it left off.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ID = "amaai-lab/MidiCaps"
TRAIN_FILENAME = "train.json"

KEEP_FIELDS = {
    "location",
    "caption",
    "genre",
    "genre_prob",
    "mood",
    "mood_prob",
    "key",
    "time_signature",
    "tempo",
    "tempo_word",
    "duration",
    "duration_word",
    "chord_summary",
    "instrument_summary",
    "instrument_numbers_sorted",
    "test_set",
}


def build_md5_index(train_json_path: str) -> dict[str, dict]:
    index: dict[str, dict] = {}
    with open(train_json_path, "r") as f:
        for line in tqdm(f, desc="Indexing MidiCaps", unit=" entries"):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            md5 = Path(entry.get("location", "")).stem
            index[md5] = {k: v for k, v in entry.items() if k in KEEP_FIELDS}
    print(f"Indexed {len(index):,} MidiCaps entries.")
    return index


def load_already_done(output_path: Path) -> set[str]:
    """Read MD5s already written to the output JSONL (for resume)."""
    if not output_path.exists():
        return set()
    done: set[str] = set()
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["md5"])
                except (KeyError, json.JSONDecodeError):
                    pass
    print(f"Resuming: {len(done):,} entries already written, skipping them.")
    return done


def match_lmd_full(lmd_dir: Path, index: dict, output_path: Path) -> None:
    already_done = load_already_done(output_path)

    # One representative physical path per unique MD5 (first one found by rglob).
    # See module docstring: LMD can duplicate the same MD5 under several
    # directories, but each MD5 is exactly one song with exactly one
    # MidiCaps caption, so only one row should exist per MD5.
    md5_to_path: dict[str, Path] = {}
    for midi_path in lmd_dir.rglob("*.mid"):
        md5_to_path.setdefault(midi_path.stem, midi_path)

    captioned_md5s = md5_to_path.keys() & index.keys()
    todo = sorted(captioned_md5s - already_done)
    print(f"Unique captioned files to process: {len(todo):,} "
          f"(skipping {len(already_done):,} already done)")

    matched = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as out_f:
        for md5 in tqdm(todo, desc="Matching LMD files", unit=" files"):
            midi_path = md5_to_path[md5]
            entry = {"md5": md5, "midi_path": str(midi_path.relative_to(lmd_dir.parent))}
            entry.update(index[md5])
            out_f.write(json.dumps(entry) + "\n")
            matched += 1

    no_local_file = len(index) - len(captioned_md5s)
    total_done = len(already_done) + matched
    print(f"New matches: {matched:,}  |  Captioned but no local file: {no_local_file:,}  |  "
          f"Total unique MD5s in file: {total_done:,}")


def main():
    parser = argparse.ArgumentParser(description="Match lmd_full MIDIs to MidiCaps captions.")
    parser.add_argument(
        "--lmd-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset" / "lmd_full",
        help="Path to lmd_full directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset" / "lmd_full_captions.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    print(f"Downloading {TRAIN_FILENAME} from {REPO_ID} (cached after first run) ...")
    train_json_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=TRAIN_FILENAME,
        repo_type="dataset",
    )
    print(f"Loaded: {train_json_path}")

    index = build_md5_index(train_json_path)
    match_lmd_full(args.lmd_dir, index, args.output)
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
