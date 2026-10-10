#!/usr/bin/env python3
"""
Download the MIDI-LLM (Wu et al., 2025) stage-1 continued-pretraining datasets
into dataset/. Stage-2 data (MidiCaps captions + LMD-full MIDI) is handled by
match_midicaps.py.

  - MusicPile (Yuan et al., 2024)  -> dataset/musicpile/       (music-adjacent text)
  - GigaMIDI  (Lee et al., 2025)   -> dataset/gigamidi/         (standalone MIDI, v2.0.0)

GigaMIDI is gated: request access at
https://huggingface.co/datasets/Metacreation/GigaMIDI and export HF_TOKEN.
MusicPile is not gated.

Usage:
    python scripts/download_stage1_data.py --which all
    python scripts/download_stage1_data.py --which musicpile
    python scripts/download_stage1_data.py --which gigamidi
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATASET_DIR = _REPO_ROOT / "dataset"

MUSICPILE_REPO = "m-a-p/MusicPile"
GIGAMIDI_REPO = "Metacreation/GigaMIDI"


def _load_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")


def download_musicpile(token: str | None) -> None:
    out_dir = _DATASET_DIR / "musicpile"
    print(f"Downloading {MUSICPILE_REPO} -> {out_dir} ...")
    snapshot_download(
        repo_id=MUSICPILE_REPO,
        repo_type="dataset",
        local_dir=str(out_dir),
        allow_patterns=["data/*.parquet", "README.md", "*.json"],
        token=token,
    )
    print(f"MusicPile done: {out_dir}")


def download_gigamidi(token: str | None) -> None:
    out_dir = _DATASET_DIR / "gigamidi"
    print(f"Downloading {GIGAMIDI_REPO} (v2.0.0 split only) -> {out_dir} ...")
    # v2.0.0 is the up-to-date config per the dataset card; it stores raw MIDI
    # bytes + metadata directly in parquet, so we skip the redundant
    # v1.0.0 / all-instruments-with-drums / drums-only / no-drums configs and
    # the standalone zip+csv mirrors of the same content.
    try:
        snapshot_download(
            repo_id=GIGAMIDI_REPO,
            repo_type="dataset",
            local_dir=str(out_dir),
            allow_patterns=["v2.0.0/*", "README.md"],
            token=token,
            max_workers=4,
        )
    except GatedRepoError:
        print(
            "\nGigaMIDI is a gated dataset. Visit "
            "https://huggingface.co/datasets/Metacreation/GigaMIDI with the "
            "HF account behind your HF_TOKEN, fill out the access form "
            "(auto-approved), then re-run this script.\n"
        )
        raise
    print(f"GigaMIDI done: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--which", choices=["all", "musicpile", "gigamidi"], default="all",
    )
    args = parser.parse_args()

    token = _load_hf_token()

    if args.which in ("all", "musicpile"):
        download_musicpile(token)
    if args.which in ("all", "gigamidi"):
        download_gigamidi(token)


if __name__ == "__main__":
    main()
