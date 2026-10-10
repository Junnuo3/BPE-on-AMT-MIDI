#!/bin/bash
# Download and prepare all datasets under dataset/.
# LMD-full must be downloaded manually first:
#   http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz  ->  dataset/lmd_full/
# GigaMIDI is gated on Hugging Face: request access, then export HF_TOKEN.
source "$(dirname "$0")/common.sh"

python scripts/match_midicaps.py                 # -> dataset/lmd_full_captions.jsonl (168,385 captions)
python scripts/download_stage1_data.py --which all
python scripts/extract_gigamidi_midi.py          # -> dataset/gigamidi/midi/, excludes all MidiCaps md5s
