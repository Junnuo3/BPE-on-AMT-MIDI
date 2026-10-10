"""
Data-loading for scripts/run_evaluation.py and scripts/generate_eval_set.py:
- the Lakh md5-prefix test (or valid) split (ground truth audio + prompts), reusing the same
  train/valid/test partition that bpe_utils.split_caption_matched_files_lakh_ef
  gives finetune_smollm2.py/build_caption_cache.py, so eval never touches files
  the model could have trained on.
- a model run's generated MIDI files + the text prompt that produced each one,
  from the directory layout written by scripts/generate_eval_set.py:
      <generations_dir>/<sample_id>/prompt.txt
      <generations_dir>/<sample_id>/sample_*.mid
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from bpe_utils import find_midi_files, split_caption_matched_files_lakh_ef


@dataclass
class TestEntry:
    md5: str
    caption: str
    midi_path: str
    wav_path: str | None = None  # filled in by eval_utils.synth_midi_cached


@dataclass
class GeneratedSample:
    sample_id: str      # "<md5>/<stem>", unique within a generations_dir
    mid_path: str
    prompt: str | None
    wav_path: str | None = None


def load_caption_map(captions_path: str) -> dict[str, str]:
    """md5 -> caption. Small/standalone on purpose (see build_caption_cache.py's
    own copy of this loader) so importing this module doesn't require torch."""
    caption_map: dict[str, str] = {}
    with open(captions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            md5, caption = entry.get("md5"), entry.get("caption")
            if md5 and caption:
                caption_map[md5] = caption
    return caption_map


def load_lakh_test_set(
    dataset_dir: str, captions_path: str, eval_ids_path: str,
) -> list[TestEntry]:
    """The held-out `test` split (not `valid`) of caption-matched Lakh files,
    per bpe_utils.split_caption_matched_files_lakh_ef. One entry per unique md5.
    """
    return load_lakh_split(dataset_dir, captions_path, eval_ids_path, split="test")


def load_lakh_split(
    dataset_dir: str, captions_path: str, eval_ids_path: str, split: str = "test",
    subset_size: int | None = None, subset_seed: int = 0,
) -> list[TestEntry]:
    """The `test` or `valid` split of caption-matched files, per
    bpe_utils.split_caption_matched_files_lakh_ef. One entry per unique md5,
    sorted by md5.

    subset_size draws a fixed random subset (seeded by subset_seed, independent of
    the directory-scan order) -- e.g. an 896-song slice of the ~10.7K-song 'e'
    validation bucket, size-matched to the test set, for decoding-hyperparameter
    selection that must not touch test.
    """
    import random

    from bpe_utils import load_lakh_evaluation_ids

    if split not in ("test", "valid"):
        raise ValueError(f"split must be 'test' or 'valid', got {split!r}")

    caption_map = load_caption_map(captions_path)
    eval_ids    = load_lakh_evaluation_ids(eval_ids_path)
    midi_files  = find_midi_files(dataset_dir)
    _train, valid, test = split_caption_matched_files_lakh_ef(
        midi_files, caption_map, eval_ids,
    )

    entries = []
    for path in (test if split == "test" else valid):
        md5 = os.path.splitext(os.path.basename(path))[0]
        entries.append(TestEntry(md5=md5, caption=caption_map[md5], midi_path=path))
    entries.sort(key=lambda e: e.md5)
    if subset_size is not None and subset_size < len(entries):
        entries = random.Random(subset_seed).sample(entries, subset_size)
        entries.sort(key=lambda e: e.md5)
    return entries


def locate_generations(generations_dir: str) -> list[GeneratedSample]:
    """Scan <generations_dir>/<sample_id>/{prompt.txt, sample_*.mid or gen_*.mid}."""
    samples: list[GeneratedSample] = []
    if not os.path.isdir(generations_dir):
        return samples

    for sample_id in sorted(os.listdir(generations_dir)):
        sub_dir = os.path.join(generations_dir, sample_id)
        if not os.path.isdir(sub_dir):
            continue

        prompt = None
        prompt_path = os.path.join(sub_dir, "prompt.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, encoding="utf-8") as f:
                prompt = f.read().strip()

        mid_files = sorted(
            f for f in os.listdir(sub_dir) if f.lower().endswith((".mid", ".midi"))
        )
        for mid_file in mid_files:
            stem = os.path.splitext(mid_file)[0]
            samples.append(GeneratedSample(
                sample_id=f"{sample_id}/{stem}",
                mid_path=os.path.join(sub_dir, mid_file),
                prompt=prompt,
            ))
    return samples
