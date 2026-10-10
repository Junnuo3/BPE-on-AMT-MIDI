"""
Offline cache builder for MIDI-LLM-style captioned anticipatory augmentation.

For each caption-matched Stage-2 training file, for each ~100s chunk, for each
anticipation pass k in [1, 9] (see anticipation_utils.augmentation_spec):
  1. Extract this pass's (items, events) from ONE build_sequence() call --
     `items` is the interleaved event/control token stream that gets trained
     on, `events` is the subset of notes this pass actually asks the model to
     generate (the "target"). Both come from the same extraction so the
     cached caption always describes the same notes as the cached token text.
  2. Render `events` to audio via FluidSynth (src/audio_utils.py).
  3. Caption that audio with a HF audio-captioning model (default Qwen2.5-Omni,
     src/caption_model.py).
  4. Cache {chunk_idx, k, chunk_text, caption} for this file.

Output layout: <out-dir>/<md5>.jsonl (one line per (chunk_idx, k)) plus
<out-dir>/<md5>.done once the file's records are fully written. Files with a
.done marker are skipped on rerun; a partial .jsonl is rebuilt.

k=0 (non-augmented) examples keep the original MidiCaps caption and are built
at training time, not here.

Usage:
    python scripts/build_caption_cache.py \
        --ant-dir tokenizers/anticipation/velocity/onset-10ms_duration-10ms_velocity-32bin \
        --out-dir outputs/stage2_cache
"""

import argparse
import json
import os
import sys
import tempfile
import warnings

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_SCRIPT_DIR)

_DEFAULT_ANT_DIR = os.path.join(_REPO_ROOT, "tokenizers", "anticipation",
                                "velocity", "onset-10ms_duration-10ms_velocity-32bin")
_DEFAULT_DATASET  = os.path.join(_REPO_ROOT, "dataset", "lmd_full")
_DEFAULT_CAPTIONS = os.path.join(_REPO_ROOT, "dataset", "lmd_full_captions.jsonl")
_DEFAULT_EVAL_IDS = os.path.join(_REPO_ROOT, "assets", "evaluation_set_lakh_ids.txt")
_DEFAULT_OUT_DIR  = os.path.join(_REPO_ROOT, "outputs", "stage2_cache")

_DONE_MARKER = ".done"

sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))


def load_caption_map(jsonl_path: str) -> dict:
    """md5 -> caption string. Duplicated from finetune_smollm2.py to avoid pulling
    in that module's torch/Trainer-only dependencies for this offline script."""
    caption_map = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            md5     = entry.get("md5")
            caption = entry.get("caption")
            if md5 and caption:
                caption_map[md5] = caption
    return caption_map


def build_file_cache(
    md5: str,
    path: str,
    *,
    b2c: dict, ctrl_b2c: dict,
    k_values: list[int],
    caption_model,
    sf2_path: str | None,
    use_fluid: bool,
    sample_rate: int,
    onset_ms: float, dur_ms: float, vel_bins: int,
    caption_prompt: str,
) -> list[dict]:
    """Build all (chunk_idx, k) cache records for one MIDI file. Pure/no I/O to out_dir."""
    from anticipation_utils import augmentation_spec, build_sequence
    from audio_utils import synth_fluidsynth, synth_numpy, write_wav
    from bpe_utils import items_to_text, stable_seed
    from midi_to_amt import amt_to_midi, midi_to_amt

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chunks = midi_to_amt(path, onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins)
    if not chunks:
        return []

    onset_steps_per_sec = 1000.0 / onset_ms
    delta_steps = max(1, int(round(5.0 * onset_steps_per_sec)))

    records: list[dict] = []
    for chunk_idx, notes in enumerate(chunks):
        instruments = list({int(n[3][12:-1]) for n in notes})

        for k in k_values:
            rng = np.random.default_rng(stable_seed(md5, chunk_idx, k))
            mode, kwargs = augmentation_spec(k, notes, instruments, delta_steps, rng)
            items, events = build_sequence(
                notes, mode, rng, onset_ms=onset_ms, return_events=True, **kwargs,
            )

            chunk_text = items_to_text(items, b2c, ctrl_b2c)
            if chunk_text is None or not events:
                continue  # OOV token, or this pass put every note into controls

            pm = amt_to_midi(events, onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins)
            if not any(inst.notes for inst in pm.instruments):
                continue

            audio = (synth_fluidsynth(pm, sf2_path, sample_rate) if use_fluid
                     else synth_numpy(pm, sample_rate))

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                write_wav(tmp.name, audio, sample_rate)
                caption = caption_model.caption(tmp.name, prompt=caption_prompt)

            records.append({
                "chunk_idx": chunk_idx,
                "k": k,
                "chunk_text": chunk_text,
                "caption": caption,
            })

    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset",  default=_DEFAULT_DATASET)
    p.add_argument("--captions", default=_DEFAULT_CAPTIONS)
    p.add_argument("--eval-ids", default=_DEFAULT_EVAL_IDS,
                   help="MIDI-LLM's 896-song evaluation md5 list (test split).")
    p.add_argument("--ant-dir",  default=_DEFAULT_ANT_DIR,
                   help="Directory containing mappings/amt_base_token_char_mapping.json")
    p.add_argument("--mapping",  default=None,
                   help="Overrides --ant-dir's default mapping path.")
    p.add_argument("--out-dir",  default=_DEFAULT_OUT_DIR)
    p.add_argument("--onset-ms", type=float, default=10.0)
    p.add_argument("--dur-ms",   type=float, default=10.0)
    p.add_argument("--vel-bins", type=int,   default=32)
    p.add_argument("--k-values", default="1,2,3,4,5,6,7,8,9",
                   help="Comma-separated augmentation passes to cache (5-10 recommended).")
    p.add_argument("--caption-model",  default=None,
                   help="HF model id for captioning. Default: src/caption_model.py's default (Qwen2.5-Omni-7B).")
    p.add_argument("--caption-prompt", default=None,
                   help="Override the default captioning prompt.")
    p.add_argument("--soundfont",    default=None)
    p.add_argument("--sample-rate",  type=int, default=44100)
    p.add_argument("--shard-id",   type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()

    from audio_utils import find_soundfont, try_fluidsynth_import
    from bpe_utils import (
        find_midi_files, load_mapping_doubled,
        load_lakh_evaluation_ids, split_caption_matched_files_lakh_ef,
    )
    from caption_model import CaptionModel, DEFAULT_PROMPT

    mapping_path = args.mapping or os.path.join(args.ant_dir, "mappings", "amt_base_token_char_mapping.json")
    if not os.path.exists(mapping_path):
        print(f"ERROR: mapping not found at {mapping_path}")
        sys.exit(1)
    b2c, _c2b, ctrl_b2c, _ctrl_c2b = load_mapping_doubled(mapping_path)

    caption_map = load_caption_map(args.captions)
    eval_ids    = load_lakh_evaluation_ids(args.eval_ids)
    midi_files  = find_midi_files(args.dataset)
    train_files, _valid, _test = split_caption_matched_files_lakh_ef(
        midi_files, caption_map, eval_ids,
    )
    print(f"Caption-matched train files: {len(train_files):,}")

    train_files = train_files[args.shard_id :: args.num_shards]
    print(f"Shard {args.shard_id}/{args.num_shards}: {len(train_files):,} files")

    use_fluid = try_fluidsynth_import()
    sf2_path  = args.soundfont or find_soundfont()
    if use_fluid and sf2_path is None:
        print("WARNING: FluidSynth available but no soundfont found "
              "(set $MIDI_SOUNDFONT or pass --soundfont); using pretty_midi's bundled TimGM6mb.sf2.")
    if not use_fluid:
        print("WARNING: pyfluidsynth not importable; falling back to the numpy sine synth "
              "(much lower audio quality -- captions will be worse).")

    caption_kwargs = {}
    if args.caption_model:
        caption_kwargs["model_id"] = args.caption_model
    caption_model  = CaptionModel(**caption_kwargs)
    caption_prompt = args.caption_prompt or DEFAULT_PROMPT

    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    n_done, n_built, n_empty = 0, 0, 0
    for path in train_files:
        md5 = os.path.splitext(os.path.basename(path))[0]
        done_path = os.path.join(args.out_dir, f"{md5}{_DONE_MARKER}")
        if os.path.exists(done_path):
            n_done += 1
            continue

        records = build_file_cache(
            md5, path,
            b2c=b2c, ctrl_b2c=ctrl_b2c, k_values=k_values,
            caption_model=caption_model, sf2_path=sf2_path, use_fluid=use_fluid,
            sample_rate=args.sample_rate,
            onset_ms=args.onset_ms, dur_ms=args.dur_ms, vel_bins=args.vel_bins,
            caption_prompt=caption_prompt,
        )

        jsonl_path = os.path.join(args.out_dir, f"{md5}.jsonl")
        with open(jsonl_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        open(done_path, "w").close()

        if records:
            n_built += 1
        else:
            n_empty += 1
        if (n_built + n_empty) % 50 == 0:
            print(f"  built={n_built:,} empty={n_empty:,} already-done={n_done:,} "
                  f"/ {len(train_files):,} in this shard")

    print(f"Done. built={n_built:,} empty={n_empty:,} already-done={n_done:,} "
          f"/ {len(train_files):,} in this shard")


if __name__ == "__main__":
    main()
