"""
Build the real-song side of the evaluation once per split, keeping only packed
files (no per-song wavs):

  shard mode:
      renders its slice of the split's real MIDI into --work-dir (scratch
      space), embeds it with VGGish, and writes one packed
      <out-npz>.shard<i>of<n>.npz.
  --merge mode (after all shards):
      concatenates the shard files into <out-npz> (then deletes them), and
      scores the split's real MIDI for chroma/groove into the shared
      evaluation/_ground_truth/{chroma,groove}/ CSVs with run_evaluation.py's
      default parameters, so concurrent evaluations only read them.

run_evaluation.py then takes --gt-embed-npz <out-npz> and skips ground-truth
synthesis entirely.

Usage:
    python scripts/build_ground_truth_cache.py --split valid \
        --work-dir /tmp/gt --shard-id 0 --num-shards 1 --synth-workers 8
    python scripts/build_ground_truth_cache.py --split valid --merge
"""

import argparse
import glob
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_SCRIPT_DIR)

sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, _SCRIPT_DIR)

import eval_utils  # noqa: E402
from eval_data import load_lakh_split  # noqa: E402

_DEFAULT_DATASET  = os.path.join(_REPO_ROOT, "dataset", "lmd_full")
_DEFAULT_CAPTIONS = os.path.join(_REPO_ROOT, "dataset", "lmd_full_captions.jsonl")
_DEFAULT_EVAL_IDS = os.path.join(_REPO_ROOT, "assets", "evaluation_set_lakh_ids.txt")


def default_npz_path(split: str, subset_size: int | None, subset_seed: int) -> str:
    name = f"fad_vggish_{split}" + (f"_n{subset_size}_s{subset_seed}" if subset_size else "")
    return os.path.join(eval_utils.GROUND_TRUTH_CACHE_DIR, name + ".npz")


def _midi_metric_specs(metrics=("chroma", "groove")) -> list[tuple]:
    """(module, shared ground-truth CSV path, _score_set kwargs) for chroma and
    groove at run_evaluation.py's defaults (window 5s, chroma w/o drums, groove
    with drums at a 50ms grid, final partial window discarded)."""
    from metrics import chroma, groove

    gt = eval_utils.GROUND_TRUTH_CACHE_DIR
    c_kw = dict(window_sec=5.0, include_drums=False, discard_partial_window=True)
    g_kw = dict(window_sec=5.0, grid_ms=50.0, include_drums=True, discard_partial_window=True)
    specs = {
        "chroma": (chroma, os.path.join(gt, "chroma", f"ground_truth_per_piece_{chroma._config_tag(**c_kw)}.csv"), c_kw),
        "groove": (groove, os.path.join(gt, "groove", f"ground_truth_per_piece_{groove._config_tag(**g_kw)}.csv"), g_kw),
    }
    if "groove_bar" in metrics:
        from metrics import groove_bar

        b_kw = dict(positions_per_bar=64, include_drums=True)
        specs["groove_bar"] = (groove_bar, os.path.join(
            gt, "groove_bar", f"ground_truth_per_piece_{groove_bar._config_tag(**b_kw)}.csv"), b_kw)
    return [specs[m] for m in metrics if m in specs]


def _score(module, entries, csv_path, kw, tracker):
    """chroma/groove score straight from MIDI; groove_bar also needs the tracker
    (and entries' rendered wav_path)."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if module.__name__.endswith("groove_bar"):
        return module._score_set(entries, "midi_path", csv_path, tracker, **kw)
    return module._score_set(entries, "midi_path", csv_path, **kw)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["test", "valid"], default="valid")
    p.add_argument("--subset-size", type=int, default=None)
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--dataset",  default=_DEFAULT_DATASET)
    p.add_argument("--captions", default=_DEFAULT_CAPTIONS)
    p.add_argument("--eval-ids", default=_DEFAULT_EVAL_IDS)
    p.add_argument("--out-npz",  default=None, help="Default: _ground_truth/fad_vggish_<split>[...].npz")

    p.add_argument("--metrics", default="fad,chroma,groove",
                   help="Ground-truth caches to build: any of fad, chroma, groove, groove_bar "
                        "(groove_bar renders audio and runs the downbeat tracker).")
    p.add_argument("--only-missing", action="store_true",
                   help="Shard mode: only songs missing from the shared per-piece CSVs of the "
                        "requested MIDI metrics (and any of their shard CSVs) -- repairs a partial cache.")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--pack-cache-dir", default=None,
                   help="With --merge: pack an existing per-song <md5>.npy VGGish cache (e.g. the "
                        "test split's _ground_truth/fad_vggish_embeddings/) instead of shard files, "
                        "so FAD numbers stay identical to the per-file evaluation path.")
    p.add_argument("--shard-id",   type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--work-dir", default=None, help="Scratch space for wavs/per-clip embeddings "
                                                    "(node-local). Default: $TMPDIR.")
    p.add_argument("--synth-workers", type=int, default=1)
    p.add_argument("--soundfont",   default=None)
    p.add_argument("--sample-rate", type=int, default=44100, help="Must match run_evaluation.py.")
    p.add_argument("--device", default=None)
    return p.parse_args()


def build_shard(args, entries, out_npz: str) -> None:
    from audio_utils import find_soundfont, try_fluidsynth_import
    from metrics.fad import _embed_items_cached, _load_vggish, save_packed_embeddings
    from run_evaluation import _synth_ground_truth

    metrics = args.metrics.split(",")
    shard = entries[args.shard_id::args.num_shards]
    sfx = f".shard{args.shard_id}of{args.num_shards}"
    if args.only_missing:
        from chroma_groove import load_csv_rows

        have = None
        for _module, csv_path, _kw in _midi_metric_specs(metrics):
            ids = set()
            for f in [csv_path] + glob.glob(f"{glob.escape(csv_path)}.*shard*of*"):
                ids |= {row["id"] for row in load_csv_rows(f)}
            have = ids if have is None else have & ids
        shard = [e for e in shard if e.md5 not in (have or set())]
        sfx += ".missing"
    shard_path = f"{out_npz}{sfx}.npz"
    done_marker = shard_path if "fad" in metrics else f"{out_npz}{sfx}.{'_'.join(metrics)}.done"
    if os.path.exists(done_marker):
        print(f"{done_marker} already exists; nothing to do.")
        return
    work_dir = args.work_dir or os.environ.get("TMPDIR")
    if not work_dir:
        sys.exit("ERROR: pass --work-dir (or set $TMPDIR).")
    print(f"Shard {args.shard_id}/{args.num_shards}: {len(shard):,} songs, work dir {work_dir}, "
          f"metrics {metrics}")

    if {"fad", "groove_bar"} & set(metrics):
        use_fluid = try_fluidsynth_import()
        if not use_fluid:
            sys.exit("ERROR: pyfluidsynth not importable; refusing to build a numpy-synth ground-truth cache.")
        sf2_path = args.soundfont or find_soundfont()
        _synth_ground_truth(shard, sf2_path=sf2_path, sample_rate=args.sample_rate, use_fluid=use_fluid,
                            workers=args.synth_workers, audio_dir=os.path.join(work_dir, "gt_audio"))

    if "fad" in metrics:
        model = _load_vggish(args.device)
        id_frames = _embed_items_cached(model, [(e.md5, e.wav_path) for e in shard if e.wav_path],
                                        os.path.join(work_dir, "gt_embeddings"))

    # per-piece metrics for this slice into per-shard CSVs; --merge folds them into
    # the shared CSVs. Written before the done marker.
    tracker = None
    if "groove_bar" in metrics:
        from downbeats import DownbeatTracker
        tracker = DownbeatTracker(device=args.device)
    for module, csv_path, kw in _midi_metric_specs(metrics):
        _score(module, shard, f"{csv_path}.{args.split}{sfx}", kw, tracker)
        print(f"{module.__name__} scored for this shard.")

    if "fad" in metrics:
        save_packed_embeddings(shard_path, id_frames)
        print(f"Wrote {len(id_frames):,} embeddings -> {shard_path}")
    else:
        open(done_marker, "w").close()


def merge(args, entries, out_npz: str) -> None:
    from metrics.fad import load_packed_embeddings, save_packed_embeddings

    metrics = args.metrics.split(",")
    shard_paths = sorted(glob.glob(f"{glob.escape(out_npz)}.shard*of*.npz"))
    if "fad" not in metrics:
        pass
    elif args.pack_cache_dir:
        import numpy as np

        id_frames = []
        for e in entries:
            path = os.path.join(args.pack_cache_dir, f"{e.md5}.npy")
            if os.path.exists(path):
                frames = np.load(path)
                if frames.size:
                    id_frames.append((e.md5, frames))
        save_packed_embeddings(out_npz, id_frames)
        print(f"Packed {len(id_frames):,}/{len(entries):,} cached embeddings from "
              f"{args.pack_cache_dir} -> {out_npz}")
    elif shard_paths:
        packed = {}
        for path in shard_paths:
            packed.update(load_packed_embeddings(path))
        id_frames = [(e.md5, packed[e.md5]) for e in entries if e.md5 in packed]
        save_packed_embeddings(out_npz, id_frames)
        for path in shard_paths:
            os.remove(path)
        print(f"Merged {len(shard_paths)} shards: {len(id_frames):,}/{len(entries):,} songs embedded "
              f"-> {out_npz}")
    elif not os.path.exists(out_npz):
        sys.exit(f"ERROR: no shard files for {out_npz}")

    from chroma_groove import append_csv_row, load_csv_rows

    wanted = {e.md5 for e in entries}
    for module, csv_path, kw in _midi_metric_specs(metrics):
        have = {row["id"] for row in load_csv_rows(csv_path)}
        # This split's shard CSVs, plus legacy unsplit-named ones (<csv>.shardIofN) --
        # those are folded in only if every row belongs to this split, so one split's
        # merge never consumes (or deletes) another split's in-progress shards.
        shard_csvs = sorted(glob.glob(f"{glob.escape(csv_path)}.{args.split}.shard*of*"))
        for legacy in sorted(glob.glob(f"{glob.escape(csv_path)}.shard*of*")):
            ids = {row["id"] for row in load_csv_rows(legacy)}
            if ids and ids <= wanted:
                shard_csvs.append(legacy)
        n_new = 0
        for shard_csv in shard_csvs:
            for row in load_csv_rows(shard_csv):
                if row["id"] not in have:
                    append_csv_row(csv_path, module._CSV_FIELDS, row)
                    have.add(row["id"])
                    n_new += 1
        # Anything still missing (e.g. no shard files) is scored here directly
        # (groove_bar can't be -- no audio here -- so it must be complete already).
        if module.__name__.endswith("groove_bar"):
            missing = [e.md5 for e in entries if e.md5 not in have]
            if missing:
                sys.exit(f"ERROR: {len(missing):,} songs missing from {csv_path} (e.g. {missing[0]}); "
                         f"rerun their shards")
        else:
            _score(module, entries, csv_path, kw, None)
        for shard_csv in shard_csvs:
            os.remove(shard_csv)
        print(f"{module.__name__}: merged {n_new:,} rows from {len(shard_csvs)} shard CSVs -> {csv_path}")
    for marker in glob.glob(f"{glob.escape(out_npz)}.shard*of*.*.done"):
        os.remove(marker)


def main() -> None:
    args = parse_args()
    out_npz = args.out_npz or default_npz_path(args.split, args.subset_size, args.subset_seed)
    entries = load_lakh_split(args.dataset, args.captions, args.eval_ids, split=args.split,
                              subset_size=args.subset_size, subset_seed=args.subset_seed)
    print(f"{len(entries):,} '{args.split}' ground-truth songs.")
    if args.merge:
        merge(args, entries, out_npz)
    else:
        build_shard(args, entries, out_npz)


if __name__ == "__main__":
    main()
