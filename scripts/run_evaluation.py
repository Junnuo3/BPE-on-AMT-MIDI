"""
Evaluate generated MIDI (scripts/generate_eval_set.py layout) against the real
songs of the same split: render audio with FluidSynth when a metric needs it,
then compute FAD, CLAP, chroma, groove and groove_bar (src/metrics/).

Every stage is resumable: synthesis skips wavs that already exist, and each
metric checkpoints its own per-item progress under evaluation/<model_name>/<metric>/
(see src/metrics/clap.py, fad.py, chroma.py, groove.py) -- interrupting and
re-running this script picks up wherever it left off.

chroma/groove read decoded MIDI directly (src/chroma_groove.py) and never
touch synthesized audio, so the FluidSynth stage above is skipped automatically
when --metrics selects only chroma and/or groove.

Output:
    evaluation/<model_name>/clap/{per_sample.jsonl, summary.json}
    evaluation/<model_name>/fad/{embeddings/, summary.json}
    evaluation/<model_name>/{chroma,groove}/{per_piece.csv, summary.json}
    evaluation/<model_name>/summary.json   (combined, this run's metadata + all metric summaries)
    evaluation/_ground_truth/               (shared real-song audio/embedding cache and
                                             chroma/groove reference CSVs)

Usage:
    python scripts/run_evaluation.py --generations-dir generations/bpe_test \
        --model-name bpe_test --split test --metrics fad,clap,chroma,groove,groove_bar
"""

import argparse
import json
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_SCRIPT_DIR)

sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, _SCRIPT_DIR)

import eval_utils  # noqa: E402
from audio_utils import find_soundfont, try_fluidsynth_import  # noqa: E402
from eval_data import GeneratedSample, TestEntry, load_lakh_split, locate_generations  # noqa: E402
from generate_smollm2 import _resolve_checkpoint  # noqa: E402
from metrics import METRICS  # noqa: E402

_DEFAULT_DATASET  = os.path.join(_REPO_ROOT, "dataset", "lmd_full")
_DEFAULT_CAPTIONS = os.path.join(_REPO_ROOT, "dataset", "lmd_full_captions.jsonl")
_DEFAULT_EVAL_IDS = os.path.join(_REPO_ROOT, "assets", "evaluation_set_lakh_ids.txt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generations-dir", default=None,
                   help="Directory of <md5>/{prompt.txt, sample_*.mid}, e.g. from "
                        "generate_eval_set.py. Default: <run_dir>/generations/<checkpoint>/eval_lakh_<split>/")
    p.add_argument("--run-dir",    default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--step",       default=None)
    p.add_argument("--model-name", default=None,
                   help="Default: derived from --run-dir (see eval_utils.derive_model_name).")

    p.add_argument("--dataset",    default=_DEFAULT_DATASET)
    p.add_argument("--captions",   default=_DEFAULT_CAPTIONS)
    p.add_argument("--eval-ids",   default=_DEFAULT_EVAL_IDS,
                   help="MIDI-LLM's 896-song evaluation md5 list (test split). Must match "
                        "generate_eval_set.py's value.")
    p.add_argument("--split",      choices=["test", "valid"], default="test",
                   help="Which held-out split to prompt with / score against. Use 'valid' "
                        "for decoding-hyperparameter selection so test stays untouched.")
    p.add_argument("--subset-size", type=int, default=None,
                   help="Fixed random subset of the split (e.g. 896 to size-match the "
                        "~10.7K-song valid split to test). Must match between "
                        "generate_eval_set.py and run_evaluation.py.")
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--limit",      type=int, default=None,
                   help="Cap number of generated samples AND ground-truth entries (smoke tests).")

    p.add_argument("--metrics",    default=None,
                   help=f"Comma-separated metric names to run. Default: all registered "
                        f"({','.join(METRICS)}).")

    p.add_argument("--soundfont",   default=None)
    p.add_argument("--sample-rate", type=int, default=44100,
                   help="Rate for the shared intermediate wav cache. Both CLAP (48kHz) and "
                        "VGGish (16kHz) resample internally regardless, so this is chosen to "
                        "avoid clipping bandwidth either metric could use, not to match either.")
    p.add_argument("--synth-workers", type=int, default=1,
                   help="Parallel FluidSynth processes (CPU-bound; set to the job's CPU count).")
    p.add_argument("--gt-embed-npz", default=None,
                   help="Packed ground-truth VGGish embeddings from build_ground_truth_cache.py. "
                        "When set, ground-truth audio is not synthesized at all.")
    p.add_argument("--gen-embed-dir", default=None,
                   help="Per-clip generated-embedding cache location (e.g. node-local $TMPDIR); "
                        "only a packed fad/generated_embeddings.npz is kept in the eval dir.")
    p.add_argument("--numpy-synth", action="store_true",
                   help="Force the numpy sine synth instead of FluidSynth (no timbre; for "
                        "environments without FluidSynth installed).")

    p.add_argument("--clap-ckpt", default=None,
                   help="Local path to a CLAP checkpoint. Default: auto-download LAION-AI's "
                        "music checkpoint (see src/metrics/clap.py).")
    p.add_argument("--device", default=None, help="Default: cuda if available, else cpu.")

    p.add_argument("--window-sec", type=float, default=5.0,
                   help="Fixed-duration window (seconds) for chroma/groove metrics -- a "
                        "practical replacement for bar-based segmentation, since MidiCaps/Lakh "
                        "have no reliable beat/bar annotations (see src/chroma_groove.py).")
    p.add_argument("--chroma-include-drums", action="store_true",
                   help="Include percussion onsets in chroma-entropy windows (excluded by default).")
    p.add_argument("--groove-exclude-drums", action="store_true",
                   help="Exclude percussion onsets from groove-similarity windows (included by default).")
    p.add_argument("--groove-grid-ms", type=float, default=50.0,
                   help="Grid resolution (ms) for groove-similarity binary onset vectors.")
    p.add_argument("--groove-bar-positions", type=int, default=64,
                   help="Subdivisions per detected bar for the bar-based groove metric (groove_bar).")
    p.add_argument("--keep-partial-window", action="store_true",
                   help="Include the final partial window instead of discarding it (chroma+groove).")
    p.add_argument("--chroma-bins", type=int, default=20,
                   help="Number of fixed histogram bins over [0, log2(12)] for Chroma JS Distance.")
    p.add_argument("--groove-bins", type=int, default=20,
                   help="Number of fixed histogram bins over [0, 1] for Groove JS Distance.")

    return p.parse_args()


def _resolve_generations_dir(args: argparse.Namespace) -> tuple[str, str | None]:
    """Returns (generations_dir, run_dir)."""
    if args.generations_dir:
        return args.generations_dir, args.run_dir

    if args.checkpoint:
        checkpoint_dir = args.checkpoint
        run_dir = args.run_dir or os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_dir)))
    else:
        if not args.run_dir:
            print("ERROR: pass --generations-dir, or --run-dir (+ optional --step)/--checkpoint.")
            sys.exit(1)
        run_dir = os.path.abspath(args.run_dir)
        checkpoint_dir = _resolve_checkpoint(run_dir, args.step)

    ckpt_name = os.path.basename(checkpoint_dir.rstrip("/"))
    return os.path.join(run_dir, "generations", ckpt_name, f"eval_lakh_{args.split}"), run_dir


def _synth_one(job: tuple) -> bool:
    mid_path, wav_path, sf2_path, sample_rate, use_fluid = job
    return eval_utils.synth_midi_cached(
        mid_path, wav_path, sf2_path=sf2_path, sample_rate=sample_rate, use_fluid=use_fluid,
    )


def _synth_many(pairs: list[tuple[str, str]], label: str, *, sf2_path, sample_rate, use_fluid,
                workers: int) -> list[bool]:
    """Render (mid_path, wav_path) pairs, in `workers` processes when > 1.
    synth_midi_cached is safe to run concurrently (per-process tmp file)."""
    jobs = [(m, w, sf2_path, sample_rate, use_fluid) for m, w in pairs]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = []
            for i, ok in enumerate(pool.map(_synth_one, jobs, chunksize=16)):
                results.append(ok)
                if (i + 1) % 1000 == 0:
                    print(f"  synthesized {i + 1}/{len(jobs)} {label} ({sum(results)} ok)")
    else:
        results = []
        for i, job in enumerate(jobs):
            results.append(_synth_one(job))
            if (i + 1) % 100 == 0:
                print(f"  synthesized {i + 1}/{len(jobs)} {label} ({sum(results)} ok)")
    return results


def _synth_generated(samples: list[GeneratedSample], *, sf2_path, sample_rate, use_fluid,
                     workers: int = 1) -> None:
    pairs = [(s.mid_path, eval_utils.wav_path_for(s.mid_path)) for s in samples]
    oks = _synth_many(pairs, "generated samples", sf2_path=sf2_path, sample_rate=sample_rate,
                      use_fluid=use_fluid, workers=workers)
    for sample, (_, wav_path), ok in zip(samples, pairs, oks):
        if ok:
            sample.wav_path = wav_path
    print(f"Generated audio: {sum(oks)}/{len(samples)} synthesized OK.")


def _synth_ground_truth(entries: list[TestEntry], *, sf2_path, sample_rate, use_fluid,
                        workers: int = 1, audio_dir: str | None = None) -> None:
    audio_dir = audio_dir or os.path.join(eval_utils.GROUND_TRUTH_CACHE_DIR, "audio")
    pairs = [(e.midi_path, os.path.join(audio_dir, f"{e.md5}.wav")) for e in entries]
    oks = _synth_many(pairs, "ground-truth samples", sf2_path=sf2_path, sample_rate=sample_rate,
                      use_fluid=use_fluid, workers=workers)
    for entry, (_, wav_path), ok in zip(entries, pairs, oks):
        if ok:
            entry.wav_path = wav_path
    print(f"Ground-truth audio: {sum(oks)}/{len(entries)} synthesized OK.")


def main() -> None:
    args = parse_args()

    generations_dir, run_dir = _resolve_generations_dir(args)
    print(f"Generations dir: {generations_dir}")

    generated = locate_generations(generations_dir)
    if args.limit is not None:
        generated = generated[:args.limit]
    print(f"Located {len(generated):,} generated samples.")
    if not generated:
        print(f"ERROR: no generations found under {generations_dir}. "
              f"Run scripts/generate_eval_set.py first.")
        sys.exit(1)

    ground_truth = load_lakh_split(
        args.dataset, args.captions, args.eval_ids, split=args.split,
        subset_size=args.subset_size, subset_seed=args.subset_seed,
    )
    if args.limit is not None:
        ground_truth = ground_truth[:args.limit]
    print(f"Located {len(ground_truth):,} MidiCaps {args.split}-split ground-truth entries.")

    # Generations must come from the same prompts as the ground truth (else FAD's
    # genre mix and CLAP's pairing silently compare against the wrong split).
    gt_md5s = {e.md5 for e in ground_truth}
    foreign = {s.sample_id.split("/")[0] for s in generated} - gt_md5s
    if foreign and args.limit is None:
        print(f"ERROR: {len(foreign):,} generation dirs (e.g. {sorted(foreign)[0]}) are not in the "
              f"{args.split} ground-truth set -- check --split/--subset-size/--subset-seed.")
        sys.exit(1)

    metric_names = args.metrics.split(",") if args.metrics else list(METRICS)
    for name in metric_names:
        if name not in METRICS:
            print(f"ERROR: unknown metric '{name}'. Available: {','.join(METRICS)}")
            sys.exit(1)

    if any(METRICS[name].needs_audio for name in metric_names):
        use_fluid = False
        sf2_path  = None
        if not args.numpy_synth:
            use_fluid = try_fluidsynth_import()
            if not use_fluid:
                print("WARNING: pyfluidsynth not importable; falling back to the numpy sine synth "
                      "(much lower audio quality -- metrics will be less meaningful).")
        if use_fluid:
            sf2_path = args.soundfont or find_soundfont()
            if sf2_path is None:
                print("WARNING: FluidSynth available but no soundfont found "
                      "(set $MIDI_SOUNDFONT or pass --soundfont); using pretty_midi's bundled TimGM6mb.sf2.")

        print("Synthesizing audio (skips files already rendered) ...")
        _synth_generated(generated, sf2_path=sf2_path, sample_rate=args.sample_rate,
                         use_fluid=use_fluid, workers=args.synth_workers)
        # Only FAD reads ground-truth audio (CLAP scores generations against their
        # prompts); packed ground-truth embeddings stand in for it.
        if "fad" in metric_names and not args.gt_embed_npz:
            _synth_ground_truth(ground_truth, sf2_path=sf2_path, sample_rate=args.sample_rate,
                                use_fluid=use_fluid, workers=args.synth_workers)
    else:
        print("Skipping audio synthesis -- no requested metric needs it "
              f"({', '.join(metric_names)} read decoded MIDI directly).")

    model_name = args.model_name or eval_utils.derive_model_name(generations_dir, run_dir)
    output_root = eval_utils.model_output_dir(model_name)
    print(f"Model name: {model_name}")
    print(f"Output dir: {output_root}")

    from metrics import EvalContext

    ctx = EvalContext(
        model_name=model_name, generated=generated, ground_truth=ground_truth,
        output_root=output_root, sample_rate=args.sample_rate, device=args.device,
    )

    results = {}
    summary_path = os.path.join(output_root, "summary.json")

    def write_summary() -> None:
        """Merge into an existing summary (e.g. adding CLAP to a run already scored
        for FAD/chroma/groove) instead of dropping metrics this invocation didn't
        run. Called after every metric, so a job killed mid-way (time limit,
        preemption) keeps every metric it finished."""
        merged = dict(results)
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                merged = {**json.load(f).get("metrics", {}), **results}
        combined = {
            "model_name": model_name,
            "generations_dir": generations_dir,
            "run_dir": run_dir,
            "n_generated_samples": len(generated),
            "n_ground_truth_entries": len(ground_truth),
            "split": args.split,
            "subset_size": args.subset_size,
            "subset_seed": args.subset_seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "metrics": merged,
        }
        tmp = f"{summary_path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(combined, f, indent=2)
        os.replace(tmp, summary_path)
    for name in metric_names:
        print(f"\n=== Metric: {name} ===")
        kwargs = {}
        if name == "clap":
            kwargs = {"ckpt_path": args.clap_ckpt, "device": args.device}
        elif name == "fad":
            kwargs = {
                "device": args.device,
                "ground_truth_embed_cache_dir": os.path.join(
                    eval_utils.GROUND_TRUTH_CACHE_DIR, "fad_vggish_embeddings"),
                "ground_truth_embed_npz": args.gt_embed_npz,
                "generated_embed_dir": args.gen_embed_dir,
            }
        elif name == "chroma":
            kwargs = {
                "window_sec": args.window_sec,
                "include_drums": args.chroma_include_drums,
                "discard_partial_window": not args.keep_partial_window,
                "n_bins": args.chroma_bins,
                "ground_truth_cache_dir": os.path.join(eval_utils.GROUND_TRUTH_CACHE_DIR, "chroma"),
            }
        elif name == "groove":
            kwargs = {
                "window_sec": args.window_sec,
                "grid_ms": args.groove_grid_ms,
                "include_drums": not args.groove_exclude_drums,
                "discard_partial_window": not args.keep_partial_window,
                "n_bins": args.groove_bins,
                "ground_truth_cache_dir": os.path.join(eval_utils.GROUND_TRUTH_CACHE_DIR, "groove"),
            }
        elif name == "groove_bar":
            kwargs = {
                "positions_per_bar": args.groove_bar_positions,
                "include_drums": not args.groove_exclude_drums,
                "n_bins": args.groove_bins,
                "ground_truth_cache_dir": os.path.join(eval_utils.GROUND_TRUTH_CACHE_DIR, "groove_bar"),
                "device": args.device,
            }
        metric = METRICS[name](**kwargs)
        summary = metric.run(ctx)
        print(json.dumps(summary, indent=2))
        results[name] = summary
        write_summary()

    print(f"\nCombined results: {os.path.join(output_root, 'summary.json')}")


if __name__ == "__main__":
    main()
