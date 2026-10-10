"""
Generate MIDI from a finetuned SmolLM2 checkpoint using HF's model.generate() API
(https://huggingface.co/docs/transformers/en/llm_tutorial), then decode the
generated AMT-BPE tokens back into a .mid file.

Outputs are written next to the checkpoint:
    <run_dir>/generations/<checkpoint>/sample_0000.mid
    <run_dir>/generations/<checkpoint>/sample_0000.json   (tokens + notes)
    <run_dir>/generations/<checkpoint>/generation_config.json

Usage:
    python scripts/generate_smollm2.py --checkpoint checkpoints/bpe/stage2 \
        --caption "A calm piano piece in C major with a slow tempo."
    python scripts/generate_smollm2.py --run-dir <run_dir> --num-samples 4
"""

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_SCRIPT_DIR)

sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, _SCRIPT_DIR)

from finetune_smollm2 import _auto_mapping, _auto_train_cfg  # noqa: E402
from bpe_utils import load_mapping, load_mapping_doubled, is_mapping_doubled, decode_generated_ids  # noqa: E402
from midi_to_amt import amt_to_midi  # noqa: E402
from audio_utils import find_soundfont, try_fluidsynth_import, synth_and_write  # noqa: E402

_DEFAULT_SYSTEM_PROMPT = (
    "You are a world-class composer. Please compose some music "
    "according to the following description: "
)


def _resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def _resolve_checkpoint(run_dir: str, step: str | None) -> str:
    """Pick a checkpoint dir under run_dir/checkpoints/: explicit step, else 'final', else latest."""
    ckpt_root = os.path.join(run_dir, "checkpoints")
    if step is not None:
        candidate = os.path.join(ckpt_root, step if step == "final" else f"checkpoint-{step}")
        if not os.path.isdir(candidate):
            print(f"ERROR: checkpoint not found: {candidate}")
            sys.exit(1)
        return candidate

    final_dir = os.path.join(ckpt_root, "final")
    if os.path.isdir(final_dir):
        return final_dir

    from transformers.trainer_utils import get_last_checkpoint
    last = get_last_checkpoint(ckpt_root)
    if last is None:
        print(f"ERROR: no checkpoints found under {ckpt_root}")
        sys.exit(1)
    return last


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-dir",    default=None,
                   help="Training run dir containing checkpoints/ and amt_config.json "
                        "(same value as --out-dir/<feature>/<bins> during finetuning).")
    p.add_argument("--checkpoint", default=None,
                   help="Explicit checkpoint dir. Overrides --run-dir/--step.")
    p.add_argument("--step",       default=None,
                   help="Checkpoint step to load from --run-dir/checkpoints/ "
                        "(int, or 'final'). Default: 'final' if present, else the latest step.")
    p.add_argument("--out-dir",    default=None,
                   help="Output dir for generations. Default: <run_dir>/generations/<checkpoint-name>/")

    p.add_argument("--mode",       choices=["autoregress", "anticipate"], default="autoregress",
                   help="Mode token prefixed to the prompt. 'anticipate' only makes sense when "
                        "paired with real control tokens, which this script does not construct — "
                        "kept for completeness/experimentation.")
    p.add_argument("--caption",    default=None,
                   help="Text description, for models finetuned with --caption-text. "
                        "Required if the checkpoint's amt_config.json has caption_text=true.")
    p.add_argument("--system-prompt", default=None,
                   help="Overrides the default system prompt prepended to --caption.")
    p.add_argument("--max-text-tokens", type=int, default=None,
                   help="Cap on tokenized caption length. Default: amt_config.json's value.")

    p.add_argument("--num-samples",  type=int,   default=1)
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Default: fill the remainder of the training window size.")
    p.add_argument("--do-sample",    action="store_true", default=True)
    p.add_argument("--greedy",       dest="do_sample", action="store_false",
                   help="Greedy decoding instead of sampling.")
    p.add_argument("--temperature",  type=float, default=1.0)
    p.add_argument("--top-p",        type=float, default=0.98)
    p.add_argument("--top-k",        type=int,   default=0,
                   help="0 disables top-k filtering.")
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--seed",         type=int,   default=None)
    p.add_argument("--device",       default=None, help="Default: cuda if available, else cpu.")

    p.add_argument("--render-audio", action="store_true",
                   help="Also render each generated .mid to a .wav using FluidSynth.")
    p.add_argument("--soundfont",    default=None,
                   help="Path to .sf2 soundfont for --render-audio. Auto-discovered from "
                        "~/FluidR3_GM.sf2, /usr/share/sounds/sf2/, or $MIDI_SOUNDFONT. "
                        "Recommended: Fluid R3 GM (musical-artifacts.com/artifacts/738).")
    p.add_argument("--sample-rate",  type=int,   default=44100,
                   help="Sample rate for --render-audio output (default: 44100).")
    p.add_argument("--numpy-synth",  action="store_true",
                   help="For --render-audio: force numpy sine synthesizer even if "
                        "FluidSynth is available (no timbre, but no system deps).")

    return p.parse_args()


def build_prompt(
    hf_tokenizer, amt_cfg: dict, *,
    mode: str = "autoregress", caption: str | None = None,
    system_prompt: str | None = None, max_text_tokens: int | None = None,
) -> list[int]:
    """[caption text tokens (if captioned)] + [mode token] + [SEPARATOR]."""
    mode_token = "[ANTICIPATE]" if mode == "anticipate" else "[AUTOREGRESS]"
    mode_id      = hf_tokenizer.convert_tokens_to_ids(mode_token)
    separator_id = hf_tokenizer.convert_tokens_to_ids("[SEPARATOR]")

    captioned = amt_cfg.get("caption_text", False)
    if captioned:
        if not caption:
            print("ERROR: this checkpoint was finetuned with --caption-text; pass --caption.")
            sys.exit(1)
        system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        max_text_tokens = max_text_tokens or amt_cfg.get("max_text_tokens") or 256
        text_ids = hf_tokenizer.encode(system_prompt + caption + " ", add_special_tokens=False)
        text_ids = text_ids[:max_text_tokens]
        return text_ids + [mode_id, separator_id]

    if caption:
        print("WARNING: --caption given but this checkpoint was not finetuned with "
              "--caption-text; ignoring it.")
    return [mode_id, separator_id]


def tokens_to_midi(
    generated_ids: list[int],
    hf_tokenizer,
    char_to_base_token: dict,
    ctrl_char_to_base_token: dict | None,
    onset_ms: float, dur_ms: float, vel_bins: int,
):
    token_strings = hf_tokenizer.convert_ids_to_tokens(generated_ids)
    notes = decode_generated_ids(
        token_strings, char_to_base_token, ctrl_char_to_base_token, strip_control=True,
    )
    pm = amt_to_midi(notes, onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins)
    return pm, token_strings, notes


def load_checkpoint_for_generation(
    run_dir: str | None, checkpoint: str | None, step: str | None, device: str | None,
) -> dict:
    """Resolve a checkpoint and load its model + tokenizer + AMT config, ready for
    build_prompt()/tokens_to_midi(). Shared by this script and
    scripts/generate_eval_set.py so both load checkpoints identically.

    Returns a dict with keys: model, hf_tokenizer, amt_cfg, char_to_base_token,
    ctrl_char_to_base_token, onset_ms, dur_ms, vel_bins, window_size, pad_token_id,
    run_dir, checkpoint_dir, device.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"ERROR: required package not found: {e}")
        sys.exit(1)

    if checkpoint:
        checkpoint_dir = checkpoint
        run_dir = run_dir or os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_dir)))
    else:
        if not run_dir:
            print("ERROR: pass --run-dir or --checkpoint.")
            sys.exit(1)
        run_dir = os.path.abspath(run_dir)
        checkpoint_dir = _resolve_checkpoint(run_dir, step)

    print(f"Run dir:    {run_dir}")
    print(f"Checkpoint: {checkpoint_dir}")

    # Released checkpoints carry their own amt_config.json; training runs keep it
    # in the run dir.
    amt_cfg_path = os.path.join(checkpoint_dir, "amt_config.json")
    if not os.path.exists(amt_cfg_path):
        amt_cfg_path = os.path.join(run_dir, "amt_config.json")
    if not os.path.exists(amt_cfg_path):
        print(f"ERROR: amt_config.json not found in {checkpoint_dir} or {run_dir}.")
        sys.exit(1)
    with open(amt_cfg_path) as f:
        amt_cfg = json.load(f)

    tokenizer_path = _resolve_path(amt_cfg["amt_tokenizer"])
    mapping_path   = _auto_mapping(tokenizer_path)
    if mapping_path is None:
        print(f"ERROR: could not auto-detect mapping for tokenizer {tokenizer_path}")
        sys.exit(1)

    cfg      = _auto_train_cfg(tokenizer_path)
    onset_ms = cfg.get("onset_ms", 10.0)
    dur_ms   = cfg.get("dur_ms",   10.0)
    vel_bins = cfg.get("vel_bins", 32)
    print(f"Quantization: onset={onset_ms}ms  dur={dur_ms}ms  vel={vel_bins}bins")

    if is_mapping_doubled(mapping_path):
        _, char_to_base_token, _, ctrl_char_to_base_token = load_mapping_doubled(mapping_path)
    else:
        _, char_to_base_token = load_mapping(mapping_path)
        ctrl_char_to_base_token = None

    window_size = amt_cfg.get("window_size", 2048)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model + tokenizer from checkpoint …")
    hf_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model        = AutoModelForCausalLM.from_pretrained(checkpoint_dir).to(device)
    model.eval()

    pad_token_id = hf_tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = hf_tokenizer.eos_token_id

    return dict(
        model=model, hf_tokenizer=hf_tokenizer, amt_cfg=amt_cfg,
        char_to_base_token=char_to_base_token, ctrl_char_to_base_token=ctrl_char_to_base_token,
        onset_ms=onset_ms, dur_ms=dur_ms, vel_bins=vel_bins, window_size=window_size,
        pad_token_id=pad_token_id, run_dir=run_dir, checkpoint_dir=checkpoint_dir, device=device,
    )


def main() -> None:
    args = parse_args()

    import torch

    loaded = load_checkpoint_for_generation(args.run_dir, args.checkpoint, args.step, args.device)
    run_dir            = loaded["run_dir"]
    checkpoint_dir      = loaded["checkpoint_dir"]
    amt_cfg            = loaded["amt_cfg"]
    hf_tokenizer       = loaded["hf_tokenizer"]
    model              = loaded["model"]
    char_to_base_token = loaded["char_to_base_token"]
    ctrl_char_to_base_token = loaded["ctrl_char_to_base_token"]
    onset_ms, dur_ms, vel_bins = loaded["onset_ms"], loaded["dur_ms"], loaded["vel_bins"]
    window_size        = loaded["window_size"]
    pad_token_id       = loaded["pad_token_id"]
    device             = loaded["device"]
    max_new_tokens     = args.max_new_tokens

    if args.seed is not None:
        torch.manual_seed(args.seed)

    prompt_ids = build_prompt(
        hf_tokenizer, amt_cfg,
        mode=args.mode, caption=args.caption,
        system_prompt=args.system_prompt, max_text_tokens=args.max_text_tokens,
    )
    prompt_len = len(prompt_ids)
    if max_new_tokens is None:
        max_new_tokens = max(1, window_size - prompt_len)
    print(f"Prompt tokens: {prompt_len}  max_new_tokens: {max_new_tokens}")

    input_ids      = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    sampling_kwargs = {}
    if args.do_sample:
        sampling_kwargs = {
            "temperature": args.temperature,
            "top_p":       args.top_p,
            "top_k":       args.top_k,
        }

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
            num_return_sequences=args.num_samples,
            pad_token_id=pad_token_id,
            **sampling_kwargs,
        )

    use_fluid = False
    sf2_path  = None
    if args.render_audio:
        if args.numpy_synth:
            print("  Synthesizer: numpy sine (--numpy-synth; no timbre differentiation)")
        elif not try_fluidsynth_import():
            print(
                "ERROR: FluidSynth is not available.\n"
                "  Install the system library:  conda install -c conda-forge fluidsynth\n"
                "                           or: sudo apt-get install -y fluidsynth libfluidsynth-dev\n"
                "  Install the Python binding:  pip install pyfluidsynth\n"
                "  Then download Fluid R3 GM:   https://musical-artifacts.com/artifacts/738\n"
                "  Or use --numpy-synth for a basic sinewave fallback (no timbre).",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            use_fluid = True

        sf2_path = args.soundfont or find_soundfont()
        if use_fluid:
            if sf2_path:
                print(f"  Synthesizer: FluidSynth  soundfont={sf2_path}")
            else:
                print(
                    "  Synthesizer: FluidSynth  (no soundfont found — using pretty_midi's "
                    "bundled TimGM6mb.sf2; for better timbre download Fluid R3 GM from "
                    "musical-artifacts.com/artifacts/738 and pass --soundfont or set $MIDI_SOUNDFONT)"
                )

    ckpt_name = os.path.basename(checkpoint_dir.rstrip("/"))
    out_dir   = args.out_dir or os.path.join(run_dir, "generations", ckpt_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "generation_config.json"), "w") as f:
        json.dump({
            "checkpoint":       checkpoint_dir,
            "mode":             args.mode,
            "caption":          args.caption,
            "num_samples":      args.num_samples,
            "prompt_tokens":    prompt_len,
            "max_new_tokens":   max_new_tokens,
            "do_sample":        args.do_sample,
            "temperature":      args.temperature,
            "top_p":            args.top_p,
            "top_k":            args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "seed":             args.seed,
        }, f, indent=2)

    for i in range(args.num_samples):
        generated_ids = output_ids[i, prompt_len:].tolist()
        pm, _, _ = tokens_to_midi(
            generated_ids, hf_tokenizer, char_to_base_token, ctrl_char_to_base_token,
            onset_ms, dur_ms, vel_bins,
        )

        stem     = f"sample_{i:04d}"
        mid_path = os.path.join(out_dir, f"{stem}.mid")
        pm.write(mid_path)

        n_notes = sum(len(inst.notes) for inst in pm.instruments)
        log_line = f"  {stem}: {len(generated_ids)} tokens → {n_notes} notes → {mid_path}"
        if args.render_audio:
            wav_path = os.path.join(out_dir, f"{stem}.wav")
            try:
                synth_and_write(pm, wav_path, sf2_path, args.sample_rate, use_fluid)
            except Exception as e:
                # pretty_midi's fluidsynth() can raise on certain note-timing edge
                # cases (per-instrument buffer length mismatch); skip audio for
                # this sample rather than losing the rest of the batch (matches
                # src/eval_utils.py's synth_midi_cached, which is defensive
                # for the same reason).
                log_line += f"  → WARNING: audio synthesis failed: {e}"
            else:
                log_line += f"  → {wav_path}"
        print(log_line)

    print(f"\nOutput dir: {out_dir}")


if __name__ == "__main__":
    main()
