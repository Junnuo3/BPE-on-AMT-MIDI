"""
Generate MIDI from a finetuned SmolLM2 checkpoint, then decode the
generated AMT-BPE tokens back into a .mid file.

Outputs are written next to the checkpoint they came from
    <run_dir>/checkpoints/checkpoint-14400/...        (trained model, from finetune_smollm2.py)
    <run_dir>/generations/checkpoint-14400/sample_0000.mid
    <run_dir>/generations/checkpoint-14400/sample_0000.json   (tokens + notes for that sample)
    <run_dir>/generations/checkpoint-14400/generation_config.json

Usage:
    python bpe/scripts/generate_smollm2.py --run-dir bpe/outputs/runs/no_bpe/velocity/onset-10ms_duration-10ms_velocity-32bin
    python bpe/scripts/generate_smollm2.py --checkpoint <path/to/checkpoints/checkpoint-10200> --num-samples 4
    python bpe/scripts/generate_smollm2.py --run-dir <run_dir> --caption "A relaxing piano ballad"
"""

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BPE_ROOT   = os.path.dirname(_SCRIPT_DIR)   # repo root

sys.path.insert(0, os.path.join(_BPE_ROOT, "src"))
sys.path.insert(0, _SCRIPT_DIR)

from finetune_smollm2 import _auto_mapping, _auto_train_cfg  # noqa: E402
from bpe_utils import load_mapping, load_mapping_doubled, is_mapping_doubled, decode_generated_ids  # noqa: E402
from midi_to_amt import amt_to_midi  # noqa: E402

_DEFAULT_SYSTEM_PROMPT = (
    "You are a world-class composer. Please compose some music "
    "according to the following description: "
)


def _resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_BPE_ROOT, path)


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
    p.add_argument("--top-p",        type=float, default=0.95)
    p.add_argument("--top-k",        type=int,   default=50)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--seed",         type=int,   default=None)
    p.add_argument("--device",       default=None, help="Default: cuda if available, else cpu.")

    return p.parse_args()


def build_prompt(hf_tokenizer, amt_cfg: dict, args: argparse.Namespace) -> list[int]:
    """[caption text tokens (if captioned)] + [mode token] + [SEPARATOR]."""
    mode_token = "[ANTICIPATE]" if args.mode == "anticipate" else "[AUTOREGRESS]"
    mode_id      = hf_tokenizer.convert_tokens_to_ids(mode_token)
    separator_id = hf_tokenizer.convert_tokens_to_ids("[SEPARATOR]")

    captioned = amt_cfg.get("caption_text", False)
    if captioned:
        if not args.caption:
            print("ERROR: this checkpoint was finetuned with --caption-text; pass --caption.")
            sys.exit(1)
        system_prompt = args.system_prompt or _DEFAULT_SYSTEM_PROMPT
        max_text_tokens = args.max_text_tokens or amt_cfg.get("max_text_tokens") or 256
        text_ids = hf_tokenizer.encode(system_prompt + args.caption + " ", add_special_tokens=False)
        text_ids = text_ids[:max_text_tokens]
        return text_ids + [mode_id, separator_id]

    if args.caption:
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


def main() -> None:
    args = parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"ERROR: required package not found: {e}")
        sys.exit(1)

    if args.checkpoint:
        checkpoint_dir = args.checkpoint
        run_dir = args.run_dir or os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_dir)))
    else:
        if not args.run_dir:
            print("ERROR: pass --run-dir or --checkpoint.")
            sys.exit(1)
        run_dir = os.path.abspath(args.run_dir)
        checkpoint_dir = _resolve_checkpoint(run_dir, args.step)

    print(f"Run dir:    {run_dir}")
    print(f"Checkpoint: {checkpoint_dir}")

    amt_cfg_path = os.path.join(run_dir, "amt_config.json")
    if not os.path.exists(amt_cfg_path):
        print(f"ERROR: amt_config.json not found at {amt_cfg_path} (written by finetune_smollm2.py).")
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
    max_new_tokens = args.max_new_tokens

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print("Loading model + tokenizer from checkpoint …")
    hf_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model        = AutoModelForCausalLM.from_pretrained(checkpoint_dir).to(device)
    model.eval()

    pad_token_id = hf_tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = hf_tokenizer.eos_token_id

    prompt_ids = build_prompt(hf_tokenizer, amt_cfg, args)
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
        pm, token_strings, notes = tokens_to_midi(
            generated_ids, hf_tokenizer, char_to_base_token, ctrl_char_to_base_token,
            onset_ms, dur_ms, vel_bins,
        )

        stem     = f"sample_{i:04d}"
        mid_path = os.path.join(out_dir, f"{stem}.mid")
        pm.write(mid_path)

        n_notes = sum(len(inst.notes) for inst in pm.instruments)
        with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
            json.dump({
                "generated_token_ids": generated_ids,
                "generated_tokens":    token_strings,
                "num_notes":           n_notes,
            }, f, indent=2)

        print(f"  {stem}: {len(generated_ids)} tokens → {n_notes} notes → {mid_path}")

    print(f"\nOutput dir: {out_dir}")


if __name__ == "__main__":
    main()
