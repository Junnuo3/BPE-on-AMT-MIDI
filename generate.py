#!/usr/bin/env python3
"""Generate MIDI from text captions with the released SmolLM2-135M checkpoints.

    python generate.py --prompt "A calm piano piece in C major with a slow tempo."
    python generate.py --variant no_bpe --prompts_file assets/example_prompts.txt --n_outputs 2

Each output is written as gen_<i>.mid (and gen_<i>.wav unless --no-synthesize) under
<output_root>/<timestamp>_prompt_<n>/. Sampling is restricted by the note grammar and stops
at the end-of-song token, as in the reported evaluation.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from bpe_amt_midi.notes import DUR_MS, ONSET_MS, VEL_BINS, MidiGrammarLogitsProcessor, amt_to_midi, decode_notes, load_mapping

MODEL_REPO = "Junnuo/BPE-on-AMT-MIDI"
SYSTEM_PROMPT = "You are a world-class composer. Please compose some music according to the following description: "
# Selected decoding settings: BPE on the test split, no BPE on the validation split.
DEFAULTS = {"bpe": dict(temperature=0.9, top_p=1.0, top_k=0), "no_bpe": dict(temperature=0.9, top_p=1.0, top_k=0)}


def load(model: str, variant: str, revision: str | None, device: str):
    path = model if os.path.isdir(model) else os.path.join(
        snapshot_download(model, revision=revision, allow_patterns=[f"{variant}/stage2/*"]), variant, "stage2")
    with open(os.path.join(path, "amt_config.json")) as f:
        amt_cfg = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(path)
    lm = AutoModelForCausalLM.from_pretrained(path).to(device).eval()
    return lm, tokenizer, amt_cfg


def prompt_ids(tokenizer, caption: str, max_text_tokens: int) -> list[int]:
    """[system prompt + caption, at most max_text_tokens] + [AUTOREGRESS] + [SEPARATOR]."""
    text = tokenizer.encode(SYSTEM_PROMPT + caption + " ", add_special_tokens=False)[:max_text_tokens]
    return text + [tokenizer.convert_tokens_to_ids("[AUTOREGRESS]"), tokenizer.convert_tokens_to_ids("[SEPARATOR]")]


def synthesize(pm, path: Path, soundfont: str | None, sample_rate: int = 44100) -> None:
    import numpy as np
    import soundfile as sf

    audio = pm.fluidsynth(fs=sample_rate, sf2_path=soundfont)
    peak = np.abs(audio).max(initial=0.0)
    sf.write(str(path), audio / peak if peak > 0 else audio, sample_rate)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="One text caption.")
    src.add_argument("--prompts_file", help="Text file with one caption per line.")
    p.add_argument("--variant", choices=["bpe", "no_bpe"], default="bpe")
    p.add_argument("--model", default=MODEL_REPO, help="Hugging Face repo id, or a local checkpoint dir.")
    p.add_argument("--revision", default=None, help="Hub revision (default: latest).")
    p.add_argument("--n_outputs", type=int, default=1)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None, help="0 disables top-k.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_root", default="generated_outputs")
    p.add_argument("--no-synthesize", dest="synthesize", action="store_false",
                   help="Write MIDI only (synthesis needs FluidSynth and pyfluidsynth).")
    p.add_argument("--soundfont", default=None, help=".sf2 file (default: pretty_midi's TimGM6mb).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    sampling = {k: v if getattr(args, k) is None else getattr(args, k) for k, v in DEFAULTS[args.variant].items()}
    prompts = [args.prompt] if args.prompt else [l.strip() for l in open(args.prompts_file) if l.strip()]

    lm, tokenizer, amt_cfg = load(args.model, args.variant, args.revision, args.device)
    char_to_base, ctrl_char_to_base = load_mapping()
    end_id = amt_cfg["end_token_id"]
    window = amt_cfg.get("window_size", 2048)
    n_scores = lm.get_output_embeddings().weight.shape[0]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    out_root = Path(args.output_root) / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    for n, caption in enumerate(prompts, 1):
        ids = prompt_ids(tokenizer, caption, amt_cfg.get("max_text_tokens", 256))
        torch.manual_seed(args.seed)
        x = torch.tensor([ids], device=args.device)
        t0 = time.time()
        with torch.no_grad():
            out = lm.generate(
                input_ids=x, attention_mask=torch.ones_like(x),
                max_new_tokens=max(1, window - len(ids)), do_sample=True, repetition_penalty=1.0,
                num_return_sequences=args.n_outputs, pad_token_id=pad_id, eos_token_id=end_id,
                logits_processor=LogitsProcessorList([MidiGrammarLogitsProcessor(
                    tokenizer, char_to_base, end_id=end_id, n_scores=n_scores)]),
                **sampling,
            )
        print(f"[{n}/{len(prompts)}] {time.time() - t0:.1f}s  {caption[:70]}")

        out_dir = out_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_prompt_{n}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "prompt.txt").write_text(caption + "\n")
        for i, row in enumerate(out[:, len(ids):].tolist(), 1):
            if end_id in row:
                row = row[:row.index(end_id)]
            notes = decode_notes(tokenizer.convert_ids_to_tokens(row), char_to_base, ctrl_char_to_base, strip_control=True)
            pm = amt_to_midi(notes, onset_ms=ONSET_MS, dur_ms=DUR_MS, vel_bins=VEL_BINS)
            pm.write(str(out_dir / f"gen_{i}.mid"))
            if args.synthesize:
                synthesize(pm, out_dir / f"gen_{i}.wav", args.soundfont)
            print(f"  gen_{i}: {len(row)} tokens, {len(notes)} notes, {pm.get_end_time():.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
