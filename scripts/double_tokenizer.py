"""
Double an existing BPE tokenizer for anticipation training.

Usage:
    python bpe/scripts/double_tokenizer.py
    python bpe/scripts/double_tokenizer.py --src-dir bpe/tokenizers/quantization_sweep/merges-8192/velocity/onset-10ms_duration-10ms_velocity-16bin
"""

import argparse
import glob
import json
import os
import shutil
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BPE_ROOT    = os.path.dirname(_SCRIPTS_DIR)
_REPO_ROOT   = os.path.dirname(_BPE_ROOT)

sys.path.insert(0, os.path.join(_BPE_ROOT, "src"))

from tokenizers import Tokenizer
from bpe_utils import build_ctrl_mapping, load_mapping, save_mapping
from bpe_train import double_tokenizer

_DEFAULT_SRC  = os.path.join(_BPE_ROOT, "tokenizers", "vocab_sweep",
                             "q_onset-10ms_duration-10ms_velocity-32bin")
_DEFAULT_OUT  = os.path.join(_BPE_ROOT, "tokenizers", "anticipation",
                             "velocity", "onset-10ms_duration-10ms_velocity-32bin")
_ANT_ROOT     = os.path.join(_BPE_ROOT, "tokenizers", "anticipation")
_QUANT_ROOT   = os.path.join(_BPE_ROOT, "tokenizers", "quantization_sweep", "merges-8192")
_VOCAB_SWEEP  = os.path.join(_BPE_ROOT, "tokenizers", "vocab_sweep")


def _derive_out_dir(src_dir: str) -> str:
    """Map a source dir to its anticipation output dir."""
    src_dir = os.path.abspath(src_dir)
    # vocab_sweep source → anticipation/velocity/<config>
    vocab_sweep = os.path.abspath(_VOCAB_SWEEP)
    if src_dir.startswith(vocab_sweep):
        # strip leading "q_" prefix from the config folder name
        config = os.path.basename(src_dir).lstrip("q_")
        return os.path.join(_ANT_ROOT, "velocity", config)
    # quantization_sweep source → mirrors the relative path under anticipation/
    quant = os.path.abspath(_QUANT_ROOT)
    if src_dir.startswith(quant):
        return os.path.join(_ANT_ROOT, os.path.relpath(src_dir, quant))
    # fallback
    return os.path.join(_ANT_ROOT, os.path.basename(src_dir))


def _find_tokenizer(tok_dir: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(tok_dir, "*.json")))
    return hits[-1] if hits else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", default=_DEFAULT_SRC,
                   help="Source quantization directory containing tokenizers/, "
                        "mappings/, train_config.json "
                        f"(default: {os.path.relpath(_DEFAULT_SRC, _REPO_ROOT)})")
    p.add_argument("--out-dir", default=None,
                   help="Destination anticipation directory. "
                        "Defaults to the anticipation/ subfolder derived from --src-dir "
                        f"(default src → {os.path.relpath(_DEFAULT_OUT, _REPO_ROOT)})")
    p.add_argument("--tokenizer", default=None,
                   help="Explicit tokenizer JSON inside src-dir/tokenizers/. "
                        "Auto-selects largest vocab if omitted.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing outputs.")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    src_dir = os.path.abspath(args.src_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else _derive_out_dir(src_dir)

    src_tok_dir = os.path.join(src_dir, "tokenizers")
    src_map     = os.path.join(src_dir, "mappings", "amt_base_token_char_mapping.json")
    src_cfg     = os.path.join(src_dir, "train_config.json")
    src_cache   = os.path.join(src_dir, "cache")

    dst_tok_dir = os.path.join(out_dir, "tokenizers")
    dst_map     = os.path.join(out_dir, "mappings", "amt_base_token_char_mapping.json")
    dst_cfg     = os.path.join(out_dir, "train_config.json")
    dst_cache   = os.path.join(out_dir, "cache")

    print(f"Source : {src_dir}")
    print(f"Output : {out_dir}")

    if not os.path.isdir(src_dir):
        print(f"ERROR: source dir not found: {src_dir}")
        sys.exit(1)
    if not os.path.exists(src_map):
        print(f"ERROR: mapping not found: {src_map}")
        sys.exit(1)

    src_tok = args.tokenizer or _find_tokenizer(src_tok_dir)
    if src_tok is None:
        print(f"ERROR: no tokenizer JSON found in {src_tok_dir}")
        sys.exit(1)
    if not os.path.exists(src_tok):
        print(f"ERROR: tokenizer not found: {src_tok}")
        sys.exit(1)

    tok_name = os.path.basename(src_tok)
    dst_tok  = os.path.join(dst_tok_dir, tok_name)

    if os.path.exists(dst_tok) and not args.force:
        print(f"Already exists: {dst_tok}")
        print("Use --force to overwrite.")
        sys.exit(0)

    os.makedirs(dst_tok_dir, exist_ok=True)
    os.makedirs(os.path.dirname(dst_map), exist_ok=True)

    b2c, c2b = load_mapping(src_map)
    print(f"Base mapping: {len(b2c):,} tokens")

    ctrl_b2c, ctrl_c2b = build_ctrl_mapping(b2c)
    print(f"Ctrl mapping: {len(ctrl_b2c):,} ctrl tokens")

    tokenizer = Tokenizer.from_file(src_tok)
    print(f"Loaded tokenizer: {tok_name}  vocab={tokenizer.get_vocab_size():,}")

    doubled      = double_tokenizer(tokenizer, b2c, ctrl_b2c)
    actual_vocab = doubled.get_vocab_size()
    print(f"Doubled tokenizer: vocab={actual_vocab:,}")

    # Save with ensure_ascii so PUA chars appear as \uXXXX
    doubled.save(dst_tok)
    with open(dst_tok, encoding="utf-8") as f:
        data = json.load(f)
    with open(dst_tok, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)
    print(f"Saved → {dst_tok}")

    save_mapping(b2c, c2b, dst_map,
                 fingerprint="derived",
                 ctrl_b2c=ctrl_b2c,
                 ctrl_c2b=ctrl_c2b)
    print(f"Saved → {dst_map}")

    cfg: dict = {}
    if os.path.exists(src_cfg):
        with open(src_cfg) as f:
            cfg = json.load(f)
    cfg["augment"] = True
    with open(dst_cfg, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved → {dst_cfg}")

    os.makedirs(dst_cache, exist_ok=True)
    for fname in ("filelist.txt", "filelist_meta.json"):
        src = os.path.join(src_cache, fname)
        dst = os.path.join(dst_cache, fname)
        if os.path.exists(src) and (not os.path.exists(dst) or args.force):
            shutil.copy2(src, dst)
            print(f"Copied cache/{fname} → {dst}")

    print(f"\nDone. Doubled vocab: {actual_vocab:,}")


if __name__ == "__main__":
    main()
