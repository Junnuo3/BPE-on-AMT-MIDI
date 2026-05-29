from typing import Iterable

from tokenizers import Tokenizer

from bpe_utils import serialize_midi


def eval_file(
    midi_path: str,
    tokenizer: Tokenizer,
    base_token_to_char: dict[str, str],
) -> tuple[int, int] | None:
    """Return (original_len, bpe_len) for one file, or None if unserializable."""
    text = serialize_midi(midi_path, base_token_to_char)
    if text is None:
        return None
    original = len(text.split()) * 5
    bpe      = len(tokenizer.encode(text).ids)
    return original, bpe


def eval_dataset(
    midi_files: Iterable[str],
    tokenizer: Tokenizer,
    base_token_to_char: dict[str, str],
) -> dict:
    total_orig = total_bpe = num_files = 0
    for path in midi_files:
        result = eval_file(path, tokenizer, base_token_to_char)
        if result is None:
            continue
        orig, bpe  = result
        total_orig += orig
        total_bpe  += bpe
        num_files  += 1

    return {
        "vocab_size":            tokenizer.get_vocab_size(),
        "num_files":             num_files,
        "total_original_tokens": total_orig,
        "total_bpe_tokens":      total_bpe,
        "reduction_ratio":       total_bpe  / total_orig if total_orig else float("nan"),
        "compression_factor":    total_orig / total_bpe  if total_bpe  else float("nan"),
    }
