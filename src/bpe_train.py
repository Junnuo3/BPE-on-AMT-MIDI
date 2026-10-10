import json as _json
from typing import Callable, Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# [UNK] is required by the tokenizers lib; [PAD]/[BOS]/[EOS] reserve IDs but are never triggered.
SPECIAL_TOKENS: list[str] = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]

ANTICIPATION_SENTINELS: list[str] = ["[REST]", "[SEPARATOR]", "[AUTOREGRESS]", "[ANTICIPATE]"]


def train_bpe(
    iterator: Iterator[str],
    vocab_size: int,
    initial_alphabet: list[str],
    show_progress: bool = True,
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=initial_alphabet,
        show_progress=show_progress,
    )
    tokenizer.train_from_iterator(iterator, trainer=trainer)
    return tokenizer


def train_multiple(
    iterator_factory: Callable[[], Iterator[str]],
    vocab_sizes: list[int],
    initial_alphabet: list[str],
    special_tokens: list[str] | None = None,
) -> dict[int, Tokenizer]:
    """Train one tokenizer per vocab size, restarting the iterator each time."""
    return {
        vs: train_bpe(iterator_factory(), vs, initial_alphabet,
                      special_tokens=special_tokens)
        for vs in sorted(vocab_sizes)
    }


def double_tokenizer(
    tokenizer: Tokenizer,
    base_b2c: dict[str, str],
    ctrl_b2c: dict[str, str],
) -> Tokenizer:
    """Add ctrl-token counterparts to a trained BPE tokenizer for anticipation.

    Appends ctrl PUA chars + mirrored merge rules, then replaces the 4 baseline specials
    with [REST][SEPARATOR][AUTOREGRESS][ANTICIPATE] as the actual special tokens.
    Baseline IDs 0-3 stay in the vocab dict for unk_token stability.
    """
    data = _json.loads(tokenizer.to_str())
    vocab: dict[str, int] = data["model"]["vocab"]
    merges: list = data["model"]["merges"]

    # event char → ctrl char for every individual PUA char
    event_to_ctrl: dict[str, str] = {
        event_char: ctrl_b2c[tok]
        for tok, event_char in base_b2c.items()
    }

    def ctrl_of(s: str) -> str:
        return "".join(event_to_ctrl[c] for c in s)

    # 1. Append ctrl PUA chars (sorted by base token string for stable ordering)
    next_id = max(vocab.values()) + 1
    for tok in sorted(base_b2c):
        vocab[ctrl_b2c[tok]] = next_id
        next_id += 1

    # 2. Append ctrl merge rules + ctrl merged-token vocab entries.
    #    Merges may be stored as ["a","b"] lists or "a b" strings — handle both.
    #    Iterate original merges only; extend after so ctrl merges don't self-refer.
    ctrl_merges: list = []
    for merge_item in list(merges):
        if isinstance(merge_item, list):
            a, b = merge_item[0], merge_item[1]
        else:
            a, b = merge_item.split(" ", 1)
        ca, cb = ctrl_of(a), ctrl_of(b)
        cab = ca + cb
        if cab not in vocab:
            vocab[cab] = next_id
            next_id += 1
        ctrl_merges.append([ca, cb] if isinstance(merge_item, list) else f"{ca} {cb}")
    merges.extend(ctrl_merges)

    # 3. Drop baseline specials from added_tokens (keep in vocab for unk_token stability),
    #    then append the 4 anticipation sentinels as new entries.
    old_specials = set(SPECIAL_TOKENS)
    data["added_tokens"] = [
        t for t in data.get("added_tokens", [])
        if t["content"] not in old_specials
    ]

    for sentinel in ANTICIPATION_SENTINELS:
        if sentinel not in vocab:
            vocab[sentinel] = next_id
            next_id += 1
        if not any(t["content"] == sentinel for t in data["added_tokens"]):
            data["added_tokens"].append({
                "id": vocab[sentinel],
                "content": sentinel,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            })

    return Tokenizer.from_str(_json.dumps(data, ensure_ascii=True))
