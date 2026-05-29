from typing import Callable, Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

SPECIAL_TOKENS: list[str] = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]


def train_bpe(
    iterator: Iterator[str],
    vocab_size: int,
    initial_alphabet: list[str],
    show_progress: bool = True,
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=initial_alphabet,
        show_progress=show_progress,
    )
    tokenizer.train_from_iterator(iterator, trainer=trainer)
    return tokenizer


def train_multiple(
    iterator_factory: Callable[[], Iterator[str]],
    vocab_sizes: list[int],
    initial_alphabet: list[str],
) -> dict[int, Tokenizer]:
    """Train one tokenizer per vocab size, restarting the iterator each time."""
    return {
        vs: train_bpe(iterator_factory(), vs, initial_alphabet)
        for vs in sorted(vocab_sizes)
    }
