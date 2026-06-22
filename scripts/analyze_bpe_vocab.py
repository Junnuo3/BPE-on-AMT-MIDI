"""
Analyze a trained BPE tokenizer on AMT MIDI sequences.

Usage:
    python bpe/scripts/analyze_bpe_vocab.py --vocab-size 11900
"""

import sys
import os
import argparse
import json
import csv
import re
from collections import Counter

_SCRIPTS_DIR     = os.path.dirname(os.path.abspath(__file__))
_BPE_ROOT        = os.path.dirname(_SCRIPTS_DIR)
_REPO_ROOT       = os.path.dirname(_BPE_ROOT)
_DEFAULT_DATASET = os.path.join(_REPO_ROOT, "dataset", "lmd_matched")
_DEFAULT_OUT     = os.path.join(_BPE_ROOT, "tokenizers", "vocab_sweep",
                                "q_onset-10ms_duration-10ms_velocity-128bin")

sys.path.insert(0, os.path.join(_SCRIPTS_DIR, "..", "src"))

from tqdm import tqdm
from tokenizers import Tokenizer

from bpe_utils import (
    load_mapping,
    load_mapping_doubled,
    is_mapping_doubled,
    load_corpus_unconditional,
    find_midi_files,
    serialize_midi,
)

BATCH_SIZE = 512
SPECIAL_TOKENS = {"[UNK]", "[PAD]", "[BOS]", "[EOS]",
                  "[REST]", "[SEPARATOR]", "[AUTOREGRESS]", "[ANTICIPATE]"}



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze a trained AMT BPE tokenizer — generates CSVs and a summary JSON."
    )
    p.add_argument("--dataset", default=_DEFAULT_DATASET,
                   help="Root directory of MIDI files (for corpus regeneration fallback).")
    p.add_argument("--out", default=_DEFAULT_OUT,
                   help="Outputs root (must contain mappings/ and cache/).")
    p.add_argument("--tokenizer-path", default=None,
                   help="Path to a trained tokenizer JSON.")
    p.add_argument("--vocab-size", type=int, default=None,
                   help="Infer tokenizer path from vocab size when --tokenizer-path is omitted.")
    p.add_argument("--limit-files", type=int, default=None,
                   help="Cap number of MIDI files when regenerating corpus.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing analysis outputs.")
    return p.parse_args()



def resolve_tokenizer_path(args: argparse.Namespace) -> str:
    if args.tokenizer_path:
        return args.tokenizer_path
    if args.vocab_size:
        fname = f"amt_compound_bpe_vocab{args.vocab_size}.json"
        # vocab_sweep layout: vocab-<N>/<fname>
        candidate = os.path.join(args.out, f"vocab-{args.vocab_size}", fname)
        if os.path.exists(candidate):
            return candidate
        # quantization experiment layout: tokenizers/<fname>
        return os.path.join(args.out, "tokenizers", fname)
    print("ERROR: provide --tokenizer-path or --vocab-size.")
    sys.exit(1)


def extract_vocab_label(path: str) -> str | None:
    m = re.search(r"vocab(\d+)", os.path.basename(path))
    return m.group(1) if m else None



def load_corpus(args: argparse.Namespace, out_dir: str) -> list[str]:
    cache_dir = os.path.join(out_dir, "cache")

    if not args.limit_files:
        corpus_file = load_corpus_unconditional(cache_dir)
        if corpus_file is not None:
            corpus = list(corpus_file)  # analysis needs random-access slicing
            print(f"Corpus loaded from cache — {len(corpus):,} strings")
            return corpus

    mapping_path = os.path.join(out_dir, "mappings", "amt_base_token_char_mapping.json")
    if not os.path.exists(mapping_path):
        print(f"ERROR: mapping not found at {mapping_path}. Run train_bpe.py first.")
        sys.exit(1)
    base_token_to_char, _ = load_mapping(mapping_path)

    print(f"Scanning {args.dataset} …")
    midi_files = find_midi_files(args.dataset)
    if args.limit_files:
        midi_files = midi_files[: args.limit_files]
    print(f"  {len(midi_files):,} files")

    corpus: list[str] = []
    for path in tqdm(midi_files, desc="serializing", unit="file"):
        text = serialize_midi(path, base_token_to_char)
        if text is not None:
            corpus.append(text)
    print(f"  {len(corpus):,} valid files serialized")
    return corpus



def get_category(amt_token: str) -> str:
    # ctrl_ prefix distinguishes control-variant tokens from events
    if amt_token.startswith("ctrl_"):
        inner = amt_token[5:]  # strip "ctrl_"
        m = re.match(r"(\w+)-", inner)
        return f"ctrl_{m.group(1)}" if m else "ctrl_unknown"
    m = re.match(r"<(\w+)-", amt_token)
    return m.group(1) if m else "unknown"


def build_vocab_tables(
    tokenizer: Tokenizer,
    char_to_base_token: dict[str, str],
    ctrl_char_to_base_token: dict[str, str] | None = None,
) -> tuple[dict[int, str], dict[int, list[str]]]:
    """Build {id→token_string} and {id→[amt_base_tokens]} for all merged vocab entries.

    In anticipation mode, ctrl chars get a "ctrl_" prefix so get_category can tag them.
    """
    vocab = tokenizer.get_vocab()
    id_to_str: dict[int, str] = {v: k for k, v in vocab.items()}
    id_to_amt: dict[int, list[str]] = {}
    ctrl_chars = set(ctrl_char_to_base_token.keys()) if ctrl_char_to_base_token else set()

    for tok_str, tok_id in vocab.items():
        if tok_str in SPECIAL_TOKENS or len(tok_str) <= 1:
            continue
        decoded: list[str] = []
        for c in tok_str:
            if ctrl_char_to_base_token and c in ctrl_chars:
                base = ctrl_char_to_base_token[c]
                inner = base[1:-1] if base.startswith("<") and base.endswith(">") else base
                decoded.append(f"ctrl_{inner}")
            else:
                decoded.append(char_to_base_token.get(c, f"<UNK:{ord(c):#x}>"))
        id_to_amt[tok_id] = decoded
    return id_to_str, id_to_amt



def count_token_frequencies(corpus: list[str], tokenizer: Tokenizer) -> Counter:
    counts: Counter = Counter()
    batches = range(0, len(corpus), BATCH_SIZE)
    for i in tqdm(batches, desc="encoding corpus", unit="batch"):
        batch = corpus[i : i + BATCH_SIZE]
        for enc in tokenizer.encode_batch(batch):
            counts.update(enc.ids)
    return counts



def save_token_frequency(
    freq: Counter,
    id_to_str: dict[int, str],
    id_to_amt: dict[int, list[str]],
    char_to_base_token: dict[str, str],
    out_dir: str,
) -> None:
    total = sum(freq.values())
    path = os.path.join(out_dir, "token_frequency.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["original_tokens", "corpus_count", "percentage", "token_length"]
        )
        writer.writeheader()
        for tok_id, count in freq.most_common():
            tok_str = id_to_str.get(tok_id, "")
            if tok_str in SPECIAL_TOKENS:
                continue
            if len(tok_str) == 1:
                amt_tokens = [char_to_base_token.get(tok_str, f"<UNK:{ord(tok_str):#x}>")]
            else:
                amt_tokens = id_to_amt.get(tok_id, [f"<UNK:{tok_str}>"])
            writer.writerow({
                "original_tokens": " ".join(amt_tokens),
                "corpus_count": count,
                "percentage": f"{count / total * 100:.4f}",
                "token_length": len(amt_tokens),
            })
    print(f"  token_frequency.csv")


def _compute_merge_counts(
    id_to_amt: dict[int, list[str]],
    freq: Counter,
    key_fn,
) -> tuple[Counter, Counter]:
    """Count how many merged vocab types and corpus occurrences each key appears in."""
    vocab_counts: Counter = Counter()
    corpus_counts: Counter = Counter()
    for tok_id, amt_tokens in id_to_amt.items():
        c = freq.get(tok_id, 0)
        for tok in amt_tokens:
            k = key_fn(tok)
            vocab_counts[k] += 1
            corpus_counts[k] += c
    return vocab_counts, corpus_counts


def save_merged_token_category_counts(
    id_to_amt: dict[int, list[str]], freq: Counter, out_dir: str
) -> tuple[Counter, Counter]:
    vocab_counts, corpus_counts = _compute_merge_counts(id_to_amt, freq, get_category)
    # sort by corpus count — actual usage matters more than vocab structure
    ordered = sorted(vocab_counts, key=lambda k: corpus_counts[k], reverse=True)
    path = os.path.join(out_dir, "merged_token_category_counts.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["category", "merged_vocab_count", "corpus_weighted_count"]
        )
        writer.writeheader()
        for cat in ordered:
            writer.writerow({
                "category": cat,
                "merged_vocab_count": vocab_counts[cat],
                "corpus_weighted_count": corpus_counts[cat],
            })
    print(f"  merged_token_category_counts.csv")
    return vocab_counts, corpus_counts


def compute_component_analysis(
    id_to_amt: dict[int, list[str]], freq: Counter
) -> dict[int, tuple[Counter, Counter]]:
    """Group merged tokens by how many base tokens they span, count vocab and corpus occurrences."""
    by_size: dict[int, list[int]] = {}
    for tok_id, amt_tokens in id_to_amt.items():
        by_size.setdefault(len(amt_tokens), []).append(tok_id)

    result: dict[int, tuple[Counter, Counter]] = {}
    for size in sorted(by_size):
        combo_vocab: Counter = Counter()
        combo_corpus: Counter = Counter()
        for tok_id in by_size[size]:
            cats = tuple(sorted(get_category(t) for t in id_to_amt[tok_id]))
            combo_vocab[cats] += 1
            combo_corpus[cats] += freq.get(tok_id, 0)
        result[size] = (combo_vocab, combo_corpus)
    return result


def save_component_analysis(
    component_data: dict[int, tuple[Counter, Counter]], out_dir: str
) -> None:
    for size, (combo_vocab, combo_corpus) in sorted(component_data.items()):
        ordered = sorted(combo_vocab, key=lambda k: combo_corpus[k], reverse=True)
        path = os.path.join(out_dir, f"component_analysis_size{size}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["categories", "vocab_count", "corpus_count"]
            )
            writer.writeheader()
            for cats in ordered:
                writer.writerow({
                    "categories": "+".join(cats),
                    "vocab_count": combo_vocab[cats],
                    "corpus_count": combo_corpus[cats],
                })
        print(f"  component_analysis_size{size}.csv")


def save_token_length_distribution(
    id_to_str: dict[int, str], out_dir: str
) -> None:
    length_counts: Counter = Counter()
    for tok_str in id_to_str.values():
        if tok_str not in SPECIAL_TOKENS:
            length_counts[len(tok_str)] += 1
    path = os.path.join(out_dir, "token_length_distribution.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["token_length", "vocab_count"])
        writer.writeheader()
        for length in sorted(length_counts):
            writer.writerow({"token_length": length, "vocab_count": length_counts[length]})
    print(f"  token_length_distribution.csv")


def save_summary(
    tokenizer: Tokenizer,
    freq: Counter,
    id_to_str: dict[int, str],
    id_to_amt: dict[int, list[str]],
    char_to_base_token: dict[str, str],
    cat_vocab_counts: Counter,
    cat_corpus_counts: Counter,
    component_data: dict[int, tuple[Counter, Counter]],
    eval_result: dict | None,
    out_dir: str,
) -> None:
    non_special = {tid: s for tid, s in id_to_str.items() if s not in SPECIAL_TOKENS}
    total_chars = sum(len(s) for s in non_special.values())
    avg_len = total_chars / len(non_special) if non_special else 0.0

    def top(vocab_c: Counter, corpus_c: Counter, key: str, n: int = 10) -> list[dict]:
        # sorted by corpus_weighted_count: how often these tokens are merged in actual data
        ordered = sorted(vocab_c, key=lambda k: corpus_c[k], reverse=True)[:n]
        return [{
            key: k,
            "merged_vocab_count": vocab_c[k],
            "corpus_weighted_count": corpus_c[k],
        } for k in ordered]

    common_bpe: list[dict] = []
    for tok_id, count in freq.most_common(50):
        tok_str = id_to_str.get(tok_id, "")
        if tok_str in SPECIAL_TOKENS:
            continue
        if len(tok_str) == 1:
            amt_repr = char_to_base_token.get(tok_str, tok_str)
        else:
            amt_repr = " ".join(id_to_amt.get(tok_id, [tok_str]))
        common_bpe.append({"original_tokens": amt_repr, "corpus_count": count})
        if len(common_bpe) >= 10:
            break

    component_summary: dict[str, list[dict]] = {}
    for size, (combo_vocab, combo_corpus) in sorted(component_data.items()):
        total_vocab = sum(combo_vocab.values())
        ordered = sorted(combo_vocab, key=lambda k: combo_corpus[k], reverse=True)[:10]
        component_summary[str(size)] = [
            {
                "categories": "+".join(cats),
                "vocab_count": combo_vocab[cats],
                "vocab_proportion": round(combo_vocab[cats] / total_vocab, 4) if total_vocab else 0.0,
                "corpus_count": combo_corpus[cats],
            }
            for cats in ordered
        ]

    summary = {
        "vocab_size": tokenizer.get_vocab_size(),
        "total_bpe_tokens": eval_result.get("total_bpe_tokens") if eval_result else None,
        "num_merged_tokens": len(id_to_amt),
        "average_bpe_token_length": round(avg_len, 4),
        "most_common_bpe_tokens": common_bpe,
        "most_common_categories": top(cat_vocab_counts, cat_corpus_counts, "category"),
        "component_analysis": component_summary,
    }

    path = os.path.join(out_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary.json")



def load_eval_result(out_dir: str, vocab_label: str) -> dict | None:
    path = os.path.join(out_dir, "results", f"eval_partial_vocab{vocab_label}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    tok_path = resolve_tokenizer_path(args)
    if not os.path.exists(tok_path):
        print(f"ERROR: tokenizer not found at {tok_path}")
        sys.exit(1)

    vocab_label = args.vocab_size and str(args.vocab_size) or extract_vocab_label(tok_path)
    analysis_dir = os.path.join(
        args.out, "analysis", f"vocab{vocab_label}" if vocab_label else "analysis"
    )

    summary_path = os.path.join(analysis_dir, "summary.json")
    if os.path.exists(summary_path) and not args.force:
        print(f"Analysis already exists at {analysis_dir}/")
        print("Use --force to overwrite.")
        sys.exit(0)

    os.makedirs(analysis_dir, exist_ok=True)

    print(f"Loading tokenizer: {tok_path}")
    tokenizer = Tokenizer.from_file(tok_path)
    print(f"  vocab size: {tokenizer.get_vocab_size():,}")

    mapping_path = os.path.join(args.out, "mappings", "amt_base_token_char_mapping.json")
    if not os.path.exists(mapping_path):
        print(f"ERROR: mapping not found at {mapping_path}. Run train_bpe.py first.")
        sys.exit(1)
    ctrl_c2b: dict[str, str] | None = None
    if is_mapping_doubled(mapping_path):
        _, char_to_base_token, _, ctrl_c2b = load_mapping_doubled(mapping_path)
        print(f"  Doubled mapping: {len(char_to_base_token):,} event + "
              f"{len(ctrl_c2b):,} ctrl base tokens")
    else:
        _, char_to_base_token = load_mapping(mapping_path)
        print(f"  {len(char_to_base_token):,} base tokens in mapping")

    corpus = load_corpus(args, args.out)

    print("Building vocab tables …")
    id_to_str, id_to_amt = build_vocab_tables(tokenizer, char_to_base_token,
                                               ctrl_char_to_base_token=ctrl_c2b)
    print(f"  {len(id_to_amt):,} merged tokens  |  "
          f"{len(id_to_str) - len(SPECIAL_TOKENS) - len(id_to_amt):,} base tokens")

    freq = count_token_frequencies(corpus, tokenizer)
    eval_result = load_eval_result(args.out, vocab_label) if vocab_label else None

    print(f"\nSaving to {analysis_dir}/")
    cat_vocab_counts, cat_corpus_counts = save_merged_token_category_counts(id_to_amt, freq, analysis_dir)
    component_data = compute_component_analysis(id_to_amt, freq)
    save_component_analysis(component_data, analysis_dir)
    save_token_length_distribution(id_to_str, analysis_dir)
    save_token_frequency(freq, id_to_str, id_to_amt, char_to_base_token, analysis_dir)
    save_summary(
        tokenizer, freq, id_to_str, id_to_amt, char_to_base_token,
        cat_vocab_counts, cat_corpus_counts,
        component_data,
        eval_result, analysis_dir,
    )

    print(f"\nDone. {analysis_dir}/")


if __name__ == "__main__":
    main()
