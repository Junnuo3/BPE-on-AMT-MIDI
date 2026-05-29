# BPE on AMT MIDI

Trains BPE tokenizers on MIDI files using AMT's 5-token compound note representation and measures sequence length compression across vocab sizes.

Each note → `<time-t> <duration-d> <pitch-p> <instrument-i> <velocity-v>` A song with N notes is 5N base tokens before BPE.

## Setup

```bash
pip install -r bpe/requirements.txt
```

Dataset at `dataset/lmd_matched/` (scanned recursively for `.mid`/`.midi`).

## Usage

Run from the project root.

```bash
# 1. Train
python bpe/scripts/train_bpe.py --dataset dataset/lmd_matched --out outputs

# 2. Evaluate
python bpe/scripts/evaluate_bpe.py --dataset dataset/lmd_matched --out outputs

# 3. Plot
python bpe/scripts/plot_results.py --out outputs
```

All scripts are resumable — interrupt and re-run to continue where they stopped. Use `--force-rescan` / `--force-reeval` to start fresh.

## Options

| Flag | Script | Default | Description |
|---|---|---|---|
| `--vocab-sizes N [N …]` | train | auto | Override target vocab sizes |
| `--limit-files N` | train/eval | all | Cap number of MIDI files used |
| `--time-resolution N` | train | 100 | AMT bins per second |
| `--force-rescan` | train | off | Ignore all caches |
| `--force-reeval` | eval | off | Re-evaluate existing results |

Auto vocab sizes: `B + 4 + {512, 1024, 2048, 4096, 8192, 16384}` where `B` is the observed base alphabet size and 4 is the special token count (`[UNK] [PAD] [BOS] [EOS]`).

## Outputs

```
outputs/
  mappings/   amt_base_token_char_mapping.json
  tokenizers/ amt_compound_bpe_vocab{N}.json
  results/    length_reduction_results.json / .csv
  plots/      vocab_vs_length_reduction.png
  cache/      filelist, vocab scan, corpus (resumable checkpoints)
```
