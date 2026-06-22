# BPE on AMT MIDI

BPE tokenizers on MIDI, using AMT's 5-token compound note representation. Measures how much sequence length shrinks as vocab size and quantization level vary.

Each note is `<time-t> <duration-d> <pitch-p> <instrument-i> <velocity-v>` — 5 tokens. A song with N notes starts as 5N base tokens before BPE.

## Setup

```bash
pip install -r bpe/requirements.txt
```

Dataset expected at `dataset/lmd_matched/` (scanned recursively for `.mid`/`.midi`).

## Pipeline

### Vocab sweep

Train across many vocab sizes at default quantization (onset=10ms, dur=10ms, vel=128).

```bash
python bpe/scripts/train_bpe.py        # train all vocab sizes
python bpe/scripts/evaluate_bpe.py     # measure compression
python bpe/scripts/plot_results.py     # plot the curve
```

### Quantization sweep

Vary onset, duration, or velocity independently at 8192 merges. Only the finest resolution per factor needs a full MIDI scan — coarser configs are derived from it.

```bash
python bpe/scripts/run_quant_experiments.py --factor onset
python bpe/scripts/run_quant_experiments.py --factor duration
python bpe/scripts/run_quant_experiments.py --factor velocity

# Same sweep but keeping onset as standalone tokens
python bpe/scripts/run_quant_experiments.py --onset-standalone --factor velocity

python bpe/scripts/plot_results.py --quant
```

### Anticipation

Doubles an existing quantization tokenizer without re-training BPE: every event token gets a matching control-variant token, and the 4 sentinels `[REST] [SEPARATOR] [AUTOREGRESS] [ANTICIPATE]` are appended.

```bash
# Double the vel=32 tokenizer (default)
python bpe/scripts/double_tokenizer.py
# → bpe/tokenizers/anticipation/velocity/onset-10ms_duration-10ms_velocity-32bin/

# Finetune SmolLM2 with on-the-fly anticipation augmentation
python bpe/scripts/finetune_smollm2.py \
    --ant-dir bpe/tokenizers/anticipation/velocity/onset-10ms_duration-10ms_velocity-32bin
# → bpe/outputs/runs/smollm2/velocity/onset-10ms_duration-10ms_velocity-32bin/
```

10 passes per file per epoch: pass 0 = `[AUTOREGRESS]` (no control), passes 1–9 = `[ANTICIPATE]` with control extraction. 

## Directory layout

```
bpe/tokenizers/
├── vocab_sweep/
│   └── q_onset-10ms_duration-10ms_velocity-128bin/
│       ├── vocab-11388/   amt_compound_bpe_vocab11388.json
│       ├── vocab-11644/   …
│       ├── vocab-76924/   …
│       ├── cache/         corpus cache (filelist, corpus.txt, …)
│       ├── mappings/      amt_base_token_char_mapping.json
│       ├── analysis/      per-vocab CSV/JSON breakdowns
│       ├── results/       length_reduction_results.json
│       └── train_config.json
│
├── quantization_sweep/
│   └── merges-8192/
│       ├── onset/
│       │   ├── onset-1ms_duration-10ms_velocity-128bin/
│       │   │   ├── tokenizers/   amt_compound_bpe_vocab*.json
│       │   │   ├── mappings/
│       │   │   ├── cache/
│       │   │   ├── results/
│       │   │   └── train_config.json
│       │   └── onset-{2,5,10,20}ms_duration-10ms_velocity-128bin/
│       ├── duration/   onset-10ms_duration-{1,2,5,10,20}ms_velocity-128bin/
│       └── velocity/   onset-10ms_duration-10ms_velocity-{8,16,32,64,128}bin/
│
├── merge_constraints/
│   └── no_onset_merge/
│       └── merges-8192/
│           ├── onset/    (same configs as quantization_sweep/onset/)
│           ├── duration/
│           └── velocity/
│
└── anticipation/
    └── velocity/
        └── onset-10ms_duration-10ms_velocity-32bin/
            ├── tokenizers/   amt_compound_bpe_vocab*.json  (doubled vocab)
            ├── mappings/     amt_base_token_char_mapping.json  (doubled)
            ├── cache/        filelist.txt
            └── train_config.json

bpe/outputs/runs/smollm2/
└── velocity/
    └── onset-10ms_duration-10ms_velocity-32bin/
        ├── amt_config.json
        └── checkpoints/
```

The `{factor}/` sub-level (onset/duration/velocity) within each sweep group disambiguates experiments that share the same quant parameters at the default values (onset=10ms, dur=10ms, vel=128bin).

## Options

| Flag | Script | Default | Description |
|---|---|---|---|
| `--vocab-sizes N [N …]` | train | auto | Override target vocab sizes |
| `--merges N` | train | — | Train to B+S+N vocab size |
| `--onset-ms F` | train/eval | 10 | Onset quantization step in ms |
| `--dur-ms F` | train/eval | 10 | Duration quantization step in ms |
| `--vel-bins N` | train/eval | 128 | Velocity quantization levels |
| `--onset-standalone` | train/eval | off | Keep onset tokens as standalone BPE words |
| `--workers N` | train/finetune | auto | Parallel workers (defaults to `SLURM_CPUS_PER_TASK`) |
| `--limit-files N` | train/eval | all | Cap number of MIDI files |
| `--force-rescan` | train | off | Ignore all caches |
| `--src-dir PATH` | double | `quantization_sweep/…/velocity-32bin` | Source quantization experiment dir |
| `--ant-dir PATH` | finetune | `anticipation/…/velocity-32bin` | Anticipation tokenizer dir |
| `--augment-passes N` | finetune | 10 | Anticipation passes per file per epoch |
| `--corpus PATH` | finetune | — | Pre-built chunks.npy for baseline (static) mode |
| `--fp16` | finetune | off | Mixed-precision training |

All training scripts are resumable — interrupt and re-run to continue.
Use `--force-rescan` / `--force-reeval` to start fresh.
