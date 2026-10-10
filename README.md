# BPE on AMT MIDI

This repository studies byte-pair encoding (BPE) over note-level MIDI tokens for
text-to-MIDI generation with a small language model. Each note is written as five
tokens taken from the compound note format of the Anticipatory Music Transformer
(AMT): onset, duration, pitch, instrument and velocity. BPE merges frequent
within-note combinations of these tokens into single tokens. We train the same
model with and without BPE, using a two-stage recipe adapted from MIDI-LLM, and
compare the generated music.

**Question.** With model, data and training recipe fixed, does applying BPE to AMT
note tokens change the quality of caption-conditioned MIDI generation?

- Listening examples: [demo page](https://junnuo3.github.io/BPE-on-AMT-MIDI/)
- Checkpoints: [huggingface.co/Junnuo/BPE-on-AMT-MIDI](https://huggingface.co/Junnuo/BPE-on-AMT-MIDI)

## Representation

| Field | Values | Tokens |
|---|---|---|
| Onset | 10 ms steps within a 100 s chunk | 10,000 |
| Duration | 10 ms steps, clipped to 10 s | 1,000 |
| Pitch | MIDI pitch | 128 |
| Instrument | 128 GM programs + drums | 129 |
| Velocity | 32 bins | 32 |

Songs longer than 100 s are split into non-overlapping 100 s chunks whose onsets
restart at 0. Each of the 11,289 field tokens is mapped to one Unicode
private-use character, and a note is serialized as one five-character word. BPE
(Hugging Face `tokenizers`) learns merges only inside a word, so no token spans
two notes.

- **No BPE:** 0 merges; 5 tokens per note.
- **BPE:** 16,384 merges, learned on a random sample of 100,000 chunks drawn from
  LMD-full and GigaMIDI. The merge count was chosen so that the BPE vocabulary
  (27,677 tokens; 55,354 with control tokens) roughly matches AMT's vocabulary
  (27,513; 55,028), which MIDI-LLM adds to its base model.

Most merged tokens combine the non-onset fields. Of the 16,384 merged tokens,
7,513 cover duration, pitch, instrument and velocity; 5,425 cover duration and
pitch; and 2,217 cover instrument and velocity. Only 89 contain an onset, so a
typical BPE note is `[onset][duration-pitch-instrument-velocity]`.

Measured on the 896 test songs (`scripts/measure_compression.py`,
`results/compression.csv`):

| | Tokens per note | Music covered by one 2,048-token window (median) | Chunks that fit in one window | MIDI tokens inside the first window |
|---|---|---|---|---|
| No BPE | 5.00 | 21.4 s | 12.5% | 22.7% |
| BPE | 2.87 | 34.2 s | 21.3% | 37.8% |

Both models use the same token context length, so the BPE model sees more music
per training window and per generated sequence. The training loader takes one
window per 100 s chunk: the chunk is tokenized and truncated to its first
2,048 tokens (fewer in stage 2, where the caption shares the window), and the
rest of the chunk is discarded rather than split into further windows. As a
result, most MIDI tokens are never trained on (last column), training covers
mainly the first 20–35 s of each chunk, and the BPE model sees about 1.7× more of
each song than the no-BPE model.

For AMT-style anticipation, each tokenizer is extended with a control copy of
every token and the sentinels `[REST]`, `[SEPARATOR]`, `[AUTOREGRESS]` and
`[ANTICIPATE]`. The result has 22,586 tokens (no BPE) or 55,354 tokens (BPE).
These are appended to SmolLM2's 49,152-token vocabulary, so the embedding tables
have 71,738 or 104,506 rows.

## Relation to AMT and MIDI-LLM

- **AMT** (Thickstun et al., 2024; [jthickstun/anticipation](https://github.com/jthickstun/anticipation)).
  We use AMT's compound note fields as base tokens. Instead of AMT's three-token
  event vocabulary (time, duration, note-instrument, no velocity), each field is
  its own token. We also port AMT's span and instrument anticipation (5 s
  anticipation interval) to this format. AMT's random-subset anticipation is not
  used.
- **MIDI-LLM** (Wu, Kim and Huang, 2025; [slSeanWU/MIDI-LLM](https://github.com/slSeanWU/MIDI-LLM)).
  We follow its recipe:
  - extend a pretrained text language model with MIDI tokens;
  - stage 1: continued pretraining on MusicPile text and GigaMIDI;
  - stage 2: caption-to-MIDI training on MidiCaps captions paired with Lakh MIDI;
  - re-caption augmented excerpts with Qwen2.5-Omni.

  We also reuse its system prompt, its Lakh validation/test split and its 896-song
  test list. MIDI-LLM releases inference code only; the training code here is our
  own implementation.
- **Differences from MIDI-LLM.** SmolLM2-135M instead of Llama-3.2-1B; velocity in
  the note representation; our own learning rates and batch size; training until
  early stopping; and CLAP averaged over the whole clip. Our numbers are therefore
  not comparable with those in the MIDI-LLM paper.

## Data

| Stage | Source | Use |
|---|---|---|
| 1 | MusicPile (text) | 28 parquet shards for training; 1 held out (300 eval windows) |
| 1 | GigaMIDI v2.0.0 (MIDI) | 1,812,336 training files / 300 validation files. All 168,385 MidiCaps-captioned Lakh files are excluded (GigaMIDI contains Lakh), as are files with fewer than 20 notes |
| 2 | MidiCaps captions × LMD-full | 168,385 captioned files, split by md5 (below) |

The stage-2 split by md5:

| Split | Rule | Files |
|---|---|---|
| Train | Everything not below | 147,198 |
| Validation | md5 starts with `e` | 10,693 |
| Test | MIDI-LLM's published list (all `f`) | 896 |
| Unused | Remaining `f` files | — |

In stage 1, each example is text with probability 0.55, otherwise MIDI.

## Training

Both tokenizers use identical settings:

| Setting | Value |
|---|---|
| Model | SmolLM2-135M, 2,048-token windows |
| Batch | 2 GPUs × 4 sequences × 31 accumulation steps = 507,904 tokens per step |
| Optimizer | AdamW, weight decay 0.1, constant learning rate after 200 warmup steps, bf16 |
| Learning rate | 3e-3 (stage 1); 1e-3 (stage 2, starting from the best stage-1 checkpoint) |
| Captions | Truncated to 256 tokens |

The learning rates were chosen by short sweeps (5,000 steps, validation loss) on
the BPE model and reused for no-BPE.

**Augmentation (both stages).**
- **Stage 1:** each GigaMIDI file is drawn three times per epoch. One draw is
  anticipatory with probability 0.9 and autoregressive otherwise; the other two
  are always anticipatory (span or instrument).
- **Stage 2:** each chunk is used once autoregressively with its MidiCaps caption.
  It is also used twice as precomputed anticipatory variants (passes k = 1, 3, 5
  span; k = 7, 9 instrument) whose target notes were rendered and re-captioned by
  Qwen2.5-Omni-7B. The realized token expansion was about 2.4×, against a target
  of 3×.

The loss covers caption text and target MIDI. Control tokens and padding are
masked. Generation always uses `[AUTOREGRESS]`.

**Stopping.** Every 200 steps we compute the autoregressive validation loss on one
fixed window per held-out file. Training stops when this loss has not improved by
more than 0.001 nats for 15 consecutive evaluations (3,000 steps). The checkpoint
with the lowest loss is kept. All four runs stopped this way.

Runs are trained to convergence rather than to matched compute: step counts
differ, and at equal steps the BPE model sees about 1.7× more notes.

| | Stage 1: best step | nats/token | nats/base token | Stage 2: best step | nats/token | nats/base token |
|---|---|---|---|---|---|---|
| No BPE | 30,200 | 0.615 | 0.614 | 35,200 | 0.712 | 0.712 |
| BPE | 26,800 | 1.140 | 0.598 | 38,400 | 1.157 | 0.675 |

Per-token losses are not comparable across tokenizers. "Nats per base token"
divides the total MIDI negative log-likelihood by the number of base tokens those
MIDI tokens represent. Each model is scored on its own first 2,048-token window
per file, so the BPE model is scored on longer excerpts of the same songs. Full
values are in `results/training.csv`.

## Decoding-parameter selection (validation only)

We tested 30 configurations per model, all with repetition penalty 1.0:
- **Step 1:** temperature T ∈ {0.9, 1.0, …, 1.4} × top-p ∈ {0.95, 0.98, 1.0},
  no top-k.
- **Step 2:** two temperatures carried over from step 1 (BPE: 0.9, 1.0; no BPE:
  1.0, 1.1) × top-p ∈ {0.95, 0.98, 1.0} × top-k ∈ {40, 100}.

Each configuration generates one sample for every one of the 10,693 validation
captions (seed 0, batch size 256). Configurations are ranked by the sum of their
per-metric ranks on FAD, CLAP, chroma JS and groove-bar JS. The step-2
temperatures were chosen with an earlier metric set (FAD, chroma JS, 5 s groove
JS and failure rate), before CLAP and bar-level groove were added.

Selected configurations:
- **BPE:** T = 1.0, top-p = 1.0, top-k = 40.
- **No BPE:** T = 1.0, top-p = 1.0, no top-k.

The test set was evaluated only after selection. All 30 configurations per model
are listed in `results/decoding_sweep_valid_{bpe,no_bpe}.csv`.

## Evaluation

The prompt is the system prompt followed by the caption. The model generates up
to 2,048 − (prompt length) tokens, which are decoded to MIDI and rendered with
FluidSynth (pretty_midi's bundled `TimGM6mb.sf2`, 44.1 kHz). References are all
real songs of the same split, rendered in full. The metrics are:

- **FAD:** Fréchet distance between VGGish embeddings of generated and real audio.
- **CLAP:** cosine similarity between the caption and the audio, using the
  LAION-CLAP music checkpoint. The audio embedding is the mean over consecutive
  10 s windows.
- **Chroma JS:** Jensen–Shannon distance between the histograms of 5 s-window
  pitch-class entropy of generated and real pieces.
- **Groove-bar JS:** Jensen–Shannon distance between histograms of per-piece
  groove similarity. Bars come from Beat This! downbeat tracking on the rendered
  audio, with 64 positions per bar.
- **Groove JS (5 s windows):** reported but not used for selection.

## Results (single seed)

One sample per caption with the selected configurations:

| Split | Model | FAD ↓ | CLAP ↑ | Chroma JS ↓ | Groove-bar JS ↓ |
|---|---|---|---|---|---|
| Validation (10,693) | No BPE | 0.875 | 0.310 | 0.160 | 0.156 |
| Validation (10,693) | BPE | 0.752 | 0.294 | 0.161 | 0.135 |
| Test (896) | No BPE | 0.785 | 0.307 | 0.154 | 0.167 |
| Test (896) | BPE | 0.655 | 0.298 | 0.144 | 0.160 |

On both splits, BPE has lower FAD and slightly lower CLAP, and similar
pitch-class and bar-level rhythm statistics. These are single samples from one
seed.

The random draw of every sample depends on the batch layout. We drew one BPE
configuration (T = 1.0, top-p = 1.0, top-k = 100) twice on the test set with
different batch sizes, and its FAD moved from 0.629 to 0.730. That shift is
comparable to the BPE vs no-BPE gap, so the FAD difference is not established.
The test results are also in `results/test_single_seed.csv`.

## Listening examples

20 test captions with one generation from each model (selected decoding
settings), side by side: [demo page](https://junnuo3.github.io/BPE-on-AMT-MIDI/)
(`docs/index.html`), or the MP3 files and captions in
[`docs/examples/`](docs/examples/README.md).

## Installation

    pip install -r requirements.txt     # Python 3.11; also needs the FluidSynth system library

## Released checkpoints

Hugging Face: [Junnuo/BPE-on-AMT-MIDI](https://huggingface.co/Junnuo/BPE-on-AMT-MIDI),
with folders `no_bpe/stage2` and `bpe/stage2`: the final (best-validation-loss)
stage-2 checkpoints used for all reported results. Each folder includes the
`amt_config.json` the generation scripts read. Stage-1 checkpoints are not
released; `run/stage2.sh` expects them from `run/stage1.sh`.

    hf download Junnuo/BPE-on-AMT-MIDI --local-dir checkpoints
    python scripts/generate_smollm2.py --checkpoint checkpoints/bpe/stage2 \
        --caption "A calm piano piece in C major with a slow tempo." --top-p 1.0 --top-k 40

## Reproduction

Run every script from the repository root. The scripts take `no_bpe` or `bpe` as
an argument.

    bash run/data.sh                      # LMD-full (manual download), MidiCaps matching, MusicPile, GigaMIDI
    bash run/tokenizers.sh                # optional: rebuild the released tokenizers
    bash run/caption_cache.sh             # stage-2 re-captioned variants (GPU, several GPU-days)
    bash run/stage1.sh bpe                # 2 GPUs
    bash run/stage2.sh bpe
    bash run/decoding_sweep.sh bpe        # 30 configurations on validation
    bash run/evaluate_test.sh bpe         # selected configuration on test

To evaluate the released checkpoints without training, set `CKPT`:

    CKPT=checkpoints/bpe/stage2 bash run/evaluate_test.sh bpe

To match the reported rendering, leave `MIDI_SOUNDFONT` unset so that pretty_midi's
bundled soundfont is used. Generation is reproducible given the seed, batch size
and GPU type, but training data order and augmentation draws are not seeded
deterministically.

## Limitations

- **Single seed:** one seed and one sample per caption; multi-seed confirmation is
  in progress.
- **One BPE size:** a single merge count (16,384, matched to AMT's vocabulary
  size) was trained; vocabulary size was not varied for language-model training.
- **Hyperparameters:** learning rates were tuned on the BPE model only.
- **Compute:** not matched across tokenizers.
- **Generation length:** fixed in tokens, not seconds; references are full-length
  songs.
- **Unused data:** only the first window of each 100 s chunk is trained on (about
  23% of MIDI tokens without BPE, 38% with BPE), so the two models see different
  amounts of each song. Splitting chunks into consecutive windows would remove
  this difference.
- **Tokenizer data:** BPE merges were learned on data that includes the
  validation and test MIDI files (unsupervised statistics only).
- **Rendering:** audio metrics depend on one General MIDI soundfont. Files that
  pretty_midi's FluidSynth wrapper fails to render fall back to a sine
  synthesizer; how often this happened was not logged.
- **Scope:** no listening study is reported, and all results are for a
  135M-parameter model.

## References

- Thickstun, Hall, Donahue, Liang. *Anticipatory Music Transformer.* TMLR, 2024.
- Wu, Kim, Huang. *MIDI-LLM: Adapting Large Language Models for Text-to-MIDI Music Generation.* arXiv:2511.03942, 2025.
- Melechovsky, Roy, Herremans. *MidiCaps: A Large-scale MIDI Dataset with Text Captions.* ISMIR, 2024.
- Raffel. *Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching* (Lakh MIDI Dataset). PhD thesis, 2016.
- Lee et al. *GigaMIDI Dataset with Features for Expressive Music Performance Detection.* TISMIR, 2025.
- Yuan et al. *ChatMusician: Understanding and Generating Music Intrinsically with LLM* (MusicPile). Findings of ACL, 2024.
- Allal et al. *SmolLM2: When Smol Goes Big.* 2025.
- Kilgour et al. *Fréchet Audio Distance.* Interspeech, 2019.
- Wu et al. *Large-Scale Contrastive Language-Audio Pretraining* (LAION-CLAP). ICASSP, 2023.
- Foscarin, Schlüter, Widmer. *Beat This! Accurate Beat Tracking Without DBN Postprocessing.* ISMIR, 2024.
- Wu, Yang. *The Jazz Transformer on the Front Line* (pitch-class entropy, grooving similarity). ISMIR, 2020.

## License

Code: MIT (see `LICENSE`). The released checkpoints were trained on Lakh MIDI,
MidiCaps, GigaMIDI and MusicPile; their use is subject to those datasets' terms.
