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
learns merges only inside a word, so no token spans
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

## Relation to MIDI-LLM

- **MIDI-LLM** (Wu, Kim and Huang, 2025; [slSeanWU/MIDI-LLM](https://github.com/slSeanWU/MIDI-LLM)).
We follow its recipe:
  - extend a pretrained text language model with MIDI tokens;
  - stage 1: continued pretraining on MusicPile text and GigaMIDI;
  - stage 2: caption-to-MIDI training on MidiCaps captions paired with Lakh MIDI;
  - re-caption augmented excerpts with Qwen2.5-Omni.

## Data

| Stage | Source                       |
| ----- | ---------------------------- |
| 1     | MusicPile (text)             |
| 1     | GigaMIDI (MIDI)              |
| 2     | MidiCaps captions + LMD-full |

The stage-2 split by md5:

| Split | Rule | Files |
|---|---|---|
| Train | Everything not below | 147,198 |
| Validation | md5 starts with `e` | 10,693 |
| Test | MIDI-LLM's published list (all `f`) | 896 |

## Training

Both tokenizers use identical settings:

| Setting       | Value                                                                        |
| ------------- | ---------------------------------------------------------------------------- |
| Model         | SmolLM2-135M, 2,048-token windows                                            |
| Batch         | 2 GPUs × 4 sequences × 31 accumulation steps = 507,904 tokens per step       |
| Optimizer     | AdamW, weight decay 0.1, constant learning rate after 200 warmup steps, bf16 |
| Learning rate | 3e-3 (stage 1); 1e-3 (stage 2)                                               |
| Captions      | Truncated to 256 tokens                                                      |

## Evaluation

- **FAD:** Fréchet distance between VGGish embeddings of generated and real audio.
- **CLAP:** cosine similarity between the caption and the audio, using the
LAION-CLAP music checkpoint. 
- **Chroma JS:** Jensen–Shannon distance between the histograms of 5 s-window
pitch-class entropy of generated and real pieces.
- **Groove-bar JS:** Jensen–Shannon distance between histograms of per-piece
groove similarity. Bars come from Beat This! downbeat tracking on the rendered
audio, with 64 positions per bar.

## Results

On the test split:

| Model | FAD ↓ | CLAP ↑ | Chroma JS ↓ | Groove-bar JS ↓ |
|---|---|---|---|---|
| No BPE | 0.785 | **0.307** | 0.154 | 0.167 |
| BPE | **0.655** | 0.298 | **0.144** | **0.160** |

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
