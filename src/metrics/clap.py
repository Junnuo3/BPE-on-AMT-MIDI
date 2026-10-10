"""
CLAP score: cosine similarity between a prompt's text embedding and its
synthesized audio's embedding, via LAION-AI's CLAP (Wu et al. 2023) music
checkpoint -- same metric/checkpoint choice as the MIDI-LLM paper (Sec 4.2).

Needs `pip install laion_clap` (see requirements.txt). The checkpoint is
downloaded once via huggingface_hub and cached by the usual HF cache dir.

Audio windowing: laion_clap's get_audio_embedding_from_filelist (non-fusion
checkpoint) scores ONE randomly placed 10 s crop of a longer clip
('rand_trunc' -- unseeded, so not reproducible, and it ignores most of a
30-100 s generation). window="all" (default) instead embeds every consecutive
10 s window (a final partial window >= 1 s is repeat-padded, as laion_clap pads
short clips) and averages the L2-normalized window embeddings: deterministic and
covering the whole clip. window="rand_trunc" keeps the library behaviour,
seeded. The mode is recorded in summary.json.
"""

from __future__ import annotations

import json
import os

from .base import EvalContext, Metric, register

DEFAULT_CLAP_REPO = "lukewys/laion_clap"
DEFAULT_CLAP_FILE = "music_audioset_epoch_15_esc_90.14.pt"


def _load_clap_ckpt(ckpt_path: str | None) -> str:
    if ckpt_path:
        return ckpt_path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=DEFAULT_CLAP_REPO, filename=DEFAULT_CLAP_FILE)


def _load_done_ids(per_sample_path: str) -> set[str]:
    done = set()
    if os.path.exists(per_sample_path):
        with open(per_sample_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["sample_id"])
    return done


@register("clap")
class ClapMetric(Metric):
    """Per-sample cosine similarity, written incrementally (resumable) to
    per_sample.jsonl; the aggregate mean/std is recomputed from that file
    every run, so a partially-completed run picks up where it left off."""

    def __init__(self, ckpt_path: str | None = None, device: str | None = None,
                 window: str = "all", batch_size: int = 32, seed: int = 0):
        if window not in ("all", "rand_trunc"):
            raise ValueError(f"window must be 'all' or 'rand_trunc', got {window!r}")
        self.ckpt_path = ckpt_path
        self.device = device
        self.window = window
        self.batch_size = batch_size
        self.seed = seed
        self._model = None

    def _model_(self):
        if self._model is None:
            import laion_clap

            model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base",
                                            device=self.device)
            model.load_ckpt(_load_clap_ckpt(self.ckpt_path))
            model.model.eval()
            self._model = model
        return self._model

    def _audio_windows(self, wav_path: str) -> list:
        """laion_clap-preprocessed input dicts, one per 10 s window."""
        import librosa
        import torch
        from laion_clap.training.data import (
            float32_to_int16, get_audio_features, int16_to_float32,
        )

        model = self._model_()
        max_len = 480000  # 10 s at 48 kHz, the checkpoint's input length
        audio, _ = librosa.load(wav_path, sr=48000)
        audio = int16_to_float32(float32_to_int16(audio))  # same quantization as laion_clap
        if self.window == "all" and len(audio) > max_len:
            chunks = [audio[i:i + max_len] for i in range(0, len(audio), max_len)]
            if len(chunks) > 1 and len(chunks[-1]) < 48000:
                chunks = chunks[:-1]  # drop a trailing sliver under 1 s
        else:
            chunks = [audio]  # rand_trunc crops inside get_audio_features
        out = []
        for chunk in chunks:
            out.append(get_audio_features(
                {}, torch.from_numpy(chunk).float(), max_len,
                data_truncating="rand_trunc", data_filling="repeatpad",
                audio_cfg=model.model_cfg["audio_cfg"], require_grad=False,
            ))
        return out

    def _embed_batch(self, samples) -> list:
        """Per-sample (audio_emb, text_emb) for a batch, windows batched together."""
        import numpy as np
        import torch

        model = self._model_()
        windows, owner = [], []
        for k, sample in enumerate(samples):
            w = self._audio_windows(sample.wav_path)
            windows += w
            owner += [k] * len(w)
        embs = []
        with torch.no_grad():
            for i in range(0, len(windows), self.batch_size):
                embs.append(model.model.get_audio_embedding(windows[i:i + self.batch_size])
                            .detach().cpu().numpy())
            text = model.get_text_embedding([s.prompt for s in samples], use_tensor=False)
        embs = np.concatenate(embs, axis=0)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        owner = np.array(owner)
        return [(embs[owner == k].mean(axis=0), text[k]) for k in range(len(samples))]

    def run(self, ctx: EvalContext) -> dict:
        import random

        import numpy as np

        out_dir = os.path.join(ctx.output_root, self.name)
        os.makedirs(out_dir, exist_ok=True)
        per_sample_path = os.path.join(out_dir, "per_sample.jsonl")

        todo = [
            s for s in ctx.generated
            if s.wav_path is not None and s.prompt
        ]
        n_no_prompt = sum(1 for s in ctx.generated if s.wav_path is not None and not s.prompt)
        if n_no_prompt:
            print(f"  clap: {n_no_prompt} synthesized sample(s) have no prompt.txt, skipping them.")

        done_ids = _load_done_ids(per_sample_path)
        todo = [s for s in todo if s.sample_id not in done_ids]

        if todo:
            random.seed(self.seed)  # only consulted by window="rand_trunc"
            np.random.seed(self.seed)
            with open(per_sample_path, "a", encoding="utf-8") as f:
                for i in range(0, len(todo), self.batch_size):
                    batch = todo[i:i + self.batch_size]
                    for sample, (audio_emb, text_emb) in zip(batch, self._embed_batch(batch)):
                        cos_sim = float(
                            (audio_emb * text_emb).sum()
                            / ((audio_emb ** 2).sum() ** 0.5 * (text_emb ** 2).sum() ** 0.5 + 1e-8)
                        )
                        f.write(json.dumps({
                            "sample_id": sample.sample_id, "cosine_similarity": cos_sim,
                        }) + "\n")
                    f.flush()
                    if (i // self.batch_size) % 20 == 0:
                        print(f"  clap: {i + len(batch)}/{len(todo)}")

        wanted = {s.sample_id for s in ctx.generated if s.wav_path is not None and s.prompt}
        all_scores = []
        with open(per_sample_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if row["sample_id"] in wanted:
                        all_scores.append(row["cosine_similarity"])

        mean_cos = sum(all_scores) / len(all_scores) if all_scores else float("nan")
        summary = {
            "params": {"window": self.window, "window_sec": 10.0, "seed": self.seed,
                       "checkpoint": DEFAULT_CLAP_FILE if not self.ckpt_path else self.ckpt_path},
            "n_samples": len(all_scores),
            "n_skipped_no_prompt": n_no_prompt,
            "clap_cosine_mean": mean_cos,
            "clap_score_mean": mean_cos * 100.0,  # x100, matches the literature's usual scale
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary
