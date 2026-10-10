"""
FAD (Frechet Audio Distance, Kilgour et al. 2019): Frechet distance between the
VGGish (Hershey et al. 2017) embedding distributions of the generated set and
the ground-truth test set -- same metric/feature-extractor choice as the
MIDI-LLM paper (Sec 4.2).

VGGish itself comes from torch.hub ('harritaylor/torchvggish'; downloaded once
and cached under ~/.cache/torch/hub). Needs `pip install scipy resampy`
(resampy is what torchvggish's preprocessing uses to get to its required 16kHz
mono input; scipy.linalg.sqrtm computes the Frechet distance's matrix sqrt term).

Per-file embeddings are cached as .npy (frames x 128), so an interrupted run
only has to re-embed whatever wasn't finished -- the Frechet distance itself is
cheap to recompute from cached embeddings every run.
"""

from __future__ import annotations

import os

import numpy as np

from .base import EvalContext, Metric, register


def _load_vggish(device: str | None):
    import torch

    # device must go through the hub kwargs (-> VGGish.__init__), not a later
    # .to(device): VGGish.forward() moves its input with `x.to(self.device)`,
    # a plain attribute set once in __init__ that a bare .to() call wouldn't update.
    kwargs = {"postprocess": False}
    if device:
        kwargs["device"] = device
    # trust_repo=True avoids an interactive prompt in non-interactive jobs.
    model = torch.hub.load("harritaylor/torchvggish", "vggish", trust_repo=True, **kwargs)
    model.eval()
    return model


def _embed_wav(model, wav_path: str) -> np.ndarray:
    import torch

    with torch.no_grad():
        emb = model.forward(wav_path)
    return emb.detach().cpu().numpy().astype(np.float64)


def _embed_items_cached(
    model, items: list[tuple[str, str]], cache_dir: str,
) -> list[tuple[str, np.ndarray]]:
    """items: list of (id, wav_path). Returns (id, frames) for every item with a
    non-empty embedding, caching each as <cache_dir>/<id>.npy."""
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for item_id, wav_path in items:
        cache_path = os.path.join(cache_dir, item_id.replace(os.sep, "__") + ".npy")
        if os.path.exists(cache_path):
            frames = np.load(cache_path)
        else:
            frames = _embed_wav(model, wav_path)
            if frames.size == 0:
                continue
            np.save(cache_path, frames)
        if frames.size:
            out.append((item_id, frames))
    return out


def _stack(id_frames: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """The FAD statistics are computed over pooled frames, not one vector per clip."""
    if not id_frames:
        return np.zeros((0, 128))
    return np.concatenate([f for _, f in id_frames], axis=0)


def _embed_set_cached(
    model, items: list[tuple[str, str]], cache_dir: str,
) -> np.ndarray:
    """items: list of (id, wav_path). Returns all per-frame embeddings for the
    set stacked together."""
    return _stack(_embed_items_cached(model, items, cache_dir))


def save_packed_embeddings(path: str, id_frames: list[tuple[str, np.ndarray]]) -> None:
    """One .npz for a whole set (ids, concatenated frames, offsets) instead of one
    .npy per clip -- keeps a 10K-clip set to a single inode. float32 is lossless:
    VGGish's own output is float32 (_embed_wav only upcasts it)."""
    lengths = np.array([f.shape[0] for _, f in id_frames], dtype=np.int64)
    tmp_path = f"{path}.tmp.{os.getpid()}.npz"
    np.savez(
        tmp_path,
        ids=np.array([i for i, _ in id_frames]),
        frames=_stack(id_frames).astype(np.float32),
        offsets=np.concatenate([[0], np.cumsum(lengths)]),
    )
    os.replace(tmp_path, path)


def load_packed_embeddings(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        ids, frames, offsets = z["ids"], z["frames"].astype(np.float64), z["offsets"]
    return {str(i): frames[offsets[k]:offsets[k + 1]] for k, i in enumerate(ids)}


def _frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    from scipy import linalg

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


@register("fad")
class FadMetric(Metric):
    def __init__(self, device: str | None = None,
                 ground_truth_embed_cache_dir: str | None = None,
                 ground_truth_embed_npz: str | None = None,
                 generated_embed_dir: str | None = None):
        """ground_truth_embed_npz: packed ground-truth embeddings (from
        scripts/build_ground_truth_cache.py); when set, ground-truth wavs are
        not needed at all. generated_embed_dir: put the per-clip generated cache
        somewhere else (e.g. node-local $TMPDIR) and keep only a packed
        <out_dir>/generated_embeddings.npz."""
        self.device = device
        self.ground_truth_embed_cache_dir = ground_truth_embed_cache_dir
        self.ground_truth_embed_npz = ground_truth_embed_npz
        self.generated_embed_dir = generated_embed_dir
        self._model = None

    def _model_(self):
        if self._model is None:
            self._model = _load_vggish(self.device)
        return self._model

    def run(self, ctx: EvalContext) -> dict:
        out_dir = os.path.join(ctx.output_root, self.name)
        gen_embed_dir = self.generated_embed_dir or os.path.join(out_dir, "embeddings")
        gt_embed_dir  = self.ground_truth_embed_cache_dir or os.path.join(out_dir, "ground_truth_embeddings")

        gen_items = [(s.sample_id, s.wav_path) for s in ctx.generated if s.wav_path]

        model = self._model_()
        packed_gen = os.path.join(out_dir, "generated_embeddings.npz")
        cached = load_packed_embeddings(packed_gen) if (
            self.generated_embed_dir and os.path.exists(packed_gen)) else {}
        if gen_items and all(i in cached for i, _ in gen_items):
            # a previous (e.g. timed-out) run already embedded every clip
            print(f"  fad: reusing {len(gen_items):,} packed generated embeddings from {packed_gen}")
            gen_id_frames = [(i, cached[i]) for i, _ in gen_items]
        else:
            gen_id_frames = _embed_items_cached(model, gen_items, gen_embed_dir)
        gen_frames = _stack(gen_id_frames)
        if self.generated_embed_dir:
            os.makedirs(out_dir, exist_ok=True)
            save_packed_embeddings(os.path.join(out_dir, "generated_embeddings.npz"), gen_id_frames)

        if self.ground_truth_embed_npz:
            packed = load_packed_embeddings(self.ground_truth_embed_npz)
            gt_id_frames = [(e.md5, packed[e.md5]) for e in ctx.ground_truth if e.md5 in packed]
            n_missing = len(ctx.ground_truth) - len(gt_id_frames)
            if n_missing:
                print(f"  fad: {n_missing:,} ground-truth entries have no packed embedding "
                      f"(failed synthesis / no notes) and are left out.")
            n_gt_files = len(gt_id_frames)
            gt_frames = _stack(gt_id_frames)
        else:
            gt_items  = [(e.md5, e.wav_path) for e in ctx.ground_truth if e.wav_path]
            n_gt_files = len(gt_items)
            gt_frames  = _embed_set_cached(model, gt_items, gt_embed_dir)

        summary = {
            "n_generated_files": len(gen_items),
            "n_generated_frames": int(gen_frames.shape[0]),
            "n_ground_truth_files": n_gt_files,
            "n_ground_truth_frames": int(gt_frames.shape[0]),
        }
        if gen_frames.shape[0] < 2 or gt_frames.shape[0] < 2:
            print("  fad: not enough embedded frames to estimate a covariance yet; "
                  "skipping distance computation.")
            summary["fad"] = None
        else:
            mu_g, mu_r = gen_frames.mean(axis=0), gt_frames.mean(axis=0)
            sigma_g = np.cov(gen_frames, rowvar=False)
            sigma_r = np.cov(gt_frames, rowvar=False)
            summary["fad"] = _frechet_distance(mu_g, sigma_g, mu_r, sigma_r)

        os.makedirs(out_dir, exist_ok=True)
        import json
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary
