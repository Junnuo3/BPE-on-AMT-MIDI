# Thin wrapper around a HF audio-captioning model, used by
# scripts/build_caption_cache.py to caption FluidSynth-rendered target
# audio. Default model matches MIDI-LLM (Qwen2.5-Omni), but the model id is
# configurable so a newer/lighter captioning model can be swapped in without
# touching the caching script.

from __future__ import annotations

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"

DEFAULT_PROMPT = (
    "Listen to this piece of music and describe it in one or two sentences, "
    "covering genre, mood, instrumentation, and tempo if apparent."
)

_SYSTEM_PROMPT = (
    "You are a music expert. Answer only with the requested description, "
    "no preamble."
)


class CaptionModel:
    """Loads once, captions many WAV files. Not thread-safe; one instance per process."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        dtype: str = "auto",
    ) -> None:
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self.model_id = model_id
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype if dtype != "auto" else "auto",
            device_map="auto" if device is None else None,
        )
        if device is not None:
            self.model = self.model.to(device)
        self.model.eval()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        self._torch = torch

    def caption(self, wav_path: str, prompt: str = DEFAULT_PROMPT, max_new_tokens: int = 100) -> str:
        """Caption a single WAV file. Returns a plain-text description."""
        from qwen_omni_utils import process_mm_info

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": wav_path},
                {"type": "text", "text": prompt},
            ]},
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True,
        )
        inputs = inputs.to(self.model.device)

        with self._torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_audio_in_video=False,
                return_audio=False,
            )
        # generate() with return_audio=False returns just the text-token ids.
        text_ids = output[0] if isinstance(output, tuple) else output
        prompt_len = inputs["input_ids"].shape[1]
        decoded = self.processor.batch_decode(
            text_ids[:, prompt_len:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()
