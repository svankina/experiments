"""NVIDIA Nemotron streaming ASR (NeMo) adapter.

Offline: ASRModel.transcribe([...]) — high confidence, this is the primary comparison.

Streaming: Nemotron is a cache-aware streaming model. We attempt NeMo's native cache-aware
loop (conformer_stream_step over a CacheAwareStreamingAudioBuffer), with the chunk_cfg knob
mapped to the encoder att_context_size ([left, right] frames, the model's latency control —
e.g. "70,1" ≈ 160 ms). NeMo's streaming API is version-sensitive; if the native path raises,
we fall back to a chunked re-decode (streaming_native=False) so a number is still produced.

  >>> VALIDATE THE NATIVE STREAMING PATH ON THE FIRST REAL GPU RUN before trusting its
  >>> numbers. Check the result's `streaming_native` flag — if False, the fallback ran.
"""
from __future__ import annotations

import time

import numpy as np

from .base import StreamEvent, StreamTrace

MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"


def _hyp_text(out) -> str:
    """NeMo .transcribe returns list of str OR list of Hypothesis depending on version."""
    item = out[0] if isinstance(out, (list, tuple)) else out
    if isinstance(item, (list, tuple)):       # nested (e.g. [[hyp]])
        item = item[0]
    return getattr(item, "text", item) if not isinstance(item, str) else item


class NemotronEngine:
    name = "nemotron"

    def __init__(self, model: str = MODEL, device: str = "cuda", **_):
        self.model_name = model
        self.device = device
        self._model = None
        self._torch = None

    def load(self) -> None:
        import torch
        import nemo.collections.asr as nemo_asr

        self._torch = torch
        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
        self._model.eval()
        if torch.cuda.is_available():
            self._model = self._model.to("cuda")

    def transcribe_file(self, path: str) -> str:
        out = self._model.transcribe([path], batch_size=1, verbose=False)
        return _hyp_text(out).strip()

    def synchronize(self) -> None:
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()

    # ------------------------------------------------------------------ streaming
    def stream_file(self, path: str, chunk: str = "70,1", **_) -> StreamTrace:
        try:
            return self._stream_native(path, chunk)
        except Exception as e:  # noqa: BLE001 - any version/API mismatch falls back
            print(f"[nemotron] native cache-aware streaming unavailable ({e!r}); "
                  f"falling back to chunked re-decode")
            return self._stream_fallback(path, chunk)

    def _att_ctx(self, chunk: str) -> list[int]:
        left, right = (int(x) for x in chunk.split(","))
        return [left, right]

    def _stream_native(self, path: str, chunk: str) -> StreamTrace:
        import soundfile as sf
        import torch
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer,
        )

        model = self._model
        att = self._att_ctx(chunk)
        # set_default_att_context_size configures streaming_cfg; do NOT also call the bare
        # setup_streaming_params() (its all-None args wipe the cache sizes).
        model.encoder.set_default_att_context_size(att)
        param_dtype = next(model.parameters()).dtype

        buf = CacheAwareStreamingAudioBuffer(model=model, online_normalization=False)
        buf.append_audio_file(path, stream_id=-1)
        # Materialize chunks so we can pace to real time using true per-step audio duration.
        chunks = list(buf)
        n_steps = max(1, len(chunks))
        audio_sec = sf.info(path).duration
        per_step_s = audio_sec / n_steps

        # Per NeMo's reference loop: drop 0 pre-encoded frames on step 0, else streaming_cfg's.
        drop_default = model.encoder.streaming_cfg.drop_extra_pre_encoded

        cache_last_channel, cache_last_time, cache_last_channel_len = (
            model.encoder.get_initial_cache_state(batch_size=1))
        previous_hypotheses = None
        pred_out_stream = None
        trace = StreamTrace(chunk_cfg=f"att={att} (native)", streaming_native=True,
                           audio_sec=round(audio_sec, 3))
        prev_text = ""
        start = time.perf_counter()
        last_audio_fed = start

        for step, (chunk_audio, chunk_lengths) in enumerate(chunks):
            is_last = step == len(chunks) - 1
            # real-time pacing: don't process step until its audio would have arrived
            ahead = (step + 1) * per_step_s - (time.perf_counter() - start)
            if ahead > 0:
                time.sleep(ahead)
            last_audio_fed = time.perf_counter()
            with torch.inference_mode():
                (pred_out_stream, transcribed_texts, cache_last_channel,
                 cache_last_time, cache_last_channel_len, previous_hypotheses) = (
                    model.conformer_stream_step(
                        processed_signal=chunk_audio.to(param_dtype),
                        processed_signal_length=chunk_lengths,
                        cache_last_channel=cache_last_channel,
                        cache_last_time=cache_last_time,
                        cache_last_channel_len=cache_last_channel_len,
                        keep_all_outputs=is_last,
                        previous_hypotheses=previous_hypotheses,
                        previous_pred_out=pred_out_stream,
                        drop_extra_pre_encoded=(0 if step == 0 else drop_default),
                        return_transcription=True,
                    ))
            self.synchronize()
            compute_ms = (time.perf_counter() - last_audio_fed) * 1000.0
            text = _hyp_text(transcribed_texts).strip() if transcribed_texts else prev_text
            if text and text != prev_text:
                trace.events.append(StreamEvent(
                    t_rel_s=time.perf_counter() - start, text=text, compute_ms=compute_ms))
                prev_text = text

        trace.final_text = prev_text
        trace.final_lag_ms = (time.perf_counter() - last_audio_fed) * 1000.0
        return trace

    def _stream_fallback(self, path: str, chunk: str, window_s: float = 1.0) -> StreamTrace:
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        trace = StreamTrace(audio_sec=len(audio) / sr,
                            chunk_cfg=f"{window_s:.2f}s window (fallback)",
                            streaming_native=False)
        step = int(window_s * sr)
        prev_text = ""
        start = time.perf_counter()
        last_audio_fed = start
        pos = step
        while pos < len(audio) + step:
            end = min(pos, len(audio))
            ahead = end / sr - (time.perf_counter() - start)
            if ahead > 0:
                time.sleep(ahead)
            last_audio_fed = time.perf_counter()
            out = self._model.transcribe([audio[:end]], batch_size=1, verbose=False)
            self.synchronize()
            compute_ms = (time.perf_counter() - last_audio_fed) * 1000.0
            text = _hyp_text(out).strip()
            if text and text != prev_text:
                trace.events.append(StreamEvent(
                    t_rel_s=time.perf_counter() - start, text=text, compute_ms=compute_ms))
                prev_text = text
            pos += step
        trace.final_text = prev_text
        trace.final_lag_ms = (time.perf_counter() - last_audio_fed) * 1000.0
        return trace
