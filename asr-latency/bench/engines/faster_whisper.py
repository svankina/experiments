"""faster-whisper (CTranslate2) adapter.

Offline: WhisperModel.transcribe returns a *generator*; we consume it inside the timed
region. Streaming: faster-whisper is not a streaming model, so we PSEUDO-stream by decoding
a growing audio buffer at fixed wall-clock intervals (a sliding re-decode). The trace is
labelled streaming_native=False — this is an approximation, not a like-for-like with
Nemotron's cache-aware streaming.
"""
from __future__ import annotations

import time

import numpy as np

from .base import StreamEvent, StreamTrace


class FasterWhisperEngine:
    name = "faster-whisper"

    def __init__(self, model: str = "large-v3", device: str = "cuda",
                 compute_type: str = "float16", beam_size: int = 5, **_):
        self.model_size = model
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self._model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel
        self._model = WhisperModel(self.model_size, device=self.device,
                                   compute_type=self.compute_type)

    def transcribe_file(self, path: str) -> str:
        segments, _info = self._model.transcribe(
            path, language="en", beam_size=self.beam_size)
        # Consume the generator HERE so the work happens inside the caller's timer.
        return " ".join(s.text.strip() for s in segments).strip()

    def _transcribe_array(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            audio, language="en", beam_size=self.beam_size)
        return " ".join(s.text.strip() for s in segments).strip()

    def stream_file(self, path: str, window_s: float = 1.0, **_) -> StreamTrace:
        """Pseudo-streaming: every `window_s` of audio, re-decode the buffer so far and
        emit. compute_ms = decode time for that step (the per-chunk processing cost)."""
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        trace = StreamTrace(audio_sec=len(audio) / sr,
                            chunk_cfg=f"{window_s:.2f}s window (pseudo)",
                            streaming_native=False)
        step = int(window_s * sr)
        prev_text = ""
        start = time.perf_counter()
        last_audio_fed = 0.0
        pos = step
        while pos < len(audio) + step:
            end = min(pos, len(audio))
            # pace to "real time": don't decode ahead of when the audio would have arrived
            target = end / sr
            ahead = target - (time.perf_counter() - start)
            if ahead > 0:
                time.sleep(ahead)
            last_audio_fed = time.perf_counter()
            text = self._transcribe_array(audio[:end])
            self.synchronize()
            compute_ms = (time.perf_counter() - last_audio_fed) * 1000.0
            if text and text != prev_text:
                trace.events.append(StreamEvent(
                    t_rel_s=time.perf_counter() - start, text=text, compute_ms=compute_ms))
                prev_text = text
            pos += step
        trace.final_text = prev_text
        trace.final_lag_ms = (time.perf_counter() - last_audio_fed) * 1000.0
        return trace

    def synchronize(self) -> None:
        # CTranslate2 transcribe is synchronous; nothing async to flush.
        pass
