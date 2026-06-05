"""NVIDIA Parakeet ASR (NeMo) adapter.

Parakeet-TDT-0.6B-v2 is NVIDIA's high-accuracy offline ASR model (TDT decoder, English),
currently at/near the top of the HuggingFace Open ASR leaderboard. It is *not* a cache-aware
streaming model like Nemotron, so the offline path is the primary (and high-confidence)
comparison; streaming is provided only as a chunked re-decode shim (streaming_native=False),
matching how faster-whisper is pseudo-streamed.

Runs in the same container as the Nemotron engine (nemo_toolkit[asr]).
"""
from __future__ import annotations

import time

from .base import StreamEvent, StreamTrace

MODEL = "nvidia/parakeet-tdt-0.6b-v2"


def _hyp_text(out) -> str:
    """NeMo .transcribe returns list of str OR list of Hypothesis depending on version."""
    item = out[0] if isinstance(out, (list, tuple)) else out
    if isinstance(item, (list, tuple)):       # nested (e.g. [[hyp]])
        item = item[0]
    return getattr(item, "text", item) if not isinstance(item, str) else item


class ParakeetEngine:
    name = "parakeet"

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
    def stream_file(self, path: str, window_s: float = 1.0, **_) -> StreamTrace:
        """Pseudo-streaming: re-decode a growing buffer (not native, labelled as such)."""
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
