"""Engine adapters behind a common interface.

Each adapter exposes:
    load()                          -> initialise model on GPU
    transcribe_file(path) -> str    -> full offline transcription (consume lazy generators!)
    stream_file(path, **cfg) -> StreamTrace
                                    -> feed audio in real time, recording a StreamEvent each
                                       time new text is produced (see engines/base.py)
    synchronize()                   -> flush async GPU work (torch.cuda.synchronize / no-op)

`get_engine(name, **kw)` is the factory used by the CLIs.
"""
from __future__ import annotations


def get_engine(name: str, **kw):
    if name in ("faster-whisper", "fw", "whisper"):
        from .faster_whisper import FasterWhisperEngine
        return FasterWhisperEngine(**kw)
    if name in ("nemotron", "nemo"):
        from .nemotron import NemotronEngine
        return NemotronEngine(**kw)
    if name in ("parakeet", "parakeet-tdt"):
        from .parakeet import ParakeetEngine
        return ParakeetEngine(**kw)
    raise ValueError(f"unknown engine: {name!r}")
