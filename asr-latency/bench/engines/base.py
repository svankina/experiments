"""Shared types for engine adapters."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StreamEvent:
    t_rel_s: float      # wall-clock seconds since stream start when text was emitted
    text: str           # cumulative transcript at this point
    compute_ms: float   # compute time the engine spent producing this step


@dataclass
class StreamTrace:
    events: list[StreamEvent] = field(default_factory=list)
    final_text: str = ""
    audio_sec: float = 0.0
    final_lag_ms: float = 0.0   # gap between last audio fed and final text ready
    chunk_cfg: str = ""
    streaming_native: bool = True
