"""Result schema, statistics, and JSONL recording.

Timing lives here (not in the engine adapters) so every engine is measured the same way.
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

RESULTS_DIR = Path(os.environ.get("BENCH_RESULTS_DIR", "/work/results"))


def _gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    # Fallback for containers without torch (faster-whisper uses CTranslate2).
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        name = out.stdout.strip().splitlines()
        if name:
            return name[0].strip()
    except Exception:
        pass
    return "unknown"


def _now_iso() -> str:
    # time.gmtime is allowed; avoids importing datetime just for a stamp.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class OfflineResult:
    engine: str
    model: str
    file: str
    audio_sec: float
    runs_ms: list[float]
    median_ms: float
    p90_ms: float
    rtf: float                      # median_ms/1000 / audio_sec  (lower = faster than realtime)
    transcript: str
    device: str = "cuda"
    gpu: str = field(default_factory=_gpu_name)
    host: str = field(default_factory=platform.node)
    ts: str = field(default_factory=_now_iso)
    kind: str = "offline"


@dataclass
class StreamingResult:
    engine: str
    model: str
    file: str
    audio_sec: float
    chunk_cfg: str                  # engine-specific knob, e.g. nemotron "70,1" or fw "1.0s window"
    ttft_ms: float                  # time from stream start to first emitted token
    emit_latencies_ms: list[float]  # per-chunk: compute time relative to chunk's audio availability
    median_emit_ms: float
    p90_emit_ms: float
    final_lag_ms: float             # gap between last audio fed and final transcript ready
    transcript: str
    streaming_native: bool = True   # False => pseudo-streamed (faster-whisper shim)
    device: str = "cuda"
    gpu: str = field(default_factory=_gpu_name)
    host: str = field(default_factory=platform.node)
    ts: str = field(default_factory=_now_iso)
    kind: str = "streaming"


def time_offline(fn: Callable[[], str], synchronize: Callable[[], None],
                 warmup: int = 1, runs: int = 5) -> tuple[list[float], str]:
    """Run `fn` (which must do the full transcription, consuming any lazy generator) and
    time it `runs` times after `warmup` untimed runs. `synchronize` flushes async GPU work
    before the clock stops. Returns (per-run ms, last transcript)."""
    transcript = ""
    for _ in range(max(0, warmup)):
        transcript = fn()
        synchronize()
    times_ms: list[float] = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        transcript = fn()
        synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return times_ms, transcript


def p90(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(0.9 * (len(s) - 1))))
    return s[idx]


def write_result(result, path: Optional[Path] = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = RESULTS_DIR / f"{result.kind}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(asdict(result)) + "\n")
    return path


def summarize_offline(engine: str, model: str, file: str, audio_sec: float,
                      runs_ms: list[float], transcript: str) -> OfflineResult:
    med = statistics.median(runs_ms)
    return OfflineResult(
        engine=engine, model=model, file=file, audio_sec=audio_sec,
        runs_ms=[round(x, 2) for x in runs_ms],
        median_ms=round(med, 2), p90_ms=round(p90(runs_ms), 2),
        rtf=round((med / 1000.0) / audio_sec, 4) if audio_sec else 0.0,
        transcript=transcript,
    )
