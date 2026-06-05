"""Streaming TTFT benchmark CLI.

    python -m bench.streaming --engine nemotron --chunk 70,1
    python -m bench.streaming --engine faster-whisper --model large-v3 --window 1.0

Feeds each clip in real time and records: time-to-first-token, per-chunk emit latency
(median/p90), and final-transcript lag. Nemotron streams natively (chunk = att_context_size);
faster-whisper is pseudo-streamed (window seconds) and flagged streaming_native=False.
"""
from __future__ import annotations

import argparse

from . import datasets, metrics
from .engines import get_engine


def main() -> None:
    ap = argparse.ArgumentParser(description="streaming ASR TTFT benchmark")
    ap.add_argument("--engine", required=True, help="faster-whisper | nemotron")
    ap.add_argument("--model", default=None)
    ap.add_argument("--chunk", default="70,1", help="nemotron att_context_size 'left,right'")
    ap.add_argument("--window", type=float, default=1.0, help="faster-whisper pseudo window (s)")
    args = ap.parse_args()

    clips = datasets.load_manifest()
    kw = {}
    if args.model:
        kw["model"] = args.model
    engine = get_engine(args.engine, **kw)
    model_id = args.model or getattr(engine, "model_name", getattr(engine, "model_size", "default"))
    print(f"Loading {args.engine} ({model_id}) ...")
    engine.load()

    cfg = {"chunk": args.chunk} if args.engine in ("nemotron", "nemo") else {"window_s": args.window}

    print(f"{'clip':<24}{'audio_s':>9}{'ttft_ms':>10}{'med_emit':>10}{'native':>8}")
    for clip in clips:
        trace = engine.stream_file(clip.path, **cfg)
        emit_ms = [e.compute_ms for e in trace.events]
        ttft_ms = (trace.events[0].t_rel_s * 1000.0) if trace.events else float("nan")
        med_emit = (sorted(emit_ms)[len(emit_ms) // 2]) if emit_ms else 0.0
        res = metrics.StreamingResult(
            engine=args.engine, model=str(model_id), file=clip.id,
            audio_sec=trace.audio_sec or clip.audio_sec, chunk_cfg=trace.chunk_cfg,
            ttft_ms=round(ttft_ms, 2),
            emit_latencies_ms=[round(x, 2) for x in emit_ms],
            median_emit_ms=round(med_emit, 2), p90_emit_ms=round(metrics.p90(emit_ms), 2),
            final_lag_ms=round(trace.final_lag_ms, 2), transcript=trace.final_text,
            streaming_native=trace.streaming_native)
        metrics.write_result(res)
        print(f"{clip.id:<24}{res.audio_sec:>9.2f}{res.ttft_ms:>10.1f}"
              f"{res.median_emit_ms:>10.1f}{str(res.streaming_native):>8}")

    print(f"\nWrote results -> {metrics.RESULTS_DIR}/streaming.jsonl")


if __name__ == "__main__":
    main()
