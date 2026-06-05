"""Offline RTF benchmark CLI.

    python -m bench.offline --engine faster-whisper --model large-v3
    python -m bench.offline --engine nemotron

For each manifest clip: warm up, then time N full transcriptions; record median ms + RTF.
"""
from __future__ import annotations

import argparse

from . import datasets, metrics
from .engines import get_engine


def main() -> None:
    ap = argparse.ArgumentParser(description="offline ASR RTF benchmark")
    ap.add_argument("--engine", required=True, help="faster-whisper | nemotron")
    ap.add_argument("--model", default=None, help="engine model id (e.g. large-v3)")
    ap.add_argument("--runs", type=int, default=5, help="timed runs per clip")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--compute-type", default="float16", help="faster-whisper compute type")
    args = ap.parse_args()

    clips = datasets.load_manifest()
    kw = {}
    if args.model:
        kw["model"] = args.model
    if args.engine in ("faster-whisper", "fw"):
        kw["compute_type"] = args.compute_type
    engine = get_engine(args.engine, **kw)

    model_id = args.model or getattr(engine, "model_name", getattr(engine, "model_size", "default"))
    print(f"Loading {args.engine} ({model_id}) ...")
    engine.load()

    print(f"{'clip':<24}{'audio_s':>9}{'median_ms':>11}{'rtf':>8}")
    for clip in clips:
        runs_ms, transcript = metrics.time_offline(
            lambda: engine.transcribe_file(clip.path),
            engine.synchronize, warmup=args.warmup, runs=args.runs)
        res = metrics.summarize_offline(
            engine=args.engine, model=str(model_id), file=clip.id,
            audio_sec=clip.audio_sec, runs_ms=runs_ms, transcript=transcript)
        metrics.write_result(res)
        print(f"{clip.id:<24}{clip.audio_sec:>9.2f}{res.median_ms:>11.1f}{res.rtf:>8.3f}")

    print(f"\nWrote results -> {metrics.RESULTS_DIR}/offline.jsonl")


if __name__ == "__main__":
    main()
