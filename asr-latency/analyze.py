"""Aggregate results/*.jsonl into comparison tables.

    python analyze.py            # text tables to stdout
    python analyze.py --md       # markdown (for pasting into the README)

No third-party deps so it runs in either container (or on the host).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("BENCH_RESULTS_DIR", "/work/results"))


def _load(name: str) -> list[dict]:
    path = RESULTS_DIR / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fmt_table(headers: list[str], rows: list[list], md: bool) -> str:
    rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
              for i in range(len(headers))]
    sep = " | " if md else "  "
    def line(cells):
        return (("| " if md else "") + sep.join(c.ljust(widths[i]) for i, c in enumerate(cells))
                + (" |" if md else ""))
    out = [line(headers)]
    if md:
        out.append("| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |")
    out += [line(r) for r in rows]
    return "\n".join(out)


def offline_table(md: bool) -> str:
    data = _load("offline.jsonl")
    if not data:
        return "(no offline results)"
    # group by (engine, model): aggregate RTF + median ms across clips
    groups = defaultdict(list)
    for d in data:
        groups[(d["engine"], d["model"])].append(d)
    rows = []
    for (engine, model), items in sorted(groups.items()):
        rtfs = [d["rtf"] for d in items]
        meds = [d["median_ms"] for d in items]
        secs = sum(d["audio_sec"] for d in items)
        rows.append([engine, model, len(items),
                     f"{statistics.mean(rtfs):.3f}",
                     f"{min(rtfs):.3f}-{max(rtfs):.3f}",
                     f"{statistics.median(meds):.0f}",
                     f"{secs:.0f}"])
    return _fmt_table(
        ["engine", "model", "clips", "mean_RTF", "RTF_range", "median_ms", "audio_s"], rows, md)


def streaming_table(md: bool) -> str:
    data = _load("streaming.jsonl")
    if not data:
        return "(no streaming results)"
    groups = defaultdict(list)
    for d in data:
        groups[(d["engine"], d["model"], d["chunk_cfg"], d["streaming_native"])].append(d)
    rows = []
    for (engine, model, cfg, native), items in sorted(groups.items()):
        ttfts = [d["ttft_ms"] for d in items if d["ttft_ms"] == d["ttft_ms"]]  # drop nan
        emits = [d["median_emit_ms"] for d in items]
        lags = [d["final_lag_ms"] for d in items]
        rows.append([engine, model, cfg, "yes" if native else "PSEUDO", len(items),
                     f"{statistics.mean(ttfts):.0f}" if ttfts else "-",
                     f"{statistics.mean(emits):.0f}" if emits else "-",
                     f"{statistics.mean(lags):.0f}" if lags else "-"])
    return _fmt_table(
        ["engine", "model", "chunk_cfg", "native", "clips",
         "TTFT_ms", "emit_ms", "final_lag_ms"], rows, md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="markdown output")
    args = ap.parse_args()
    print("\n=== Offline RTF (lower RTF = faster than realtime) ===")
    print(offline_table(args.md))
    print("\n=== Streaming (TTFT = time-to-first-token; PSEUDO = not native streaming) ===")
    print(streaming_table(args.md))


if __name__ == "__main__":
    main()
