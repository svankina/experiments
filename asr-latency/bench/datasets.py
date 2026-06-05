"""Benchmark audio: fetch a curated LibriSpeech test-clean set and load it.

`fetch` streams samples from the HuggingFace `openslr/librispeech_asr` dataset (no full
download), buckets them by duration into short/medium/long, writes 16 kHz mono WAVs into
./audio, and records reference transcripts in manifest.json (for optional WER).

Run inside the faster-whisper container (it has `datasets` + `soundfile`):
    python -m bench.datasets fetch --n 8
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

AUDIO_DIR = Path(os.environ.get("BENCH_AUDIO_DIR", "/work/audio"))
MANIFEST = AUDIO_DIR / "manifest.json"
SAMPLE_RATE = 16000


@dataclass
class Clip:
    id: str
    path: str
    audio_sec: float
    text: str          # reference transcript ("" if unknown)
    bucket: str        # short | medium | long
    subset: str = "clean"   # librispeech config the clip came from: clean | other


def load_manifest() -> list[Clip]:
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"{MANIFEST} not found. Fetch audio first: python -m bench.datasets fetch")
    raw = json.loads(MANIFEST.read_text())
    return [Clip(**c) for c in raw]


def _bucket(sec: float) -> str:
    if sec < 6:
        return "short"
    if sec < 15:
        return "medium"
    return "long"


def _decode_audio(audio: dict):
    """Decode a non-decoded HF audio cell ({'bytes':..,'path':..}) with soundfile.
    Avoids the `datasets` torchcodec/torch dependency (we don't want torch in the
    faster-whisper image just to fetch data)."""
    import io

    import soundfile as sf

    if audio.get("bytes"):
        arr, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    else:
        arr, sr = sf.read(audio["path"], dtype="float32")
    if getattr(arr, "ndim", 1) > 1:
        arr = arr.mean(axis=1)
    return arr, sr


def _fetch_config(config: str, n: int, sf, Audio, load_dataset) -> list[Clip]:
    """Pull ~n clips from one librispeech config (clean|other), balanced across buckets."""
    ds = load_dataset("openslr/librispeech_asr", config, split="test", streaming=True)
    # decode=False -> raw bytes/path, decoded by soundfile (no torchcodec needed)
    ds = ds.cast_column("audio", Audio(decode=False))

    per_bucket = max(1, n // 3)
    counts = {"short": 0, "medium": 0, "long": 0}
    clips: list[Clip] = []
    for ex in ds:
        if sum(counts.values()) >= n:
            break
        arr, sr = _decode_audio(ex["audio"])
        sec = len(arr) / sr
        b = _bucket(sec)
        if counts[b] >= per_bucket and sum(counts.values()) >= per_bucket * 2:
            if all(counts[x] >= per_bucket for x in counts):
                break
            continue
        if counts[b] >= per_bucket + 2:
            continue
        # prefix id with config so clean/other never collide on disk or in the manifest
        cid = f"{config}-{ex.get('id') or f'clip{len(clips):03d}'}"
        path = AUDIO_DIR / f"{cid}.wav"
        sf.write(str(path), arr, sr, subtype="PCM_16")
        clips.append(Clip(id=cid, path=str(path), audio_sec=round(sec, 3),
                          text=ex.get("text", "").strip(), bucket=b, subset=config))
        counts[b] += 1
        print(f"  + [{config}] {cid:<30} {sec:5.1f}s  [{b}]")
    print(f"  {config}: {counts}")
    return clips


def fetch(n: int = 8, configs: str = "clean") -> None:
    """Pull ~n clips per config (clean,other) spread across buckets; write WAVs + manifest."""
    import soundfile as sf
    from datasets import Audio, load_dataset

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    config_list = [c.strip() for c in configs.split(",") if c.strip()]
    clips: list[Clip] = []
    for config in config_list:
        clips += _fetch_config(config, n, sf, Audio, load_dataset)

    clips.sort(key=lambda c: (c.subset, c.audio_sec))
    MANIFEST.write_text(json.dumps([c.__dict__ for c in clips], indent=2))
    print(f"\nWrote {len(clips)} clips ({', '.join(config_list)}) -> {MANIFEST}")


def main() -> None:
    ap = argparse.ArgumentParser(description="benchmark audio fetch/inspect")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download benchmark clips")
    f.add_argument("--n", type=int, default=8, help="clips per config")
    f.add_argument("--configs", default="clean",
                   help="comma-separated librispeech configs, e.g. 'clean,other'")
    sub.add_parser("list", help="list current manifest")
    args = ap.parse_args()

    if args.cmd == "fetch":
        fetch(args.n, args.configs)
        # `datasets` streaming spawns background threads whose interpreter-teardown can
        # raise a benign "Fatal Python error" after we're done. Hard-exit to avoid it.
        os._exit(0)
    elif args.cmd == "list":
        for c in load_manifest():
            print(f"{c.id:<24} {c.audio_sec:6.2f}s  [{c.bucket}]  {c.text[:60]}")


if __name__ == "__main__":
    main()
