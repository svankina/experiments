# ASR latency: faster-whisper vs Nemotron 3.5-ASR

Relative latency of `faster-whisper` (GPU) vs NVIDIA `nemotron-speech-streaming-en-0.6b`
(GPU) on the same machine (RTX 3090 Ti), same audio, batch size 1. We ignore absolute
throughput — the point is the head-to-head.

## Design

Each engine runs in its **own** container but mounts the **same** `bench/` harness, so the
timing code is byte-identical across engines. Two metrics:

| metric | script | what it measures |
|---|---|---|
| **Offline RTF** | `bench/offline.py` | full-file transcribe, median ms + `RTF = proc_time/audio_len`. Primary, apples-to-apples. |
| **Streaming TTFT** | `bench/streaming.py` | real-time feed → time-to-first-token, per-chunk emit latency, final lag. |

- faster-whisper sizes swept: `small`, `medium`, `large-v3`.
- Nemotron streaming exposes a **latency knob** via encoder `att_context_size` (`70,0`≈80 ms …
  `70,13`≈1.12 s) — we sweep it.
- faster-whisper has **no native streaming**; its streaming numbers are a pseudo-streamed
  re-decode shim, flagged `native=PSEUDO` in the output. Treat the offline table as the real
  comparison and streaming as indicative.

## Prerequisites

1. **nvidia-container-toolkit** (one-time, needs sudo — see `../CLAUDE.md`):
   ```bash
   sudo apt install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   command docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi   # verify
   ```
2. Build images (no GPU needed): `command docker compose build`
   - faster-whisper: quick. nemotron: **long** (pulls torch + full NeMo) and multi-GB.

## Run

```bash
./run_all.sh                 # fetch audio + full sweep + analysis
# or step by step:
command docker compose run --rm faster-whisper python3 -m bench.datasets fetch --n 8
command docker compose run --rm faster-whisper python3 -m bench.offline --engine faster-whisper --model large-v3
command docker compose run --rm nemotron      python3 -m bench.offline --engine nemotron
command docker compose run --rm nemotron      python3 -m bench.streaming --engine nemotron --chunk 70,1
command docker compose run --rm faster-whisper python3 analyze.py --md
```

Results append to `results/offline.jsonl` and `results/streaming.jsonl`. `analyze.py` groups
them into comparison tables (`--md` for markdown).

## Caveats / validation

- **Native Nemotron streaming is version-sensitive.** `bench/engines/nemotron.py` attempts
  NeMo's cache-aware `conformer_stream_step` loop and falls back to a chunked re-decode if the
  API differs. **Check `native` in the streaming table** — `PSEUDO`/`no` means the fallback
  ran and the streaming numbers aren't the model's true streaming latency. The offline path is
  unaffected and is the trustworthy comparison.
- Warm runs only (1 warmup + 5 timed); first-ever run also pays a one-time model download.
- Optional WER: reference transcripts are in `audio/manifest.json` (not yet wired into a WER
  metric — add if quality, not just latency, matters).
