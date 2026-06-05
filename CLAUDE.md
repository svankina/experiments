# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A workbench for self-contained ML/systems **experiments**. Each experiment lives in its own
top-level directory and is fully isolated: its own containers, harness, and results. The first
experiment is `asr-latency/`. New experiments get a sibling directory and their own README.

## Host environment (important constraints)

- **GPU:** single RTX 3090 Ti, 24 GB, Ampere (sm_86). Driver 580 / CUDA 13-capable.
- **Docker:** runs **without sudo** (user is in the `docker` group). In CLAUDE.md the shell aliases
  `docker` to `sudo docker`; bypass that with `command docker ...` in non-interactive shells.
- **GPU-in-Docker prerequisite:** `nvidia-container-toolkit` is **NOT installed** by default and
  Docker has no `nvidia` runtime. Containers cannot see the GPU until a human runs (needs sudo):
  ```
  sudo apt install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
  Verify with: `command docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi`.
  **You cannot run sudo non-interactively here** — ask the user to run the install step.

## asr-latency experiment

Goal: measure **relative** latency of `faster-whisper` (GPU) vs NVIDIA `nemotron-speech-streaming-en-0.6b`
(GPU). We deliberately ignore absolute throughput / batching — both run batch_size=1 on the same
machine, same audio, so the comparison is apples-to-apples.

Two engines, each in its **own** container, both mounting the **same** `bench/` harness package so the
measurement code is identical across engines. Two metrics:

- **Offline RTF** (`bench/offline.py`): transcribe fixed files end-to-end, report median wall-clock ms
  and Real-Time Factor (`processing_time / audio_duration`). Cleanest comparison.
- **Streaming TTFT** (`bench/streaming.py`): feed audio in real-time frames, report time-to-first-token
  and per-chunk emission latency. Nemotron streams natively (chunk-size latency knob 80 ms–1.12 s);
  faster-whisper is **pseudo-streamed** via a chunking shim — labelled as such, not a like-for-like.

faster-whisper is swept across model sizes (`small`, `medium`, `large-v3`); Nemotron is the fixed
0.6 B model. Audio is a curated LibriSpeech test-clean set (varied lengths + reference transcripts for
optional WER), fetched by the harness.

### Layout

```
asr-latency/
  bench/            shared harness (mounted into BOTH containers, identical code)
    metrics.py      result schema (OfflineResult/StreamingResult) + JSONL writer + RTF
    datasets.py     manifest loader; fetch_audio() pulls LibriSpeech samples
    offline.py      offline RTF CLI:   python -m bench.offline   --engine ... --model ...
    streaming.py    streaming TTFT CLI: python -m bench.streaming --engine ... --model ...
    engines/        engine adapters (faster_whisper.py, nemotron.py) behind a common interface
  engines/
    faster-whisper/Dockerfile   CUDA 12.x + cuDNN + faster-whisper (+ datasets for audio fetch)
    nemotron-asr/Dockerfile     CUDA 12.x + nemo_toolkit[asr]
  audio/            fetched wavs + manifest.json (gitignored)
  results/          per-run JSONL (gitignored)
  compose.yaml      both services, GPU-enabled
  analyze.py        aggregate results/*.jsonl into a comparison table
```

### Commands

All commands assume cwd `asr-latency/` and `command docker` (not the sudo alias).

```bash
# 0. one-time: build images (no GPU needed to build)
command docker compose build

# 1. fetch benchmark audio into ./audio (runs inside the faster-whisper image)
command docker compose run --rm faster-whisper python -m bench.datasets fetch --n 8

# 2. offline RTF sweep
command docker compose run --rm faster-whisper python -m bench.offline --engine faster-whisper --model small
command docker compose run --rm faster-whisper python -m bench.offline --engine faster-whisper --model large-v3
command docker compose run --rm nemotron      python -m bench.offline --engine nemotron

# 3. streaming TTFT
command docker compose run --rm nemotron      python -m bench.streaming --engine nemotron --chunk 70,1
command docker compose run --rm faster-whisper python -m bench.streaming --engine faster-whisper --model large-v3

# 4. aggregate
command docker compose run --rm faster-whisper python analyze.py
```

### Engine adapter contract

`bench/engines/*.py` each expose a class with: `load()`, `transcribe_file(path) -> str` (offline),
`stream_file(path, on_emit, **cfg)` (streaming), and `synchronize()` (call `torch.cuda.synchronize()`
for NeMo; no-op for CTranslate2/faster-whisper which blocks synchronously). The harness owns all
timing so the two engines are measured identically. To add an engine: add an adapter + a Dockerfile +
a compose service; the CLIs dispatch by `--engine`.

### Gotchas

- **faster-whisper `transcribe()` is lazy** — it returns a generator; the harness must consume the
  segments *inside* the timed region or the measurement is meaningless.
- **NeMo timing needs `torch.cuda.synchronize()`** around the measured region (async CUDA kernels).
- Nemotron streaming uses NeMo's cache-aware pipeline and is **version-sensitive**; the streaming
  adapter is best-effort and should be validated on the first real GPU run before trusting its numbers.
  The offline path is the high-confidence comparison.
- Model weights cache to `~/.cache/huggingface` on the host (mounted into containers) — first run
  downloads ~2.4 GB (Nemotron) and the Whisper sizes.
```
