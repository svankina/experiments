# ASR latency: Nemotron 3.5-ASR vs Parakeet (vs faster-whisper)

Relative latency **and accuracy** of three GPU ASR stacks on the same machine (RTX 3090 Ti),
same audio, batch size 1:

- NVIDIA `nemotron-speech-streaming-en-0.6b` (600M, native cache-aware streaming)
- NVIDIA `parakeet-tdt-0.6b-v2` (600M, offline FastConformer-TDT)
- `faster-whisper` swept across `small` / `medium` / `large-v3`

We ignore absolute throughput — the point is the head-to-head. See `SUMMARY.md` for the
one-table digest and `RESULTS.md` for the full writeup.

## Design

Each engine runs in its **own** container but mounts the **same** `bench/` harness, so the
timing code is byte-identical across engines. Metrics:

| metric | script | what it measures |
|---|---|---|
| **Offline RTF + WER** | `bench/offline.py` | full-file transcribe, `RTF = proc_time/audio_len` + first-party WER vs references. Primary, apples-to-apples. |
| **Streaming latency** | `bench/streaming.py` | real-time feed → first-token compute, per-update emit latency, final lag. |

- The two NeMo models (Nemotron, Parakeet) and faster-whisper each get an adapter under
  `bench/engines/` behind a common interface; CLIs dispatch by `--engine`.
- Nemotron streaming exposes a **latency knob** via encoder `att_context_size` (`70,0`≈80 ms …
  `70,13`≈1.12 s).
- **Only Nemotron streams natively.** faster-whisper and Parakeet have no native streaming path
  (Parakeet's encoder is full-context — verified, see RESULTS.md), so their streaming numbers come
  from a pseudo-streamed re-decode shim flagged `native=False`. Treat the offline table as the real
  comparison.
- **Wall-clock TTFT is not reported** — it's ~1 s for every engine (clip lead-in + the pseudo
  window), a content/pacing artifact, not a model property. The streaming table reports first-token
  *compute* and final-lag, which actually separate the models.

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
command docker compose run --rm faster-whisper python3 -m bench.datasets fetch --n 15 --configs clean,other
command docker compose run --rm faster-whisper python3 -m bench.offline --engine faster-whisper --model large-v3
command docker compose run --rm nemotron      python3 -m bench.offline --engine nemotron
command docker compose run --rm nemotron      python3 -m bench.offline --engine parakeet
command docker compose run --rm nemotron      python3 -m bench.streaming --engine nemotron --chunk 70,1
command docker compose run --rm faster-whisper python3 analyze.py --md
```

`fetch --configs clean,other` pulls both LibriSpeech splits (tagged per clip) so WER can be broken
out by clean vs other. Both NeMo models (nemotron, parakeet) share the `nemotron` image. Results
append to `results/offline.jsonl` and `results/streaming.jsonl`.

## Caveats / validation

- **Native Nemotron streaming is version-sensitive.** `bench/engines/nemotron.py` attempts
  NeMo's cache-aware `conformer_stream_step` loop and falls back to a chunked re-decode if the
  API differs. **Check `native` in the streaming table** — `False` means the fallback ran and the
  streaming numbers aren't the model's true streaming latency. The offline path is unaffected.
- **WER is first-party**, computed by the harness against `audio/manifest.json` references
  (word-level, lowercased/punctuation-stripped). Our numbers run below published full-set figures
  because the 30-clip set is a small, easier subset — the ordering and gaps are the signal.
- Warm runs only (1 warmup + 3 timed); first-ever run also pays a one-time model download.
