# ASR results summary — Nemotron 3.5-ASR vs Parakeet

RTX 3090 Ti · batch=1 · FP16 · 30 LibriSpeech clips (15 clean + 15 other, 887 ref words).

**🏆 Winner: Parakeet TDT 0.6b-v2** — best accuracy (lowest WER) *and* fastest offline
(lowest median latency / RTF). Nemotron wins only on native streaming latency (Parakeet
has no native streaming path); for offline transcription Parakeet leads on every axis.

| Model | params | Median Response Time (ms) | RTF | WER | TTFT (ms) | Final Lag (ms) |
|-------|--------|---------------------------|-----|-----|-----------|----------------|
| **Parakeet TDT 0.6b-v2** 🏆 | 600M | **80** | **0.013** | **1.69%** | 57 | 71 |
| Nemotron 3.5 ASR (en) | 600M | 84 | 0.013 | 3.04% | 45 | 45 |
| faster-whisper small | 244M | 235 | 0.035 | 3.27% | 88 | 235 |
| faster-whisper medium | 769M | 373 | 0.054 | 5.98% | 135 | 359 |
| faster-whisper large-v3 | 1.55B | 431 | 0.065 | 1.80% | 184 | 407 |

- **Median Response Time (ms)** = median wall-clock to transcribe a clip (offline, per-clip). Length-
  dependent — **RTF** normalizes it out and is the cleaner speed metric; both are shown.
- **RTF** = Real-Time Factor (processing ÷ audio duration; lower = faster).
- **WER** = first-party, overall across clean+other (per-split breakdown in RESULTS.md). faster-whisper
  medium is inflated by a Whisper repetition hallucination on one clip.
- **TTFT (ms)** = first-token *compute* (model time to emit first token); wall-clock TTFT is a
  content/pacing artifact (~1 s for all) and is not reported. Add chunk lookahead for real-world
  algorithmic latency (Nemotron att=[70,1] ≈ 160 ms).
- **Final Lag (ms)** = compute to settle final transcript after last audio frame. Only Nemotron streams
  natively; Parakeet & faster-whisper are pseudo-streamed — so Nemotron owns the streaming-latency
  columns, while Parakeet owns offline accuracy + speed.
