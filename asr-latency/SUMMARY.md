# ASR results summary — Nemotron 3.5-ASR vs Parakeet

RTX 3090 Ti · batch=1 · FP16 · 30 LibriSpeech clips (15 clean + 15 other, 887 ref words).

| Model | params | TTFT (ms) | RTF | WER | Final Lag (ms) |
|-------|--------|-----------|-----|-----|----------------|
| Parakeet TDT 0.6b-v2 | 600M | 57 | 0.013 | 1.69% | 71 |
| Nemotron 3.5 ASR (en) | 600M | 45 | 0.013 | 3.04% | 45 |
| faster-whisper small | 244M | 88 | 0.035 | 3.27% | 235 |
| faster-whisper medium | 769M | 135 | 0.054 | 5.98% | 359 |
| faster-whisper large-v3 | 1.55B | 184 | 0.065 | 1.80% | 407 |

- **TTFT (ms)** = first-token *compute* (model time to emit first token); wall-clock TTFT is a
  content/pacing artifact (~1 s for all) and is not reported. Add chunk lookahead for real-world
  algorithmic latency (Nemotron att=[70,1] ≈ 160 ms).
- **RTF** = Real-Time Factor (processing ÷ audio duration; lower = faster).
- **WER** = first-party, overall across clean+other (per-split breakdown in RESULTS.md). faster-whisper
  medium is inflated by a Whisper repetition hallucination on one clip.
- **Final Lag (ms)** = compute to settle final transcript after last audio frame. Only Nemotron streams
  natively; Parakeet & faster-whisper are pseudo-streamed.
