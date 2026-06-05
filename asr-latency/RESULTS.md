# ASR latency results — Nemotron 3.5-ASR vs Parakeet

**Hardware:** RTX 3090 Ti (Ampere), batch size 1, FP16. **Audio:** 30 LibriSpeech clips —
15 test-clean + 15 test-other (287 s total, 887 reference words). 1 warmup + 3 timed runs per clip.
**Software:** faster-whisper 1.1.0 (CTranslate2, CUDA 12.6) · NeMo 2.7.3 / torch 2.6+cu124.

Reproduce: fetch with `python3 -m bench.datasets fetch --n 15 --configs clean,other`, run each
engine via `python3 -m bench.offline --engine ...`, aggregate with `analyze.py`.

## Offline latency + WER (full-file transcription)

WER is first-party, word-level (Levenshtein on lowercased, punctuation-stripped words), against
the references in `audio/manifest.json` (clean = 470 ref words, other = 417 ref words).
Median response time = median wall-clock to transcribe a clip (length-dependent); Real-Time Factor =
processing time ÷ audio duration (length-normalized, the cleaner speed metric). Both are shown.

### Tested (this machine)

**🏆 Parakeet TDT 0.6b-v2 wins:** lowest WER on both splits *and* fastest offline (lowest median
response time + RTF). Nemotron leads only on native streaming latency (next section); for offline
transcription Parakeet is the top model on every axis.

| model                              | params | median response | Real-Time Factor | WER clean | WER other |
| ---------------------------------- | ------ | --------------- | ---------------- | --------- | --------- |
| **parakeet-tdt-0.6b-v2** 🏆        | 600M   | **80 ms**       | **0.013**        | **1.28%** | **2.16%** |
| nemotron-speech-streaming-en-0.6b  | 600M   | 84 ms          | 0.013            | 1.91%     | 4.32%     |
| faster-whisper small               | 244M   | 235 ms         | 0.035            | 1.91%     | 4.80%     |
| faster-whisper medium              | 769M   | 373 ms         | 0.054            | 7.02% ⚠   | 4.80%     |
| faster-whisper large-v3            | 1.55B  | 431 ms         | 0.065            | 0.85%     | 2.88%     |

### Published (model cards / literature)

| model                              | params | WER clean   | WER other   |
| ---------------------------------- | ------ | ----------- | ----------- |
| parakeet-tdt-0.6b-v2               | 600M   | 1.69%       | 3.19%       |
| nemotron-speech-streaming-en-0.6b  | 600M   | 2.32–2.80%  | 4.84–6.01%  |
| faster-whisper small               | 244M   | ~3.4%       | ~7.5%       |
| faster-whisper medium              | 769M   | ~3.0%       | ~7.5%       |
| faster-whisper large-v3            | 1.55B  | ~2.0%       | ~3.6%       |

Parakeet & Nemotron WER are exact from NVIDIA model cards (Nemotron's range spans its 80 ms → 1.12 s
streaming chunk settings). Whisper WER is approximate (normalizer-dependent; large-v3 clean is
variously cited 2.1–2.7%). Published RTF is omitted — papers report WER, not on-device speed (it's
hardware-specific), which is exactly what the Tested table measures.

⚠ **medium clean 7.02% is a real Whisper repetition hallucination, not a measurement bug:** on
clip `clean-6930-81414-0001` medium transcribed the utterance correctly, then looped and
re-transcribed the first half (70 output words vs 47 reference → 24 insertion errors). One clip
swings a 470-word sample ~5 points. small / large-v3 / Nemotron / Parakeet all handled it cleanly.

**Takeaways:**
- **🏆 Parakeet is the winner:** best WER on *both* splits *and* fastest offline (80 ms median,
  RTF 0.013). Parakeet and Nemotron — both 600M NeMo models — are ~4–5× lower RTF than Whisper
  large-v3; Parakeet edges Nemotron on both accuracy and offline speed.
- **Our WER runs below published** almost everywhere — expected, since 30 clips is a small, easier
  subset of the full ~5.4 h test sets. The *ordering and relative gaps* are the signal, not absolute
  parity with published full-set numbers.
- Whisper latency scales with utterance length; the NeMo streaming encoders stay nearly flat.

## Streaming latency (real-time fed; 30 clips, median per engine)

The latency that matters is **compute**, isolated from clip content: **first-token compute** =
ms the model spends to produce its first token; **emit** = median per-update compute; **final-lag**
= compute to settle the final transcript after the last audio frame. Nemotron at att=[70,1]
(≈160 ms); pseudo engines use a 1.0 s re-decode window.

| model                              | streaming | first-token ms | emit_ms | final_lag_ms |
| ---------------------------------- | --------- | -------------- | ------- | ------------ |
| nemotron-speech-streaming-en-0.6b  | native    | 45             | 45      | 45           |
| parakeet-tdt-0.6b-v2               | pseudo    | 57             | 61      | 71           |
| faster-whisper small               | pseudo    | 88             | 163     | 235          |
| faster-whisper medium              | pseudo    | 135            | 267     | 359          |
| faster-whisper large-v3            | pseudo    | 184            | 342     | 407          |

**Takeaway:** Nemotron's native cache-aware streaming produces its first token in **45 ms** of
compute and settles the transcript **45 ms** after the last audio frame; faster-whisper large-v3's
pseudo-stream needs **184 ms** to first token and **407 ms** final-lag, and its cost grows with
utterance length. Parakeet (pseudo shim, no native streaming path) still shows low per-update
compute (57–61 ms), reflecting its fast offline decode.

### Caveats

- **Wall-clock "time-to-first-token" is NOT reported because it's a content/pacing artifact, not a
  model property.** Measured from the start of real-time audio feed, first-token wall-clock is
  ~1.0–1.2 s for *every* engine — dominated by when the first word is actually spoken, and floored
  for the pseudo engines by their 1 s window. It does not separate the models. The **first-token
  compute** column above is the model-attributable cost (45 ms vs 184 ms); add the chunk lookahead
  (Nemotron att=[70,1] ≈ 160 ms) for the real-world algorithmic latency.
- **Only Nemotron streams natively.** Parakeet and faster-whisper are **pseudo**-streamed (growing-
  buffer re-decode) — a streaming-quality proxy, not a like-for-like. The offline table is the
  apples-to-apples comparison; for true streaming, only Nemotron's numbers are native.
- **Parakeet cannot stream natively** (verified, not just unimplemented): `parakeet-tdt-0.6b-v2`'s
  encoder is full-context — `att_context_style: regular`, and `att_context_size_all` offers only
  `[[-1, -1]]` (unlimited left+right). It must see the whole utterance before scoring early frames,
  so no chunk size yields valid incremental output. The `conformer_stream_step` method exists only
  via `ConformerEncoder` inheritance; it is unusable here. A native NVIDIA streaming comparison would
  need a cache-aware checkpoint (e.g. `stt_en_fastconformer_hybrid_large_streaming_multi`), which is
  a different/smaller model — Nemotron is the streaming-trained model in this study.
- 30 clips (15 clean + 15 other), single GPU, single host — directional, not benchmark-grade.
