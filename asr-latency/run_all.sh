#!/usr/bin/env bash
# End-to-end ASR latency sweep. Assumes images are built and audio is fetched.
# Usage: ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

DC="command docker compose"

echo "== fetch audio (if missing) =="
[ -f audio/manifest.json ] || $DC run --rm faster-whisper python3 -m bench.datasets fetch --n 8

echo "== offline RTF: faster-whisper sweep =="
for m in small medium large-v3; do
  $DC run --rm faster-whisper python3 -m bench.offline --engine faster-whisper --model "$m"
done

echo "== offline RTF: nemotron =="
$DC run --rm nemotron python3 -m bench.offline --engine nemotron

echo "== streaming: nemotron latency-knob sweep =="
for c in 70,0 70,1 70,6 70,13; do
  $DC run --rm nemotron python3 -m bench.streaming --engine nemotron --chunk "$c"
done

echo "== streaming: faster-whisper (pseudo) =="
$DC run --rm faster-whisper python3 -m bench.streaming --engine faster-whisper --model large-v3 --window 1.0

echo "== results =="
$DC run --rm faster-whisper python3 analyze.py
