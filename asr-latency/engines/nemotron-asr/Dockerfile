# NVIDIA Nemotron streaming ASR via NeMo. Minimal ASR-only image (NOT the repo's 2-3hr
# Blackwell unified build) — the 0.6B model officially supports Ampere (RTX 3090).
#
# NeMo's dep tree is large; this build pulls torch (cu124, includes sm_86 for Ampere) then
# nemo_toolkit[asr]. Expect a long first build (many minutes) and a multi-GB image.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git build-essential \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Torch first (pinned CUDA build) so NeMo doesn't pull a CPU wheel.
RUN pip3 install --upgrade pip setuptools wheel \
    && pip3 install Cython packaging \
    && pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# NeMo ASR (>=2.6 / NeMo 25.11 is required by nemotron-speech-streaming-en-0.6b).
RUN pip3 install "nemo_toolkit[asr]>=2.6.0" "soundfile>=0.12" "numpy>=1.26"

WORKDIR /work
COPY bench /work/bench
COPY analyze.py /work/analyze.py

ENV BENCH_AUDIO_DIR=/work/audio \
    BENCH_RESULTS_DIR=/work/results

CMD ["python3", "-m", "bench.offline", "--engine", "nemotron"]
