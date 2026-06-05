# faster-whisper on GPU (CTranslate2). cuDNN-runtime base provides the cuDNN 9 that
# CTranslate2 needs for CUDA 12. This image also carries `datasets`+`soundfile` so it
# doubles as the audio-fetch runner (python -m bench.datasets fetch).
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# faster-whisper pulls CTranslate2; soundfile/datasets are for streaming + audio fetch.
RUN pip3 install --upgrade pip && pip3 install \
        "faster-whisper==1.1.0" \
        "soundfile>=0.12" \
        "datasets>=2.18" \
        "numpy>=1.26"

WORKDIR /work
# bench/ is mounted at runtime (compose), but copy it too so the image is self-contained.
COPY bench /work/bench
COPY analyze.py /work/analyze.py

ENV BENCH_AUDIO_DIR=/work/audio \
    BENCH_RESULTS_DIR=/work/results

CMD ["python3", "-m", "bench.offline", "--engine", "faster-whisper", "--model", "large-v3"]
