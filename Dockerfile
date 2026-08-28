# ODIN VOICE — RunPod Serverless Worker
# STT: Faster-Whisper | Brain: DeepSeek | TTS: XTTS v2 (Jarvis clone)
FROM runpod/base:1.1.0-cuda1281-ubuntu2204

WORKDIR /odin-voice

# System deps for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Torch pinned to cu128 wheels (matches base image CUDA 12.8)
# torchaudio required by coqui-tts (XTTS) — MUST match torch version + wheel index
RUN pip install --no-cache-dir torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Python deps (no torch in requirements — pinned above)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Worker code + Jarvis voice reference
COPY handler.py .
COPY refs/jarvis.wav ./refs/jarvis.wav

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
