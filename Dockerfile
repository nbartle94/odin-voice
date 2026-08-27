# ODIN VOICE — RunPod Serverless Worker
# STT: Faster-Whisper | Brain: OpenClaw gateway | TTS: Kokoro bm_lewis
FROM runpod/base:1.1.0-cuda1281-ubuntu2204

WORKDIR /odin-voice

# System deps for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Worker code
COPY handler.py .

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
