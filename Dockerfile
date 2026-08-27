# ODIN VOICE — RunPod Serverless Worker
# STT: Faster-Whisper | Brain: DeepSeek | TTS: Kokoro bm_lewis
FROM runpod/base:1.1.0-cuda1281-ubuntu2204

WORKDIR /odin-voice

# System deps for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Torch pinned to cu128 wheels (matches base image CUDA 12.8)
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Python deps (no torch in requirements — pinned above)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Worker code
COPY handler.py .

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
