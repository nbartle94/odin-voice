"""
ODIN VOICE — RunPod Serverless Worker
=====================================
GPU worker that handles Discord voice turns for Odin (Nick's AI chief of staff).

Pipeline (per job):
  1. Decode base64 audio from Discord voice capture
  2. STT: Faster-Whisper (small.en, float16, CUDA)
  3. Live context fetch:
       - Hyperspell live autocontext (api.hyperspell.com, cloud)
       - Vault MCP (.md memory files) via https://odin-mcp.douggie.au/vaultmcp/mcp
  4. Brain: DeepSeek API (direct) — persona'd as Odin with SOUL.md + live context
  5. TTS: Kokoro-82M bm_lewis (British male) → base64 wav
  6. Return { text, audio_b64 }

Env vars:
  DEEPSEEK_API_KEY     — DeepSeek API key (required)
  HYPERSPELL_API_KEY   — Hyperspell key for live autocontext (required)
  HYPERSPELL_USER_ID   — Hyperspell userId (default nick.bartle94@gmail.com)
  VAULT_MCP_URL        — default https://odin-mcp.douggie.au/vaultmcp/mcp
  VAULT_MCP_TOKEN      — Bearer token for vault MCP (required)
  KOKORO_VOICE         — default bm_lewis
  WHISPER_MODEL        — default small.en
"""

import base64
import json
import os
import time
import urllib.request

import runpod
import torch
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

HYPERSPELL_KEY = os.environ.get("HYPERSPELL_API_KEY", "")
HYPERSPELL_USER = os.environ.get("HYPERSPELL_USER_ID", "nick.bartle94@gmail.com")
HYPERSPELL_URL = "https://api.hyperspell.com"

VAULT_MCP_URL = os.environ.get("VAULT_MCP_URL", "https://odin-mcp.douggie.au/vaultmcp/mcp")
VAULT_MCP_TOKEN = os.environ.get("VAULT_MCP_TOKEN", "")

KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "bm_lewis")
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "small.en")

# Odin persona — this is what makes the worker "another instance of me"
ODIN_PERSONA = (
    "You are Odin, Nick Bartle's AI chief of staff and personal assistant. "
    "You are speaking to Nick by voice in a Discord voice channel. You are "
    "calm, competent, dry-witted, and warm when it counts. You call him 'bro' "
    "and 'man' occasionally. Reply in a concise, conversational way as if "
    "talking out loud. Do NOT use markdown, asterisks, bullet points, emojis, "
    "or any formatting that would be read aloud awkwardly. Keep it short — a "
    "couple of sentences is ideal. Give a brief answer and stop; offer to "
    "expand only if genuinely useful. No newlines or special characters "
    "besides plain words and basic punctuation."
)

# ---------------------------------------------------------------------------
# Models (loaded once at cold start)
# ---------------------------------------------------------------------------
print("[odin-voice] CUDA available:", torch.cuda.is_available(), flush=True)

from faster_whisper import WhisperModel  # noqa: E402

stt_model = None
tts_pipeline = None


def load_models():
    global stt_model, tts_pipeline
    if stt_model is not None:
        return
    t0 = time.time()
    print("[odin-voice] loading Whisper", WHISPER_MODEL_NAME, flush=True)
    stt_model = WhisperModel(WHISPER_MODEL_NAME, device="cuda", compute_type="float16")
    print("[odin-voice] loading Kokoro (voice", KOKORO_VOICE + ")", flush=True)
    from kokoro import KPipeline  # noqa: E402
    tts_pipeline = KPipeline(lang_code="b")  # b = British English
    print(f"[odin-voice] models loaded in {time.time()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Hyperspell live autocontext
# ---------------------------------------------------------------------------
def hyperspell_context(query: str, limit: int = 5) -> str:
    """Pull relevant memories from Hyperspell (same API the OpenClaw plugin uses)."""
    if not HYPERSPELL_KEY:
        return ""
    try:
        url = f"{HYPERSPELL_URL}/v1/search?q={urllib.parse.quote(query)}&limit={limit}&user_id={urllib.parse.quote(HYPERSPELL_USER)}"
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {HYPERSPELL_KEY}",
            "X-As-User": HYPERSPELL_USER,
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        # Normalize response shape: could be {"results":[...]} or {"memories":[...]}
        results = data.get("results") or data.get("memories") or []
        if isinstance(results, list) and results:
            parts = []
            for r in results[:limit]:
                text = r.get("text") or r.get("content") or r.get("summary") or ""
                if text:
                    parts.append(text[:300])
            return "\n".join(parts)
    except Exception as e:
        print(f"[odin-voice] hyperspell err: {e}", flush=True)
    return ""


# ---------------------------------------------------------------------------
# Vault MCP (local .md memory files)
# ---------------------------------------------------------------------------
def vault_search(query: str) -> str:
    """Search the local vault via MCP (through the Cloudflare tunnel)."""
    if not VAULT_MCP_TOKEN:
        return ""
    try:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "vault_search", "arguments": {"query": query}},
        }
        req = urllib.request.Request(
            VAULT_MCP_URL,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {VAULT_MCP_TOKEN}",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        # MCP tool result shape: {result: {content: [{text: ...}]}}
        content = data.get("result", {}).get("content", [])
        texts = [c.get("text", "") for c in content if isinstance(c, dict)]
        return "\n".join(texts)[:2000]
    except Exception as e:
        print(f"[odin-voice] vault err: {e}", flush=True)
    return ""


# ---------------------------------------------------------------------------
# DeepSeek brain
# ---------------------------------------------------------------------------
def ask_deepseek(transcript: str, context: str) -> str:
    messages = [
        {"role": "system", "content": ODIN_PERSONA},
    ]
    if context.strip():
        messages.append({
            "role": "system",
            "content": (
                "Relevant recalled context from Nick's memory (hyperspell + vault). "
                "Use it only if relevant to what he says; don't force it:\n\n" + context[:4000]
            ),
        })
    messages.append({"role": "user", "content": transcript})

    body = json.dumps({
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(
        DEEPSEEK_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        return clean_for_voice(data["choices"][0]["message"]["content"])
    except Exception as e:
        return f"Brain error: {e}"


def clean_for_voice(text: str) -> str:
    import re
    t = re.sub(r"[*_#`~>|\\-]{1,}", " ", text)
    t = re.sub(r"\[([^]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
def decode_audio(audio_b64: str):
    raw = base64.b64decode(audio_b64)
    if raw[:4] == b"RIFF" or raw[:2] == b"\xff\xfb" or raw[:3] == b"ID3":
        import io
        import scipy.io.wavfile as wavfile
        try:
            rate, data = wavfile.read(io.BytesIO(raw))
            if data.dtype != np.float32:
                data = data.astype(np.float32) / 32768.0
            return data, rate
        except Exception:
            pass
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return data, 16000


def transcribe(audio_b64: str) -> str:
    audio, rate = decode_audio(audio_b64)
    import wave
    with open("/tmp/input.wav", "wb") as f:
        with wave.open(f, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes((audio * 32768).astype(np.int16).tobytes())
    segments, _ = stt_model.transcribe("/tmp/input.wav", beam_size=1)
    text = " ".join(s.text for s in segments).strip()
    print(f"[odin-voice] STT: {text!r}", flush=True)
    return text


def synthesize(text: str) -> str:
    generator = tts_pipeline(text, voice=KOKORO_VOICE, speed=1.0)
    chunks = []
    for _, _, audio in generator:
        chunks.append(audio)
    if not chunks:
        return ""
    full = np.concatenate(chunks)
    import wave
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes((full * 32767).astype(np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
def handler(job):
    job_input = job.get("input", {})
    audio_b64 = job_input.get("audio")
    user_name = job_input.get("user_name", "Nick")
    if not audio_b64:
        return {"error": "no audio provided"}

    load_models()
    t0 = time.time()

    transcript = transcribe(audio_b64)
    if not transcript.strip():
        return {"text": "", "audio": None}

    # Live context: hyperspell + vault (this is what makes it "me")
    context = hyperspell_context(transcript)
    if not context.strip():
        context = vault_search(transcript)

    reply = ask_deepseek(transcript, context)
    audio_b64_out = synthesize(reply) if reply.strip() else None

    print(f"[odin-voice] turn done in {time.time()-t0:.1f}s | reply: {reply[:80]!r}", flush=True)
    return {"text": reply, "audio": audio_b64_out}


@runpod.serverless.register_fitness_check
def check_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available")


runpod.serverless.start({"handler": handler})
