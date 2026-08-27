"""
ODIN VOICE — RunPod Serverless Worker
=====================================
GPU worker for Odin's Discord voice.

Two modes:
  1. Full turn:  input {audio: b64, user_name} -> whisper STT -> DeepSeek (Odin) -> Kokoro TTS -> {text, audio}
  2. TTS only:   input {text} -> DeepSeek (Odin) -> Kokoro TTS -> {text, audio}

Live context: Hyperspell (POST /memories/query) + vault MCP via Cloudflare tunnel.
"""

import base64
import json
import os
import time
import urllib.request

import runpod
import torch
import numpy as np

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

ODIN_PERSONA = (
    "You are Odin, Nick Bartle's AI chief of staff and personal assistant. "
    "You are speaking to Nick by voice in a Discord voice channel. You are "
    "calm, competent, dry-witted, and warm when it counts. You call him 'bro' "
    "and 'man' occasionally. Reply in a concise, conversational way as if "
    "talking out loud. Do NOT use markdown, asterisks, bullet points, emojis, "
    "or any formatting that would be read aloud awkwardly. Keep it short - a "
    "couple of sentences is ideal. Give a brief answer and stop; offer to "
    "expand only if genuinely useful. No newlines or special characters "
    "besides plain words and basic punctuation."
)

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
    tts_pipeline = KPipeline(lang_code="b")
    print(f"[odin-voice] models loaded in {time.time()-t0:.1f}s", flush=True)


def hyperspell_context(query: str, limit: int = 5) -> str:
    if not HYPERSPELL_KEY:
        return ""
    try:
        body = json.dumps({"query": query, "max_results": limit}).encode()
        req = urllib.request.Request(
            f"{HYPERSPELL_URL}/memories/query",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {HYPERSPELL_KEY}",
                "X-As-User": HYPERSPELL_USER,
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        results = data.get("documents") or []
        parts = []
        for r in results[:limit]:
            text = r.get("text") or r.get("content") or r.get("title") or ""
            if text:
                parts.append(text[:300])
        return "\n".join(parts)
    except Exception as e:
        print(f"[odin-voice] hyperspell err: {e}", flush=True)
    return ""


class VaultMCPClient:
    """Minimal MCP Streamable-HTTP client for the vault."""

    def __init__(self, url: str, token: str):
        self.url = url
        self.token = token
        self.session_id = None

    def _headers(self):
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        if self.session_id:
            h["mcp-session-id"] = self.session_id
        return h

    def _post(self, payload: dict, timeout: int = 15):
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers=self._headers(),
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            sid = resp.headers.get("mcp-session-id")
            if sid:
                self.session_id = sid
            body = resp.read().decode()
        texts = []
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("data: "):
                try:
                    texts.append(json.loads(line[6:]))
                except Exception:
                    pass
        if not texts:
            try:
                texts.append(json.loads(body))
            except Exception:
                pass
        return texts[-1] if texts else {}

    def ensure_session(self):
        if self.session_id:
            return
        self._post({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "odin-voice", "version": "1.0"}},
        })
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call_tool(self, name: str, args: dict, timeout: int = 15):
        self.ensure_session()
        resp = self._post({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": name, "arguments": args},
        }, timeout=timeout)
        content = resp.get("result", {}).get("content", [])
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))


_vault_client = None


def get_vault_client():
    global _vault_client
    if _vault_client is None:
        _vault_client = VaultMCPClient(VAULT_MCP_URL, VAULT_MCP_TOKEN)
    return _vault_client


def vault_search(query: str) -> str:
    if not VAULT_MCP_TOKEN:
        return ""
    try:
        return get_vault_client().call_tool("vault_search", {"query": query})[:2000]
    except Exception as e:
        print(f"[odin-voice] vault err: {e}", flush=True)
    return ""


def ask_deepseek(transcript: str, context: str) -> str:
    messages = [{"role": "system", "content": ODIN_PERSONA}]
    if context.strip():
        messages.append({
            "role": "system",
            "content": (
                "Relevant recalled context from Nick's memory (hyperspell + vault). "
                "Use it only if relevant; don't force it:\n\n" + context[:4000]
            ),
        })
    messages.append({"role": "user", "content": transcript})
    body = json.dumps({
        "model": DEEPSEEK_MODEL, "messages": messages,
        "max_tokens": 300, "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(
        DEEPSEEK_URL, data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_KEY}"},
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


def handler(job):
    job_input = job.get("input", {})
    text_in = job_input.get("text", "")
    audio_b64 = job_input.get("audio", "")
    user_name = job_input.get("user_name", "Nick")

    load_models()
    t0 = time.time()

    # Mode 1: full turn (audio in)
    if audio_b64:
        transcript = transcribe(audio_b64)
        if not transcript.strip():
            return {"text": "", "audio": None}
    # Mode 2: text in (TTS only path for Discord native voice)
    elif text_in.strip():
        transcript = text_in.strip()
    else:
        return {"error": "no audio or text provided"}

    # Live context
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
