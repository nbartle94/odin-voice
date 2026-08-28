"""
ODIN VOICE — RunPod Serverless Worker
=====================================
GPU worker for Odin's Discord voice. THE FAST BRAIN.

Architecture (as requested by Nick 2026-08-28):
  Discord voice -> audio -> THIS WORKER:
    1. STT:  Faster-Whisper (small.en, float16, CUDA)
    2. Brain: DeepSeek API DIRECT (api.deepseek.com) — instant, NO gateway in the loop
    3. Context: Hyperspell (live memory) + Vault MCP (via odin-mcp.douggie.au tunnel)
       + OPTIONAL gateway tool execution when Nick asks for tasks/info that need tools
    4. TTS:  XTTS v2 (Jarvis voice clone) -> base64 wav
    5. Return { text, audio }

The OpenClaw gateway is NOT in the chat path. It is only consulted when the
worker's DeepSeek brain decides the request needs real tool access (calendar,
email, files, etc.). That call goes through the vault MCP tunnel to the gateway,
and the result is folded into the reply.

Env vars:
  DEEPSEEK_API_KEY      — DeepSeek API key (required)
  HYPERSPELL_API_KEY    — Hyperspell key for live autocontext (required)
  HYPERSPELL_USER_ID    — Hyperspell userId (default nick.bartle94@gmail.com)
  VAULT_MCP_URL         — default https://odin-mcp.douggie.au/vaultmcp/mcp
  VAULT_MCP_TOKEN       — Bearer token for vault MCP (required)
  GATEWAY_CHAT_URL      — OpenClaw gateway chat completions (for tool turns), default https://odin-mcp.douggie.au/v1/chat/completions
  GATEWAY_TOKEN         — OpenClaw gateway token (for tool turns)
  XTTS_VOICE            — default "" (uses ref wav)
  XTTS_REF_WAV          — default /odin-voice/refs/jarvis.wav
  WHISPER_MODEL         — default small.en
"""

import base64
import json
import os
import re
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

# Gateway chat completions for TOOL turns only (not the chat path)
GATEWAY_CHAT_URL = os.environ.get("GATEWAY_CHAT_URL", "https://odin-mcp.douggie.au/v1/chat/completions")
GATEWAY_TOKEN = os.environ.get("GATEWAY_TOKEN", "")

XTTS_VOICE = os.environ.get("XTTS_VOICE", "")
XTTS_REF_WAV = os.environ.get("XTTS_REF_WAV", "/odin-voice/refs/jarvis.wav")
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
    print("[odin-voice] loading XTTS v2", flush=True)
    from TTS.tts.configs.xtts_config import XttsConfig  # noqa: E402
    from TTS.tts.models.xtts import Xtts  # noqa: E402
    model_path = os.environ.get("XTTS_MODEL_PATH", "")
    if model_path:
        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        tts_pipeline = Xtts.init_from_config(config)
        tts_pipeline.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
        if torch.cuda.is_available():
            tts_pipeline.cuda()
    else:
        from TTS.api import TTS  # noqa: E402
        tts_pipeline = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    print(f"[odin-voice] models loaded in {time.time()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Hyperspell live autocontext
# ---------------------------------------------------------------------------
def hyperspell_context(query: str, limit: int = 5) -> str:
    """Pull relevant memories from Hyperspell (same API the OpenClaw plugin uses)."""
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
        if isinstance(results, list) and results:
            parts = []
            for r in results[:limit]:
                text = r.get("text") or r.get("content") or r.get("title") or ""
                if text:
                    parts.append(text[:300])
            return "\n".join(parts)
    except Exception as e:
        print(f"[odin-voice] hyperspell err: {e}", flush=True)
    return ""


# ---------------------------------------------------------------------------
# Vault MCP (local .md memory files)
# ---------------------------------------------------------------------------
class VaultMCPClient:
    """Minimal MCP Streamable-HTTP client for the vault (session + SSE handling)."""

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
                       "clientInfo": {"name": "odin-voice", "version": "2.0"}},
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


# ---------------------------------------------------------------------------
# Gateway tool execution (ONLY when the brain needs real tools)
# ---------------------------------------------------------------------------
def gateway_tool_turn(transcript: str, fast_reply: str) -> str:
    """Send a task to the OpenClaw gateway (full agent with tools) and return its text.

    Used when Nick asks for something that needs real tool access (calendar,
    email, files, web, etc.). This is the SLOW path but it's only invoked when
    the fast DeepSeek brain decides it's needed. Returns the agent's reply.
    """
    if not GATEWAY_TOKEN:
        return ""
    try:
        body = json.dumps({
            "model": "openclaw/default",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Odin, Nick's AI chief of staff. Nick just spoke via "
                        "voice. The fast brain drafted a quick reply; if the task "
                        "needs real tools (calendar, email, files, web, memory), "
                        "perform it and give a concise spoken-style answer. "
                        "No markdown, no emojis, couple of sentences.\n\n"
                        f"Fast brain draft: {fast_reply}"
                    ),
                },
                {"role": "user", "content": transcript},
            ],
            "user": "voice-runpod",
        }).encode()
        req = urllib.request.Request(
            GATEWAY_CHAT_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GATEWAY_TOKEN}",
            },
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode())
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return clean_for_voice(content) if content else ""
    except Exception as e:
        print(f"[odin-voice] gateway tool turn err: {e}", flush=True)
        return ""


# ---------------------------------------------------------------------------
# DeepSeek brain (fast path)
# ---------------------------------------------------------------------------
def ask_deepseek(transcript: str, context: str) -> str:
    messages = [{"role": "system", "content": ODIN_PERSONA}]
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


def needs_tools(transcript: str) -> bool:
    """Heuristic: does this request need live tool access (calendar, email, files, web)?"""
    patterns = [
        r"\b(calendar|schedule|appointment|meeting|remind|reminder)\b",
        r"\b(email|mail|inbox|send)\b",
        r"\b(file|document|drive|folder)\b",
        r"\b(check|look up|lookup|search|find|pull up|fetch|get)\b",
        r"\b(weather|news|price|stock|score|result)\b",
        r"\b(who|what time|what day|when is|where is)\b",
        r"\b(task|todo|to[- ]do)\b",
    ]
    return any(re.search(p, transcript, re.IGNORECASE) for p in patterns)


def clean_for_voice(text: str) -> str:
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
    """XTTS v2 TTS -> base64 wav (24k mono). Uses voice cloning if ref wav provided."""
    import io
    import wave

    speaker = XTTS_VOICE or "jarvis"
    kwargs = {}
    if XTTS_REF_WAV and os.path.exists(XTTS_REF_WAV):
        kwargs = {"speaker_wav": XTTS_REF_WAV}

    if hasattr(tts_pipeline, "tts"):
        wav = tts_pipeline.tts(text=text, speaker=speaker, language="en", **kwargs)
        import numpy as _np
        if not isinstance(wav, _np.ndarray):
            wav = _np.array(wav)
    else:
        out = tts_pipeline.synthesize(text, config=tts_pipeline.config, speaker_wav=XTTS_REF_WAV or speaker, language="en")
        wav = out["wav"] if isinstance(out, dict) else out[0]
        wav = torch.tensor(wav).cpu().numpy() if hasattr(wav, "cpu") else wav

    wav = wav.squeeze()
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)
    if wav.abs().max() > 1.0:
        wav = wav / 32768.0

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes((wav * 32767).astype(np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
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

    # Live context (fast, non-blocking-ish)
    context = hyperspell_context(transcript)
    if not context.strip():
        context = vault_search(transcript)

    # Fast brain: DeepSeek direct
    reply = ask_deepseek(transcript, context)

    # Tool path: if the request needs real tools, consult the gateway agent
    if needs_tools(transcript):
        t_tool = time.time()
        tool_reply = gateway_tool_turn(transcript, reply)
        if tool_reply.strip():
            reply = tool_reply
        print(f"[odin-voice] gateway tool turn took {time.time()-t_tool:.1f}s", flush=True)

    audio_b64_out = synthesize(reply) if reply.strip() else None

    print(f"[odin-voice] turn done in {time.time()-t0:.1f}s | reply: {reply[:80]!r}", flush=True)
    return {"text": reply, "audio": audio_b64_out}


@runpod.serverless.register_fitness_check
def check_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available")


runpod.serverless.start({"handler": handler})
