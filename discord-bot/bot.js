#!/usr/bin/env node
/**
 * ODIN VOICE — Standalone Discord Voice Bot (v2)
 * ==============================================
 * The FAST mouth/ears for Odin's voice.
 *
 * This bot is deliberately SEPARATE from the OpenClaw gateway. It:
 *   1. Joins a Discord voice channel (slash command /odin join)
 *   2. Listens for speech (VAD via @discordjs/voice SpeakingMap)
 *   3. Captures audio -> sends to RunPod serverless endpoint (STT+DeepSeek+TTS in ONE shot)
 *   4. Plays the returned audio back into the voice channel
 *   5. Also posts the text reply to the text channel (so Nick sees it instantly)
 *
 * The RunPod worker is the brain (DeepSeek direct). The gateway is only
 * consulted by the worker when the request needs real tools.
 *
 * Token: second Discord bot token (voice bot) — configured via env DISCORD_TOKEN
 * Run:   node bot.js
 */

import {
  Client,
  GatewayIntentBits,
  SlashCommandBuilder,
  Events,
  ChannelType,
} from "discord.js";
import {
  joinVoiceChannel,
  createAudioPlayer,
  createAudioResource,
  AudioPlayerStatus,
  entersState,
  VoiceConnectionStatus,
  getVoiceConnection,
  EndBehaviorType,
} from "@discordjs/voice";
import prism from "prism-media";
import { Readable } from "node:stream";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import crypto from "node:crypto";
import { spawn } from "node:child_process";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------- Config ----------
const DISCORD_TOKEN = process.env.DISCORD_TOKEN || "";
const RUNPOD_URL = process.env.RUNPOD_URL || "https://api.runpod.ai/v2/k8y9qs4tj5108s";
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY || "";
const AUTO_JOIN_CHANNEL = process.env.AUTO_JOIN_CHANNEL || ""; // voice channel ID
const OWNER_ID = process.env.OWNER_ID || ""; // restrict voice commands to this user
const WHISPER_SAMPLE_RATE = 48000; // discord voice is 48k; runpod decodes to 16k

if (!DISCORD_TOKEN) {
  console.error("[odin-voice-bot] DISCORD_TOKEN not set. Export it and re-run.");
  process.exit(1);
}
if (!RUNPOD_API_KEY) {
  console.error("[odin-voice-bot] RUNPOD_API_KEY not set. Export it and re-run.");
  process.exit(1);
}

// ---------- State ----------
const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildVoiceStates,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
});

const audioPlayer = createAudioPlayer();
let currentConnection = null;
let currentChannel = null; // text channel to post replies to
let currentVoiceChannel = null;
let recording = false;
let audioChunks = [];
let silenceTimer = null;
let userSpeakingSince = null;
let isProcessing = false;
let voiceStream = null;

// ---------- RunPod call ----------
async function callRunPod(input) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000); // 2min max for cold start

  try {
    const res = await fetch(`${RUNPOD_URL}/runsync`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${RUNPOD_API_KEY}`,
      },
      body: JSON.stringify({ input }),
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`RunPod HTTP ${res.status}: ${text.slice(0, 200)}`);
    }
    const data = await res.json();
    return data.output || data;
  } catch (err) {
    clearTimeout(timeout);
    throw err;
  }
}

// ---------- Audio capture ----------
function startListening(connection) {
  if (recording) return;
  recording = true;
  audioChunks = [];
  userSpeakingSince = null;
  console.log("[odin-voice-bot] listening...");

  const receiver = connection.receiver;
  receiver.speaking.on("start", (userId) => {
    console.log(`[odin-voice-bot] SPEAKING START user=${userId}`);
    // Only listen to the owner (if set) or anyone (if not)
    if (OWNER_ID && userId !== OWNER_ID) return;
    if (isProcessing) return; // don't interrupt while replying

    userSpeakingSince = Date.now();
    if (silenceTimer) clearTimeout(silenceTimer);

    // Create a per-user audio stream
    const audioStream = receiver.subscribe(userId, {
      end: { behavior: EndBehaviorType.AfterSilence, duration: 700 },
    });
    // Convert Opus (48k) -> PCM s16le via prism
    const decoder = new prism.opus.Decoder({ rate: 48000, channels: 1, frameSize: 960 });
    const pcmStream = audioStream.pipe(decoder);
    audioChunks = [];

    pcmStream.on("data", (chunk) => {
      audioChunks.push(chunk);
    });
    pcmStream.on("end", () => {
      if (audioChunks.length === 0) return;
      const pcm = Buffer.concat(audioChunks);
      // Convert PCM 48k s16le -> WAV (RunPod handler decodes raw PCM as 16k; send 48k wav and let whisper handle)
      const wav = pcmToWav(pcm, 48000);
      console.log(`[odin-voice-bot] captured ${wav.length} bytes wav, sending to RunPod...`);
      handleTurn(wav);
    });
  });

  // Fallback: if no speaking event within 10s, reset
  receiver.speaking.on("end", (userId) => {
    if (OWNER_ID && userId !== OWNER_ID) return;
    silenceTimer = setTimeout(() => {
      if (recording && userSpeakingSince && Date.now() - userSpeakingSince > 5000) {
        // stop listening
      }
    }, 10000);
  });
}

function pcmToWav(pcm, sampleRate) {
  const numChannels = 1;
  const bitsPerSample = 16;
  const blockAlign = numChannels * bitsPerSample / 8;
  const byteRate = sampleRate * blockAlign;
  const dataSize = pcm.length;
  const buffer = Buffer.alloc(44 + dataSize);

  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write("WAVE", 8);
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16); // fmt chunk size
  buffer.writeUInt16LE(1, 20); // PCM
  buffer.writeUInt16LE(numChannels, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(byteRate, 28);
  buffer.writeUInt16LE(blockAlign, 32);
  buffer.writeUInt16LE(bitsPerSample, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(dataSize, 40);
  pcm.copy(buffer, 44);
  return buffer;
}

// ---------- Turn handling ----------
async function handleTurn(wavBuffer) {
  if (isProcessing) return;
  isProcessing = true;
  const t0 = Date.now();
  try {
    const audioB64 = wavBuffer.toString("base64");
    const result = await callRunPod({ audio: audioB64, user_name: "Nick" });

    if (result && result.audio) {
      const audioBuf = Buffer.from(result.audio, "base64");
      const resource = createAudioResource(Readable.from(audioBuf), {
        inputType: "arbitrary", // we pass wav bytes directly
      });
      audioPlayer.play(resource);
      if (currentChannel) {
        const latency = ((Date.now() - t0) / 1000).toFixed(1);
        currentChannel.send(`🎙️ **Odin** (${latency}s): ${result.text || "(no reply)"}`).catch(() => {});
      }
    } else if (result && result.text) {
      if (currentChannel) {
        const latency = ((Date.now() - t0) / 1000).toFixed(1);
        currentChannel.send(`🎙️ **Odin** (${latency}s): ${result.text}`).catch(() => {});
      }
    } else if (result && result.error) {
      console.error("[odin-voice-bot] RunPod error:", result.error);
      if (currentChannel) currentChannel.send(`⚠️ ${result.error}`).catch(() => {});
    }
  } catch (err) {
    console.error("[odin-voice-bot] turn error:", err.message);
    if (currentChannel) currentChannel.send(`⚠️ Voice link error: ${err.message}`).catch(() => {});
  } finally {
    isProcessing = false;
  }
}

// ---------- Join/leave ----------
async function joinVoice(channel) {
  if (!channel || !channel.isVoiceBased()) {
    throw new Error("Not a voice channel");
  }
  currentVoiceChannel = channel;

  const connection = joinVoiceChannel({
    channelId: channel.id,
    guildId: channel.guild.id,
    adapterCreator: channel.guild.voiceAdapterCreator,
    selfDeaf: false,
    selfMute: false,
  });
  currentConnection = connection;

  connection.on(VoiceConnectionStatus.Signalling, () => console.log("[odin-voice-bot] voice: signalling"));
  connection.on(VoiceConnectionStatus.Connecting, () => console.log("[odin-voice-bot] voice: connecting"));
  connection.on(VoiceConnectionStatus.Ready, () => {
    console.log("[odin-voice-bot] connected to voice");
    connection.subscribe(audioPlayer);
    startListening(connection);
  });
  connection.on("error", (err) => console.error("[odin-voice-bot] voice error:", err.message));
  connection.on(VoiceConnectionStatus.Disconnected, async () => {
    try {
      await Promise.race([
        entersState(connection, VoiceConnectionStatus.Connecting, 5000),
        entersState(connection, VoiceConnectionStatus.Signalling, 5000),
      ]);
    } catch {
      console.log("[odin-voice-bot] voice disconnected, destroying");
      connection.destroy();
      currentConnection = null;
    }
  });

  await entersState(connection, VoiceConnectionStatus.Ready, 25000);
}

async function leaveVoice() {
  if (currentConnection) {
    currentConnection.destroy();
    currentConnection = null;
  }
  currentVoiceChannel = null;
  recording = false;
}

// ---------- Slash commands ----------
const commands = [
  new SlashCommandBuilder()
    .setName("odin")
    .setDescription("Odin voice controls")
    .addSubcommand((s) => s.setName("join").setDescription("Join the voice channel you're in"))
    .addSubcommand((s) => s.setName("leave").setDescription("Leave the voice channel"))
    .addSubcommand((s) => s.setName("status").setDescription("Check voice status")),
];

client.on(Events.ClientReady, async () => {
  console.log(`[odin-voice-bot] logged in as ${client.user.tag}`);

  // Register slash commands
  try {
    const guilds = [...client.guilds.cache.values()];
    for (const guild of guilds) {
      await guild.commands.set(commands);
      console.log(`[odin-voice-bot] registered commands in ${guild.name}`);
    }
  } catch (err) {
    console.error("[odin-voice-bot] command registration failed:", err.message);
  }

  // Auto-join if configured
  if (AUTO_JOIN_CHANNEL) {
    const tryJoin = async (attempt) => {
      try {
        const ch = await client.channels.fetch(AUTO_JOIN_CHANNEL);
        if (ch && ch.isVoiceBased()) {
          await joinVoice(ch);
          console.log(`[odin-voice-bot] auto-joined ${ch.name}`);
        } else {
          console.error(`[odin-voice-bot] auto-join channel ${AUTO_JOIN_CHANNEL} is not voice`);
        }
      } catch (err) {
        console.error(`[odin-voice-bot] auto-join attempt ${attempt} failed: ${err.message}`);
        console.log(`[odin-voice-bot] retrying auto-join in 20s (attempt ${attempt})...`);
        setTimeout(() => tryJoin(attempt + 1), 20000);
      }
    };
    // Give the client a moment to fully hydrate, then join
    setTimeout(() => tryJoin(1), 5000);
  }
});

client.on(Events.InteractionCreate, async (interaction) => {
  if (!interaction.isChatInputCommand()) return;
  if (interaction.commandName !== "odin") return;

  const sub = interaction.options.getSubcommand();
  if (sub === "join") {
    const member = interaction.member;
    const voiceChannel = member?.voice?.channel;
    if (!voiceChannel) {
      await interaction.reply({ content: "You're not in a voice channel, bro.", ephemeral: true });
      return;
    }
    currentChannel = interaction.channel;
    try {
      await joinVoice(voiceChannel);
      await interaction.reply({ content: `🎙️ Joined **${voiceChannel.name}** — talk to me.`, ephemeral: false });
    } catch (err) {
      await interaction.reply({ content: `❌ Couldn't join: ${err.message}`, ephemeral: true });
    }
  } else if (sub === "leave") {
    await leaveVoice();
    await interaction.reply({ content: "👋 Left the voice channel.", ephemeral: false });
  } else if (sub === "status") {
    await interaction.reply({
      content: currentConnection
        ? `🔊 Connected to **${currentVoiceChannel?.name || "voice"}**${isProcessing ? " (processing…)" : ""}`
        : "🔇 Not connected.",
      ephemeral: true,
    });
  }
});

// ---------- Text command fallback (if slash not available) ----------
client.on(Events.MessageCreate, async (message) => {
  if (message.author.bot) return;
  if (!message.content.startsWith("!odin")) return;

  const args = message.content.slice(6).trim().split(/\s+/);
  const cmd = args[0]?.toLowerCase();
  if (cmd === "join") {
    const vc = message.member?.voice?.channel;
    if (!vc) return message.reply("You're not in a voice channel.");
    currentChannel = message.channel;
    try {
      await joinVoice(vc);
      message.reply(`Joined ${vc.name}.`);
    } catch (err) {
      message.reply(`Failed: ${err.message}`);
    }
  } else if (cmd === "leave") {
    await leaveVoice();
    message.reply("Left.");
  } else if (cmd === "status") {
    message.reply(currentConnection ? "Connected." : "Not connected.");
  }
});

// ---------- Startup ----------
client.on(Events.Error, (err) => console.error("[odin-voice-bot] client error:", err));
client.login(DISCORD_TOKEN).catch((err) => {
  console.error("[odin-voice-bot] login failed:", err.message);
  process.exit(1);
});
