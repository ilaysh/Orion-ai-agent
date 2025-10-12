from __future__ import annotations
import asyncio
import base64
import traceback
import numpy as np
import sounddevice as sd

from orion_core.base_component import BaseComponent, ComponentState
from orion_core.tts.vad import SileroVAD
from orion_core.tts.speak import tts_bytes
from orion_core.tts.transcriber import Transcriber

THRESHOLD = 0.25
MIN_SPEECH_MS = 120
MIN_SILENCE_MS = 900


class SpeechEngine(BaseComponent):
    """
    Handles full speech capture + STT + TTS.
    Core calls:
        text = await engine.transcribe(audio_chunk)
        audio_b64 = await engine.speak(reply)
    """

    name = "speech"

    def __init__(self, sample_rate: int = 16000, chunk: int = 1024):
        super().__init__()
        self.sr = sample_rate
        self.chunk = chunk
        self.vad: SileroVAD | None = None
        self.transcriber: Transcriber | None = None
        self._loop = None
        self._tts_ready = False
        self._running = False

        # capture params
        self._max_len_s = 15.0
        self._end_silence_s = 0.8

    # ------------------------------------------------------------------ #
    #                           LIFECYCLE
    # ------------------------------------------------------------------ #

    async def init(self):
        """Initialize VAD, Transcriber, prepare TTS, and mark ready."""
        async with self._lock:
            self.state = ComponentState.INITIALIZING
            await self._report_state()
            try:
                self._loop = asyncio.get_running_loop()
                self.vad = SileroVAD(sr=self.sr,
                                     threshold=THRESHOLD,
                                     min_speech_ms=MIN_SPEECH_MS,
                                     min_silence_ms=MIN_SILENCE_MS)
                self.transcriber = Transcriber()

                # Preload TTS (warm-up)
                try:
                    _ = await tts_bytes(" ")
                    self._tts_ready = True
                except Exception:
                    pass

                self.state = ComponentState.READY
                await self._report_state()
                print("[SpeechEngine] ✅ Ready.")
            except Exception as e:
                traceback.print_exc()
                self.state = ComponentState.ERROR
                await self._report_state()
                print(f"[SpeechEngine] ❌ Init failed: {e}")

    async def start(self):
        """Marks active for use by core."""
        async with self._lock:
            if self.state != ComponentState.READY:
                await self.init()
            self.active = True
            await self._report_state(extra={"active": True})
            print("[SpeechEngine] ▶ Active.")

    async def stop(self):
        async with self._lock:
            self.active = False
            self.state = ComponentState.STOPPED
            await self._report_state(extra={"active": False})
            print("[SpeechEngine] ⏹️ Stopped.")

    def set_reporter(self, reporter):
        """Called by OrionCore to attach event reporter."""
        self.reporter = reporter

    # ------------------------------------------------------------------ #
    #                         MAIN TRANSCRIBE (STREAM)
    # ------------------------------------------------------------------ #
    async def transcribe(self, audio_chunk: bytes | np.ndarray) -> str:
        if not self.active or self.state != ComponentState.READY:
            return ""

        # decode to float32
        audio = (np.frombuffer(audio_chunk, dtype=np.float32)
                 if isinstance(audio_chunk, (bytes, bytearray))
                 else np.asarray(audio_chunk, dtype=np.float32))
        if audio.size == 0:
            return ""

        # --- rolling buffer (max 3s), flush after ~0.9s silence ---
        sr = self.sr
        min_buf = int(sr * 1.8)          # start once we have 1.5s
        max_buf = int(sr * 3.0)          # never exceed 3s
        silence_gate = int(sr * 1.0)     # ~0.9s of trailing silence

        if not hasattr(self, "_vad_buf"):
            self._vad_buf = np.zeros(0, dtype=np.float32)
        self._vad_buf = np.concatenate([self._vad_buf, audio])[-max_buf:]

        if self._vad_buf.size < min_buf:
            return ""

        window = self._vad_buf

        # --- pre-normalize for stable VAD/Whisper ---
        peak = float(np.max(np.abs(window))) or 1.0
        window = np.clip(window / peak, -1.0, 1.0)

        # --- adaptive flush: if last 0.9s is non-speech, transcribe; else wait ---
        tail = window[-silence_gate:]
        if self.vad and self.vad.is_speech(tail):
            return ""  # keep collecting; user still talking

        overlap = int(sr * 0.3)
        window = self._vad_buf[-int(sr * 3.0 + overlap):]
        try:
            # synchronous call, but fast enough (~200ms) from datetime import datetime
            text = self.transcriber.transcribe(window)  # sync call
            # print timestamp of transcription hh:mm;ss
            self._vad_buf = np.zeros(0, dtype=np.float32)
            if text:
                print(f"[SpeechEngine] 💬 Transcribed: {text}")
            return text or ""
        except Exception as e:
            print(f"[SpeechEngine] ⚠️ Transcribe error: {e}")
            return ""

    # ------------------------------------------------------------------ #
    #                     LOCAL MIC MODE (LEGACY / OFFLINE)
    # ------------------------------------------------------------------ #

    async def listen_and_transcribe(self) -> str:
        """
        Capture live audio from mic until silence, then return transcription text.
        """
        if not self.active or self.state != ComponentState.READY:
            print("[SpeechEngine] ⚠️ Not ready for live transcription.")
            return ""

        print("[SpeechEngine] 🎙️ Listening via mic...")
        self._running = True

        buf = np.zeros(0, dtype=np.float32)
        last_voice_t = self._loop.time()
        speech_started = False
        max_samples = int(self._max_len_s * self.sr)
        silence_limit = self._end_silence_s

        try:
            with sd.InputStream(
                samplerate=self.sr,
                channels=1,
                dtype="float32",
                blocksize=self.chunk,
            ) as stream:
                while self._running:
                    await asyncio.sleep(self.chunk / self.sr / 2)
                    data, _ = stream.read(self.chunk)
                    audio = np.asarray(data, dtype=np.float32).squeeze()
                    if audio.size == 0:
                        continue

                    if self.vad.is_speech(audio):
                        speech_started = True
                        last_voice_t = self._loop.time()
                        buf = np.concatenate([buf, audio])
                    elif speech_started:
                        if (self._loop.time() - last_voice_t) > silence_limit:
                            break

                    if buf.size >= max_samples:
                        print("[SpeechEngine] ⚠️ Max length reached.")
                        break

            # --- Run STT ---
            text = await self.transcriber.transcribe(buf)
            print(f"[SpeechEngine] 🗣️ {text}")
            return text

        except Exception as e:
            print(f"[SpeechEngine] ⚠️ Mic transcription error: {e}")
            return ""

    # ------------------------------------------------------------------ #
    #                             SPEAK
    # ------------------------------------------------------------------ #
    async def speak(self, text: str) -> str | None:
        """TTS wrapper. Returns Base64-encoded WAV."""
        try:
            audio_bytes = await tts_bytes(text)
            if not audio_bytes:
                return None
            return base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as e:
            print(f"[SpeechEngine] ⚠️ TTS error: {e}")
            return None
