# orion_core/tts/engine.py
from __future__ import annotations
import asyncio
import base64
import traceback
from orion_core.base_component import BaseComponent, ComponentState
from orion_core.tts.speak import tts_bytes


class SpeechEngine(BaseComponent):
    """
    Unified speech interface for Orion.
    Provides:
      - VAD-based recording (Silero)
      - STT (speech to text)
      - TTS (text to speech)
      - Translation stubs
    """
    name = "speech"

    def __init__(self):
        super().__init__()
        self.vad = None
        self.stt_model = None
        self.tts_ready = False
        print("[SpeechEngine] Created instance.")

    # ------------------------------------------------------------------ #
    # INIT
    # ------------------------------------------------------------------ #
    async def init(self):
        async with self._lock:
            self.state = ComponentState.INITIALIZING
            await self._report_state()
            print("[SpeechEngine] 🚀 Initializing...")

            try:
                # Lazy import to avoid circular import
                from orion_core.vad import SileroVAD
                self.vad = SileroVAD()

                # Load STT model (stub for now)
                self.stt_model = await self._load_stt_model()

                # Warm up TTS
                _ =  tts_bytes("Orion system initializing")
                self.tts_ready = True

                self.state = ComponentState.READY
                await self._report_state()
                print("[SpeechEngine] ✅ All subsystems ready.")
            except Exception as e:
                traceback.print_exc()
                self.state = ComponentState.ERROR
                print(f"[SpeechEngine] ❌ Init failed: {e}")
                await self._report_state()

    async def _load_stt_model(self):
        """Placeholder for Whisper or other STT models."""
        await asyncio.sleep(0.2)
        print("[SpeechEngine] 🧠 (Stub) STT model loaded.")
        return "stt_model_stub"

    # ------------------------------------------------------------------ #
    # MAIN FUNCTIONS
    # ------------------------------------------------------------------ #

    async def transcribe(self) -> str:
        """
        Capture audio via VAD and transcribe to text.
        """
        if not self.vad:
            print("[SpeechEngine] ⚠️ No VAD initialized.")
            return ""

        print("[SpeechEngine] 🎙️ Listening for speech...")
        try:
            # You will later replace this with your real VAD+mic pipeline.
            # Example: audio = await self.vad.listen_once()
            await asyncio.sleep(0.5)
            print("[SpeechEngine] 🪄 Captured audio chunk. Running STT...")

            # Placeholder for Whisper STT
            text = "hello world"
            print(f"[SpeechEngine] 🗣️ Transcribed: {text}")
            return text
        except Exception as e:
            print(f"[SpeechEngine] ❌ Transcription failed: {e}")
            return ""

    async def speak(self, text: str) -> str:
        """
        Convert text to speech and return audio bytes.
        """
        if not self.tts_ready:
            print("[SpeechEngine] ⚠️ TTS not initialized yet.")
            return b""

        try:
            audio_bytes = await tts_bytes(text)
            print(f"[SpeechEngine] 🔊 Spoke: {text}")
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            print(f"[SpeechEngine] 🔊 Encoded audio, type={type(audio_b64)}")
            return audio_b64
        except Exception as e:
            print(f"[SpeechEngine] ❌ Speak error: {e}")
            return b""

    async def translate(self, text: str, target_lang: str = "en") -> str:
        """
        Simple translation stub (can connect to model or MarianMT).
        """
        await asyncio.sleep(0.1)
        print(f"[SpeechEngine] 🌐 Translating to {target_lang} (stub).")
        return text

    # ------------------------------------------------------------------ #
    # LIFECYCLE CONTROL
    # ------------------------------------------------------------------ #

    async def start(self):
        async with self._lock:
            if self.state != ComponentState.READY:
                await self.init()
            self.active = True
            print("[SpeechEngine] ▶ Active.")
            await self._report_state(extra={"active": True})

    async def stop(self):
        async with self._lock:
            self.active = False
            self.state = ComponentState.STOPPED
            print("[SpeechEngine] ⏹ Stopped.")
            await self._report_state(extra={"active": False})
