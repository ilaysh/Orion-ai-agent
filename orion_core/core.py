# orion_core/core.py
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Optional

from .tts.engine import SpeechEngine
from .wakeword import WakeMonitor


class OrionCore:
    """
    Orchestrates wake-word gating, VAD/STT, and TTS.
    States: "idle" -> "listening" -> "speaking" -> back to "idle"
    """

    def __init__(self):
        self.state: str = "idle"
        self.events: asyncio.Queue[dict] = asyncio.Queue()
        self.wake_triggered: bool = False

        # Speech/TTS pipeline (do nothing until wake fires)
        self.speech_engine = SpeechEngine()

        # Wake word monitor (background thread)
        model_path = Path("models/orion_wake.onnx")
        self.wake: Optional[WakeMonitor] = None
        if model_path.exists():
            self.wake = WakeMonitor(
                model_path=model_path,
                on_detect=self._on_wake_detect,
                threshold=0.80,
                consecutive=2,
                provider=None,  # set to "CUDAExecutionProvider" if your ORT build supports it
                verbose=True,   # flip to False to silence confidence logs
            )
            self.wake.start()

    # ------------------- wake callback -------------------

    def _set_state(self, new_state: str):
        self.state = new_state
        # enqueue UI state change
        try:
            self.events.put_nowait({"type": "state", "state": new_state})
        except asyncio.QueueFull:
            pass

    def _on_wake_detect(self):
        # called from background thread → schedule onto event loop
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(self._wake_armed)

    def _wake_armed(self):
        # now on event loop thread
        self.wake_triggered = True
        self.speech_engine.reset_feed()  # fresh buffer for the utterance after wake
        self._set_state("listening")

    # ------------------- public handlers -------------------

    async def handle_user_text(self, text: str) -> AsyncGenerator[dict, None]:
        """
        Text entry path (UI textbox). No wake gating required.
        """
        if not text:
            return
        # Emit user's text
        yield {"type": "user_text", "text": text}

        # Produce Orion reply (your existing LLM/skills logic goes here)
        reply_text = await self._llm_reply(text)

        # TTS
        audio_bytes = self.speech_engine.tts_generate(reply_text)
        self._set_state("speaking")
        yield {"type": "orion_reply", "text": reply_text}
        yield {"type": "audio", "audio": audio_bytes}

    async def handle_user_audio(self, audio_chunk: bytes) -> AsyncGenerator[dict, None]:
        """
        Streaming mic path. Completely locked until the wake word fires.
        """
        # gate everything until wake fired
        if not self.wake_triggered or self.state not in ("listening", "idle"):
            return

        out = self.speech_engine.transcribe_feed(audio_chunk)
        if out is None:
            return

        eng_text, lang = out  # tuple from engine.transcribe_feed
        if not eng_text:
            return

        # show user transcription
        yield {"type": "user_text", "text": eng_text}

        # very short blurbs often are filler; you can choose to ignore
        if len(eng_text.strip()) < 2:
            return

        # reply
        reply_text = await self._llm_reply(eng_text)
        audio_bytes = self.speech_engine.tts_generate(reply_text)

        self._set_state("speaking")
        yield {"type": "orion_reply", "text": reply_text}
        yield {"type": "audio", "audio": audio_bytes}

    def playback_finished(self):
        """
        Called from router when the browser finishes playing TTS audio.
        Resets back to idle and disarms the wake latch.
        """
        self.speech_engine.reset_feed()
        self.wake_triggered = False
        self._set_state("idle")

    # ------------------- your app brain -------------------

    async def _llm_reply(self, user_text: str) -> str:
        """
        Replace this stub with your actual response builder.
        Keep it async to avoid blocking the event loop.
        """
        # naive echo for now:
        return f"You said: {user_text}"
