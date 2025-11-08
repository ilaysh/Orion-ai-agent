# orion_core/core.py
from __future__ import annotations
import asyncio
import time
from enum import Enum, auto
from typing import Optional
from datetime import datetime
import numpy as np

from orion_core.base_component import BaseComponent
from orion_core.brain.brain import Brain
from orion_core.skills.skills import Skills
from orion_core.tts.engine import SpeechEngine
from orion_core.wakeword_component import WakeWordComponent


class CoreState(Enum):
    STARTUP = auto()
    INITIALIZING = auto()
    IDLE = auto()
    LISTEN = auto()
    THINK = auto()
    SPEAK = auto()
    ERROR = auto()
    STOPPED = auto()


class OrionCore:
    def __init__(self):
        print("[Core] 🚀 Orion initialization started (entered _on_init)")
        self.state = CoreState.STARTUP
        self.events: asyncio.Queue = asyncio.Queue()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.engine: Optional[SpeechEngine] = None
        self.wakeword: Optional[WakeWordComponent] = None
        self.llm = None
        self.db = None
        self.face = None
        self._pcm_buf = np.zeros(0, dtype=np.float32)
        self._pcm_target = 16000  # 1s at 16kHz
        self.skills = Skills()
        self.brain = Brain()

    # ------------------------- init -------------------------
    def _not_ready(self) -> bool:
        return (
            self.engine is None
            or not hasattr(self.engine, "is_ready")
            or not self.engine.is_ready()
        )

    async def _on_init(self):
        print("[Core] 🚀 Orion initialization started")
        self.loop = asyncio.get_running_loop()
        self.state = CoreState.INITIALIZING
        await self._broadcast_state("initializing")
        await self.events.put("[Init] Orion system initializing...")

        self.engine = SpeechEngine()
        self.wakeword = WakeWordComponent()
        self.engine.set_reporter(self._component_report)
        self.wakeword.set_reporter(self._component_report)
        await asyncio.gather(
            self._safe_init("speech", self.engine.init),
            self._safe_init("wakeword", self.wakeword.init),
        )
        await self.engine.start()

        self.wakeword.on_detect = self._on_wake_detect_sync
        await self.wakeword.start()

        if self.engine.is_ready() and self.wakeword.is_ready():
            await self.events.put("[Init] ✅ All systems ready. Entering idle state.")
            self.state = CoreState.IDLE
            await self._broadcast_state("idle")
            print("[Core] ✅ All systems ready.")
            print(f"[Core] 🕓 Ready at {time.strftime('%H:%M:%S')}")
        else:
            await self.events.put("[Init] ⚠️ Some subsystems failed to initialize.")
            self.state = CoreState.ERROR
            await self._broadcast_state("error")

    async def _safe_init(self, name: str, func):
        try:
            r = func()
            if asyncio.iscoroutine(r):
                await r
        except Exception as e:
            print(f"[Core] ⚠️ Failed to init {name}: {e}")

    async def _component_report(self, name: str, ready: bool, extra: dict):
        msg = f"[Init] {name.capitalize()}: {'ready ✅' if ready else 'initializing...'}"
        print(msg)
        await self.events.put(msg)

    # ------------------------- unified input handling -------------------------
    async def _process_input(self, text: str):
        """Common logic for both text and transcribed audio."""
        if not text:
            await self._broadcast_state("idle")
            return

        if self.state in (CoreState.THINK, CoreState.SPEAK):
            print("[Core] ⚠️ Busy, ignoring new input.")
            return

        try:
            self.state = CoreState.THINK
            await self._broadcast_state("thinking")
            reply = await self.brain.think(text)

            self.state = CoreState.SPEAK
            await self._broadcast_state("speaking")

            audio_b64 = None
            if self.engine and self.engine.is_ready():
                try:
                    audio_b64 = await self.engine.speak(reply)
                except Exception as e:
                    print(f"[Core] ⚠️ TTS error: {e}")

            payload = {"type": "orion_reply", "text": reply}
            if audio_b64:
                payload["audio"] = audio_b64

            await self.events.put(payload)
            print(f"[Core] 💭 {text[:80]} → {reply[:80]}")

        finally:
            self.state = CoreState.IDLE
            await self._broadcast_state("idle")
            if self.wakeword:
                self.wakeword.arm()

    async def handle_user_text(self, text: str):
        print(f"[Core] Received user text: {text}")
        await self._process_input(text)

    async def handle_user_audio(self, audio_chunk: bytes):
        if self.engine is None or self.wakeword is None:
            return

        # decode + wake-word check
        try:
            chunk = self._decode_pcm_f32_mono(audio_chunk)
        except Exception as e:
            print(f"[Core] ⚠️ PCM decode error: {e}")
            return
        if chunk.size == 0:
            return

        # if idle, feed wake-word
        if self.state == CoreState.IDLE:
            self._pcm_buf = np.concatenate([self._pcm_buf, chunk])
            if self._pcm_buf.size >= self._pcm_target:
                window = self._pcm_buf[-self._pcm_target:]
                if self.wakeword.feed(window):
                    print("[Core] 🔔 Wake word fired → switching to LISTEN mode")
                    self.state = CoreState.LISTEN
                    await self._broadcast_state("listening")
                    self.wakeword.disarm()
                self._pcm_buf = np.zeros(0, dtype=np.float32)
            return

        # if listening, transcribe then reuse shared path
        if self.state == CoreState.LISTEN:
            try:
                text = await self.engine.transcribe(chunk)
            except Exception as e:
                print(f"[Core] ⚠️ STT error: {e}")
                return
            if text:
                await self._process_input(text)

    # ------------------------- playback end -------------------------
    def playback_finished(self):
        print("[Core] ▶ Playback finished.")
        self.state = CoreState.IDLE
        try:
            if self.wakeword:
                self.wakeword.arm()
                print("[Core] 🔁 Wakeword re-armed after playback.")
        except Exception as e:
            print(f"[Core] ⚠️ Failed to re-arm wakeword: {e}")
        asyncio.create_task(self._broadcast_state("idle"))
        asyncio.create_task(self.events.put("[Core] 💤 Returning to idle."))

    # ------------------------- LLM / skills -------------------------
    async def chat_reply(self, text: str) -> str:
        # 1) intents → skills
        intent = self.intents.parse(text)
        if intent:
            if intent.name == "time.now":
                from datetime import datetime
                return f"The time is {datetime.now():%H:%M}, sir."
            if intent.name == "dashboard.open":
                try:
                    return self.skills.handle("open dashboard")
                except Exception:
                    return "I attempted to open the dashboard, sir."
            if intent.name == "image.generate":
                return "Image generation is wired but disabled in this mode, sir."
            if intent.name == "weather.get":
                return "Weather skill pending, sir."

        # 2) LLM fallback
        return self.brain.think(text)

    # ------------------------- helpers -------------------------
     # ------------------------- wake integration -------------------------
    def _on_wake_detect_sync(self):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.handle_wake(), self.loop)
        else:
            print("[Core] ⚠️ event loop missing; cannot schedule handle_wake()")

    async def _broadcast_state(self, state: str):
        await self.events.put({"type": "state", "state": state})

    def _delayed_init(self):
        asyncio.create_task(self._on_init())

    async def start_component(self, name: str):
        comp = self._get_component(name)
        if comp:
            await comp.start()

    async def stop_component(self, name: str):
        comp = self._get_component(name)
        if comp:
            await comp.stop()

    def _get_component(self, name: str) -> Optional[BaseComponent]:
        lookup = {"speech": self.engine, "wakeword": self.wakeword}
        return lookup.get(name)

    def _decode_pcm_f32_mono(self, audio_chunk: bytes) -> np.ndarray:
        if audio_chunk is None or len(audio_chunk) == 0:
            return np.zeros(0, dtype=np.float32)
        try:
            x = np.frombuffer(audio_chunk, dtype=np.float32)
            if x.size > 0 and np.isfinite(x).all() and np.max(np.abs(x)) <= 1.5:
                return x.astype(np.float32, copy=False)
        except Exception:
            pass
        try:
            i16 = np.frombuffer(audio_chunk, dtype=np.int16)
            if i16.size == 0:
                return np.zeros(0, dtype=np.float32)
            x = i16.astype(np.float32) / 32768.0
        except Exception:
            return np.zeros(0, dtype=np.float32)
        if x.ndim != 1:
            x = x.reshape(-1)
        x = np.clip(x, -1.0, 1.0)
        return x

    def _guess_sample_rate(length: int) -> int:
        if 40000 < length < 52000:
            return 48000
        return 16000
