# orion_core/core.py
from __future__ import annotations
import asyncio, time
from enum import Enum, auto
from typing import Optional
from orion_core.base_component import BaseComponent
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

        # bind before start
        self.wakeword.on_detect = self._on_wake_detect_sync
        await self.wakeword.start()

        if self.engine.is_ready() and self.wakeword.is_ready():
            await self.events.put("[Init] ✅ All systems ready. Entering idle state.")
            self.state = CoreState.IDLE
            await self._broadcast_state("idle")
            print("[Core] ✅ All systems ready.")
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

    # ------------------------- wake integration -------------------------
    def _on_wake_detect_sync(self):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.handle_wake(), self.loop)
        else:
            print("[Core] ⚠️ event loop missing; cannot schedule handle_wake()")

    async def handle_wake(self):
        # strict single-command: only in IDLE
        if self.state != CoreState.IDLE:
            print(f"[Core] 🛑 Ignoring wake (busy: {self.state.name})")
            return
        if self._not_ready():
            await self.events.put("[Core] still initializing… please wait.")
            return

        print("[Core] Wake-word detected → activating core")
        self.state = CoreState.LISTEN
        await self._broadcast_state("listening")

        # mute wakeword while we listen/speak
        if self.wakeword and self.wakeword.thread:
            self.wakeword.thread.disarm()

        # capture + stt
        text = await self.engine.transcribe()
        if not text:
            await self.events.put("[Core] ⚠️ No speech detected.")
            self.state = CoreState.IDLE
            await self._broadcast_state("idle")
            if self.wakeword and self.wakeword.thread:
                self.wakeword.thread.arm()
            return

        await self.events.put(f"You said: {text}")

        self.state = CoreState.THINK
        await self._broadcast_state("thinking")
        reply = await self.chat_reply(text)
        await self.events.put(f"Orion: {reply}")

        self.state = CoreState.SPEAK
        await self._broadcast_state("speaking")
        audio_b64 = await self.engine.speak(reply)
        if audio_b64:
            await self.events.put({"type": "orion_reply", "text": reply, "audio": audio_b64})
        # remain SPEAK until playback_finished()

    # ------------------------- router path -------------------------
    async def handle_user_audio(self, audio_chunk: bytes | None):
        """
        Router calls this with raw mic chunks.
        SpeechEngine handles recording internally, so we ignore the chunk.
        """
        if self.state != CoreState.IDLE:
            print(f"[Core] 🛑 Ignoring user audio (busy: {self.state.name})")
            return

        self.state = CoreState.LISTEN
        await self._broadcast_state("listening")

        # also mute wakeword when manual audio comes in
        if self.wakeword and self.wakeword.thread:
            self.wakeword.thread.disarm()

        text = await self.engine.transcribe()  # ignore audio_chunk by design
        if not text:
            await self.events.put("[Core] ⚠️ No speech detected.")
            self.state = CoreState.IDLE
            await self._broadcast_state("idle")
            if self.wakeword and self.wakeword.thread:
                self.wakeword.thread.arm()
            return

        await self.events.put(f"You said: {text}")

        self.state = CoreState.THINK
        await self._broadcast_state("thinking")
        reply = await self.chat_reply(text)

        self.state = CoreState.SPEAK
        await self._broadcast_state("speaking")
        audio_b64 = await self.engine.speak(reply)
        if audio_b64:
            yield {"type": "orion_reply", "text": reply, "audio": audio_b64}
        # remain SPEAK until playback_finished()

    # ------------------------- playback end -------------------------
    def playback_finished(self):
        print("[Core] ▶ Playback finished.")
        self.state = CoreState.IDLE
        if self.wakeword and self.wakeword.thread:
            try:
                time.sleep(0.4)  # settle output audio
                self.wakeword.thread.arm()
                print("[Core] 🔁 Wakeword re-armed after playback.")
            except Exception as e:
                print(f"[Core] ⚠️ Failed to re-arm wakeword: {e}")

    # ------------------------- stubs -------------------------
    async def chat_reply(self, text: str) -> str:
        await asyncio.sleep(0.1)
        return f"Very good sir, you said {text}"

    # ------------------------- helpers -------------------------
    async def _broadcast_state(self, state: str):
        await self.events.put({"type": "state", "state": state})

    def _delayed_init(self):
        asyncio.create_task(self._on_init())

    # optional component helpers (kept for compatibility)
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
