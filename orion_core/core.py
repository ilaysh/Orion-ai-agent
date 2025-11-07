# orion_core/core.py
from __future__ import annotations
import asyncio
import time
from enum import Enum, auto
from typing import Optional
from datetime import datetime
import numpy as np
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
        self._pcm_buf = np.zeros(0, dtype=np.float32)
        self._pcm_target = 16000   # 1 second at 16kHz

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

        # bind before start
        self.wakeword.on_detect = self._on_wake_detect_sync
        await self.wakeword.start()

        if self.engine.is_ready() and self.wakeword.is_ready():
            await self.events.put("[Init] ✅ All systems ready. Entering idle state.")
            self.state = CoreState.IDLE
            await self._broadcast_state("idle")
            print("[Core] ✅ All systems ready.")
            # log current time
            [print(f"[Core] 🕓 Ready at {time.strftime('%H:%M:%S')}")]

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

    async def handle_user_text(self, text: str):
        # simple stub for now
        print(f"[Core] Received user text: {text}")
        return {"type": "orion_reply", "text": f"Very good sir, you said: {text}"}

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
        if self.wakeword:
            self.wakeword.disarm()

        # capture + stt
        text = await self.engine.transcribe()  # None = use internal capture
        if not text:
            await self.events.put("[Core] ⚠️ No speech detected.")
            self.state = CoreState.IDLE
            await self._broadcast_state("idle")
            if self.wakeword:
                self.wakeword.arm()
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
    # inside orion_core/core.py

    async def handle_user_audio(self, audio_chunk: bytes):
        if self.engine is None or self.wakeword is None:
            return

        if self.state not in (CoreState.IDLE, CoreState.LISTEN):
            return

        # Decode to float32 mono
        try:
            chunk = self._decode_pcm_f32_mono(audio_chunk)
        except Exception as e:
            print(f"[Core] ⚠️ PCM decode error: {e}")
            return
        if chunk.size == 0:
            return

        # Auto-resample if 48kHz
        def _guess_sample_rate(length: int) -> int:
            if 40000 < length < 52000:
                return 48000
            return 16000

        sr_guess = _guess_sample_rate(len(chunk))
        if sr_guess == 48000:
            try:
                import librosa
                chunk = librosa.resample(chunk, orig_sr=48000, target_sr=16000)
            except Exception as e:
                print(f"[Core] ⚠️ Resample failed: {e}")

        # --------------------------- IDLE → WAKEWORD ---------------------------
        if self.state == CoreState.IDLE:
            if not hasattr(self, "_pcm_buf"):
                self._pcm_buf = np.zeros(0, dtype=np.float32)
            if not hasattr(self, "_pcm_target"):
                self._pcm_target = 16000

            self._pcm_buf = np.concatenate([self._pcm_buf, chunk])
            if self._pcm_buf.size >= self._pcm_target:
                window = self._pcm_buf[-self._pcm_target:]
                fired = self.wakeword.feed(window)
                self._pcm_buf = np.zeros(0, dtype=np.float32)

                if fired:
                    print("[Core] 🔔 Wake word fired → switching to LISTEN mode")
                    self.state = CoreState.LISTEN
                    await self._broadcast_state("listening")
                    self.wakeword.disarm()
            return

        # --------------------------- LISTEN → STT ---------------------------
        if self.state == CoreState.LISTEN:
            try:
                text = await self.engine.transcribe(chunk)
            except Exception as e:
                print(f"[Core] ⚠️ STT error: {e}")
                return

            # ✅ Case 1: final text returned
            if text:
                await self.events.put({"type": "user_text", "text": text})

                self.state = CoreState.THINK
                await self._broadcast_state("thinking")

                reply = await self.chat_reply(text)
                await self.events.put({"type": "orion_reply", "text": reply})

                self.state = CoreState.SPEAK
                await self._broadcast_state("speaking")

                audio_b64 = await self.engine.speak(reply)
                payload = {"type": "orion_reply", "text": reply}
                if audio_b64:
                    payload["audio"] = audio_b64
                await self.events.put(payload)

                # ✅ After speaking → return to idle and arm wakeword
                self.state = CoreState.IDLE
                await self._broadcast_state("idle")
                self.wakeword.arm()
                return

            # ✅ Case 2: engine ended stream but no text (silence cutoff)
            if not self.engine._stream_active:
                print("[Core] 🤫 No speech detected — back to IDLE")
                self.state = CoreState.IDLE
                await self._broadcast_state("idle")
                self.wakeword.arm()
                return

            # ✅ Case 3: no final text yet → let engine accumulate time
            await asyncio.sleep(0)
            return

    # ------------------------- playback end -------------------------

    def playback_finished(self):
        print("[Core] ▶ Playback finished.")
        self.state = CoreState.IDLE

        try:
            if self.wakeword:
                # Use the new WakeWordComponent interface
                self.wakeword.arm()
                print("[Core] 🔁 Wakeword re-armed after playback.")
        except Exception as e:
            print(f"[Core] ⚠️ Failed to re-arm wakeword: {e}")

        asyncio.create_task(self._broadcast_state("idle"))
        asyncio.create_task(self.events.put("[Core] 💤 Returning to idle."))

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

    def _decode_pcm_f32_mono(self, audio_chunk: bytes) -> np.ndarray:
        """
        Return float32 mono in [-1, 1]. 100% safe, no ambiguous truth-value checks.
        Tries Float32 first, fallback to Int16 if invalid.
        """
        # audio_chunk may be bytes or empty
        if audio_chunk is None or len(audio_chunk) == 0:
            return np.zeros(0, dtype=np.float32)

        # ---- Try float32 ----
        try:
            x = np.frombuffer(audio_chunk, dtype=np.float32)
            # Validate: non-empty, finite, values in sane range
            if x.size > 0 and np.isfinite(x).all() and np.max(np.abs(x)) <= 1.5:
                # Already float32 mono
                return x.astype(np.float32, copy=False)
        except Exception:
            pass  # fall through to int16

        # ---- Fallback: int16 → float32 ----
        try:
            i16 = np.frombuffer(audio_chunk, dtype=np.int16)
            if i16.size == 0:
                return np.zeros(0, dtype=np.float32)
            x = i16.astype(np.float32) / 32768.0
        except Exception:
            # worst case
            return np.zeros(0, dtype=np.float32)

        # x is float32 now. Ensure 1D.
        if x.ndim != 1:
            x = x.reshape(-1)

        # (Optional) Clamp extremes
        x = np.clip(x, -1.0, 1.0)

        return x

    def _guess_sample_rate(length: int) -> int:
        """Roughly infer if chunk is 16 kHz or 48 kHz based on size."""
        # Orion wakeword buffers 1 s = 16000 samples normally
        # If a chunk looks ~3× larger, assume 48 kHz.
        if 40000 < length < 52000:
            return 48000
        return 16000
