import asyncio
import numpy as np
import time
from enum import Enum
from orion_core.vad import SileroVAD
from orion_core.wakeword import WakeWordThread


class State(Enum):
    IDLE = 1
    LISTEN = 2
    THINK = 3
    SPEAK = 4


class OrionCore:
    def __init__(self, model_path="models/orion_speechbrain_full_finetune.pt"):
        self.state = State.IDLE
        self.app_loop = None
        self.vad = SileroVAD()
        self.wake = WakeWordThread(model_path, self._on_wake_detect)
        self.wake.start()

        # Audio buffers for VAD and capture
        self._buf = np.zeros(0, dtype=np.float32)
        self._last_speech_t = 0.0
        self._sr = 16000
        self._end_silence_s = 1.0
        self._max_len_s = 15

        print("[Core] OrionCore initialized.")

    # ------------------------------------------------------------------ #
    #                        WAKE WORD CALLBACK
    # ------------------------------------------------------------------ #
    def _on_wake_detect(self):
        if self.state is not State.IDLE:
            return

        print("[Core] Wake-word detected → activating core")
        self.state = State.LISTEN
        self._ensure_loop()

        try:
            self.wake.disarm()
            self.wake._next_arm = time.time() + self.wake.cooldown_s

            # Notify UI
            if hasattr(self, "ws"):
                asyncio.run_coroutine_threadsafe(
                    self.ws.send_json({"type": "state", "state": "listening"}),
                    self.app_loop,
                )

            # Schedule async handling
            asyncio.run_coroutine_threadsafe(self._handle_wake(), self.app_loop)

        except Exception as e:
            print(f"[Core] Wake handling failed: {e}")
            self.state = State.IDLE
            if hasattr(self, "ws"):
                asyncio.run_coroutine_threadsafe(
                    self.ws.send_json({"type": "state", "state": "idle"}),
                    self.app_loop,
                )

    # ------------------------------------------------------------------ #
    #                        ASYNC PIPELINE
    # ------------------------------------------------------------------ #
    def _ensure_loop(self):
        try:
            self.app_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.app_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.app_loop)

    async def _handle_wake(self):
        print("[Core] 🧠 _handle_wake triggered")
        self.state = State.LISTEN
        await asyncio.sleep(0.2)

        audio = await self.capture_until_silence()
        text = await self.transcribe(audio)
        reply = await self.chat_reply(text)
        await self.speak(reply)

        self.state = State.IDLE
        self.wake.arm()
        print("[Core] Back to IDLE")

        # Push final state to UI
        if hasattr(self, "ws"):
            await self.ws.send_json({"type": "state", "state": "idle"})

    # ------------------------------------------------------------------ #
    #                        STUBS (replace later)
    # ------------------------------------------------------------------ #
    async def capture_until_silence(self, max_len_s=10, end_silence_s=1.0):
        print("[VAD] Capturing (stub, returning fake audio)...")
        await asyncio.sleep(2.0)
        return np.zeros(16000 * 2, dtype=np.float32)

    async def transcribe(self, audio):
        print("[STT] (stub) transcribing...")
        await asyncio.sleep(0.5)
        return "hello world"

    async def chat_reply(self, text):
        print(f"[Chat] You said: {text}")
        await asyncio.sleep(0.5)
        return f"Very good sir, you said {text}"

    async def speak(self, text):
        print("[TTS]", text)

        # --- send to client ---
        if hasattr(self, "ws"):
            try:
                await self.ws.send_json({
                    "type": "tts",
                    "text": text
                })
            except Exception as e:
                print(f"[Core] ⚠️ Could not send TTS to UI: {e}")

        await asyncio.sleep(1.0)  # keep delay for stub

    # ------------------------------------------------------------------ #
    #                        HELPER FUNCTIONS
    # ------------------------------------------------------------------ #
    def _is_speech(self, x: np.ndarray, th=0.005):
        if x.size == 0:
            return False
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        rms = np.sqrt(np.mean(np.clip(x, -1.0, 1.0) ** 2))
        return float(rms) > th

    async def handle_user_audio(self, audio_chunk: bytes):
        chunk = np.frombuffer(audio_chunk, dtype=np.float32)

        if self.state is not State.LISTEN:
            return

        self._buf = np.concatenate([self._buf, chunk])
        if self._is_speech(chunk):
            self._last_speech_t = time.time()

        too_long = len(self._buf) >= self._sr * self._max_len_s
        long_silence = (time.time() - self._last_speech_t) > self._end_silence_s

        if not (too_long or long_silence):
            return

        audio = self._buf.copy()
        self._buf = np.zeros(0, dtype=np.float32)
        self.state = State.THINK
        yield {"type": "state", "state": self.state.name}

        text = await self.transcribe(audio)
        yield {"type": "stt", "text": text}

        reply = await self.chat_reply(text)
        yield {"type": "reply", "text": reply}
        await self.speak(reply)

        self.state = State.IDLE
        self.wake.arm()
        print("[Core] Wake re-armed.")
        yield {"type": "state", "state": self.state.name}
