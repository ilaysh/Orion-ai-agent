# orion_core/core.py
from __future__ import annotations
import asyncio
import contextlib
import signal
import numpy as np
from enum import Enum, auto
from typing import Optional
from orion_core.brain.llm.hearing_engine import get_hearing_engine

# --- CORE PARADIGM IMPORTS ---
from orion_core.brain.brain import Brain
from orion_core.intent_queue import IntentQueue
from orion_core.skills.skills import Skills
from orion_core.tts.engine import SpeechEngine
from orion_core.wakeword_component import WakeWordComponent
from system.scheduler import Scheduler
from system.message_bus import global_bus
from system.telemetry.telemetry import timed
from orion_core.wakeword_component import WakeWordComponent
from system.message_bus import global_bus


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
        print("[Core] 🚀 Orion initialization (ctor)")
        self.state = CoreState.STARTUP
        self.events: asyncio.Queue = asyncio.Queue()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        self.engine: Optional[SpeechEngine] = None
        self.wakeword: Optional[WakeWordComponent] = None
        self.brain: Optional[Brain] = None
        self.scheduler: Optional[Scheduler] = None

        self.intent_queue = IntentQueue()
        self._tasks: list[asyncio.Task] = []

        # --- Speech serialization (fixes the double-speech race) ---
        self._speak_lock: Optional[asyncio.Lock] = None
        self._last_spoken_norm = ""      # last FINAL utterance
        # Non-blocking speech: callers enqueue, a single worker plays serially. This
        # stops the graph freezing during playback (the old speak() awaited full audio
        # playback, so brain.think stalled for the whole spoken sentence).
        self._speech_queue: Optional["asyncio.Queue"] = None
        self._last_thought_norm = ""     # last interstitial THOUGHT bubble

    async def start(self, run_init: bool = True):
        try:
            self.loop = asyncio.get_running_loop()
            self._speak_lock = asyncio.Lock()
            self._speech_queue = asyncio.Queue()
            self._tasks.append(self.loop.create_task(self._speech_worker()))
            if run_init:
                await self._startup_sequence()

            print("[Core] 🚌 All components ready. Connecting to Global Bus...")
            global_bus.subscribe("intent", self._on_intent)

            self._tasks.append(self.loop.create_task(global_bus.run()))

            # --- 2. INTENT WORKER ---
            self._tasks.append(self.loop.create_task(self._intent_worker()))
            self._tasks.append(self.loop.create_task(self._scheduler_worker()))

            print("[Core] ▶ System fully operational.")
        except Exception as e:
            print(f"[Core] ⚠️ Error starting core: {e}")
            raise
    
    async def _startup_sequence(self):
        print("[Core] ⏳ Initializing Internal Systems...")
        self.state = CoreState.INITIALIZING

        async def _bubble_thought_callback(text: str):
            """Interstitial 'thinking' narration — a DIFFERENT channel from the
            final response, with its own dedup so it can never collide with or
            double the final answer."""
            if not text or not self.engine:
                return
            clean = text.strip()
            norm = "".join(c for c in clean.lower() if c.isalnum())
            if not norm or norm == self._last_thought_norm:
                return
            self._last_thought_norm = norm

            print(f"[Core] 🫧 Bubbling thought: {clean}")
            await self.events.put({"type": "thought", "text": clean})
            await self.speak(clean, is_thought=True)

        try:
            print("[Core] 🧠 Init Brain...")
            self.brain = Brain(bubble_thought=_bubble_thought_callback)
            await self.brain.init()

            hearing_engine = get_hearing_engine()
            await hearing_engine.load()
            self.engine = SpeechEngine(hearing_engine)
            self.wakeword = WakeWordComponent()
            await asyncio.gather(self.engine.init(), self.wakeword.init())

            print("[Core] 📅 Scheduler Online.")
            self.scheduler = Scheduler()

            self.wakeword.on_detect = self._on_wake_detect_sync
            await self.wakeword.start()

            if self.engine:
                print("[Core] 🌊 Starting Passive Audio Stream...")
                await self.loop.run_in_executor(
                    None, self.engine.start_passive_stream,
                    self._handle_audio_stream_sync
                )

            self.state = CoreState.IDLE
            print("[Core] ✅ Startup Sequence Complete.")
        except Exception as e:
            print(f"[Core] 💥 Initialization Critical Failure: {e}")
            self.state = CoreState.ERROR
            raise

    @timed("Core.ProcessInput")
    async def _process_input(self, text: str, source: str = "user"):
        if not text:
            return
        if self.state in [CoreState.STARTUP, CoreState.INITIALIZING]:
            print(f"[Core] ⚠️ Input ignored; still initializing: {text}")
            return

        # Reset the thought channel at the start of each new turn.
        self._last_thought_norm = ""

        self.state = CoreState.THINK
        await self._broadcast_state("thinking")

        reply = await self.brain.think(text)
        await self._handle_brain_output(reply)
        self.playback_finished()

    @timed("Core.HandleOutput")
    async def _handle_brain_output(self, content: str):
        if not content:
            return
        print("[Core] 🧠 Brain returned final response.")
        await self.speak(content, is_thought=False)

    async def handle_user_audio(self, audio_chunk: bytes):
        """Entry point for Audio Blob from UI (Mic button)."""
        # Future: Decode bytes -> np.array -> Transcribe
        print(f"[Core] 📩 UI Audio Input (Not implemented yet)")
        pass

    def _handle_audio_stream_sync(self, chunk: np.ndarray):
        if self.loop:
            self.loop.call_soon_threadsafe(self._process_audio_chunk, chunk)

    def _process_audio_chunk(self, chunk: np.ndarray):
        if self.state == CoreState.IDLE and self.wakeword:
            self.wakeword.feed(chunk)

    def _on_wake_detect_sync(self):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.handle_wake(), self.loop)

    async def handle_wake(self):
        print("[Core] 🔔 Wake detected. Switching to Active Listen.")
        self.state = CoreState.LISTEN
        await self._broadcast_state("listening")
        if self.wakeword:
            self.wakeword.disarm()
        text = await self.engine.listen(timeout=10.0)
        if text:
            await self._process_input(text)
        else:
            self.playback_finished()

    async def _process_input(self, text: str):
        if not text: return
        clean = text.lower().strip()
        if clean in ["orion", "hey orion", "hi orion"]:
             print(f"[Core] 🤝 Greeting only.")
             await self.speak("Yes sir?")
             self.playback_finished()
             return

        self.state = CoreState.THINK
        await self._broadcast_state("thinking")
        
        reply = await self.brain.think(text)
        print(f"[Core] 🧠 user text:{text} \n Reply: {reply}")
        if reply: await self.speak(str(reply))
        
        self.playback_finished()

    def playback_finished(self):
        self.state = CoreState.IDLE
        if self.wakeword:
            self.wakeword.arm()

    async def speak(self, text: str, is_thought: bool = False):
        """Enqueue speech and return IMMEDIATELY — never blocks the caller. This is
        what stops the graph freezing while Orion talks. A single worker plays items
        one at a time, so utterances never overlap and their order is preserved
        (an acknowledgement queued before the final answer is heard first)."""
        if not text or not self.engine or self._speech_queue is None:
            return
        norm = "".join(c for c in text.lower() if c.isalnum())
        if not norm:
            return
        await self._speech_queue.put((text, is_thought, norm))

    async def _speech_worker(self):
        """Owns playback. Generate → display → play-to-completion, one utterance at a
        time. Holds SPEAK state while the queue is active (so the wakeword can't hear
        Orion's own voice) and returns to IDLE only once the queue drains."""
        while True:
            text, is_thought, norm = await self._speech_queue.get()
            try:
                if norm == self._last_spoken_norm:
                    continue  # dedup a repeat of the last final utterance
                if not is_thought:
                    self._last_spoken_norm = norm

                self.state = CoreState.SPEAK
                await self._broadcast_state("speaking")

                result = await self.engine.speak(text)
                if not result:
                    continue

                # Display clients get TEXT ONLY — no audio payload. Python is the sole
                # TTS engine; sending the b64 audio made the browser ALSO play it,
                # which is the duplicated-first-response you heard.
                await self.events.put({
                    "type": "speak", "text": text, "audio": None, "role": "assistant",
                })

                samples = result.get("samples")
                sr = result.get("sample_rate", 24000)
                if samples is not None:
                    try:
                        await self.loop.run_in_executor(None, self._play_blocking, samples, sr)
                    except Exception as e:
                        print(f"[Core] ⚠️ Playback error: {e}")
            except Exception as e:
                print(f"[Core] ⚠️ Speech worker error: {e}")
            finally:
                self._speech_queue.task_done()
                # Back to IDLE only when nothing else is queued, so the wakeword stays
                # gated across a burst (e.g. acknowledgement … progress … final).
                if self._speech_queue.empty():
                    self.state = CoreState.IDLE
                    await self._broadcast_state("idle")

    def _play_blocking(self, samples, sample_rate):
        """Synchronous playback; runs in an executor thread so the event loop
        isn't frozen while Orion speaks. sd.wait() returns only when audio ends."""
        try:
            import sounddevice as sd
            sd.play(samples, sample_rate)
            sd.wait()
        except Exception as e:
            print(f"[Core] ⚠️ sounddevice playback failed: {e}")

    async def _scheduler_worker(self):
        while True:
            await asyncio.sleep(1.0)
            if self.scheduler and self.state == CoreState.IDLE:
                try:
                    for task in self.scheduler.check_schedule():
                        # Simple reminders speak directly — no LangGraph pipeline.
                        if task.get("action", "notify") == "notify":
                            self._last_spoken_norm = ""  # allow the reminder through
                            await self.speak(f"Sir, a reminder: {task['data']}",
                                             is_thought=False)
                        else:
                            # Computational/insight jobs go through the brain.
                            prompt = f"[SYSTEM EVENT]: {task['data']}"
                            await self._process_input(prompt, source="system")
                except Exception as e:
                    print(f"[Core] ⚠️ Scheduler Error: {e}")

        # 2. Generate Audio
        audio_base64 = await self.engine.speak(text)
        
        # 3. Send Audio to UI
        if audio_base64:
            await self.events.put({"type": "speak", "audio": audio_base64})

    # ------------------------- HELPERS -------------------------
    def display_text(self, text: str):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.events.put({"type": "display", "text": text}), self.loop)

    async def _intent_worker(self):
        while True:
            intent = await self.intent_queue.next_intent()
            if intent:
                t = intent.get("type")
                data = intent.get("data", {})
                if t == "speak": await self.speak(data.get("text", ""))
                elif t == "display": self.display_text(data.get("text", ""))
            else:
                await asyncio.sleep(0.1) 

    async def handle_user_text(self, text: str):
        if self.brain and self.brain.session:
            self.brain.session.set_speaker("ilay")
        await self._process_input(text)

    async def handle_user_audio(self, audio_bytes: bytes):
        """Transcribe an inbound audio blob (from the UI) and process as text.
        NOTE: confirm your SpeechEngine's transcription method name — this assumes
        `engine.transcribe(bytes) -> str`. Adjust if yours differs (e.g. .listen_bytes)."""
        if not audio_bytes or not self.engine:
            return
        try:
            transcribe = getattr(self.engine, "transcribe", None)
            if transcribe is None:
                print("[Core] ⚠️ SpeechEngine has no transcribe(); audio path disabled.")
                return
            text = await transcribe(audio_bytes) if asyncio.iscoroutinefunction(transcribe) \
                else await self.loop.run_in_executor(None, transcribe, audio_bytes)
        except Exception as e:
            print(f"[Core] ⚠️ Transcription failed: {e}")
            return
        if text:
            await self.handle_user_text(text)

    async def shutdown(self):
        """Graceful async shutdown for the router lifespan hook."""
        print("[Core] 🔻 Shutdown requested via lifespan...")
        self.state = CoreState.STOPPED
        try:
            from system.telemetry.telemetry import telemetry_summary
            summary = telemetry_summary()
            if summary:
                print(summary)
        except Exception as e:
            print(f"[Core] ⚠️ Telemetry dump failed: {e}")

        for task in self._tasks:
            task.cancel()
        if self.brain:
            with contextlib.suppress(Exception):
                await self.brain.close()
        print("[Core] 🔻 Shutdown complete.")

    async def _on_intent(self, data):
        pass

    async def _broadcast_state(self, state):
        await self.events.put({"type": "state", "state": state})

    def run(self):
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)
        print("[Core] 🟢 Orion is Online. Press Ctrl+C to stop.")
        self.loop = asyncio.new_event_loop()
        try:
            self.loop.run_until_complete(self.start())
            self.loop.run_forever()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            self.stop()

    def _handle_exit(self, signum, frame):
        print("[Core] 🛑 Shutting down gracefully...")
        try:
            from system.telemetry.telemetry import telemetry_summary
            summary = telemetry_summary()
            if summary:
                print(summary)
        except Exception as e:
            print(f"[Core] ⚠️ Could not dump telemetry: {e}")

        if self.brain:
            asyncio.run_coroutine_threadsafe(self.brain.close(), self.loop)
        import os
        os._exit(0)

    def stop(self):
        print("[Core] 🔻 Engine stopped.")