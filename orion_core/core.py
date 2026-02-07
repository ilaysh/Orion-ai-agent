# orion_core/core.py
from __future__ import annotations
import asyncio
import time
from enum import Enum, auto
from typing import Optional
from datetime import datetime
import numpy as np
import signal
import sys
from orion_core.base_component import BaseComponent
from orion_core.brain.brain import Brain
from orion_core.intent_queue import IntentQueue
from orion_core.skills.skills import Skills
from orion_core.tts.engine import SpeechEngine
from system.scheduler import Scheduler
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
        self.brain = None 
        self.intent_queue = IntentQueue()
        self._tasks: list[asyncio.Task] = []
        self.scheduler = None

    async def start(self, run_init: bool = True):
        try:
            self.loop = asyncio.get_running_loop()
            
            # --- 1. GLOBAL BUS ENABLED ---
            print("[Core] 🚌 Connecting to Global Bus...")
            global_bus.subscribe("intent", self._on_intent)
            global_bus.subscribe("status", self._on_status)
            self._tasks.append(self.loop.create_task(global_bus.run()))

            # --- 2. INTENT WORKER ---
            self._tasks.append(self.loop.create_task(self._intent_worker()))

            if run_init:
                self._tasks.append(self.loop.create_task(self._startup_sequence()))

            # --- 3. SCHEDULER ENABLED ---
            print("[Core] 📅 Starting Scheduler...")
            self.scheduler = Scheduler(pulse_callback=self._on_time_tick, interval=30)
            self._tasks.append(self.loop.create_task(self.scheduler.start()))

            print("[Core] ▶ Started background tasks")

        except Exception as e:
            print(f"[Core] ⚠️ Error starting core: {e}")
            raise
    
    async def _startup_sequence(self):
        print("[Core] ⏳ Startup Sequence Begin...")
        await self._on_init()
        await asyncio.sleep(2.0)
        
        if self.engine:
            print("[Core] 🌊 Starting Passive Stream (Background)...")
            await self.loop.run_in_executor(
                None, 
                self.engine.start_passive_stream, 
                self._handle_audio_stream_sync
            )

    async def _on_init(self):
        print("[Core] 🚀 Orion initialization started")
        self.state = CoreState.INITIALIZING
        self._last_spoken_text = None
        
        async def _bubble_thought_callback(text: str):
            if text == self._last_spoken_text: return
            print(f"[Core] 🛁 Bubbling thought: {text}")
            self._last_spoken_text = text
            await self.intent_queue.enqueue({"type": "speak", "data": {"text": text, "priority": 0.8}})

        try:
            print("[Core] 🧠 Init Brain...")
            self.brain = Brain(bubble_thought=_bubble_thought_callback)
            await self.brain.init()

            print("[Core] 👂 Init Engine...")
            shared_ears = self.brain.cortex.get_shared_ears()
            self.engine = SpeechEngine(shared_model=shared_ears)
            self.wakeword = WakeWordComponent()

            await asyncio.gather(
                self.engine.init(),
                self.wakeword.init(),
            )

            print("[Core] 🔔 Init Wakeword...")
            self.wakeword.on_detect = self._on_wake_detect_sync
            await self.wakeword.start()

            self.state = CoreState.IDLE
            print("[Core] ✅ All systems ready.")

        except Exception as e:
            print(f"[Core] 💥 Fatal Init Error: {e}")
            self.state = CoreState.ERROR

    # --- PUBLIC HANDLERS (Fixes AttributeError) ---
    async def handle_user_text(self, text: str):
        """Entry point for Text Chat from UI."""
        print(f"[Core] 📩 UI Text Input: {text}")
        await self._process_input(text)

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
        if self.loop: asyncio.run_coroutine_threadsafe(self.handle_wake(), self.loop)

    async def handle_wake(self):
        print("[Core] 🔔 Wake detected. Switching to Active Listen.")
        self.state = CoreState.LISTEN
        await self._broadcast_state("listening")
        
        if self.wakeword: self.wakeword.disarm()

        # Increased timeout happens inside engine.listen(), but we handle null result here
        text = await self.engine.listen(timeout=30.0) # Passed explicitly
        
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
        if self.wakeword: self.wakeword.arm()

    async def speak(self, text: str):
        self.state = CoreState.SPEAK
        await self._broadcast_state("speaking")
        
        # 1. Send text to UI first (Visual feedback)
        await self.events.put({
            "type": "speak", 
            "text": text,
            "role": "assistant"
        })

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

    async def _on_intent(self, data): pass
    async def _on_status(self, data): pass
    async def _on_time_tick(self): pass
    
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
        except (KeyboardInterrupt, SystemExit): pass
        finally: self.stop()

    def _handle_exit(self, signum, frame):
        print("\n[Core] 🛑 FORCE SHUTDOWN.")
        import os
        os._exit(0)

    def stop(self):
        self.running = False
    
    async def shutdown(self):
        print("[Core] 🔻 Shutdown started")
        self.state = CoreState.STOPPED
        if self.wakeword: await self.wakeword.stop()
        if self.engine: await self.engine.stop()
        if self.scheduler: await self.scheduler.stop()
        for t in self._tasks: t.cancel()
        print("[Core] ✅ Shutdown complete")