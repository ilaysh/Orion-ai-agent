# orion_core/wakeword_component.py
from __future__ import annotations
import asyncio
from datetime import datetime
from typing import Optional, Callable

import numpy as np

from orion_core.base_component import BaseComponent, ComponentState
from orion_core.wakeword import WakeWordDetector

class WakeWordComponent(BaseComponent):
    """
    Wrapper around WakeWordDetector that adds BUFFERING.
    The detector expects ~2 seconds of audio to run inference.
    We accumulate streaming chunks here until we have enough data.
    """
    name = "wakeword"

    def __init__(self, model_path: str = "models/orion_speechbrain_full_finetune.pt"):
        super().__init__()
        self.detector = WakeWordDetector(model_path=model_path, log_every_eval=False, debug_save_all=False)
        self.on_detect: Optional[Callable[[], None]] = None
        
        # RING BUFFER: Stores the last ~2 seconds of audio
        self._buffer = np.zeros(0, dtype=np.float32)
        self.window_size = 32000  # 2s @ 16kHz
        self.min_context = 16000  # Min 1s to start checking

    async def init(self):
        async with self._lock:
            self.state = ComponentState.INITIALIZING
            await self._report_state()
            try:
                self.state = ComponentState.READY
                await self._report_state()
                print("[WakeWord] ✅ Ready.")
            except Exception as e:
                self.state = ComponentState.ERROR
                await self._report_state()
                print(f"[WakeWord] ❌ Init failed: {e}")

    async def start(self):
        async with self._lock:
            if self.state != ComponentState.READY:
                await self.init()
            self.active = True
            self.detector.arm()
            await self._report_state(extra={"active": True})
            print("[WakeWord] ▶ Streaming mode active.")

    async def stop(self):
        async with self._lock:
            self.active = False
            self.state = ComponentState.STOPPED
            self.detector.disarm()
            await self._report_state(extra={"active": False})

    # --------- streaming entry ----------
    def feed(self, chunk: bytes | memoryview | bytearray | np.ndarray) -> bool:
        if not self.active or self.state != ComponentState.READY:
            return False
        
        # 1. Conversion: Ensure we have a flat float32 array
        if isinstance(chunk, (bytes, bytearray)):
            data = np.frombuffer(chunk, dtype=np.float32)
        elif isinstance(chunk, memoryview):
            data = np.frombuffer(chunk, dtype=np.float32)
        else:
            data = chunk # Assuming np.ndarray

        # 2. Accumulate: Add new chunk to our memory
        self._buffer = np.concatenate((self._buffer, data))

        # 3. Slide: Keep only the last 2 seconds (Sliding Window)
        if len(self._buffer) > self.window_size:
            self._buffer = self._buffer[-self.window_size:]

        # 4. Check: Do we have enough data to bother checking?
        # (Checking every 0.25s is fine, but we need at least 1s of context)
        if len(self._buffer) < self.min_context:
            return False
        
        # 5. Feed the FULL WINDOW to the detector
        # Now the detector receives 32000 samples, not 4096.
        fired = self.detector.feed(self._buffer)
        
        if fired:
            print(f"[WakeWord] 🔔 Triggered at {datetime.now().strftime('%H:%M:%S')}")
            self._buffer = np.zeros(0, dtype=np.float32) # Clear memory to prevent double-firing
            
            if self.on_detect:
                try:
                    self.on_detect()
                except Exception as e:
                    print(f"[WakeWord] ⚠️ on_detect() error: {e}")
        return fired

    def arm(self):
        self.detector.arm()

    def disarm(self):
        self.detector.disarm()
        # IMPORTANT: Clear buffer on disarm so we don't carry over old audio
        self._buffer = np.zeros(0, dtype=np.float32)