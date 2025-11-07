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
    Thin wrapper around WakeWordDetector that:
      - exposes async init/start/stop like other components
      - supports streaming .feed(...)
      - calls on_detect() once when fired
    """
    name = "wakeword"

    def __init__(self, model_path: str = "models/orion_speechbrain_full_finetune.pt"):
        super().__init__()
        self.detector = WakeWordDetector(model_path=model_path,log_every_eval=False, debug_save_all=False)
        self.on_detect: Optional[Callable[[], None]] = None

    async def init(self):
        async with self._lock:
            self.state = ComponentState.INITIALIZING
            await self._report_state()
            try:
                # optional quick calibration (streaming approx)
                # self.detector.calibrate(noise_seconds=2.0)
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
        
        fired = self.detector.feed(chunk)
        #log current time
        print(f"[WakeWord] 🕓 Feed evaluated at {datetime.now().strftime('%H:%M:%S')}")
        if fired and self.on_detect:
            try:
                self.on_detect()
            except Exception as e:
                print(f"[WakeWord] ⚠️ on_detect() error: {e}")
        return fired

    def arm(self):
        self.detector.arm()

    def disarm(self):
        self.detector.disarm()
