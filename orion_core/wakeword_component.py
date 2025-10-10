# orion_core/components/wakeword_component.py
import asyncio
from orion_core.base_component import BaseComponent, ComponentState
from orion_core.wakeword import WakeWordThread


class WakeWordComponent(BaseComponent):
    name = "wakeword"

    def __init__(self):
        super().__init__()
        self.on_detect = None
        self.thread = None
        
    async def init(self):
        async with self._lock:
            self.state = ComponentState.INITIALIZING
            await self._report_state()
            print("[WakeWord] Loading model...")
            try:
                await asyncio.sleep(1.0)  # simulate warm-up
                self.thread = WakeWordThread(
                    "models/orion_speechbrain_full_finetune.pt", 
                    
                    lambda: self.on_detect() if self.on_detect else None
                )
                self.state = ComponentState.READY
                print("[WakeWord] ✅ Ready.")
            except Exception as e:
                self.state = ComponentState.ERROR
                print(f"[WakeWord] ❌ Init failed: {e}")
            await self._report_state()

    async def start(self):
        async with self._lock:
            if self.state is not ComponentState.READY:
                await self.init()
            self.thread.start()
            self.active = True
            print("[WakeWord] ▶ Listening started.")
            await self._report_state(extra={"active": True})

    async def stop(self):
        async with self._lock:
            if self.active and hasattr(self, "thread"):
                self.thread.stop()
            self.active = False
            self.state = ComponentState.STOPPED
            print("[WakeWord] ⏹ Stopped.")
            await self._report_state(extra={"active": False})
