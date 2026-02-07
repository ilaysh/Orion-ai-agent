# orion_core/system/event_bus.py
import asyncio
from collections import defaultdict


class EventBus:
    """Lightweight async event system for Orion."""

    def __init__(self):
        self._subscribers = defaultdict(list)
        self._queue = asyncio.Queue()

    def on(self, event_name: str, callback):
        """Register a coroutine or function to an event."""
        self._subscribers[event_name].append(callback)

    async def emit(self, event_name: str, data=None):
        """Put an event into the async queue."""
        await self._queue.put((event_name, data))

    async def _dispatch_loop(self):
        """Internal loop that dispatches queued events."""
        while True:
            event_name, data = await self._queue.get()
            for cb in self._subscribers.get(event_name, []):
                try:
                    if asyncio.iscoroutinefunction(cb):
                        await cb(data)
                    else:
                        cb(data)
                except Exception as e:
                    print(f"[EventBus] ⚠️ Error handling '{event_name}': {e}")
