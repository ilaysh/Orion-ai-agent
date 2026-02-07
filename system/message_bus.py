# orion_core/system/message_bus.py
"""
MessageBus — central async event and thought queue for Orion.
Allows all components (Core, Brain, Thinker, Monitor) to publish and subscribe
without direct dependencies between them.
"""

import asyncio
from collections import defaultdict


class MessageBus:
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.queue = asyncio.Queue()

    # --------------------------------------------------------------
    def subscribe(self, topic: str, callback):
        """Register callback for a specific topic."""
        self.subscribers[topic].append(callback)
        print(f"[Bus] 📡 Subscribed to '{topic}'")

    async def publish(self, topic: str, data=None):
        """Publish an event (asynchronously queued)."""
        await self.queue.put((topic, data))
        print(f"[Bus] 📨 Event '{topic}' queued.")

    # --------------------------------------------------------------
    async def run(self):
        """Main dispatcher loop."""
        print("[Bus] 🧠 Dispatcher started.")
        while True:
            topic, data = await self.queue.get()
            if topic in self.subscribers:
                for cb in self.subscribers[topic]:
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            await cb(data)
                        else:
                            cb(data)
                    except Exception as e:
                        print(f"[Bus] ⚠️ Error dispatching {topic}: {e}")
            else:
                print(f"[Bus] ⚠️ No subscribers for topic '{topic}'")


global_bus = MessageBus()
# create a single global instance for the app to import/use
