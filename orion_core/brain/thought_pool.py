# orion_core/brain/thought_pool.py
"""
ThoughtPool — inner short-term memory of Orion's mind.
Collects low-level thoughts from modules (Thinker, Monitor, etc.)
before they are consciously processed by the Brain.
"""

import asyncio
from collections import deque


class ThoughtPool:
    """Shared pool of inner thoughts and events before being raised to consciousness."""

    def __init__(self, max_size=50):
        self.queue = deque(maxlen=max_size)
        self.lock = asyncio.Lock()

    async def add(self, source: str, text: str, importance: float = 0.5):
        """Store a new thought with metadata."""
        async with self.lock:
            thought = {
                "source": source,
                "text": text.strip(),
                "importance": importance,
            }
            self.queue.append(thought)
            print(f"[ThoughtPool] 🧩 Added ({source}): {text[:60]}")

    async def get_recent(self, n: int = 5):
        """Return N most recent thoughts."""
        async with self.lock:
            return list(self.queue)[-n:]

    async def pop_important(self, threshold: float = 0.7):
        """Extract thoughts above a certain importance threshold."""
        async with self.lock:
            important = [t for t in self.queue if t["importance"] >= threshold]
            self.queue = deque(
                [t for t in self.queue if t not in important], maxlen=self.queue.maxlen)
            return important
