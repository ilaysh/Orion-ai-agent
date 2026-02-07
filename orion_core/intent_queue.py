# orion_core/intent_queue.py
import asyncio
from typing import Optional, Dict, Any

class IntentQueue:
    """
    Async Queue for managing System Intents.
    Uses asyncio.Queue for robust producer-consumer signaling.
    """
    def __init__(self):
        self._queue = asyncio.Queue()

    async def enqueue(self, intent: Dict[str, Any]):
        """Add intent to the queue."""
        # Simple FIFO for now to ensure stability. 
        # (Priorities can be re-added later if needed, but stability first)
        await self._queue.put(intent)

    async def next_intent(self) -> Optional[Dict[str, Any]]:
        """
        Waits efficiently for the next intent.
        """
        return await self._queue.get()

    def empty(self) -> bool:
        return self._queue.empty()