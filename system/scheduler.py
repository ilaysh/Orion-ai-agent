# orion_core/brain/scheduler.py
import asyncio

class Scheduler:
    """
    A dumb metronome. It just sends a 'pulse' event to the Brain.
    It knows nothing about reminders, users, or memory.
    """
    def __init__(self, pulse_callback, interval=5):
        self.pulse_callback = pulse_callback # The Brain's function
        self.interval = interval
        self.active = False

    async def start(self):
        self.active = True
        print(f"[Scheduler] ⏱️ Pulse active (Interval: {self.interval}s).")
        while self.active:
            await asyncio.sleep(self.interval)
            # Notify the Brain that time has passed
            if self.active:
                await self.pulse_callback()

    def stop(self):
        self.active = False