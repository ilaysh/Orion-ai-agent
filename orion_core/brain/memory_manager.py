# orion_core/skills/memory_manager.py
"""
Orion Skill — Memory Manager
Monitors GPU/CPU memory and handles heavy-task preemption.
"""
from system.message_bus import global_bus
import asyncio
import psutil
import subprocess
import torch
import time


class MemoryManager:
    def __init__(self):
        self.gpu_supported = torch.cuda.is_available()
        self.warn_threshold_vram = 0.90   # 90%
        self.warn_threshold_ram = 0.90    # 90%
        self.last_status = None

    # ------------------------------------------------------------------
    def get_gpu_memory(self):
        """Return (used, total, percent) of GPU VRAM if available."""
        if not self.gpu_supported:
            return (0, 0, 0)
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            reserved = torch.cuda.memory_reserved(0)
            allocated = torch.cuda.memory_allocated(0)
            used = reserved if reserved > allocated else allocated
            return used, total, used / total
        except Exception:
            return (0, 0, 0)

    def get_system_memory(self):
        """Return (used, total, percent) of system RAM."""
        mem = psutil.virtual_memory()
        return mem.used, mem.total, mem.percent / 100

    # ------------------------------------------------------------------
    def check_load(self):
        """Check both GPU & RAM usage and return status dict."""
        gpu_used, gpu_total, gpu_ratio = self.get_gpu_memory()
        ram_used, ram_total, ram_ratio = self.get_system_memory()
        return {
            "gpu_used": gpu_used,
            "gpu_total": gpu_total,
            "gpu_ratio": gpu_ratio,
            "ram_used": ram_used,
            "ram_total": ram_total,
            "ram_ratio": ram_ratio,
        }

    # ------------------------------------------------------------------
    async def monitor(self, interval: float = 10.0):
        """Background loop that checks system load periodically."""
        print("[MemoryManager] 🧠 Monitor started.")
        while True:
            try:
                status = self.check_load()
                gpu_ratio, ram_ratio = status["gpu_ratio"], status["ram_ratio"]

                # Log summary every minute
                if self.last_status != status:
                    print(
                        f"[MemoryManager] GPU {gpu_ratio*100:.1f}%, RAM {ram_ratio*100:.1f}%")
                    self.last_status = status

                if gpu_ratio > self.warn_threshold_vram or ram_ratio > self.warn_threshold_ram:
                    msg = "אדוני, עומס גבוה במערכת. אמליץ לשחרר מודלים זמנית."
                    if self.brain and hasattr(self.brain, "bubble_thought"):
                        await self.brain.bubble_thought(msg)
                    else:
                        print(f"[MemoryManager] ⚠️ {msg}")

                await asyncio.sleep(interval)
            except Exception as e:
                print(f"[MemoryManager] ⚠️ Monitor error: {e}")
                await asyncio.sleep(interval)

    async def report_high_load(self, gpu_ratio):
        await global_bus.publish("thought", {
            "source": "Monitor",
            "text": f"זיכרון ה-GPU כמעט מלא ({gpu_ratio*100:.1f}%)",
            "importance": 0.9
        })
