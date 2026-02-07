# orion_core/brain/cortex.py
import asyncio
import gc
from typing import Optional, Callable, Any

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from orion_core.brain.llm.chat_engine import get_chat_engine
from orion_core.brain.llm.hearing_engine import get_hearing_engine

# Singleton access
_cortex_instance = None

def get_cortex():
    return _cortex_instance

class Cortex:
    """
    The Hardware Lord.
    Holds references to the persistent Engines (Sight, Sound, Thought).
    """
    def __init__(self, bubble_thought: Optional[Callable[[str], Any]] = None):
        global _cortex_instance
        _cortex_instance = self
        
        self.bubble = bubble_thought
        
        # 1. Grab Persistent Engines (Singletons)
        self.chat_engine = get_chat_engine()   # Qwen-14B
        self.hearing = get_hearing_engine()    # Whisper
        
        self.is_llm_active = False
        self._lock = asyncio.Lock()

    async def init_core_senses(self):
        """Boot up the heavy models via their Engines."""
        # Load Ears (Whisper)
        # The HearingEngine handles the GPU/CPU fallback logic internally
        await self.hearing.load()

    def get_shared_ears(self):
        """
        Access for SpeechEngine. 
        Returns the raw WhisperModel object from the engine.
        """
        if not self.hearing or not self.hearing.model:
            print("[Cortex] ⚠️ Warning: Hearing Engine not ready.")
            return None
        return self.hearing.model

    # --- GENERIC VRAM MANAGEMENT ---
    async def ensure_chat_mode(self):
        """Alias for wake_chat."""
        await self.wake_chat()
    
    async def wake_chat(self):
        """Ensures Qwen is loaded in VRAM."""
        async with self._lock:
            if self.is_llm_active: return
            
            print("[Cortex] 🧠 Waking up Qwen...")
            gc.collect()
            if HAS_TORCH: torch.cuda.empty_cache()
            
            await self.chat_engine.load()
            await self.hearing.load()
            self.is_llm_active = True

    async def hibernate_chat(self):
        """Offloads Qwen to RAM/Disk."""
        async with self._lock:
            if not self.is_llm_active: return
            
            print("[Cortex] 💤 Hibernating Qwen (Freeing VRAM)...")
            await self.chat_engine.unload()
            gc.collect()
            if HAS_TORCH: torch.cuda.empty_cache()
            
            self.is_llm_active = False

    async def generate_chat(self, prompt: str, system_prompt: str, **kwargs) -> str:
        """Passthrough to Qwen (Auto-Wakes)."""
        await self.wake_chat()
        return await self.chat_engine.generate_chat(prompt, system_prompt=system_prompt, **kwargs)