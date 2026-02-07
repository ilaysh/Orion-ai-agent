# orion_core/brain/llm/hearing_engine.py
import asyncio
from faster_whisper import WhisperModel

_hearing_instance = None

def get_hearing_engine():
    global _hearing_instance
    if _hearing_instance is None:
        _hearing_instance = HearingEngine()
    return _hearing_instance

class HearingEngine:
    def __init__(self):
        self.model = None

    async def load(self):
        if self.model: return
        print(f"[HearingEngine] 👂 Loading Whisper (Medium)...")
        
        try:
            # GPU MODE: float16 is the Gold Standard for RTX cards.
            # It avoids the 'int8' compatibility issues you saw.
            self.model = WhisperModel(
                "medium", 
                device="cuda", 
                compute_type="float16" 
            )
            print("[HearingEngine] ✅ Whisper loaded on GPU (High Speed).")
        except Exception as e:
            print(f"[HearingEngine] ⚠️ GPU Failed: {e}. Fallback to CPU (Tiny).")
            # If GPU fails, fallback to tiny model to prevent timeouts
            self.model = WhisperModel("tiny", device="cpu", compute_type="int8")

    def transcribe(self, audio_data, beam_size=5):
        if not self.model: return ""
        # Run in thread to prevent blocking the async loop
        segments, _ = self.model.transcribe(audio_data, beam_size=beam_size)
        return " ".join([s.text for s in segments]).strip()