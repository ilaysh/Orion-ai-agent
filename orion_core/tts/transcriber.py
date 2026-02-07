# orion_core/tts/transcriber.py
import numpy as np
import torch
import asyncio
from faster_whisper import WhisperModel

class Transcriber:
    def __init__(self, model=None, model_size="medium", device=None, compute_type="float16"):
        # 1. USE SHARED MODEL (Priority)
        if model is not None:
            self.model = model
            return

        # 2. LOAD FRESH MODEL (Fallback)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[Transcriber] ⚠️ Loading own Whisper ({model_size} {compute_type})...")
        try:
            self.model = WhisperModel(model_size, device=self.device, compute_type=compute_type)
        except Exception as e:
            print(f"[Transcriber] ⚠️ Init Failed: {e}")
            self.model = None

    def transcribe_array(self, audio: np.ndarray, sr: int = 16000, language: str = None) -> str:
        if self.model is None: return ""
            
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Use beam_size=1 for speed during voice commands
        segments, info = self.model.transcribe(
            audio, 
            beam_size=1, 
            language=language,
            vad_filter=True 
        )
        
        text = " ".join([segment.text for segment in segments]).strip()
        return text

    def stop(self):
        pass