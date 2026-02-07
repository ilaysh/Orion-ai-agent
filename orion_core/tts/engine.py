# orion_core/speech/engine.py
import numpy as np
import time
import asyncio
from typing import Optional, Callable
from orion_core.base_component import BaseComponent
from orion_core.tts.vad import VADEngine
from orion_core.tts.transcriber import Transcriber
from orion_core.tts.jarvis_tts import speak as jarvis_speak, preload_voice

try:
    import sounddevice as sd
except Exception:
    sd = None

class SpeechEngine(BaseComponent):
    def __init__(self, shared_model, sr: int = 16000, vad: Optional[VADEngine] = None):
        super().__init__()
        self.sr = sr
        
        # STT Model
        if shared_model:
             self.transcriber = Transcriber(model=shared_model)
        else:
             self.transcriber = None
             print("[SpeechEngine] ⚠️ No shared model injected.")

        self.vad = vad or VADEngine(sr=sr)
        
        self._stream = None
        self._passive_callback = None
        self._listening = False
        self._audio_queue = asyncio.Queue()

    async def init(self):
        await super().init()
        preload_voice()
        print("[SpeechEngine] ✅ Ready.")

    def start_passive_stream(self, callback: Callable[[np.ndarray], None]):
        if self._stream: return
        if sd is None: return
        self._passive_callback = callback
        print("[SpeechEngine] 👂 Opening Mic Stream...")

        def _sd_callback(indata, frames, time_info, status):
            if status and "overflow" not in str(status).lower():
                pass # Silently ignore overflows for performance
            audio = indata.copy().flatten().astype(np.float32)
            if self._listening:
                try:
                    self._audio_queue.put_nowait(audio)
                except asyncio.QueueFull:
                    pass
            elif self._passive_callback:
                self._passive_callback(audio)

        self._stream = sd.InputStream(
            channels=1, samplerate=self.sr, dtype="float32",
            blocksize=4096, callback=_sd_callback
        )
        self._stream.start()

    async def listen(self, timeout: float = 30.0) -> str:
        if not self.transcriber: return ""
        
        self._listening = True 
        self.vad.reset()
        
        buffer = []
        start_time = time.time()
        print("[SpeechEngine] 🎙️ Active Listening...")
        
        try:
            while True:
                if time.time() - start_time > timeout:
                    print("[SpeechEngine] ⏱️ Listen timeout.")
                    break

                try:
                    chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                buffer.append(chunk)
                status = self.vad.update(chunk)
                if status == "silence":
                    print("[SpeechEngine] 🛑 Silence detected. Processing...")
                    break
        finally:
            self._listening = False 
            while not self._audio_queue.empty():
                self._audio_queue.get_nowait()

        if not buffer: return ""
        full_audio = np.concatenate(buffer)
        
        # --- SAFE TRANSCRIPTION ---
        try:
            # Increased timeout to 8.0s to accommodate model loading if needed
            return await asyncio.wait_for(
                self.transcribe(full_audio), 
                timeout=8.0
            )
        except asyncio.TimeoutError:
            print("[SpeechEngine] 💥 Transcription Timed Out")
            return ""
        except Exception as e:
            print(f"[SpeechEngine] 💥 Transcription Failed: {e}")
            return ""

    async def transcribe(self, audio: np.ndarray) -> str:
        if not self.transcriber: return ""
        if np.sqrt(np.mean(audio**2)) < 0.002: return ""
        
        # This calls the wrapper which calls hearing_engine.transcribe
        # It's already threaded in hearing_engine, but we wrap it here for safety
        return await asyncio.to_thread(
            self.transcriber.transcribe_array, 
            audio, 
            self.sr
        )

    async def speak(self, text: str):
        return await jarvis_speak(text)
    
    async def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        await super().stop()