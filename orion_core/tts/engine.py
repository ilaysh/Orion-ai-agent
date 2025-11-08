# orion_core/speech/engine.py
import base64
from dataclasses import dataclass
import os
import time
import asyncio
import wave
import contextlib
import numpy as np
from typing import Optional, Tuple

from orion_core.base_component import BaseComponent
from orion_core.tts.vad import VADEngine
from orion_core.tts.transcriber import Transcriber   # your faster-whisper class

try:
    import sounddevice as sd
except Exception:
    sd = None


@dataclass
class STTDebug:
    save_wavs: bool = True
    dir: str = "logs/stt_debug"


class SpeechEngine(BaseComponent):
    """
    Streaming STT pipeline with energy validation:
      - feed() audio chunks via Listener (Silero VAD)
      - validate audio has sufficient energy (not silence)
      - when silence is detected -> transcribe whole utterance
      - normalize for Orion using bridge (Hebrew -> English for processing)
      - return (english_text_for_orion, lang_hint)
    """

    def __init__(
        self,
        sr: int = 16000,
        transcriber: Optional[Transcriber] = None,
        vad: Optional[VADEngine] = None,
        debug: Optional[STTDebug] = None,
        min_capture_sec: float = 1.0,
        hard_timeout_sec: float = 8.0,
        min_avg_energy: float = 0.0025,
        frame_ms: int = 30,
    ) -> None:
        super().__init__()
        self.sr = sr
        self.transcriber = transcriber or Transcriber(model_size="medium.en")
        self.vad = vad or VADEngine(sr=sr)
        self.debug = debug or STTDebug()
        self.min_capture_sec = float(min_capture_sec)
        self.hard_timeout_sec = float(hard_timeout_sec)
        self.min_avg_energy = float(min_avg_energy)
        self.frame_ms = int(frame_ms)
        if self.debug.save_wavs:
            os.makedirs(self.debug.dir, exist_ok=True)

    # ---------------- Lifecycle ----------------
    async def init(self) -> None:
        await super().init()
        print("[SpeechEngine] ✅ Ready.")

    async def start(self) -> None:
        await super().start()
        print("[SpeechEngine] ▶ Active.")

    async def stop(self) -> None:
        await super().stop()
        print("[SpeechEngine] ⏹ Stopped.")

    # ---------------- Public API ----------------
    async def listen_and_transcribe(self, language: Optional[str] = None) -> str:
        """Continuously listen on mic, detect speech end with VAD, and transcribe once silence detected."""
        if sd is None:
            print("[SpeechEngine] ⚠️ sounddevice not available; mic capture disabled.")
            return ""

        if not self.active or not self.is_ready():
            print("[SpeechEngine] ⚠️ Not ready for live transcription.")
            return ""

        print("[SpeechEngine] 🎙️ Listening via mic...")
        self.vad.reset()
        self._stream_active = True

        frame_len = int(self.sr * (self.frame_ms / 1000.0))
        cap: list[np.ndarray] = []
        t0 = time.time()
        last_state = "idle"

        def _callback(indata, frames, time_info, status):
            """Audio callback fired by sounddevice stream."""
            x = indata.copy().astype(np.float32).reshape(-1)
            state = self.vad.update(x)
            cap.append(x)

            nonlocal last_state
            last_state = state

            # log transitions (useful for debugging)
            if state != "idle":
                print(f"[VAD] → {state}")

        try:
            with sd.InputStream(
                channels=1,
                samplerate=self.sr,
                dtype="float32",
                blocksize=frame_len,
                callback=_callback,
            ):
                # wait until silence or timeout
                while True:
                    await asyncio.sleep(0.01)
                    if last_state == "silence":
                        print("[VAD] 📴 Silence detected, ending capture.")
                        break
                    if time.time() - t0 > self.hard_timeout_sec:
                        print(
                            "[VAD] ⏰ Hard timeout after extended quiet — stopping capture.")
                        break
        except KeyboardInterrupt:
            pass
        finally:
            self._stream_active = False

        if not cap:
            print("[SpeechEngine] ⚠️ No audio captured.")
            self.vad.reset()
            return ""

        wav = np.concatenate(cap).astype(np.float32)
        print(f"[SpeechEngine] 💾 saved STT input: {len(wav)} samples")
        self.vad.reset()

        try:
            text = await self.transcribe(wav, language=language)
            print(f"[SpeechEngine] 💬 Transcribed: {text}")
            return text
        except Exception as e:
            print(f"[SpeechEngine] ⚠️ STT error: {e}")
            return ""

    async def transcribe(self, wav_or_bytes: np.ndarray | bytes, language: Optional[str] = "en") -> str:
        """Main STT entry using Faster-Whisper."""
        if isinstance(wav_or_bytes, (bytes, bytearray)):
            wav = np.frombuffer(wav_or_bytes, dtype=np.float32)
        else:
            wav = np.asarray(wav_or_bytes, dtype=np.float32)

        dur = wav.size / float(self.sr)
        if dur < self.min_capture_sec:
            print(f"[SpeechEngine] ⚠️ Too little audio ({dur:.2f}s).")
            return ""

        avg = float(np.sqrt(np.mean(np.square(wav))) or 0.0)
        if avg < self.min_avg_energy:
            print(f"[SpeechEngine] ⚠️ Too quiet ({avg:.5f}).")
            return ""

        if self.debug.save_wavs:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.debug.dir, f"stt_input_{ts}.wav")
            self._save_wav(path, wav, self.sr)
            print(f"[SpeechEngine] 💾 saved STT input: {path}")

        t0 = time.time()
        try:
            text = self.transcriber.transcribe_array(
                wav, sr=self.sr, language=language)
        except Exception as e:
            print(f"[SpeechEngine] ⚠️ STT error: {e}")
            text = ""
        print(f"[SpeechEngine] ⏱️ STT took {time.time()-t0:.2f}s")
        if text:
            print(f"[SpeechEngine] 💬 Transcribed: {text}")
        return text

    async def speak(self, text: str):
        """Return base64 WAV via best available TTS backend (Jarvis or Edge)."""
        try:
            from orion_core.tts.jarvis_tts import speak as jarvis_speak
            audio_b64 = await jarvis_speak(text, rate="1.0")
            if audio_b64:
                return audio_b64
        except Exception as e:
            print(f"[SpeechEngine] Piper failed: {e}")

        try:
            from orion_core.tts.speak import speak as edge_speak
            return await edge_speak(text)
        except Exception as e:
            print(f"[SpeechEngine] Edge-TTS failed: {e}")
            return None

    # ---------------- Utils ----------------
    @staticmethod
    def _save_wav(path: str, wav: np.ndarray, sr: int) -> None:
        pcm16 = np.clip(wav, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        with contextlib.closing(wave.open(path, "wb")) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())
