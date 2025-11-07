import os
import time
import asyncio
import wave
import contextlib
import numpy as np
from dataclasses import dataclass
from typing import Optional

from orion_core.base_component import BaseComponent
from orion_core.tts.vad import VADEngine
from orion_core.tts.transcriber import Transcriber

# optional mic dependency
try:
    import sounddevice as sd
except Exception:
    sd = None


@dataclass
class STTDebug:
    save_wavs: bool = True
    dir: str = "logs/stt_debug"


class SpeechEngine(BaseComponent):
    def __init__(
        self,
        sr: int = 16000,
        transcriber: Optional[Transcriber] = None,
        vad: Optional[VADEngine] = None,
        debug: Optional[STTDebug] = None,
        min_capture_sec: float = 1.2,
        hard_timeout_sec: float = 8.0,
        min_avg_energy: float = 0.0025,
        frame_ms: int = 30,
    ) -> None:
        super().__init__()
        self._stream_active = False
        self.sr = sr

        # your Transcriber doesn't take sample_rate
        self.transcriber = transcriber or Transcriber()
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
        if sd is None:
            print("[SpeechEngine] ⚠️ sounddevice not available; mic capture disabled.")
            return ""

        if not self.active or not self.is_running():
            print("[SpeechEngine] ⚠️ Not ready for live transcription.")
            return ""

        print("[SpeechEngine] 🎙️ Listening via mic...")
        self.vad.reset()
        self._stream_active = True  # <-- added

        frame_len = int(self.sr * (self.frame_ms / 1000.0))
        cap: list[np.ndarray] = []
        t0 = time.time()
        last_state = "idle"

        def _callback(indata, frames, time_info, status):
            x = indata.copy().astype(np.float32).reshape(-1)
            state = self.vad.update(x)
            cap.append(x)
            nonlocal last_state
            last_state = state

        try:
            with sd.InputStream(
                channels=1,
                samplerate=self.sr,
                dtype="float32",
                blocksize=frame_len,
                callback=_callback,
            ):
                while True:
                    await asyncio.sleep(0.01)
                    if last_state == "silence":
                        break
                    if time.time() - t0 > self.hard_timeout_sec:
                        print("[VAD] ⏰ Hard timeout after extended quiet — stopping capture.")
                        break
        except KeyboardInterrupt:
            pass
        finally:
            self._stream_active = False  # <-- added

        if not cap:
            self.vad.reset()
            return ""

        wav = np.concatenate(cap).astype(np.float32)
        self.vad.reset()
        return await self.transcribe(wav, language=language)


    async def transcribe(self, wav_or_bytes: np.ndarray | bytes, language: Optional[str] = None) -> str:
        """Main entry-point used by Core. No VAD logic here."""
        if isinstance(wav_or_bytes, (bytes, bytearray)):
            wav = np.frombuffer(wav_or_bytes, dtype=np.float32)
        else:
            wav = np.asarray(wav_or_bytes, dtype=np.float32)

        dur = wav.size / float(self.sr)
        if dur < self.min_capture_sec:
            print(f"[SpeechEngine] ⏱️ captured {dur:.2f}s of audio")
            print("[SpeechEngine] ⚠️ Too little audio, skipping transcription.")
            return ""

        avg = float(np.sqrt(np.mean(np.square(wav))) or 0.0)
        peak = float(np.max(np.abs(wav)) or 1.0)
        if peak > 0:
            wav = np.clip(wav / peak, -1.0, 1.0) * 0.95

        avg_after = float(np.sqrt(np.mean(np.square(wav))) or 0.0)
        if avg_after < self.min_avg_energy:
            print(f"[SpeechEngine] ⏱️ captured {dur:.2f}s of audio")
            print(f"[SpeechEngine] ⚙️ avg_energy={avg_after:.5f}")
            print(
                "[SpeechEngine] ⚠️ Too little or too quiet audio, skipping transcription.")
            return ""

        if self.debug.save_wavs:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(
                self.debug.dir, f"whisper_input_{ts}_{int((time.time() % 1)*1e6):06d}.wav"
            )
            self._save_wav(path, wav, self.sr)
            print(f"[SpeechEngine] 🎧 saved Whisper input: {path}")

        t0 = time.time()
        text = ""
        try:
            text = self.transcriber.transcribe(wav, language or "en") or ""
        except Exception as e:
            print(f"[SpeechEngine] ⚠️ STT error: {e}")
        t_stt = time.time() - t0

        print(
            f"[SpeechEngine] ⏱️ STT time={t_stt:.2f}s (dur={dur:.2f}s, avg_energy={avg_after:.5f})")

        out = text.strip()
        if out == ".":
            print("[SpeechEngine] ⚠️ Ignored low-confidence text: '.'")
            return ""

        if out:
            print(f"[SpeechEngine] 💬 Transcribed: {out}")
        return out

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
