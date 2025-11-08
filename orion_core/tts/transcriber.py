# orion_core/speech/transcriber.py
# Fast, accurate, fully-offline STT using faster-whisper
# pip install faster-whisper soundfile librosa
from __future__ import annotations
import os
import io
import tempfile
import numpy as np
import soundfile as sf
import librosa

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

from faster_whisper import WhisperModel


class Transcriber:
    """
    Usage:
        stt = Transcriber(model_size="medium.en")
        text = stt.transcribe_file("/tmp/clip.wav")
        # or:
        text = stt.transcribe_array(audio_float32_16k, sr=16000)
    """

    def __init__(
        self,
        # good speed/quality. Use "small.en"/"base.en" for faster.
        model_size: str = "medium.en",
        # "float16" on NVIDIA; fallback to "int8_float16" on CPU
        compute_type: str = "float16",
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if (
                HAS_TORCH and torch.cuda.is_available()) else "cpu"

        device = "cpu"
        # if CPU, prefer int8 mixed for speed
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"

        self.sample_rate = 16000
        self.model = WhisperModel(
            model_size, device="cpu", compute_type=compute_type)

        # sensible default VAD
        self._vad = dict(vad_filter=True, vad_parameters={
                         "min_silence_duration_ms": 200})

    # ---------- helpers ----------
    def _ensure_16k_mono_f32(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        wav = wav.astype(np.float32, copy=False)
        # normalize softly if needed
        peak = np.max(np.abs(wav)) if wav.size else 0.0
        if peak > 1.0:
            wav /= peak
        return wav

    # ---------- main APIs ----------
    def transcribe_file(self, path: str, language: str = "en") -> str:
        segments, _ = self.model.transcribe(
            path, language=language, **self._vad)
        return " ".join(s.text.strip() for s in segments if s.text).strip()

    def transcribe_array(self, wav: np.ndarray, sr: int, language: str = "en") -> str:
        wav = self._ensure_16k_mono_f32(wav, sr)
        # faster-whisper can take a float32 array at 16k directly
        segments, _ = self.model.transcribe(
            wav, language=language, **self._vad)
        return " ".join(s.text.strip() for s in segments if s.text).strip()

    # convenience for raw bytes → temp wav (if you already save bytes)
    def transcribe_bytes(self, wav_bytes: bytes, language: str = "en") -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp.flush()
            path = tmp.name
        try:
            return self.transcribe_file(path, language=language)
        finally:
            try:
                os.remove(path)
            except Exception:
                pass
