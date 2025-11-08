# orion_core/tts/jarvis_tts.py
import asyncio
import base64
import io
import os
import subprocess
import tempfile
import numpy as np
from piper.voice import PiperVoice

MODEL_PATH = "models/voice/jarvis.onnx"
_voice = None


MODEL_DIR = "models/voice"
MODEL_NAME = "jarvis"
MODEL_ONNX = os.path.join(MODEL_DIR, f"{MODEL_NAME}.onnx")
MODEL_JSON = os.path.join(MODEL_DIR, f"{MODEL_NAME}.onnx.json")
DEFAULT_RATE = "0.9"   # 1.0 = normal, <1.0 slower, >1.0 faster

_voice = None


def _get_voice():
    global _voice
    if _voice is None:
        if not os.path.exists(MODEL_ONNX):
            raise FileNotFoundError(f"Missing {MODEL_ONNX}")
        if not os.path.exists(MODEL_JSON):
            raise FileNotFoundError(
                f"Missing {MODEL_JSON} — required for Piper!")

        print(f"[PiperTTS] Loading voice model: {MODEL_ONNX}")
        print(f"[PiperTTS] Config: {MODEL_JSON}")
        _voice = PiperVoice.load(MODEL_ONNX, config_path=MODEL_JSON)
        print(f"[PiperTTS] Voice ready. "
              f"Sample rate: {_voice.config.sample_rate} Hz | "
              f"Speakers: {getattr(_voice.config, 'num_speakers', 1)}")
    return _voice


def _resample_wav(in_bytes: bytes, factor: float) -> bytes:
    import io
    import soundfile as sf
    import numpy as np
    data, sr = sf.read(io.BytesIO(in_bytes), dtype="int16")
    new_sr = int(sr * factor)
    buf = io.BytesIO()
    sf.write(buf, data, new_sr, format="WAV")
    return buf.getvalue()


async def speak(text: str, rate: str = DEFAULT_RATE) -> str:
    """Offline Jarvis voice using Piper CLI (async, adjustable speed)."""
    try:
        # create temp wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        # run Piper asynchronously
        cmd = [
            "piper",
            "--model", MODEL_PATH,
            "--output_file", wav_path,
            "--length_scale", rate,  # tempo control (1.0=normal, 0.9=slower)
            "--text", text,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()

        # read audio
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            with open(wav_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            os.remove(wav_path)
            return b64
        else:
            raise RuntimeError("Piper produced no audio output.")

    except Exception as e:
        print(f"[PiperTTS] ⚠️ CLI synthesis failed: {e}")
        return None
