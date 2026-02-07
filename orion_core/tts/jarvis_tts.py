# orion_core/tts/jarvis_tts.py
import base64
import io
import os
import re
import numpy as np
from typing import Optional
from kokoro_onnx import Kokoro
from pydub import AudioSegment, effects

# --- CONFIGURATION ---
MODEL_DIR = "models/voice"
MODEL_NAME = "kokoro-v1.0.onnx"
VOICES_NAME = "voices-v1.0.bin"

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
VOICES_PATH = os.path.join(MODEL_DIR, VOICES_NAME)

# Voice Settings
VOICE_NAME = "bm_george"  # The Butler
SPEED = 1.1 
ENABLE_PUNCH = True

_voice: Optional[Kokoro] = None

def preload_voice() -> Kokoro:
    global _voice
    if _voice is not None: return _voice

    if not os.path.exists(MODEL_PATH):
        print(f"[JarvisTTS] ❌ Model missing: {MODEL_PATH}")
        return None

    print(f"[JarvisTTS] Loading Kokoro Voice: {MODEL_PATH}")
    try:
        _voice = Kokoro(MODEL_PATH, VOICES_PATH)
        print("[JarvisTTS] ✅ Voice System Online.")
    except Exception as e:
        print(f"[JarvisTTS] ❌ Load Failed: {e}")
        return None
    return _voice

def clean_for_speech(text: str) -> str:
    """
    Sanitizes LLM output to prevent TTS crashes.
    1. Removes Markdown (*, #, _, `).
    2. Expands common symbols.
    3. Adds a period at the end (Safety Pad).
    """
    if not text: return ""
    
    # Remove Markdown bold/italic/code
    text = re.sub(r'[*_`#]', '', text)
    
    # Remove complex brackets [Source: 1] -> ""
    text = re.sub(r'\[.*?\]', '', text)
    
    # Ensure it ends with punctuation (Helps alignment model)
    text = text.strip()
    if not text.endswith(('.', '!', '?')):
        text += "."
        
    return text

def apply_jarvis_polish(audio_data: np.ndarray, sample_rate: int) -> AudioSegment:
    # Convert float32 -> int16
    audio_int16 = (audio_data * 32767).astype(np.int16)
    audio_segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    if ENABLE_PUNCH:
        audio_segment = effects.normalize(audio_segment, headroom=0.1)
    return audio_segment

async def speak(text: str) -> Optional[str]:
    """
    Generates audio and returns Base64 string.
    Includes Crash Protection & Fallback.
    """
    # print(f"[JarvisTTS] 🗣️ speak() called with: {text[:50]}...")
    if not text or not text.strip(): 
        print("[JarvisTTS] ⚠️ Empty text, skipping")
        return None

    # 1. Sanitize (The Fix for 'words_mismatch')
    safe_text = clean_for_speech(text)
    print(f"[JarvisTTS] ✅ Sanitized: {safe_text}")
    
    try:
        if not _voice:
            print("[JarvisTTS] ⚠️ Voice not loaded. Attempting load...")
            preload_voice()
            if not _voice: 
                print("[JarvisTTS] ❌ Failed to load voice")
                return None

        # 2. Generate
        print(f"[JarvisTTS] 🎙️ Generating audio...")
        samples, sample_rate = _voice.create(
            safe_text,
            voice=VOICE_NAME,
            speed=SPEED,
            lang="en-gb"
        )
        print(f"[JarvisTTS] ✅ Generated {len(samples)} samples at {sample_rate}Hz")

        # 3. Polish
        final_segment = apply_jarvis_polish(samples, sample_rate)

        # 4. Export
        buf = io.BytesIO()
        final_segment.export(buf, format="wav")
        audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        print(f"[JarvisTTS] ✅ Exported {len(audio_b64)} bytes as base64")
        return audio_b64

    except Exception as e:
        print(f"[JarvisTTS] 💥 Generation Failed: {e}")
        import traceback
        traceback.print_exc()
        # FALLBACK: If Kokoro crashes, use system voice so you aren't left silent
        # This ensures you always get an answer.
        import subprocess, shutil
        if shutil.which("spd-say"):
            subprocess.run(["spd-say", "-w", safe_text])
        return None