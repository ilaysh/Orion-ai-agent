# orion_core/tts/engine.py
import numpy as np
from typing import Optional, Tuple

from orion_core.tts.listener import Listener
from orion_core.tts.transcriber import Transcriber
from orion_core.tts import bridge


class SpeechEngine:
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
        sample_rate: int = 16000,
        stt_model: str = "small",
        debug: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.debug = debug

        # VAD + stream accumulator
        self.listener = Listener(sample_rate=sample_rate)
        
        # Try to calibrate ambient noise (optional, won't fail if no mic)
        try:
            self.listener.calibrate_ambient(duration=1.0)
        except:
            if self.debug:
                print("[ENGINE] Could not calibrate ambient noise")

        # Whisper transcriber
        self.transcriber = Transcriber(model_size=stt_model)

        # Energy thresholds to filter out silence
        self.min_energy_rms = 0.02  # Minimum RMS energy
        self.min_energy_peak = 0.1  # Minimum peak amplitude
        
        # Track when we're outputting audio (to ignore echo)
        self.is_speaking = False

    def reset(self) -> None:
        """Reset internal buffers & listener feed state."""
        if hasattr(self.listener, "reset_feed"):
            self.listener.reset_feed()

    # inside class SpeechEngine
    def reset_feed(self):
        """Gracefully stop and reset the audio input stream."""
        try:
            if hasattr(self, "stream") and self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            print("[SpeechEngine] 🧹 Microphone stream reset.")
        except Exception as e:
            print(f"[SpeechEngine] ⚠️ Reset error: {e}")
        
    def _validate_audio_energy(self, wav: np.ndarray) -> bool:
        """
        Check if audio has sufficient energy to be real speech.
        This prevents Whisper from hallucinating on silence/noise.
        
        Returns:
            bool: True if audio seems like real speech
        """
        # Calculate RMS (average energy)
        rms = np.sqrt(np.mean(wav**2))
        
        # Calculate peak amplitude
        peak = np.max(np.abs(wav))
        
        # Check dynamic range (difference between loud and quiet parts)
        percentile_90 = np.percentile(np.abs(wav), 90)
        percentile_10 = np.percentile(np.abs(wav), 10)
        dynamic_range = percentile_90 - percentile_10
        
        has_energy = rms > self.min_energy_rms
        has_peaks = peak > self.min_energy_peak
        has_variation = dynamic_range > 0.01
        
        if self.debug:
            print(f"[ENERGY] RMS={rms:.4f} Peak={peak:.4f} Range={dynamic_range:.4f}")
            print(f"[ENERGY] Valid: energy={has_energy} peaks={has_peaks} variation={has_variation}")
        
        return has_energy and has_peaks and has_variation

    def listen_and_transcribe(self) -> Tuple[str, Optional[str]]:
        """
        Legacy one-shot mic mode (if you still use it elsewhere).
        Uses listener.listen_once() then transcribes.
        Returns *localized to English for Orion*.
        """
        wav_bytes = self.listener.listen_once()
        if not wav_bytes:
            return "", None

        wav = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Validate energy before transcribing
        if not self._validate_audio_energy(wav):
            if self.debug:
                print("[ONE-SHOT STT] Audio rejected - insufficient energy")
            return "", None
        
        text = self.transcriber.transcribe(wav)
        if self.debug:
            print(f"[ONE-SHOT STT] raw='{text}'")

        eng_text, _lang_hint = text, "en"
        return eng_text, _lang_hint

    def transcribe_feed(self, audio_chunk: bytes) -> Tuple[Optional[str], Optional[str]]:
        """
        Feed PCM16 chunks (bytes) from the client microphone.
        When the user stops speaking (VAD silence), returns:
           (english_text_for_orion, lang_hint)
        Otherwise returns (None, None).
        """
        # Ask the listener to accumulate and decide when we have a full utterance
        utterance_bytes = self.listener.feed(audio_chunk)
        
        if utterance_bytes is None:
            # Still collecting speech / no end-of-utterance yet
            return None, None

        if self.debug:
            print(f"[STREAM STT] Got utterance: {len(utterance_bytes)} bytes")

        # Convert to float32 waveform in [-1, 1] for validation and transcription
        wav = np.frombuffer(utterance_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # CRITICAL: Validate that this is actually speech, not silence/noise
        if not self._validate_audio_energy(wav):
            print("[STREAM STT] Utterance rejected - insufficient energy (probably silence)")
            return None, None

        # Save audio for debugging if needed
        if self.debug:
            import soundfile as sf
            sf.write(f"debug_utterance_{len(utterance_bytes)}.wav", wav, self.sample_rate)

        # Transcribe with Whisper - FORCE ENGLISH to avoid language detection issues
        text = self.transcriber.transcribe(wav, language='en')
        
        if self.debug:
            print(f"[STREAM STT] raw='{text}'")

        if not text or not text.strip():
            # Nothing meaningful recognized
            return None, None

        # Filter common hallucinations that Whisper produces on silence
        hallucinations = [
            "thank you", "thanks for watching", "bye", "bye-bye",
            "you", ".", "...", "thank you for watching",
            "thanks", "subscribe", "like and subscribe"
        ]
        
        text_lower = text.lower().strip()
        if text_lower in hallucinations:
            print(f"[STREAM STT] Filtered hallucination: '{text}'")
            return None, None

        # Normalize for Orion
        eng_text, lang_hint = text, "en"

        # Return the English command for Orion + the original language hint
        return eng_text, lang_hint