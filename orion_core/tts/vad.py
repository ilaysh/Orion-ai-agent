# orion_core/tts/vad.py
import numpy as np
import time

class VADEngine:
    """
    Orion VAD - Fixed Calibration Logic
    """

    def __init__(
        self,
        sr: int = 16000,
        smooth: float = 0.05,       # Slower adaptation to sudden noises
        scale: float = 1.3,         # Lower threshold (was 1.5) to catch softer voice
        min_silence: float = 0.8,   # Snappy response (0.8s silence = done)
        max_silence: float = 6.0,   # Allow long dictation
        boost: float = 1.0,         # CHANGED: Removed aggressive 80.0x boost that clipped audio
    ) -> None:
        self.sr = sr
        self.smooth = smooth
        self.scale = scale
        self.min_silence = float(min_silence)
        self.max_silence = float(max_silence)
        self.boost = float(boost)

        # State
        self.noise_floor = 0.005 # Start with reasonable default
        self.last_voice_t = time.time()
        self.speech_active = False
        self._speech_start = None

        # Calibration
        self._calibrated = False
        self._calib_frames = []
        self._calib_start = time.time()

        self._silence_avg = self.min_silence
        self._timeout_logged = False

    def update(self, frame: np.ndarray) -> str:
        if frame is None or frame.size == 0:
            return "idle"

        now = time.time()

        # 1. Normalize
        frame = frame.astype(np.float32)
        # Only boost if really quiet, otherwise leave natural
        if self.boost > 1.0:
            frame = np.clip(frame * self.boost, -1.0, 1.0)
            
        energy = float(np.sqrt(np.mean(frame ** 2)))

        # 2. Calibration (One Time Only)
        if not self._calibrated:
            self._calib_frames.append(energy)
            if now - self._calib_start > 0.6:
                # Remove outliers (loud noises during startup)
                if len(self._calib_frames) > 5:
                    sorted_frames = sorted(self._calib_frames)
                    # Take median-ish low end to find true silence
                    baseline = np.mean(sorted_frames[:int(len(sorted_frames)*0.8)])
                    self.noise_floor = max(float(baseline * 1.2), 0.002)
                
                self._calibrated = True
                print(f"[VAD] 🎚️ Calibrated noise floor={self.noise_floor:.5f}")
            return "idle"

        # 3. Adaptive Noise Floor (Slow Drift)
        # We only adapt if NO speech is active to prevent adapting to the user's voice
        if not self.speech_active:
             self.noise_floor = (1.0 - self.smooth) * self.noise_floor + self.smooth * energy

        gate = self.noise_floor * self.scale
        
        # 4. Speech Logic
        if energy > gate:
            if not self.speech_active:
                self._speech_start = now
                # print(f"[VAD] 🔊 Started (E={energy:.4f} > G={gate:.4f})")
            self.speech_active = True
            self.last_voice_t = now
            return "speech"

        # 5. Silence Logic
        silence_for = now - self.last_voice_t

        # Hard Timeout
        if silence_for > (self.max_silence + 1.0):
            if self.speech_active:
                self.speech_active = False
                print("[VAD] ⏰ Timeout.")
                return "silence"
            return "idle"

        # End of Utterance
        if self.speech_active and silence_for > self.min_silence:
            self.speech_active = False
            print(f"[VAD] 🤫 Silence ({silence_for:.2f}s)")
            return "silence"

        return "idle"

    def reset(self) -> None:
        """
        Resets utterance state but DOES NOT RE-CALIBRATE.
        This fixes the bug where speaking immediately broke the VAD.
        """
        self.last_voice_t = time.time()
        self.speech_active = False
        self._speech_start = None
        self._timeout_logged = False
        # Do NOT set self._calibrated = False
        print("[VAD] 🔄 Ready.")