import numpy as np
import time


class VADEngine:
    """
    Orion unified VAD (black-box style)
    -----------------------------------
    • Adaptive energy gate with noise floor
    • Dynamic silence threshold based on speech rhythm
    • Detects speech/silence transitions
    • Logs internal state for debugging
    """

    def __init__(self, sr=16000,
                 smooth=0.07,       # noise floor smoothing
                 scale=1.5,         # energy above floor considered speech
                 min_silence=1.8,   # base silence threshold (sec)
                 max_silence=3.0):  # maximum patience for long sentences
        self.sr = sr
        self.smooth = smooth
        self.scale = scale
        self.min_silence = min_silence
        self.max_silence = max_silence

        self.noise_floor = 0.002
        self.last_voice_t = time.time()
        self.speech_active = False
        self._speech_start = None
        self._calibrated = False
        self._calib_start = time.time()

    # --------------------------------------------------------------

    def update(self, frame: np.ndarray) -> str:
        if frame is None or frame.size == 0:
            return "idle"

        frame = frame.astype(np.float32)
        energy = float(np.sqrt(np.mean(frame ** 2)))
        now = time.time()

        # --- calibration phase ---
        if not self._calibrated:
            if not hasattr(self, "_calib_frames"):
                self._calib_frames = []
                self._calib_start = now
            self._calib_frames.append(energy)
            if now - self._calib_start > 0.5:
                self.noise_floor = max(
                    np.mean(self._calib_frames) * 1.1, 0.002)
                self._calibrated = True
                print(
                    f"[VAD] 🎚️ Calibrated noise floor={self.noise_floor:.5f}")
            return "idle"

        # --- adaptive smoothing ---
        self.noise_floor = (1 - self.smooth) * \
            self.noise_floor + self.smooth * energy
        gate = self.noise_floor * self.scale

        # --- dynamic silence tuning ---
        speech_intensity = energy / (self.noise_floor + 1e-6)
        adaptive_silence = self.min_silence * \
            (1.0 + 0.5 * min(speech_intensity / 5.0, 1.0))
        adaptive_silence = np.clip(
            adaptive_silence, self.min_silence, self.max_silence)

        # --- rhythm memory ---
        if not hasattr(self, "_silence_avg"):
            self._silence_avg = self.min_silence
        if self.speech_active and self._speech_start:
            duration = now - self._speech_start
            if duration > 2.5:
                self._silence_avg = min(
                    self.max_silence, self._silence_avg * 1.15)
            else:
                self._silence_avg = max(
                    self.min_silence, self._silence_avg * 0.97)
        adaptive_silence = (adaptive_silence + self._silence_avg) / 2.0

        # --- speech / silence detection ---
        if energy > gate:
            if not self.speech_active:
                self._speech_start = now
            self.speech_active = True
            self.last_voice_t = now
            print(f"[VAD] 🔊 speech | energy={energy:.5f} > gate={gate:.5f}")
            return "speech"
        # how long since last speech
        silence_for = now - self.last_voice_t

        # extended quiet: hard timeout guard (trigger only once)
        if silence_for > self.max_silence + 1.0:
            if not getattr(self, "_timeout_logged", False):
                print("[VAD] ⏰ Hard timeout after extended quiet — stopping capture.")
                self._timeout_logged = True
                # mark speech as inactive once and return silence
                self.speech_active = False
                return "silence"
            # already logged timeout — stay idle quietly
            return "idle"

        # normal silence condition
        if self.speech_active and silence_for > self.min_silence:
            self.speech_active = False
            self.noise_floor *= 0.95
            print(
                f"[VAD] 🤫 silence | energy={energy:.5f} <= gate={gate:.5f} (silence_for={silence_for:.2f}s)")
            return "silence"

        # otherwise classify quietly
        if energy < self.noise_floor * 0.9:
            print(f"[VAD] ... quiet | energy={energy:.5f} gate={gate:.5f}")
        else:
            print(f"[VAD] ... idle | energy={energy:.5f} gate={gate:.5f}")
        return "idle"

    # --------------------------------------------------------------

    def calibrate_noise(self, duration: float = 0.6):
        """Force quick ambient calibration (like Google)."""
        print(f"[VAD] 🎚️ Manual calibration for {duration:.1f}s of silence...")
        self._calibrated = False
        self._calib_frames = []
        self._calib_start = time.time()
        end = self._calib_start + duration
        while time.time() < end:
            self._calib_frames.append(self.noise_floor)
            time.sleep(0.05)
        self.noise_floor = max(np.mean(self._calib_frames) * 1.1, 0.002)
        self._calibrated = True
        print(f"[VAD] ✅ Calibrated noise floor={self.noise_floor:.5f}")

    def detect_silence(self, frame: np.ndarray) -> bool:
        """Return True if current frame indicates end of speech."""
        return self.update(frame) == "silence"

    def is_active(self) -> bool:
        return self.speech_active

    def is_silent(self) -> bool:
        return not self.speech_active and (time.time() - self.last_voice_t) > self.min_silence

        # --------------------------------------------------------------
    # Reset internal state (used by SpeechEngine between utterances)
    # --------------------------------------------------------------
    def reset(self):
        self._calibrated = False
        self._calib_frames = []
        self.last_voice_t = time.time()
        self.speech_active = False
        self._speech_start = None
        self._timeout_logged = False
        self._calib_start = time.time()
        print("[VAD] 🔄 Reset state.")
