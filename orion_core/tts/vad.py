import numpy as np
import time


class VADEngine:
    """
    Orion VAD

    - Uses RMS energy with adaptive noise floor
    - Optional boost for quiet mics
    - Exposes the original stable API:

        update(frame) -> "idle" | "speech" | "silence"
        calibrate_noise(duration=...)
        detect_silence(frame) -> bool
        is_active() -> bool
        is_silent() -> bool
        reset()

    No external call sites need to change.
    """

    def __init__(
        self,
        sr: int = 16000,
        smooth: float = 0.07,       # noise floor smoothing factor
        scale: float = 1.5,         # energy multiplier for speech gate
        min_silence: float = 1.8,   # minimum silence to end utterance
        max_silence: float = 3.0,   # max allowed before hard timeout logic
        boost: float = 80.0,        # amplify quiet mics
    ) -> None:
        self.sr = sr
        self.smooth = smooth
        self.scale = scale
        self.min_silence = float(min_silence)
        self.max_silence = float(max_silence)
        self.boost = float(boost)

        # dynamic state
        self.noise_floor = 0.002
        self.last_voice_t = time.time()
        self.speech_active = False
        self._speech_start = None

        # calibration
        self._calibrated = False
        self._calib_frames = []
        self._calib_start = time.time()

        # rhythm memory
        self._silence_avg = self.min_silence

        # timeout logging flag
        self._timeout_logged = False

    # --------------------------------------------------------------
    # Core update
    # --------------------------------------------------------------
    def update(self, frame: np.ndarray) -> str:
        """
        Ingest one audio frame and return:
        - "speech": speech currently detected / ongoing
        - "silence": end-of-utterance detected
        - "idle": no decision yet (noise / waiting / timeout already handled)
        """
        if frame is None or frame.size == 0:
            return "idle"

        now = time.time()

        # normalize + boost (helps low-level USB mics)
        frame = frame.astype(np.float32)
        frame = np.clip(frame * self.boost, -1.0, 1.0)
        energy = float(np.sqrt(np.mean(frame ** 2)))

        # --- calibration phase (first ~0.5s treated as ambient) ---
        if not self._calibrated:
            self._calib_frames.append(energy)
            if now - self._calib_start > 0.5 and self._calib_frames:
                self.noise_floor = max(
                    float(np.mean(self._calib_frames) * 1.1), 0.002
                )
                self._calibrated = True
                print(
                    f"[VAD] 🎚️ Calibrated noise floor={self.noise_floor:.5f}")
            # During calibration we don't emit speech/silence decisions
            return "idle"

        # --- adaptive noise floor ---
        self.noise_floor = (
            (1.0 - self.smooth) * self.noise_floor + self.smooth * energy
        )
        gate = self.noise_floor * self.scale

        # --- dynamic silence based on intensity & history ---
        speech_intensity = energy / (self.noise_floor + 1e-6)
        adaptive_silence = self.min_silence * (
            1.0 + 0.5 * min(speech_intensity / 5.0, 1.0)
        )
        adaptive_silence = float(
            np.clip(adaptive_silence, self.min_silence, self.max_silence)
        )

        # incorporate learned rhythm
        adaptive_silence = (adaptive_silence + self._silence_avg) / 2.0

        # --------------------------------------------------
        # Speech / silence decision
        # --------------------------------------------------
        if energy > gate:
            # speech detected
            if not self.speech_active:
                self._speech_start = now
            self.speech_active = True
            self.last_voice_t = now
            # reset timeout logging if we speak again
            self._timeout_logged = False
            print(f"[VAD] 🔊 speech | energy={energy:.5f} > gate={gate:.5f}")
            return "speech"

        # how long since last speech frame
        silence_for = now - self.last_voice_t

        # hard timeout guard: once we've been quiet too long total
        if silence_for > (self.max_silence + 1.0):
            if not self._timeout_logged:
                print("[VAD] ⏰ Hard timeout after extended quiet — stopping capture.")
                self._timeout_logged = True
            # mark as not in speech; treat as silence boundary
            if self.speech_active:
                self.speech_active = False
                self._update_silence_avg(now)
                print(
                    f"[VAD] 🤫 silence | energy={energy:.5f} "
                    f"<= gate={gate:.5f} (silence_for={silence_for:.2f}s)"
                )
                return "silence"
            # if we were already idle, let caller decide; we stay "idle"
            return "idle"

        # normal end-of-utterance:
        if self.speech_active and silence_for > adaptive_silence:
            self.speech_active = False
            self._update_silence_avg(now)
            # slightly decay noise floor after an utterance
            self.noise_floor *= 0.95
            print(
                f"[VAD] 🤫 silence | energy={energy:.5f} "
                f"<= gate={gate:.5f} (silence_for={silence_for:.2f}s)"
            )
            return "silence"

        # otherwise just log & idle
        if energy < self.noise_floor * 0.9:
            print(f"[VAD] ... quiet | energy={energy:.5f} gate={gate:.5f}")
        else:
            print(f"[VAD] ... idle | energy={energy:.5f} gate={gate:.5f}")
        return "idle"

    # --------------------------------------------------------------
    # Helpers (internal)
    # --------------------------------------------------------------
    def _update_silence_avg(self, now: float) -> None:
        """Update rhythm-based silence average using last utterance duration."""
        if self._speech_start:
            dur = max(0.0, now - self._speech_start)
            if dur > 2.5:
                # user tends to speak long → be more patient
                self._silence_avg = min(
                    self.max_silence, self._silence_avg * 1.15)
            else:
                # short utterances → slightly quicker cutoff
                self._silence_avg = max(
                    self.min_silence, self._silence_avg * 0.97)
        self._speech_start = None

    # --------------------------------------------------------------
    # Public API (preserved)
    # --------------------------------------------------------------
    def calibrate_noise(self, duration: float = 0.6) -> None:
        """Explicit ambient calibration (optional; blocking)."""
        print(f"[VAD] 🎚️ Manual calibration for {duration:.1f}s of silence...")
        self._calibrated = False
        self._calib_frames = []
        self._calib_start = time.time()
        end = self._calib_start + duration
        # NOTE: uses current noise_floor as proxy; caller normally runs this at rest
        while time.time() < end:
            self._calib_frames.append(self.noise_floor)
            time.sleep(0.05)
        if self._calib_frames:
            self.noise_floor = max(
                float(np.mean(self._calib_frames) * 1.1), 0.002)
        self._calibrated = True
        print(f"[VAD] ✅ Calibrated noise floor={self.noise_floor:.5f}")

    def detect_silence(self, frame: np.ndarray) -> bool:
        """Return True if this frame triggers an end-of-utterance."""
        return self.update(frame) == "silence"

    def is_active(self) -> bool:
        """True if currently inside a speech segment."""
        return self.speech_active

    def is_silent(self) -> bool:
        """
        True if not in speech and enough time passed since last voice
        (based on min_silence).
        """
        return (not self.speech_active) and (
            time.time() - self.last_voice_t > self.min_silence
        )

    def reset(self) -> None:
        """Reset between utterances (used by SpeechEngine/core)."""
        self._calibrated = False
        self._calib_frames = []
        self.last_voice_t = time.time()
        self.speech_active = False
        self._speech_start = None
        self._timeout_logged = False
        self._silence_avg = self.min_silence
        self._calib_start = time.time()
        print("[VAD] 🔄 Reset state.")
