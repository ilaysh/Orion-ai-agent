# orion_core/wakeword.py
from __future__ import annotations
import os, time
from collections import deque
from datetime import datetime
from typing import Optional, Deque

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from speechbrain.inference import EncoderClassifier
import soundfile as sf


class WakeWordDetector:
    """
    Test-parity streaming wake-word detector.
    - Expect float32 mono PCM in [-1, 1] (any length).
    - Internally keeps only the last 1s (16k) window per evaluation.
    - Same preprocessing & scoring as wakeword_infer_speechbrain_live.py
    """

    def __init__(
        self,
        model_path: str = "models/orion_speechbrain_full_finetune.pt",
        sample_rate: int = 16000,
        thresh: float = 0.90,          # stricter (use 0.75 if you want test default)
        avg_boost: float = 0.05,       # test used THRESHOLD+0.05
        smooth_window: int = 3,        # deque length like test
        debounce_s: float = 1.5,
        cooldown_s: float = 1.5,
        log_every_eval: bool = False,  # True to print conf/energy every eval
        debug_save_all: bool = False,  # True to save every eval window to logs/
    ):
        self.sr = int(sample_rate)
        self.model_path = model_path
        self.THRESH = float(thresh)
        self.AVG_BOOST = float(avg_boost)
        self.smooth_window = int(smooth_window)
        self.debounce_s = float(debounce_s)
        self.cooldown_s = float(cooldown_s)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model: Optional[EncoderClassifier] = None

        self._noise_floor = 0.002
        self._armed = True
        self._last_fire = 0.0
        self._conf_buf: Deque[float] = deque(maxlen=self.smooth_window)

        self._log_every_eval = log_every_eval
        self._debug_save_all = debug_save_all

        self._load_model()

    # ------------------------------------------------------------------ #
    #                             MODEL
    # ------------------------------------------------------------------ #
    def _load_model(self):
        print(f"[Wake] Loading SpeechBrain model from {self.model_path} on {self._device} ...")
        base = EncoderClassifier.from_hparams(
            source="speechbrain/google_speech_command_xvector",
            run_opts={"device": self._device}
        )
        # Replace last classifier to (in_dim -> 1) + Sigmoid, then load fine-tuned weights
        last_linear = next(m for m in base.mods.classifier.modules() if isinstance(m, nn.Linear))
        in_dim = last_linear.in_features
        base.mods.classifier = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid()).to(self._device)

        state = torch.load(self.model_path, map_location=self._device)
        base.load_state_dict(state)
        base.eval()

        self._model = base
        print(f"✅ Wake word model ready from: {self.model_path}")

    # ------------------------------------------------------------------ #
    #                             CONTROL
    # ------------------------------------------------------------------ #
    def arm(self):
        self._armed = True

    def disarm(self):
        self._armed = False

    # ------------------------------------------------------------------ #
    #                             FEED
    # ------------------------------------------------------------------ #
    def feed(self, pcm_f32_mono: np.ndarray | bytes) -> bool:
        """
        Evaluate a ~1s window for the wake word.
        Accepts float32 mono [-1,1] or raw bytes (float32/int16).
        Returns True exactly when the detector fires.
        """
        now = time.time()

        # --- guard rails ---
        if not self._armed or self._model is None:
            print("[Wake] Not armed or model not loaded.")
            return False

        # --- normalize input to float32 mono in [-1, 1] ---
        try:
            if isinstance(pcm_f32_mono, (bytes, bytearray)):
                # Try float32 first
                x = np.frombuffer(pcm_f32_mono, dtype=np.float32, count=len(pcm_f32_mono)//4)
                looks_bad = (x.size == 0) or (not np.isfinite(x).all()) or (np.max(np.abs(x)) > 1.5)
                if looks_bad:
                    i16 = np.frombuffer(pcm_f32_mono, dtype=np.int16)
                    if i16.size == 0:
                        print("[Wake] Empty/invalid byte chunk.")
                        return False
                    x = (i16.astype(np.float32) / 32768.0)
            else:
                x = np.asarray(pcm_f32_mono, dtype=np.float32)

            if x.ndim > 1:
                x = x.reshape(-1)
            # hard clamp stray values
            x = np.clip(x, -1.0, 1.0).astype(np.float32)
        except Exception as e:
            print(f"[Wake] PCM normalize error: {e}")
            return False

        # --- require ~1s window ---
        if x.size < self.sr:
            print(f"[Wake] Need more audio, got {x.size} samples.")
            return False

        # use exactly last 1s like the tester
        wav = x[-self.sr:]

        # --- light prefilter (match tester) ---
        try:
            wav_t = torch.tensor(wav, dtype=torch.float32)
            wav_t = torchaudio.functional.highpass_biquad(wav_t, self.sr, 100)
            wav = wav_t.cpu().numpy().astype(np.float32, copy=False)
        except Exception as e:
            print(f"[Wake] Highpass error (continuing without HPF): {e}")

        # --- energy gating & adaptive noise floor ---
        raw_energy = float(np.mean(np.abs(wav)))
        if not np.isfinite(raw_energy):
            print("[Wake] Non-finite energy. Skipping.")
            return False

        # smooth/track noise floor
        old_nf = self._noise_floor
        self._noise_floor = 0.98 * self._noise_floor + 0.02 * raw_energy
        gate = max(self._noise_floor * 1.5, 0.003)  # conservative minimum gate

        # helpful input stats (always log for now)
        print(f"[Wake] In: len={wav.size}, min={np.min(wav):+0.4f}, max={np.max(wav):+0.4f}, mean={np.mean(wav):+0.4f}")
        if raw_energy < gate:
            print(f"[Wake] Energy {raw_energy:.4f} below gate {gate:.4f}, skipping eval.")
            # optional debug save of the skipped window
            if getattr(self, "_debug_save_all", False):
                try:
                    os.makedirs("logs", exist_ok=True)
                    sf.write(
                        f"logs/wake_eval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav",
                        wav, self.sr
                    )
                except Exception:
                    pass
            return False

        print(f"[Wake] raw_energy={raw_energy:.4f}, noise_floor={old_nf:.4f}->{self._noise_floor:.4f}, gate={gate:.4f}")

        # --- inference (same path as tester) ---
        try:
            conf = self._predict(wav)  # 0..1
        except Exception as e:
            print(f"[Wake] ⚠️ predict error: {e}")
            conf = 0.0

        # smooth with rolling window
        self._conf_buf.append(conf)
        avg = float(np.mean(self._conf_buf)) if len(self._conf_buf) else conf

        # always log (you asked to log everything)
        print(f"[Wake] conf={conf:.3f} | avg={avg:.3f} | energy={raw_energy:.4f}")

        # optional: save every eval window for inspection
        if getattr(self, "_debug_save_all", False):
            try:
                os.makedirs("logs", exist_ok=True)
                sf.write(
                    f"logs/wake_eval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav",
                    wav, self.sr
                )
            except Exception:
                pass

        # --- decision: threshold OR boosted rolling avg ---
        fired = (conf >= self.THRESH) or (avg >= (self.THRESH + self.AVG_BOOST))

        if not fired:
            print("[Wake] Below thresholds, no fire.")
            return False

        # --- debounce/cooldown ---
        since = now - self._last_fire
        min_gap = max(self.debounce_s, self.cooldown_s)
        if since < min_gap:
            print(f"[Wake] Fire suppressed by debounce/cooldown ({since:.2f}s < {min_gap:.2f}s).")
            return False

        # --- FIRE! ---
        self._last_fire = now
        self.disarm()  # prevent immediate re-trigger until re-armed by core

        # save success window
        try:
            os.makedirs("logs", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            sf.write(f"logs/wake_success_{ts}.wav", wav, self.sr)
            print(f"✅ Wake word fired! saved: logs/wake_success_{ts}.wav")
        except Exception:
            pass

        # reset smoother so we don't immediately keep a high avg
        self._conf_buf.clear()
        return True

    # ------------------------------------------------------------------ #
    #                         INTERNAL HELPERS
    # ------------------------------------------------------------------ #
    def _predict_conf(self, wav_np: np.ndarray) -> float:
        # Test does: resample to 16k (already 16k), encode_batch, classifier, .item()
        wav = torch.tensor(wav_np, dtype=torch.float32)
        if wav.ndim > 1:
            wav = wav.mean(1)
        if wav.numel() == 0:
            return 0.0

        wav = torchaudio.functional.resample(wav, self.sr, 16000)
        with torch.no_grad():
            conf = self._model.mods.classifier(self._model.encode_batch(wav)).item()
        return float(np.clip(conf, 0.0, 1.0))

    def _save_eval_wav(self, wav: np.ndarray, tag: str):
        try:
            os.makedirs("logs", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            sf.write(f"logs/wake_eval_{tag}_{ts}.wav", wav, self.sr)
        except Exception:
            pass
    def _predict(self, wav: np.ndarray) -> float:
        """Run inference just like the live tester."""
        with torch.no_grad():
            t = torch.tensor(wav, dtype=torch.float32, device=self._device)
            if t.ndim == 1:
                t = t.unsqueeze(0)  # (1, T)
            # encode → classifier
            emb = self._model.encode_batch(t)
            conf = self._model.mods.classifier(emb).item()
            return float(np.clip(conf, 0.0, 1.0))
