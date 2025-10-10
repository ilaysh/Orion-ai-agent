# orion_core/wakeword.py
import os
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
from speechbrain.inference import EncoderClassifier


class WakeWordThread(threading.Thread):
    """
    Threaded wake-word detector using SpeechBrain encoder classifier.
    - If model_path is a directory with hyperparams.yaml -> load via from_hparams.
    - If model_path is a .pt file (finetuned weights) -> load base, swap head, load state_dict (like your live script).
    """

    def __init__(self, model_path: str, on_detect=None):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.on_detect = on_detect

        # Audio / stream
        self.sample_rate = 16000
        self.block_size = 1024

        # Control flags
        self.stop_flag = threading.Event()
        self.armed = threading.Event()
        self.armed.set()

        # Detection params (close to your live script, slightly stricter)
        self.THRESHOLD = 0.90          # main threshold (you asked for >0.9)
        self.SMOOTH_WINDOW = 3         # rolling average window
        self.ENERGY_MIN = 0.004        # minimum RMS to consider a frame
        self.DEBOUNCE_TIME = 1.5       # seconds lockout after trigger

        # Smoothing buffers
        self.conf_buffer = deque(maxlen=self.SMOOTH_WINDOW)
        self.noise_floor = 0.002
        self.last_trigger = 0.0

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model
        print(f"[Wake] Loading SpeechBrain model on {self.device}...")
        self.model = self._load_model(self.model_path, self.device)
        if self.model is not None:
            print(f"✅ Wake word model ready from: {self.model_path}")
        else:
            print("[Wake] ❌ Model failed to load.")

    # ---------------- Model loading (matches your live infer script) ---------------- #

    def _load_model(self, model_path: str, device: str):
        try:
            if os.path.isdir(model_path):
                # SpeechBrain checkpoint directory (with hyperparams.yaml, etc.)
                model = EncoderClassifier.from_hparams(
                    source=model_path,
                    run_opts={"device": device},
                    savedir=model_path,
                )
                model.eval()
                return model

            if os.path.isfile(model_path) and model_path.endswith(".pt"):
                # Your working path: load base, replace classifier, then load .pt state_dict
                base = EncoderClassifier.from_hparams(
                    source="speechbrain/google_speech_command_xvector",
                    run_opts={"device": device}
                )
                # find last linear layer to get in_features
                last_linear = next(
                    m for m in base.mods.classifier.modules()
                    if isinstance(m, torch.nn.Linear)
                )
                in_dim = last_linear.in_features
                base.mods.classifier = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, 1),
                    torch.nn.Sigmoid()
                ).to(device)

                state = torch.load(model_path, map_location=device)
                base.load_state_dict(state)
                base.eval()
                return base

            print(f"[Wake] ⚠️ Invalid model_path: {model_path}")
            return None

        except Exception as e:
            print(f"[Wake] ❌ Failed to load SpeechBrain model: {e}")
            return None

    # ---------------- External control ---------------- #

    def arm(self):
        self.armed.set()
        print("[Wake] armed")

    def disarm(self):
        self.armed.clear()
        print("[Wake] disarmed")

    def stop(self):
        self.stop_flag.set()
        print("[Wake] ⏹ Detection stopped.")

    # ---------------- Inference helpers (same logic as live script) ---------------- #

    def _predict_chunk(self, wav_np: np.ndarray) -> float:
        """Run inference on 1s waveform chunk (float32)."""
        wav = torch.tensor(wav_np, dtype=torch.float32)
        if wav.ndim > 1:
            wav = wav.mean(1)
        if len(wav) == 0:
            return 0.0

        # ensure 16k, then feed through encoder+head
        wav = torchaudio.functional.resample(wav, self.sample_rate, 16000)
        with torch.no_grad():
            conf = self.model.mods.classifier(self.model.encode_batch(wav)).item()
        return float(conf)

    # ---------------- Main loop ---------------- #

    def run(self):
        if self.model is None:
            print("[Wake] ❌ No model loaded, aborting thread.")
            return

        print("[Wake] ▶ Listening for wake word...")
        # Accumulate 1s chunks (like your live script)
        window = np.zeros(0, dtype=np.float32)
        target_len = self.sample_rate  # 1.0s

        with sd.InputStream(
            channels=1, samplerate=self.sample_rate, dtype="float32", blocksize=self.block_size
        ) as stream:
            while not self.stop_flag.is_set():
                if not self.armed.is_set():
                    time.sleep(0.05)
                    continue

                data, _ = stream.read(self.block_size)
                block = data.astype(np.float32).squeeze()

                # append to rolling window
                window = np.concatenate([window, block])
                if len(window) < target_len:
                    continue

                # take last 1s
                wav = window[-target_len:]
                # light HP filter like your script
                wav_t = torchaudio.functional.highpass_biquad(
                    torch.tensor(wav), self.sample_rate, 100
                ).numpy()

                # energy gate and adaptive floor (same idea as script)
                energy = float(np.mean(np.abs(wav_t)))
                if energy < self.noise_floor * 1.5 or energy < self.ENERGY_MIN:
                    self.noise_floor = 0.98 * self.noise_floor + 0.02 * energy
                    continue
                self.noise_floor = 0.98 * self.noise_floor + 0.02 * energy

                # predict + smooth
                try:
                    conf = self._predict_chunk(wav_t)
                except Exception as e:
                    print(f"[Wake] ⚠️ Inference error: {e}")
                    conf = 0.0

                self.conf_buffer.append(conf)
                avg_conf = float(np.mean(self.conf_buffer)) if self.conf_buffer else conf

                print(f"[Wake] conf={conf:.3f} | avg={avg_conf:.3f} | energy={energy:.4f}")

                now = time.time()
                fired = (conf > self.THRESHOLD) or (avg_conf > (self.THRESHOLD + 0.05))
                if fired and (now - self.last_trigger) > self.DEBOUNCE_TIME:
                    self.last_trigger = now
                    # save snippet (like your script)
                    try:
                        os.makedirs("logs", exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        fname = f"logs/wake_success_{ts}.wav"
                        sf.write(fname, wav, self.sample_rate)
                        print(f"✅ Wake word fired! saved: {fname}")
                    except Exception as e:
                        print(f"[Wake] ⚠️ Could not save snippet: {e}")

                    # disarm & signal core
                    self.disarm()
                    if self.on_detect:
                        try:
                            self.on_detect()
                        except Exception as e:
                            print(f"[Wake] ⚠️ on_detect error: {e}")

                    # clear smoothing
                    self.conf_buffer.clear()

                # slide window to keep ~1s
                window = window[-target_len:]
                time.sleep(0.005)

        print("[Wake] Thread exiting...")
