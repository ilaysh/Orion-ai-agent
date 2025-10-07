import numpy as np
import sounddevice as sd
import threading, time, queue, torch, torchaudio, os
from datetime import datetime
from speechbrain.inference import EncoderClassifier
import soundfile as sf


class WakeWordThread(threading.Thread):
    def __init__(self, model_path, on_detect, cooldown_s: float = 2.0):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.on_detect = on_detect
        self.cooldown_s = cooldown_s
        self._stop = threading.Event()
        self._next_arm = 0.0
        self.armed = True
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.q = queue.Queue()

        # Load SpeechBrain fine-tuned model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Wake] Loading SpeechBrain model on {device}...")
        self.base = EncoderClassifier.from_hparams(
            source="speechbrain/google_speech_command_xvector",
            run_opts={"device": device}
        )
        last_linear = next(m for m in self.base.mods.classifier.modules() if isinstance(m, torch.nn.Linear))
        in_dim = last_linear.in_features
        self.base.mods.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 1),
            torch.nn.Sigmoid()
        ).to(device)
        self.base.load_state_dict(torch.load(model_path, map_location=device))
        self.base.eval()
        self.device = device
        print(f"✅ Wake word model loaded: {model_path}")

    # ---- Control ----
    def stop(self):
        self._stop.set()

    def arm(self):
        self.armed = True
        self._next_arm = time.time() + self.cooldown_s
        print("[Wake] re-armed")

    def disarm(self):
        self.armed = False
        print("[Wake] disarmed")

    # ---- Audio ----
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print("[Wake] audio status:", status)
        self.q.put(indata.copy())

    def run(self):
        """Background audio listening loop"""
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        ):
            print("[Wake] Listening for wake word...")

            while not self._stop.is_set():
                if not self.armed:
                    time.sleep(0.05)
                    continue

                try:
                    chunk = self.q.get(timeout=0.1).squeeze()
                except queue.Empty:
                    continue

                if not np.isfinite(chunk).any():
                    continue

                # Normalize safely
                peak = np.max(np.abs(chunk))
                if peak > 0:
                    chunk = chunk / peak
                else:
                    continue

                # Convert to tensor
                wav = torch.tensor(chunk, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    conf = self.base.mods.classifier(self.base.encode_batch(wav)).item()

                print(f"[Wake] conf={conf:.3f}")

                if conf > 0.75 and time.time() > self._next_arm:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    os.makedirs("logs", exist_ok=True)
                    sf.write(f"logs/wake_success_{ts}.wav", chunk, self.sample_rate)
                    print(f"✅ Wake word fired! saved: logs/wake_success_{ts}.wav")

                    # Disarm to prevent multiple triggers
                    self.disarm()
                    self._next_arm = time.time() + self.cooldown_s
                    try:
                        self.on_detect()
                    except Exception as e:
                        print(f"[Wake] on_detect error: {e}")

                time.sleep(0.01)
