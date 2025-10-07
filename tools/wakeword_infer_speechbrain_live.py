import os
import numpy as np
import sounddevice as sd
import torch, torchaudio, time
from datetime import datetime
from collections import deque
from speechbrain.inference import EncoderClassifier
import soundfile as sf
from datetime import datetime
        
# ====== CONFIGURATION ======
SAMPLE_RATE = 16000
CHUNK_DURATION = 1.0     # seconds
THRESHOLD = 0.75         # wake-word confidence threshold
ENERGY_MIN = 0.004         # skip chunks with RMS below this
SMOOTH_WINDOW = 3         # rolling avg window (frames)
DEBOUNCE_TIME = 1.5       # seconds to wait after detection
MODEL_PATH = "models/orion_speechbrain_full_finetune.pt"
LOG_TIMESTAMPS = True
# ===========================

print("🎧 Initializing SpeechBrain wake-word model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

base = EncoderClassifier.from_hparams(
    source="speechbrain/google_speech_command_xvector",
    run_opts={"device": device}
)
# find last linear layer
last_linear = next(m for m in base.mods.classifier.modules() if isinstance(m, torch.nn.Linear))
in_dim = last_linear.in_features
base.mods.classifier = torch.nn.Sequential(
    torch.nn.Linear(in_dim, 1),
    torch.nn.Sigmoid()
).to(device)
base.load_state_dict(torch.load(MODEL_PATH, map_location=device))
base.eval()
print(f"✅ Model loaded: {MODEL_PATH}")

# Rolling average buffer for smoother detection
conf_buffer = deque(maxlen=SMOOTH_WINDOW)

def predict_chunk(wav_np: np.ndarray) -> float:
    """Run inference on raw waveform chunk"""
    wav = torch.tensor(wav_np, dtype=torch.float32)
    if wav.ndim > 1:
        wav = wav.mean(1)
    if len(wav) == 0:
        return 0.0
    wav = torchaudio.functional.resample(wav, SAMPLE_RATE, 16000)
    with torch.no_grad():
        conf = base.mods.classifier(base.encode_batch(wav)).item()
    return float(conf)

def listen_once(duration=CHUNK_DURATION):
    """Record a single chunk of audio from mic"""
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze()

print("🎤 Listening for 'Orion' (Ctrl+C to exit)\n")
try:
    noise_floor = 0.002
    
    while True:
        wav = listen_once()
        wav = torchaudio.functional.highpass_biquad(torch.tensor(wav), SAMPLE_RATE, 100).numpy()
        energy = np.mean(np.abs(wav))

        # Skip silence or low-energy ambient chunks
        if energy < noise_floor * 1.5:
            continue
    
        noise_floor = 0.98 * noise_floor + 0.02 * energy
        conf = predict_chunk(wav)
        conf_buffer.append(conf)
        avg_conf = np.mean(conf_buffer)

        print(f"🧠 Prediction: {conf:.3f} | avg={avg_conf:.3f} | energy={energy:.3f}")

        if conf > THRESHOLD or avg_conf > (THRESHOLD + 0.05):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"logs/wake_trigger_{ts}.wav"
            os.makedirs("logs", exist_ok=True)
            sf.write(fname, wav, SAMPLE_RATE)
            print(f"✅ [{ts}] Wake word detected! Saved: {fname}\n")
            conf_buffer.clear()
            time.sleep(DEBOUNCE_TIME)

except KeyboardInterrupt:
    print("\n👋 Exiting.")
