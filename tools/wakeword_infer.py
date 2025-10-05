import numpy as np
import sounddevice as sd
import onnxruntime as ort
import librosa
import time

# ===== Settings =====
SAMPLE_RATE = 16000
N_MFCC = 40
CHUNK_DURATION = 1.0   # seconds
THRESHOLD = 0.6
MODEL_PATH = "models/orion_wake.onnx"

# ===== Init =====
print("🎧 Initializing ONNX runtime...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
print(f"✅ Model loaded: {MODEL_PATH}")

def extract_mfcc(wav):
    """Convert raw waveform → fixed-size MFCC tensor (1, 1, N_MFCC, 48 frames)."""
    mfcc = librosa.feature.mfcc(
        y=wav, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        n_fft=512, hop_length=320  # 20 ms hop
    )
    target_frames = 48
    if mfcc.shape[1] < target_frames:
        # Pad with zeros on the right
        pad_width = target_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        # Crop if too long
        mfcc = mfcc[:, :target_frames]

    mfcc = mfcc[np.newaxis, np.newaxis, :, :]  # [B, C, 40, 48]
    return mfcc.astype(np.float32)

def predict(wav):
    mfcc = extract_mfcc(wav)
    pred = session.run(None, {"input": mfcc})[0]
    return float(pred.squeeze())

def listen_once(duration=CHUNK_DURATION):
    """Record a single chunk of audio."""
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze()

# ===== Main Loop =====
print("🎤 Say 'Orion' to trigger (Ctrl+C to exit)\n")
try:
    while True:
        wav = listen_once()
        energy = np.mean(np.abs(wav))
        if energy < 0.005:
            # Skip silent parts
            continue
        pred = predict(wav)
        print(f"🧠 Prediction: {pred:.3f}")
        if pred > THRESHOLD:
            print("✅ Wake word detected!\n")
            time.sleep(1.5)  # debounce
except KeyboardInterrupt:
    print("\n👋 Exiting.")
