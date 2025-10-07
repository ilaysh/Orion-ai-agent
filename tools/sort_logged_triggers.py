import os, shutil, torchaudio
from pathlib import Path
import numpy as np
from datetime import datetime

# Paths
LOG_DIR = Path("logs")
NEG_DIR = Path("models/orion_dataset/negatives_auto")
os.makedirs(NEG_DIR, exist_ok=True)

# Settings
SILENCE_THRESHOLD = 0.003   # skip true silence
MAX_DURATION = 2.0          # skip accidental long recordings (seconds)

count = 0
for wav_file in sorted(LOG_DIR.glob("wake_trigger_*.wav")):
    wav, sr = torchaudio.load(wav_file)
    wav_np = wav.numpy()            # ✅ convert tensor → numpy
    dur = len(wav_np[0]) / sr
    energy = float(np.mean(np.abs(wav_np)))
    if dur > MAX_DURATION or energy < SILENCE_THRESHOLD:
        continue  # ignore silence or corrupted
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = NEG_DIR / f"neg_auto_{ts}.wav"
    shutil.copy(wav_file, dst)
    print(f"⚙️ Copied {wav_file.name} → {dst.name} (energy={energy:.4f})")
    count += 1

print(f"\n✅ {count} files moved to {NEG_DIR}")
