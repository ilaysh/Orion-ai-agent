import sounddevice as sd
import soundfile as sf
import os, time

SAMPLE_RATE = 16000
DURATION = 2.0
COUNT = 1
OUT_DIR = "models/orion_dataset/positives"
os.makedirs(OUT_DIR, exist_ok=True)

print("\n=== Orion Positive Recorder ===")
print(f"It will record {COUNT} short 2-second clips.")
print("Say 'Orion' naturally each time (change tone, distance, volume).")
input("Press ENTER to begin...\n")

for i in range(1, COUNT + 1):
    filename = os.path.join(OUT_DIR, f"orion_auto_{i:02d}.wav")
    print(f"[{i}/{COUNT}] Recording {filename} ...")
    audio = sd.rec(int(SAMPLE_RATE * DURATION),
                   samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    sf.write(filename, audio, SAMPLE_RATE)
    print("✅ Saved.")
    time.sleep(1.0)

print("\n✅ Finished recording new positives at:", OUT_DIR)
