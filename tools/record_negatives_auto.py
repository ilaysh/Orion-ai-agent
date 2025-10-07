import sounddevice as sd
import soundfile as sf
import os, time

SAMPLE_RATE = 16000
DURATION = 2.0
COUNT = 15
OUT_DIR = "models/orion_dataset/poitives"
os.makedirs(OUT_DIR, exist_ok=True)

print("\n=== Orion Negative Recorder ===")
print("It will record 15 short clips (2s each).")
print("Talk normally, move around, play TV/radio, etc.")
input("Press ENTER to start...\n")

for i in range(1, COUNT + 1):
    filename = os.path.join(OUT_DIR, f"neg_auto_{i:02d}.wav")
    print(f"[{i}/{COUNT}] Recording {filename} ...")
    audio = sd.rec(int(SAMPLE_RATE * DURATION),
                   samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    sf.write(filename, audio, SAMPLE_RATE)
    print("✅ Saved.")
    time.sleep(1.0)

print("\n✅ Finished recording negatives at:", OUT_DIR)
