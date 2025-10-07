import sounddevice as sd
import soundfile as sf
import time, os

SAMPLE_RATE = 16000
DURATION = 2.0   # seconds per clip
BASE = "models/orion_dataset"
POS_DIR = os.path.join(BASE, "positives")
NEG_DIR = os.path.join(BASE, "negatives")
os.makedirs(POS_DIR, exist_ok=True)
os.makedirs(NEG_DIR, exist_ok=True)

def record_clip(filename):
    print(f"🎙️ Recording {filename} ({DURATION}s)...")
    audio = sd.rec(int(SAMPLE_RATE * DURATION),
                   samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    sf.write(filename, audio, SAMPLE_RATE)
    print(f"✅ Saved {filename}\n")
    time.sleep(0.8)

# ---------- PROMPTS ----------
positives = [
    "Say clearly: 'Orion'",
    "Say: 'Hey Orion'",
    "Say softly: 'Orion'",
    "Say: 'Orion' from 1 m away",
    "Say: 'Orion' with fan noise",
    "Say: 'Orion' while typing",
    "Say: 'Hey Orion' loud",
    "Say: 'Orion' normally again",
    "Ask another person to say 'Orion'",
    "Say: 'Orion' fast and quiet",
    "Say: 'Orion' after a pause",
    "Say: 'Orion' from another room",
    "Say: 'Orion' with background music",
    "Say: 'Orion' slowly",
    "Say: 'Hey Orion' clearly",
    "Say: 'Orion' from 2 m away",
    "Say: 'Orion' quietly but clear",
    "Say: 'Orion' with TV noise",
    "Say: 'Orion' normal tone again",
    "Say: 'Hey Orion' gently",
    "Say: 'Orion' short and sharp",
    "Say: 'Orion' with window open",
    "Say: 'Orion' near the mic",
    "Say: 'Orion' while walking",
    "Say: 'Hey Orion' lower tone",
    "Say: 'Orion' higher pitch",
    "Say: 'Orion' mid-distance",
    "Say: 'Orion' normally (again)",
    "Say: 'Orion' whispering",
    "Say: 'Orion' final sample",
]

negatives = [
    "Stay silent (ambient noise)",
    "Say: 'Google'",
    "Say: 'Alexa'",
    "Say: 'Jarvis'",
    "Say: 'Computer'",
    "Talk about your day",
    "Play short music clip",
    "Move a chair / keyboard sounds",
    "Ask someone to talk randomly",
    "Cough or make noise",
    "Say: 'Weather today?'",
    "Say random numbers",
    "Say: 'Hey there'",
    "Say: 'Hello world'",
    "Say: 'Service'",
    "Say: 'Orionis'",
    "Say: 'Siri'",
    "Say: 'Okay Google'",
    "Say: 'Hey assistant'",
    "Say: 'What's the time?'",
    "Say: 'Lights on'",
    "Say: 'Hey computer'",
    "Say: 'Hey system'",
    "Say: 'Cereal'",
    "Say: 'Hey Google'",
    "Say: 'Jarvis open browser'",
    "Say: 'Music please'",
    "Say: 'Hello'",
    "Say: 'Temperature today'",
    "Stay silent again (background)",
]

# ---------- MAIN ----------
print("\n=== Orion Wake Word Recorder v2 ===")
input("Press ENTER when ready.\n")

print("🎯 POSITIVE RECORDINGS (wake word: 'Orion')\n")
for i, prompt in enumerate(positives, 1):
    input(f"[{i}/{len(positives)}] {prompt} → ENTER to start.")
    record_clip(os.path.join(POS_DIR, f"orion_{i:02d}.wav"))

print("\n🚫 NEGATIVE RECORDINGS (non-wake words)\n")
for i, prompt in enumerate(negatives, 1):
    input(f"[{i}/{len(negatives)}] {prompt} → ENTER to start.")
    record_clip(os.path.join(NEG_DIR, f"neg_{i:02d}.wav"))

print("\n✅ All recordings complete. Files saved to:", BASE)
