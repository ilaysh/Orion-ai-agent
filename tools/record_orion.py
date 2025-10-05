import sounddevice as sd
import soundfile as sf
import os, time

# =====================
# Configuration
# =====================
SAMPLE_RATE = 16000
DURATION = 1.5  # seconds per clip
BASE_DIR = "models/orion"
POS_DIR = os.path.join(BASE_DIR, "positives")
NEG_DIR = os.path.join(BASE_DIR, "negatives")
os.makedirs(POS_DIR, exist_ok=True)
os.makedirs(NEG_DIR, exist_ok=True)

# =====================
# Recording Helper
# =====================
def record_clip(filename):
    print(f"🎙️ Recording... ({DURATION}s)")
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    sf.write(filename, audio, SAMPLE_RATE)
    print(f"✅ Saved: {filename}\n")
    time.sleep(1)

# =====================
# Instruction Sequences
# =====================
positive_prompts = [
    "Say clearly: 'Orion'",
    "Say softly: 'Orion'",
    "Say 'Orion' while facing away slightly",
    "Say 'Orion' from a few meters away",
    "Say 'Orion' with a fan or background noise on",
    "Say 'Orion' while typing or moving a chair",
    "Ask your wife to say 'Orion'",
    "Ask your wife to say 'Orion' again, a bit louder",
    "Say 'Hey Orion'",
    "Say 'Orion' fast and low",
]

negative_prompts = [
    "Stay silent for a moment (ambient noise only)",
    "Say something else: 'Computer'",
    "Say: 'Hey there'",
    "Talk normally about your day for 1–2 seconds",
    "Play a song in background for 1–2 seconds",
    "Ask your wife to say something random (not 'Orion')",
    "Make typing or chair movement sounds",
    "Say: 'Ok Google'",
    "Say: 'Jarvis'",
    "Say: 'Weather today?'",
]

# =====================
# Guided Recording Loop
# =====================
print("\n=== Guided Wake Word Dataset Recorder ===")
print("Press ENTER to start each instruction.\n")

# --- Positives ---
print("🎯 Recording POSITIVES (Wake word: 'Orion')\n")
for i, prompt in enumerate(positive_prompts, 1):
    input(f"[{i}/{len(positive_prompts)}] {prompt}  → Press ENTER to record.")
    record_clip(os.path.join(POS_DIR, f"orion_{i:02d}.wav"))

# --- Negatives ---
print("\n🚫 Recording NEGATIVES (non-wake words / background)\n")
for i, prompt in enumerate(negative_prompts, 1):
    input(f"[{i}/{len(negative_prompts)}] {prompt}  → Press ENTER to record.")
    record_clip(os.path.join(NEG_DIR, f"neg_{i:02d}.wav"))

print("\n✅ All done! Your dataset is ready at:")
print(f"  → {POS_DIR}")
print(f"  → {NEG_DIR}")
