import torch
import numpy as np
import pyaudio
import time

# --- Load Silero VAD model from torch.hub ---
print("Loading Silero VAD model...")
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
(get_speech_ts, _, _, _, _) = utils
model.eval()

sr = 16000
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=sr, input=True,
                frames_per_buffer=1024)

print("\n🎙️ Speak naturally for ~30 seconds, then press Ctrl+C to stop.\n")
frames = []
try:
    while True:
        data = np.frombuffer(stream.read(1024, exception_on_overflow=False), dtype=np.int16)
        frames.append(data)
except KeyboardInterrupt:
    pass

stream.stop_stream()
stream.close()
p.terminate()

print("Processing...")
audio = np.concatenate(frames).astype(np.float32) / 32768.0

segments = get_speech_ts(audio, model, sampling_rate=sr, threshold=0.25)

# compute silent gaps between speech segments
pauses = []
for i in range(1, len(segments)):
    gap = (segments[i]['start'] - segments[i-1]['end']) / sr
    pauses.append(gap)

if pauses:
    print(f"Detected {len(pauses)} pauses")
    print(f"Average pause: {np.mean(pauses):.2f}s")
    print(f"Median pause:  {np.median(pauses):.2f}s")
    print(f"Longest pause: {np.max(pauses):.2f}s")
else:
    print("No pauses detected.")
