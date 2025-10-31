# tools/test_vad_record.py
import torch
import numpy as np
import soundfile as sf
import pyaudio
import datetime
import os
model, utils = torch.hub.load(
    'snakers4/silero-vad', 'silero_vad', force_reload=False)
(get_speech_ts, _, _, _, _) = utils
model.eval()
sr = 16000
buf = []
os.makedirs("logs/vad_test", exist_ok=True)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1,
                rate=sr, input=True, frames_per_buffer=1024)

print("🎙️ Speak naturally — stops automatically after 1 s of silence.")
silence_frames, max_silence = 0, int(sr/1024*1.0)   # ~1 s
while True:
    data = np.frombuffer(stream.read(
        1024, exception_on_overflow=False), dtype=np.int16)
    buf.append(data)
    audio = np.concatenate(buf).astype(np.float32)/32768.0
    segs = get_speech_ts(audio, model, sampling_rate=sr, threshold=0.27)
    if len(segs) == 0:
        silence_frames += 1
    else:
        silence_frames = 0
    if silence_frames > max_silence:
        break

stream.stop_stream()
stream.close()
p.terminate()
audio = np.concatenate(buf).astype(np.float32)/32768.0
path = f"logs/vad_test/vad_capture_{datetime.datetime.now().strftime('%H%M%S')}.wav"
sf.write(path, audio, sr)
print(f"💾 Saved {path}")
