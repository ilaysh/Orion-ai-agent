import torch, torchaudio
from speechbrain.inference import EncoderClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load base model and your fine-tuned weights
base = EncoderClassifier.from_hparams(source="speechbrain/google_speech_command_xvector", run_opts={"device":device})
# Find the last linear layer
last_linear = None
for m in base.mods.classifier.modules():
    if isinstance(m, torch.nn.Linear):
        last_linear = m
        break
in_dim = last_linear.in_features

# Replace with your fine-tuned classifier
base.mods.classifier = torch.nn.Sequential(
    torch.nn.Linear(in_dim, 1),
    torch.nn.Sigmoid()
).to(device)

base.load_state_dict(torch.load("models/orion_speechbrain.pt", map_location=device))
base.eval()

def predict(path):
    wav, sr = torchaudio.load(path)
    wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.mean(0) if wav.ndim > 1 else wav
    with torch.no_grad():
        conf = base.mods.classifier(base.encode_batch(wav)).item()
    print(f"🧠 {path}: {conf:.3f}")

# --- Test examples ---
predict("models/orion_dataset/positives/orion_auto_02.wav")
predict("models/orion_dataset/negatives/neg_auto_01.wav")
