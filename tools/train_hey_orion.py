import os, random, torch, torch.nn as nn, torch.optim as optim
import torchaudio, numpy as np, librosa
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

# ===========================
# Config
# ===========================
SAMPLE_RATE = 16000
N_MFCC = 40
FIXED_FRAMES = 48     # ensure uniform length
EPOCHS = 25
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE = Path("models/merged")
POS_DIRS = [BASE / "positives", BASE / "positives_aug"]
NEG_DIRS = [BASE / "negatives", BASE / "negatives_aug"]
MODEL_OUT = "models/orion_wake.onnx"

# ===========================
# Data loading
# ===========================
def load_wav(path):
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav.mean(0).numpy()
    # normalize to [-1, 1]
    wav = wav / np.max(np.abs(wav) + 1e-8)
    return wav

def extract_mfcc(wav):
    mfcc = librosa.feature.mfcc(y=wav, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    # Pad or trim to FIXED_FRAMES
    if mfcc.shape[1] < FIXED_FRAMES:
        pad = FIXED_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode="constant")
    elif mfcc.shape[1] > FIXED_FRAMES:
        mfcc = mfcc[:, :FIXED_FRAMES]
    return torch.tensor(mfcc).unsqueeze(0)

class WakeDataset(Dataset):
    def __init__(self, pos_dirs, neg_dirs):
        self.data = []
        for label, dirs in [(1, pos_dirs), (0, neg_dirs)]:
            for d in dirs:
                for f in Path(d).glob("*.wav"):
                    self.data.append((f, label))
        random.shuffle(self.data)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        path, label = self.data[idx]
        wav = load_wav(path)
        mfcc = extract_mfcc(wav)
        return mfcc, torch.tensor(label, dtype=torch.float32)

# ===========================
# Model
# ===========================
class WakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # compute flattened size dynamically
        test_in = torch.zeros(1, 1, N_MFCC, FIXED_FRAMES)
        flat_dim = self.conv(test_in).view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)

# ===========================
# Train
# ===========================
def train():
    dataset = WakeDataset(POS_DIRS, NEG_DIRS)
    print(f"🧠 Using device: {DEVICE}")
    print(f"📦 Total samples: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = WakeCNN().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for epoch in range(EPOCHS):
        model.train()
        losses = []
        correct = 0
        for mfcc, label in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            mfcc, label = mfcc.to(DEVICE), label.to(DEVICE)
            out = model(mfcc).squeeze()
            loss = loss_fn(out, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            correct += ((out > 0.5) == (label > 0.5)).sum().item()
        acc = correct / len(dataset) * 100
        print(f"Epoch {epoch+1}: Loss={np.mean(losses):.4f} | Accuracy={acc:.2f}%")

    torch.save(model.state_dict(), "models/orion_wake.pth")
    print("✅ Saved: models/orion_wake.pth")

    dummy = torch.randn(1, 1, N_MFCC, FIXED_FRAMES).to(DEVICE)
    torch.onnx.export(model, dummy, MODEL_OUT, input_names=["input"], output_names=["output"])
    print(f"📦 Exported ONNX → {MODEL_OUT}")

if __name__ == "__main__":
    train()
