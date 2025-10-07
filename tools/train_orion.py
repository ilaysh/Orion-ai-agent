import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import librosa
from sklearn.model_selection import train_test_split
import onnx

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
N_MFCC = 40
N_FRAMES = 48
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
MODEL_PATH = "models/orion_wake.onnx"

POS_DIR = Path("models/orion_dataset/positives_aug")
NEG_DIR = Path("models/orion_dataset/negatives_aug")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- DATASET ----------------
def extract_mfcc(file):
    wav, sr = librosa.load(file, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(
        y=wav, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=512, hop_length=320
    )
    if mfcc.shape[1] < N_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, N_FRAMES - mfcc.shape[1])), mode="constant")
    elif mfcc.shape[1] > N_FRAMES:
        mfcc = mfcc[:, :N_FRAMES]
    return mfcc.astype(np.float32)

class WakewordDataset(Dataset):
    def __init__(self, pos_files, neg_files):
        # balance dataset
        min_len = min(len(pos_files), len(neg_files))
        pos_files = random.sample(pos_files, min_len)
        neg_files = random.sample(neg_files, min_len)
        self.data = [(f, 1) for f in pos_files] + [(f, 0) for f in neg_files]
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        f, label = self.data[idx]
        x = extract_mfcc(f)
        x = torch.tensor(x).unsqueeze(0)  # [1, 40, 48]
        return x, torch.tensor(label, dtype=torch.float32)

# ---------------- MODEL ----------------
class WakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)     # <— new line
        self.fc1 = nn.Linear(64 * 5 * 6, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)                # <— new line
        x = torch.sigmoid(self.fc2(x))
        return x

# ---------------- TRAIN ----------------
def train_model(model, loader, optimizer, criterion):
    model.train()
    total, correct, loss_sum = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        preds = (out > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item()
    return loss_sum / len(loader), correct / total

def eval_model(model, loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            out = model(x)
            preds = (out > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# ---------------- MAIN ----------------
if __name__ == "__main__":
    pos_files = list(POS_DIR.glob("*.wav"))
    neg_files = list(NEG_DIR.glob("*.wav"))

    if len(pos_files) == 0 or len(neg_files) == 0:
        raise RuntimeError("No training data found! Check your paths.")

    train_pos, val_pos = train_test_split(pos_files, test_size=0.1, random_state=42)
    train_neg, val_neg = train_test_split(neg_files, test_size=0.1, random_state=42)

    train_set = WakewordDataset(train_pos, train_neg)
    val_set = WakewordDataset(val_pos, val_neg)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = WakeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_model(model, train_loader, optimizer, criterion)
        val_acc = eval_model(model, val_loader)
        print(f"Epoch {epoch:02d}: loss={loss:.4f}  train_acc={acc:.3f}  val_acc={val_acc:.3f}")

    # Save ONNX
    dummy = torch.randn(1, 1, N_MFCC, N_FRAMES).to(device)
    torch.onnx.export(model, dummy, MODEL_PATH, input_names=["input"], output_names=["output"], opset_version=12)
    print(f"\n✅ Model exported to {MODEL_PATH}")
