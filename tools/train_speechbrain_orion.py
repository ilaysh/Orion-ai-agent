import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio, random
from pathlib import Path
from speechbrain.inference import EncoderClassifier
from sklearn.model_selection import train_test_split

# ---------- CONFIG ----------
BASE = Path("models/orion_dataset")
POS = list((BASE / "positives_aug").glob("*.wav"))
NEG = list((BASE / "negatives_aug").glob("*.wav"))
BATCH = 32          # bigger batch
EPOCHS = 15         # more training
LR = 1e-4           # stronger adaptation
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- DATASET ----------
class OrionDataset(Dataset):
    def __init__(self, pos, neg):
        m = min(len(pos), len(neg))
        pos, neg = random.sample(pos, m), random.sample(neg, m)
        self.data = [(f, 1) for f in pos] + [(f, 0) for f in neg]
        random.shuffle(self.data)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        f, label = self.data[idx]
        wav, sr = torchaudio.load(f)
        wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.mean(0) if wav.ndim > 1 else wav
        target_len = 32000  # 2 s @ 16kHz
        if len(wav) > target_len:
            wav = wav[:target_len]
        elif len(wav) < target_len:
            wav = torch.nn.functional.pad(wav, (0, target_len - len(wav)))
        # add slight noise for robustness
        wav = wav + 0.005 * torch.randn_like(wav)
        return wav, torch.tensor(label, dtype=torch.float32)

train_p, val_p = train_test_split(POS, test_size=0.1, random_state=42)
train_n, val_n = train_test_split(NEG, test_size=0.1, random_state=42)
train_loader = DataLoader(OrionDataset(train_p, train_n), batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(OrionDataset(val_p, val_n), batch_size=BATCH)

# ---------- LOAD PRETRAINED ----------
base = EncoderClassifier.from_hparams(
    source="speechbrain/google_speech_command_xvector",
    run_opts={"device": device}
)

# find last linear layer
last_linear = None
for m in base.mods.classifier.modules():
    if isinstance(m, nn.Linear):
        last_linear = m
        break
in_dim = last_linear.in_features

# replace classifier for our binary output
base.mods.classifier = nn.Sequential(
    nn.Linear(in_dim, 1),
    nn.Sigmoid()
).to(device)

# unfreeze ALL encoder parameters for full fine-tune
encoder_module = None
if hasattr(base.mods, "encoder"):
    encoder_module = base.mods.encoder
elif "embedding_model" in base.mods:
    encoder_module = base.mods.embedding_model
else:
    encoder_module = list(base.mods.values())[0]
for p in encoder_module.parameters():
    p.requires_grad = True

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(base.parameters(), lr=LR)

# optional LR scheduler for long runs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)

# ---------- TRAIN ----------
for ep in range(1, EPOCHS + 1):
    base.train()
    total, correct, loss_sum = 0, 0, 0
    for wav, lab in train_loader:
        wav, lab = wav.to(device), lab.to(device)
        out = base.mods.classifier(base.encode_batch(wav))
        loss = criterion(out.squeeze(), lab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        preds = (out > 0.5).float()
        correct += (preds.squeeze() == lab).sum().item()
        total += lab.size(0)
        loss_sum += loss.item()
    train_acc = correct / total

    # validation
    base.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for wav, lab in val_loader:
            wav, lab = wav.to(device), lab.to(device)
            out = base.mods.classifier(base.encode_batch(wav))
            val_correct += (out > 0.5).float().squeeze().eq(lab).sum().item()
            val_total += lab.size(0)
    val_acc = val_correct / val_total
    scheduler.step(val_acc)
    print(f"Epoch {ep:02d}: loss={loss_sum/len(train_loader):.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

torch.save(base.state_dict(), "models/orion_speechbrain_full_finetune.pt")
print("\n✅ Full fine-tune complete → models/orion_speechbrain_full_finetune.pt")
