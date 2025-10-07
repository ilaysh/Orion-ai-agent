import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from pathlib import Path
import traceback

# ------------------ Config ------------------
BASE_DIR = Path("models/orion_dataset")  # fixed typo from 'mmodels'
POS_DIR = BASE_DIR / "positives"
NEG_DIR = BASE_DIR / "negatives"
AUG_POS = BASE_DIR / "positives_aug"
AUG_NEG = BASE_DIR / "negatives_aug"
AUG_POS.mkdir(parents=True, exist_ok=True)
AUG_NEG.mkdir(parents=True, exist_ok=True)

TARGET_RATE = 16000
VARIANTS = [
    ("pitch_up", {"n_steps": 2}),
    ("pitch_down", {"n_steps": -2}),
    ("fast", {"rate": 1.1}),
    ("slow", {"rate": 0.9}),
    ("noise", {"noise_db": -25}),
    ("reverb", {"reverb": True}),
]

# ------------------ Core Logic ------------------
def augment(wav, sr, params):
    """Apply augmentation variants."""
    try:
        if "n_steps" in params:
            wav = librosa.effects.pitch_shift(wav, sr=sr, n_steps=params["n_steps"])
        if "rate" in params:
            import torch, torchaudio
            wav_t = torch.tensor(wav).unsqueeze(0)
            effect = [["tempo", str(params["rate"])]]
            wav_t, _ = torchaudio.sox_effects.apply_effects_tensor(wav_t, TARGET_RATE, effect)
            wav = wav_t.squeeze().numpy()

        if "noise_db" in params:
            noise = np.random.randn(len(wav))
            wav = librosa.util.normalize(wav + 10 ** (params["noise_db"] / 20) * noise)
        if params.get("reverb"):
            tail = np.convolve(wav, np.ones(int(0.05 * sr)) / 20, mode="full")
            wav = np.pad(tail[: len(wav)], (0, max(0, len(wav) - len(tail))))
        return librosa.util.normalize(wav)
    except Exception as e:
        print(f"⚠️ Augment error: {e}")
        traceback.print_exc()
        return wav  # fallback: return original


def process_folder(in_dir, out_dir):
    """Augment all wavs in a folder."""
    files = list(Path(in_dir).glob("*.wav"))
    if not files:
        print(f"⚠️ No WAV files found in {in_dir}")
        return

    print(f"\n🎧 Processing {len(files)} files from {in_dir} → {out_dir}")

    for f in tqdm(files, desc=f"Augmenting {in_dir.name}"):
        try:
            wav, sr = librosa.load(f, sr=TARGET_RATE)
            sf.write(out_dir / f.name, wav, sr)  # original copy
            for name, params in VARIANTS:
                aug = augment(wav, sr, params)
                out_file = out_dir / f"{f.stem}_{name}.wav"
                sf.write(out_file, aug, sr)
        except Exception as e:
            print(f"❌ Error processing {f}: {e}")
            traceback.print_exc()

# ------------------ Entry ------------------
if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "all"

    if t in ("all", "positives"):
        process_folder(POS_DIR, AUG_POS)
    if t in ("all", "negatives"):
        process_folder(NEG_DIR, AUG_NEG)

    pos_count = len(list(AUG_POS.glob("*.wav")))
    neg_count = len(list(AUG_NEG.glob("*.wav")))
    print("\n✅ Augmentation complete!")
    print(f"Positives generated: {pos_count}")
    print(f"Negatives generated: {neg_count}")
