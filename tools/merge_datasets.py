from pathlib import Path
import shutil

SRC_DIRS = ["models/hey_orion", "models/orion"]
DEST = Path("models/merged")

for label in ["positives", "negatives"]:
    dest_dir = DEST / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in SRC_DIRS:
        src_dir = Path(src) / label
        if src_dir.exists():
            for f in src_dir.glob("*.wav"):
                shutil.copy(f, dest_dir / f"{src_dir.parent.name}_{f.name}")
print("✅ Merged datasets into models/merged/")