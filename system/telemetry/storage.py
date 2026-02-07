
# storage.py
import json
import time
from pathlib import Path

LOG_DIR = Path(__file__).parent
TEXT_FILE = LOG_DIR / "telemetry.log"
JSONL_FILE = LOG_DIR / "telemetry.jsonl"


def write_text(text: str):
    with open(TEXT_FILE, "a") as f:
        f.write(text + "\n")


def write_jsonl(entry: dict):
    entry["timestamp"] = time.time()
    with open(JSONL_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_last_jsonl(max_lines: int = 200):
    """Read last N items from telemetry.jsonl safely."""
    if not JSONL_FILE.exists():
        return []

    with JSONL_FILE.open("r", encoding="utf-8") as f:
        lines = f.readlines()[-max_lines:]

    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
