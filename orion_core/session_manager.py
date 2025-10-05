# orion_core/session_manager.py
import json, os, time
from datetime import datetime

class SessionManager:
    def __init__(self, base_dir="sessions"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.current_file = None
        self.session_data = []

    def start(self, wake_word="orion"):
        """Start a new conversation session and create a JSON log file"""
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_file = os.path.join(self.base_dir, f"{wake_word}_{ts}.json")
        self.session_data = [{
            "timestamp": ts,
            "wake_word": wake_word,
            "events": []
        }]
        self._save()
        print(f"[SESSION] Started new session: {self.current_file}")

    def log(self, role, text):
        """Append a new message to the current session"""
        if not self.current_file:
            self.start("unknown")
        self.session_data[0]["events"].append({
            "time": time.time(),
            "role": role,
            "text": text
        })
        self._save()

    def end(self):
        """End the current session and finalize"""
        if self.current_file:
            print(f"[SESSION] Ended: {self.current_file}")
            self.current_file = None
            self.session_data = []

    def _save(self):
        if self.current_file:
            with open(self.current_file, "w", encoding="utf-8") as f:
                json.dump(self.session_data, f, ensure_ascii=False, indent=2)
