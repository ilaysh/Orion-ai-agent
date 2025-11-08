# orion_core/session_manager.py
from datetime import datetime
import json
import os


class SessionManager:
    """
    Keeps a simple in-memory log of conversation turns.
    Can optionally persist each session to /logs/sessions.
    """

    def __init__(self, save_dir: str = "logs/sessions", autosave: bool = True):
        self.turns: list[dict] = []
        self.autosave = autosave
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------- Public API -------------------------
    def add_turn(self, role: str, text: str):
        """Append a message turn (user/orion/system)."""
        if not text:
            return
        entry = {
            "role": role,
            "text": text.strip(),
            "timestamp": datetime.now().isoformat(timespec="seconds")
        }
        self.turns.append(entry)
        if self.autosave:
            self._save()

    def get_context(self, n: int = 6) -> str:
        """
        Return last n exchanges as context string.
        Useful for RAG or LLM context injection.
        """
        subset = self.turns[-n:]
        lines = [f"{t['role']}: {t['text']}" for t in subset]
        return "\n".join(lines)

    def clear(self):
        """Reset current conversation."""
        self.turns.clear()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------- Internal -------------------------
    def _save(self):
        """Write conversation log to disk (rotating JSON)."""
        path = os.path.join(self.save_dir, f"{self.session_id}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.turns, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Session] ⚠️ Save failed: {e}")
