# orion_core/brain/session_manager.py
import os
import json
from datetime import datetime
from typing import List, Dict, Any

from orion_core.brain.people_manager import PeopleManager
from orion_core.brain.directives import DirectivesManager

class SessionManager:
    """
    The System Monitor & Context Aggregator.
    """
    def __init__(self, save_dir: str = "logs/sessions", autosave: bool = True):
        self.save_dir = save_dir
        self.autosave = autosave
        
        # MANAGERS
        self.people = PeopleManager()
        self.directives = DirectivesManager()
        
        self.turns: List[Dict[str, str]] = []
        self.current_speaker = "Unknown"
        
        os.makedirs(save_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def set_speaker(self, name: str):
        self.current_speaker = name

    def get_world_state(self) -> str:
        """Builds the context block for the LLM."""
        # 1. Identity
        owner = self.people.get_owner_name()
        now = datetime.now().strftime('%H:%M, %A, %Y-%m-%d')
        speaker = self.current_speaker
        
        # 2. Hard Constraints (The Law)
        laws = self.directives.get_directives()
        laws_block = ""
        if laws:
            laws_str = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(laws)])
            laws_block = f"\n[PRIME DIRECTIVES]\n{laws_str}"

        # 3. Status & Paths
        # Minimal file context (Just root). Agent uses `fs` to explore more.
        root_path = os.getcwd()
        status_line = "System Status: ONLINE"

        return (
            f"WORLD STATE:\n"
            f"- Time: {now}\n"
            f"- Owner: {owner}\n"
            f"- Current Speaker: {speaker}\n"
            f"- Project Root: {root_path}\n"
            f"{laws_block}\n"
            f"- {status_line}"
        )

    def add_turn(self, role: str, text: str):
        if not text: return
        self.turns.append({"role": role, "text": text.strip(), "timestamp": datetime.now().isoformat()})
        if len(self.turns)>50: self.turns.pop(0)
        if self.autosave: self._save()

    def format_history(self, limit=10):
        # Format: "User (Name): text"
        return "\n".join([f"{m['role'].upper()}: {m['text']}" for m in self.turns[-limit:]]) or "No history."

    def _save(self):
        try:
            with open(os.path.join(self.save_dir, f"{self.session_id}.json"), "w") as f:
                json.dump(self.turns, f, indent=2)
        except: pass