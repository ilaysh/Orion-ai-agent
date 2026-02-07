# orion_core/brain/memory_unit.py
import json
import os
from typing import List

class MemoryUnit:
    """
    Hybrid Memory Manager.
    - Directives: Hard rules injected 100% of the time. (JSON)
    - Entities: Structured data like People/Devices. (JSON)
    - Episodic: Fuzzy conversation history. (RAG - Optional/Later)
    """
    def __init__(self):
        self.mem_dir = "orion_core/memory"
        self.directives_path = os.path.join(self.mem_dir, "directives.json")
        self.people_path = os.path.join(self.mem_dir, "people.json")
        self._ensure_paths()

    def _ensure_paths(self):
        if not os.path.exists(self.mem_dir): os.makedirs(self.mem_dir)
        if not os.path.exists(self.directives_path):
            with open(self.directives_path, "w") as f: json.dump([], f)

    def get_directives(self) -> List[str]:
        """Returns the list of laws Orion must obey."""
        try:
            with open(self.directives_path, "r") as f:
                return json.load(f)
        except: return []

    def add_directive(self, rule: str):
        """User adds a new law."""
        rules = self.get_directives()
        if rule not in rules:
            rules.append(rule)
            with open(self.directives_path, "w") as f: json.dump(rules, f, indent=2)

    def get_people_context(self) -> str:
        """Flattens people tree for the prompt (Simplified)."""
        # Load JSON, format as "Wife: Neomi", etc.
        return ""