# orion_core/brain/directives.py
import json
import os
from typing import List

class DirectivesManager:
    """
    Manages Hard Constraints (The Laws of Orion).
    Stored in JSON. Injected 100% of the time.
    """
    def __init__(self, storage_path="orion_core/memory/directives.json"):
        self.path = storage_path
        self._ensure_storage()

    def _ensure_storage(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            # Default constraints
            default_rules = [
                "Always verify a file exists before reading it.",
                "If a user intent is unclear, ask for clarification.",
                "Do not simulate heavy processes; execute them."
            ]
            self._save(default_rules)

    def get_directives(self) -> List[str]:
        try:
            with open(self.path, 'r') as f: return json.load(f)
        except: return []

    def add_directive(self, rule: str) -> str:
        rules = self.get_directives()
        if rule not in rules:
            rules.append(rule)
            self._save(rules)
            return f"Directive added: {rule}"
        return "Directive already exists."

    def remove_directive(self, index: int) -> str:
        rules = self.get_directives()
        if 0 <= index < len(rules):
            removed = rules.pop(index)
            self._save(rules)
            return f"Removed directive: {removed}"
        return "Index out of bounds."

    def _save(self, rules: List[str]):
        with open(self.path, 'w') as f: json.dump(rules, f, indent=2)