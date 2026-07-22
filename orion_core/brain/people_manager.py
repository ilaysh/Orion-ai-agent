# orion_core/brain/people_manager.py
import json
import os
import time
from typing import Dict, Optional

class PeopleManager:
    """
    The Social Graph.
    Stores Identity (Who) and Connections (Relationships).
    """
    def __init__(self, storage_path="data/people_tree.json"):
        self.path = storage_path
        self._ensure_storage()
        self.tree = self._load()

    def _ensure_storage(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            self._save({"owner": None, "profiles": {}})

    def _load(self) -> Dict:
        try:
            with open(self.path, 'r') as f: return json.load(f)
        except: return {"owner": None, "profiles": {}}

    def _save(self, data=None):
        if data: self.tree = data
        with open(self.path, 'w') as f: json.dump(self.tree, f, indent=2)

    # --- LOOKUPS ---
    def get_person(self, name: str) -> Optional[Dict]:
        """Finds a person by profile key OR by their stored name (case-insensitive).

        Matching only the dict key broke whenever the key's case differed from the
        name (a hand-edited "Yafit" key made lookups fail while the person was
        plainly listed) — a contradiction the model cannot recover from. Matching
        either makes the store tolerant of however the JSON was written.
        """
        if not name:
            return None
        target = name.strip().lower()
        profiles = self.tree.get("profiles", {})
        # 1. Exact key match (the common, fast path).
        if target in profiles:
            return profiles[target]
        # 2. Case-insensitive key match.
        for key, prof in profiles.items():
            if key.strip().lower() == target:
                return prof
        # 3. Match on the stored display name.
        for prof in profiles.values():
            if str(prof.get("name", "")).strip().lower() == target:
                return prof
        return None

    def get_owner_name(self) -> str:
        """Returns the name of the System Owner."""
        key = self.tree.get("owner")
        if key and key in self.tree["profiles"]:
            return self.tree["profiles"][key]["name"]
        return "Unknown"

    # --- SOCIAL ACTIONS ---
    def register_person(self, name: str, relation: str = "Acquaintance", role: str = "User"):
        """
        Adds a new person to the memory.
        - relation: "Wife", "Son", "Brother" (Relative to Owner)
        - role: "Owner", "Family", "Guest" (System Privilege)
        """
        name_key = name.lower()
        
        # Auto-set Owner if claimed
        if role == "Owner":
            self.tree["owner"] = name_key
            relation = "Self"

        if name_key not in self.tree["profiles"]:
            self.tree["profiles"][name_key] = {
                "name": name,
                "relation": relation, # e.g. "Wife"
                "role": role,         # e.g. "Family"
                "links": {},          # e.g. {"Brother": "Matt"}
                "face_ids": [],
                "met_at": time.strftime("%Y-%m-%d")
            }
        else:
            # Update existing person
            self.tree["profiles"][name_key]["relation"] = relation
            self.tree["profiles"][name_key]["role"] = role

        self._save()
        return f"Registered {name} (Relation: {relation})."

    def link_people(self, person_a: str, relationship: str, person_b: str):
        """
        Connects two people.
        Example: link_people("Matt", "Brother", "Ilays")
        """
        p_a = self.get_person(person_a)
        p_b = self.get_person(person_b)
        
        if p_a and p_b:
            # Store the link on Person A's profile
            p_a["links"][relationship] = p_b["name"]
            self._save()
            return f"Note made: {person_a} is {relationship} of {person_b}."
        return "Could not link people. One or both profiles missing."

    def add_hardware_id(self, name: str, id_type: str, value: str):
        """Stores Face/Voice IDs for recognition."""
        p = self.get_person(name)
        if p:
            key = f"{id_type}_ids"
            if value not in p.get(key, []):
                p.setdefault(key, []).append(value)
                self._save()