# orion_core/brain/session_manager.py
import os
import subprocess
import platform
from datetime import datetime
from typing import Dict, Any

from orion_core.brain.people_manager import PeopleManager
from orion_core.brain.directives import DirectivesManager

class SessionManager:
    """
    The System Context Aggregator for Orion v2.5.
    Initializes the LangGraph state with dynamic RAG constraints and OS probes.
    """
    def __init__(self, bridge):
        self.bridge = bridge
        self.people = PeopleManager()
        self.directives = DirectivesManager()
        self.current_speaker = "Unknown"
        
        # Hardware context is expensive to probe, so we cache it on boot.
        self.hardware_profile = self._fetch_hardware_specs()

    def _fetch_hardware_specs(self) -> str:
        """Dynamically probes Ubuntu Wayland for CPU, RAM, GPU, and Desktop Environment."""
        try:
            cpu = platform.processor() or "Unknown CPU"
            ram_gb = "Unknown"
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemTotal' in line:
                            kb = int(line.split()[1])
                            ram_gb = f"{round(kb / (1024**2))}GB"
                            break
            except Exception: pass
                        
            gpu_info = "Unknown GPU"
            try:
                gpu_info = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                    encoding="utf-8"
                ).strip()
            except Exception:
                gpu_info = "No Nvidia GPU detected."

            de = os.environ.get("XDG_CURRENT_DESKTOP", "Unknown Desktop")
            try:
                distro = subprocess.getoutput("grep PRETTY_NAME /etc/os-release | cut -d'=' -f2 | tr -d '\"'")
            except:
                distro = "Unknown Linux"

            return (
                f"- System OS: {distro} | Desktop Environment: {de}\n"
                f"- Hardware Profile: CPU: {cpu} | RAM: {ram_gb} | GPU: {gpu_info}\n"
                f"- VRAM Law: Max 16GB capacity. Strict memory management required."
            )
        except Exception as e:
            return f"- Hardware Profile: Error fetching specs ({e})"

    def set_speaker(self, name: str):
        self.current_speaker = name

    def _get_local_constraints_via_rag(self, location: str) -> str:
        """
        Dynamically pulls electrical, hardware, or regional rules from the Vector DB.
        This completely decouples bias from the python execution tools.
        """
        query = f"mandatory electrical, hardware, and shipping constraints for {location}"
        
        # Query your MetaBridge SOP RAG
        rag_results = self.bridge.query_sops(query)
        
        if rag_results:
            # Assuming your legacy setup returns a list of dictionaries or strings
            extracted_text = rag_results[0].get("text", str(rag_results[0])) if isinstance(rag_results[0], dict) else str(rag_results[0])
            return f"\n[LOCALIZED RAG DIRECTIVE: {location}]\n{extracted_text}"
            
        return ""

    def build_initial_state(self, user_request: str) -> dict:
        """
        Constructs the exact starting dictionary required by the LangGraph OrionState.
        """
        # Re-read the people store from disk each turn. PeopleManager loads the tree
        # once in __init__ and never reloads, and the write-tools use their OWN
        # instance — so without this, anything Orion learned this session (saved to
        # disk correctly) stayed invisible in [PEOPLE] until a restart. The file is
        # tiny (a household), so re-reading per turn is free.
        self.people = PeopleManager()

        # Pull dynamic variables
        now = datetime.now().strftime('%H:%M:%S, %A, %Y-%m-%d')

        # NOTE: the owner is NOT silently assumed from the speaker. A butler doesn't
        # hand a stranger the keys. Establishing the owner is a deliberate, one-time
        # ASKED bootstrap (handled in brain.think before the graph runs), never an
        # implicit side-effect here.

        # THE PEOPLE Orion knows — the WHOLE tree, not just the owner. It is small
        # (a household, not a database), so the entire thing fits in context for a few
        # hundred tokens. This is deliberate: exposing only one person at a time via a
        # lookup API meant Orion could not SEE the household and so could not make
        # connections ("a gift for my wife" → which wife? → her recorded tastes).
        # Reading is context; WRITING still goes through tools.
        owner_key = self.people.tree.get("owner")
        profiles = self.people.tree.get("profiles", {}) or {}
        owner_profile = profiles.get(owner_key) if owner_key else None

        lines = []
        for pkey, prof in profiles.items():
            bits = [f"{prof.get('name', pkey)}"]
            role = prof.get("role")
            relation = prof.get("relation")
            if relation:
                bits.append(f"relation to owner: {relation}")
            if role:
                bits.append(f"role: {role}")
            if pkey == owner_key:
                bits.append("THE OWNER")
            attrs = prof.get("attributes", {}) or {}
            if attrs:
                bits.append("; ".join(f"{k}: {v}" for k, v in attrs.items()))
            links = prof.get("links", {}) or {}
            if links:
                bits.append("; ".join(f"{r}: {n}" for r, n in links.items()))
            restrictions = prof.get("restrictions", []) or []
            if restrictions:
                bits.append("restrictions: " + "; ".join(restrictions))
            lines.append("- " + " | ".join(bits))

        people_block = "\n".join(lines) if lines else "- (nobody is on record yet)"
        if not owner_profile:
            people_block += "\n- NOTE: no owner is established on this system yet."

        owner_attrs = (owner_profile or {}).get("attributes", {}) or {}

        # Localized constraints are only meaningful once we know something about the
        # owner. No facts → skip the RAG call entirely (don't tax every turn/greeting).
        geographic_rules = ""
        if owner_attrs:
            locale_hint = " ".join(str(v) for v in owner_attrs.values())
            geographic_rules = self._get_local_constraints_via_rag(locale_hint)

        # Fetch global prime directives
        laws = self.directives.get_directives()
        laws_str = "\n".join([f"- {rule}" for rule in laws]) if laws else "None"

        # Assemble the dynamic directives block for the prompt engine
        active_directives = f"[PRIME DIRECTIVES]\n{laws_str}\n{geographic_rules}".strip()

        # OS / people context — Orion reads [PEOPLE] to answer personal questions and
        # to resolve who someone is, rather than assuming or searching the filesystem.
        # [MACHINE] is kept in its OWN block: hardware lines are '- ' bullets too, so
        # without a header they merged into the [PEOPLE] list and the model reported
        # the OS and GPU as facts about the user.
        os_context = (
            f"Time: {now}\n"
            f"Speaker: {self.current_speaker}\n"
            f"[PEOPLE] (everyone you know, and what you know about them — use these "
            f"facts, never assume or invent):\n"
            f"{people_block}\n\n"
            f"[MACHINE] (the computer you run on — NOT facts about any person; only "
            f"relevant for technical tasks):\n"
            f"{self.hardware_profile}"
        )

        # Return the partial state. LangGraph will automatically handle 'messages'
        return {
            "user_request": user_request,
            "os_context": os_context,
            "current_working_dir": os.getcwd(),
            "active_tool_directives": active_directives,
            
            # Reset operational variables for the new turn
            "intent_id": "pending",
            "blueprint": "Pending Planner Node.",
            "allowed_tools": [],
            "completed_steps": [],
            "strike_count": 0,
            "loop_count": 0,
            "active_model": "gemma-4-12b",
            "repair_mode_active": False,
            "last_error_trace": None,
            "spoken_update": None,
            "final_response": None
        }

    def _build_speaker_slice(self) -> str:
        """
        Lean, ambient identity block for the CURRENT speaker only.
        Resolved every turn so Orion always knows who he's speaking with,
        their attributes (age/language/prefs), direct relations, and any
        active restrictions. Kept small on purpose to protect the context window.
        """
        speaker = self.current_speaker or "Unknown"
        person = self.people.get_person(speaker)

        if not person:
            # Unknown speaker — tell Orion explicitly rather than leaving it blank.
            return (
                f"[SPEAKER]\n"
                f"- Name: {speaker} (UNRECOGNIZED — not in people_tree)\n"
                f"- Role: Guest (treat with default/guest permissions)"
            )

        # Attributes inline (age, language, favorite_color, etc.) — small dict.
        attrs = person.get("attributes", {})
        attr_str = ", ".join(f"{k}: {v}" for k, v in attrs.items()) or "none recorded"

        # Direct relations by name/role — one line, not full nodes.
        links = person.get("links", {})
        link_str = ", ".join(f"{rel}: {name}" for rel, name in links.items()) or "none recorded"

        # Restrictions surfaced ambiently so Orion honors them without a lookup.
        restrictions = person.get("restrictions", [])
        restr_str = "; ".join(restrictions) if restrictions else "none"

        return (
            f"[SPEAKER]\n"
            f"- Name: {person.get('name', speaker)}\n"
            f"- Role: {person.get('role', 'Guest')}\n"
            f"- Relation to owner: {person.get('relation', 'Unknown')}\n"
            f"- Attributes: {attr_str}\n"
            f"- Direct relations: {link_str}\n"
            f"- Active restrictions: {restr_str}"
        )