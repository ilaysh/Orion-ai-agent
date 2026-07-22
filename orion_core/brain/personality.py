import os
import platform
import getpass
from pathlib import Path
from orion_core.brain.project_mapper import ProjectMapper

class Personality:
    """
    The Soul of Orion.
    Manages the System Prompt by combining Identity, Protocols, and the LIVE MAP.
    """
    def __init__(self):
        self.root_dir = Path(__file__).parents[2]
        self.prompts_dir = self.root_dir / "orion_core" / "prompts"
        
        # --- FILE MAPPINGS ---
        self.soul_path = self.prompts_dir / "orion_personality.md"  # The Butler (Final Response)
        self.kernel_path = self.prompts_dir / "kernel_system.md"    # The Architect (Planning)
        self.worker_path = self.prompts_dir / "worker_system.md"    # The Coder (Execution)
        self.qa_path = self.prompts_dir / "qa_prompt.md"            # The Reviewer (Error Fixing)
        self.forge_path = self.prompts_dir / "skill_forge.md"       # The Builder (Writing Tools)
        
        # --- CACHE ---
        self.soul_text = self._load_text(self.soul_path, "You are Orion, a refined AI assistant.")
        self.kernel_text = self._load_text(self.kernel_path, "You are the Orion Architect. Analyze the user request.")
        self.worker_text = self._load_text(self.worker_path, "You are the Orion Execution Worker. Output ONLY tool calls.")
        self.qa_text = self._load_text(self.qa_path, "You are the QA Reviewer. Fix the failing code execution.")
        self.forge_text = self._load_text(self.forge_path, "You are the Skill Forge. Write perfect LangChain Python tools.")

    def _load_text(self, path: Path, default: str) -> str:
        """Safely loads the markdown file, falling back to a default if missing."""
        if path.exists():
            return path.read_text("utf-8")
        print(f"[Personality] ⚠️ Warning: {path.name} not found. Using fallback.")
        return default

    def _get_system_context(self, override_cwd: str = None) -> str:
        """Dynamically grabs the live OS state to append to the bottom of prompts."""
        os_info = platform.system() + " " + platform.release()
        cwd = override_cwd or os.getcwd()
        return (
            f"\n\n--- CURRENT SYSTEM STATE ---\n"
            f"System Host: {getpass.getuser()}@localhost\n"
            f"Current Working Dir: {cwd}\n"
            f"OS Environment: {os_info}\n"
        )

    # =========================================================================
    # PHASE-SPECIFIC PROMPT BUILDERS
    # =========================================================================

    def get_planner_prompt(self, cwd: str = None) -> str:
        """Turn 1: Used by the planner node to decide actions without conversational bloat."""
        return self.kernel_text + self._get_system_context(cwd)

    def get_worker_prompt(self, cwd: str = None) -> str:
        """Turn 2: Used by the execution node. Highly mechanical."""
        return self.worker_text + self._get_system_context(cwd)

    def get_qa_prompt(self, error_trace: str, cwd: str = None) -> str:
        """Turn 3: Used when a tool fails. Injects the raw stack trace for debugging."""
        prompt = self.qa_text + self._get_system_context(cwd)
        if error_trace:
            prompt += f"\n\n--- FAILING EXECUTION TRACE ---\n{error_trace}\n"
        return prompt
        
    def get_skill_forge_prompt(self, cwd: str = None) -> str:
        """Used specifically when forging new capabilities to ensure Python syntax rules."""
        return self.forge_text + self._get_system_context(cwd)

    def get_butler_prompt(self, cwd: str = None) -> str:
        """Final Turn: Used ONLY when speaking the final response to the user."""
        execution_directive = (
            "\n\n[SPEAK TO THE MASTER]\n"
            "Now say your piece. Be strictly truthful to what the tools actually "
            "returned — never present a failed or unconfirmed step as done. But do "
            "not deliver a status report: the master does not care that a record was "
            "written or a command returned zero. He cares about the thing itself. "
            "Lead with what it means for him, not with your own bookkeeping. One or "
            "two sentences, in your own manner."
        )
        return self.soul_text + execution_directive + self._get_system_context(cwd)