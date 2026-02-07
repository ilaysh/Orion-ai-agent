# orion_core/brain/coder.py
"""
CODER — Orion's secure analytical and constructive hemisphere.
Handles safe programming, debugging, analysis and structured problem-solving.
Communicates actions to the Thinker via the MetaBridge.
"""

import os
from orion_core.brain.llm_local import generate_mistral
from orion_core.brain.meta_bridge import MetaBridge
from system.telemetry.telemetry import timed


CODER_SYSTEM_PROMPT = """
You are Orion's internal Secure Coding Engine.

Core principles:
1. SECURITY FIRST — All code must be safe, validated and sandbox-friendly.
2. NO RISKY IMPORTS — Never use: os.system, subprocess.*, eval, exec, runpy,
   pickle, torch.load(untrusted), or anything with shell access.
3. SAFE DEPENDENCIES — When suggesting libraries (especially from HuggingFace
   or GitHub):
   - Explain security considerations.
   - Prefer official or reputable packages.
   - Warn about unknown or unmaintained code.
4. NO NETWORK DOWNLOADS unless the user explicitly approves it.
5. Use Python by default unless specified otherwise.
6. Code must be deterministic, minimal, readable, and dependency-aware.
7. If a request seems unsafe, warn the user and propose safer alternatives.
8. NEVER return markdown. Strictly return plain text or raw code if needed.

If you are unsure, ask clarifying questions before generating code.
"""

CREATE_SKILL_PROMPT = """
Create a new Orion skill module.

Requirements:
- Use UTF-8 encoding.
- Start with a short module-level docstring.
- Define the function:

      def run(text: str | None = None, **kwargs) -> str:
          \"\"\"One-line description\"\"\"
          ...

- Return a human-readable string summary.
- Avoid heavy external dependencies unless necessary.
- No tests, no __main__, no markdown.
- Code must follow the SECURITY rules:
    * No eval / exec
    * No os.system / subprocess
    * No unsafe file access unless approved by user
"""


class Coder:
    def __init__(self, skills_dir: str = "orion_core/skills"):
        self.bridge = MetaBridge()
        self.skills_dir = skills_dir
        os.makedirs(self.skills_dir, exist_ok=True)

    # --------------------------------------------------------------

    @timed("coder execute")
    async def execute(self, request: str) -> str:
        """
        Main entry point for coding / analysis requests.
        Now uses Mistral instruct format + secure system prompt.
        """
        user_prompt = (
            "User request (coding / debugging / analysis):\n"
            f"{request}\n\n"
            "Respond according to the SECURITY RULES."
        )

        reply = generate_mistral(
            prompt=user_prompt,
            system_prompt=CODER_SYSTEM_PROMPT,
            context=""
        )

        self.bridge.record_action("Coder", f"Executed request: {request[:80]}")
        return reply

    # --------------------------------------------------------------

    @timed("coder create_skill")
    async def create_skill(self, name: str, description: str = "") -> str:
        """
        Generate a new Python skill module under orion_core/skills/.
        Now with secure-coding rules and Mistral instruct formatting.
        """
        # Sanitize file name
        safe = name.strip().replace(" ", "_")
        safe = "".join(ch for ch in safe if ch.isalnum() or ch in ("_", "-"))
        if not safe:
            safe = "generated_skill"

        path = os.path.join(self.skills_dir, f"{safe}.py")

        # Prompt LLM
        prompt = f"""
        {CREATE_SKILL_PROMPT}

        Skill name: {name}
        Description: {description}
        """

        code = generate_mistral(
            prompt=prompt,
            system_prompt=CODER_SYSTEM_PROMPT,
            context=""
        )

        # Write file
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)

        print(f"[Coder] 🧩 Created new skill: {path}")
        self.bridge.record_action("Coder", f"Created new skill: {path}")
        return path
