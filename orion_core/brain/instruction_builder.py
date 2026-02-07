# orion_core/brain/instruction_builder.py

"""
InstructionBuilder
------------------
Converts Thinker → Brain task parameters into deterministic,
structured instructions for CoderRunner.

This prevents Thinker from producing messy, ambiguous
coder prompts and centralizes all task → code generation logic.

The builder is intentionally simple so Orion can extend it.
"""

import os


class InstructionBuilder:

    # -------------------------------------------------------------
    # MAIN ENTRY POINT
    # -------------------------------------------------------------
    def build(self, params: dict) -> str:
        """
        Convert Thinker params into coder instructions.

        Expected params:
            - files: list of {path, content?, ask_content?}
            - action: "create_file", "modify_file", "multi_file", etc.

        Returns a SINGLE string that the coder model will receive.
        """
        # If Thinker did not format instructions yet:
        if "files" not in params:
            return self._fallback(params)

        files = params["files"]
        instructions = []

        # Multi-file deterministic structure
        for file_info in files:
            path = file_info.get("path")
            content = file_info.get("content")
            ask_content = file_info.get("ask_content", False)

            # Normalize path
            path = self._clean_path(path)

            # Folder creation instruction (if needed)
            folder = os.path.dirname(path)
            if folder and folder not in ("", "."):
                instructions.append(f"Ensure folder '{folder}' exists.")

            # File instruction
            if content is None and ask_content:
                instructions.append(
                    f"Create an empty file at '{path}'. The user will later provide content."
                )
            elif content is None:
                instructions.append(
                    f"Create an empty file at '{path}'."
                )
            else:
                instructions.append(
                    f"Create file '{path}' with the following EXACT content:\n{content}"
                )

        # Combine into coder-ready text
        return "\n\n".join(instructions)

    # -------------------------------------------------------------
    def _fallback(self, params: dict) -> str:
        """
        Last-resort fallback if Thinker gives only 'instructions' text.
        """
        raw = params.get("instructions", "")
        return (
            "Interpret the following user instructions literally and produce ONLY code or file content.\n"
            "Instructions:\n" + raw
        )

    # -------------------------------------------------------------
    def _clean_path(self, path: str) -> str:
        """Normalize incoming paths: trim, fix slashes."""
        if not isinstance(path, str):
            return "generated_file.txt"
        path = path.strip().lstrip("/\\")
        path = path.replace("\\", "/")
        return path or "generated_file.txt"
