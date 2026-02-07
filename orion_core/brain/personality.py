# orion_core/brain/personality.py
import os
from pathlib import Path

class Personality:
    """
    The Soul of Orion.
    Manages the System Prompt and injects Context.
    """
    def __init__(self):
        self.root_dir = Path(__file__).parents[2]
        self.prompts_dir = self.root_dir / "orion_core" / "prompts"
        self.docs_dir = self.root_dir / "docs"
        
        self.soul_path = self.docs_dir / "orion_presonality.md"
        self.kernel_path = self.prompts_dir / "kernel_system.txt"
        
        self.soul_text = ""
        self.kernel_text = ""
        self.load_cache()

    def load_cache(self):
        """Loads static text files (Identity/Rules)."""
        if self.soul_path.exists():
            self.soul_text = self.soul_path.read_text("utf-8")
        else:
            self.soul_text = "You are Orion, an intelligent autonomous agent."

        if self.kernel_path.exists():
            self.kernel_text = self.kernel_path.read_text("utf-8")
        else:
            self.kernel_text = "[CORE TOOLS]\n1. fs\n2. cmd"

    def _get_flash_map(self) -> str:
        """
        ⚡ FLASH SCAN: Lists filenames only.
        Takes ~0.002s. Runs on every request so Orion is always 100% sync.
        """
        tree = []
        try:
            # Only scan the 'projects' folder (limit depth for speed)
            for root, dirs, files in os.walk(self.root_dir):
                # 1. SKIP HEAVY FOLDERS (Speed Optimization)
                if any(x in root for x in [".venv", ".git", "__pycache__", "models", "node_modules", "ui/static"]):
                    continue
                
                rel_root = os.path.relpath(root, self.root_dir)
                if rel_root == ".": rel_root = ""
                
                # 2. FILTER RELEVANT FILES
                valid_files = [f for f in files if f.endswith(".py") or f.endswith(".md") or f.endswith(".txt")]
                
                if valid_files:
                    path_prefix = f"{rel_root}/" if rel_root else ""
                    # Add to tree (e.g. "runners/launcher.py")
                    for f in valid_files:
                        tree.append(f"- {path_prefix}{f}")
                        
        except Exception:
            return "(Map unavailable)"
            
        # 3. SAFETY LIMIT (Prevent Context Overflow)
        return "\n".join(tree[:80])

    def get_system_prompt(self, tools_list: list) -> str:
        """
        Builds the Master Context with Real-Time Awareness.
        """
        # This runs instantly (~2ms)
        project_map = self._get_flash_map()
        
        prompt = (
            f"{self.soul_text}\n\n"
            f"### 🗺️ LIVE PROJECT STRUCTURE (READ-ONLY)\n"
            f"{project_map}\n\n"
            f"{self.kernel_text}"
        )
        
        if tools_list:
            tools_str = "\n".join([f"- {tool}" for tool in tools_list])
            prompt += f"\n\n[DYNAMIC SKILLS AVAILABLE]\n{tools_str}"
        
        return prompt