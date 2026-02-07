# orion_core/brain/personality.py
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
        self.docs_dir = self.root_dir / "docs"
        
        # Link to the authoritative source of truth
        self.mapper = ProjectMapper(root_dir=str(self.root_dir))
        
        self.soul_path = self.docs_dir / "orion_presonality.md"
        self.kernel_path = self.prompts_dir / "kernel_system.txt"
        
        self.soul_text = ""
        self.kernel_text = ""
        self.load_cache()

    def load_cache(self):
        if self.soul_path.exists():
            self.soul_text = self.soul_path.read_text("utf-8")
        else:
            self.soul_text = "You are Orion, an intelligent autonomous agent."

        if self.kernel_path.exists():
            self.kernel_text = self.kernel_path.read_text("utf-8")
        else:
            self.kernel_text = "[CORE TOOLS]\n1. fs\n2. cmd"

    def get_system_prompt(self, tools_list: list) -> str:
        """
        Builds the Master Context with Real-Time Awareness.
        """
        # Fetch the optimized summary from ProjectMapper
        project_map = self.mapper.get_live_map_summary()
        
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