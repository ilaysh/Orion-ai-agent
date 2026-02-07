# orion_core/skills/skills.py
import os
import sys
import subprocess
import importlib
import importlib.util
from typing import List, Dict, Any, Optional, Tuple

# We wrap RAG directly for raw data access, or MetaBridge if we want context-aware retrieval.
# For the Kernel (Raw Tools), direct RAG access is often cleaner, but wrapping MetaBridge unifies logic.
from orion_core.brain.rag_chroma import RAGMemory

DYNAMIC_SKILLS_DIR = "orion_core/skills/dynamic"

class Kernel:
    """
    The Hardware Abstraction Layer (HAL).
    The Heavy Model uses these tools to build, execute, and remember.
    """
    def __init__(self):
        # We use RAGMemory directly here to keep the Kernel "raw" and fast.
        self.rag = RAGMemory()
        
        # Ensure the dynamic skills directory exists and is valid
        os.makedirs(DYNAMIC_SKILLS_DIR, exist_ok=True)
        if DYNAMIC_SKILLS_DIR.replace("/", ".") not in sys.modules:
             sys.path.append(os.path.abspath(DYNAMIC_SKILLS_DIR))

    # =========================================================================
    # 1. FILE SYSTEM (fs) - The Hands
    # =========================================================================
    def list_dir(self, path: str = ".") -> List[str]:
        """List files in a directory."""
        if not os.path.exists(path): return []
        return [f for f in os.listdir(path) if not f.startswith("__")]

    def read_file(self, path: str) -> str:
        """Read file content. Returns error string if failed (doesn't crash)."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading {path}: {e}"

    def write_file(self, path: str, content: str) -> str:
        """Atomic write. Creates directories if needed."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote {len(content)} bytes to {path}."
        except Exception as e:
            return f"Error writing {path}: {e}"

    def delete_file(self, path: str) -> str:
        """Delete a file."""
        try:
            if os.path.exists(path):
                os.remove(path)
                return f"Deleted {path}."
            return f"File {path} not found."
        except Exception as e:
            return f"Error deleting {path}: {e}"

    # =========================================================================
    # 2. COMMAND LINE (cmd) - The Terminal
    # =========================================================================
    def execute_cmd(self, command: str, timeout: int = 60) -> str:
        """
        Run a shell command.
        CRITICAL: Returns combined STDOUT + STDERR so the Architect can see errors.
        """
        try:
            # Security: In a real deployment, we'd sandbox this.
            # For local personal AI, we trust the Architect.
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            output = f"--- STDOUT ---\n{result.stdout}\n"
            if result.stderr:
                output += f"\n--- STDERR ---\n{result.stderr}\n"
            
            return output.strip()
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s."
        except Exception as e:
            return f"Command execution failed: {e}"

    def get_python_path(self) -> str:
        """Helper for the Architect to install pip packages correctly."""
        return sys.executable

    # =========================================================================
    # 3. SYSTEM (system) - The Self
    # =========================================================================
    def reload_module(self, module_name: str) -> str:
        """Hot-reload a module (used after writing a new skill)."""
        try:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
                return f"Reloaded {module_name}."
            
            # Try to import it if it's new
            importlib.import_module(module_name)
            return f"Imported new module {module_name}."
        except Exception as e:
            return f"Failed to load/reload {module_name}: {e}"

    # =========================================================================
    # 4. MEMORY (memory) - The Database
    # =========================================================================
    def query_memory(self, query: str, top_k: int = 3) -> str:
        """
        Retrieves FACTUAL data.
        Note: The Planner uses this to check documentation or past learnings.
        """
        results = self.rag.retrieve(query, top_k=top_k)
        if not results:
            return "No relevant memory found."
        return "\n---\n".join(results)

    def store_memory(self, text: str, metadata: Dict[str, Any]) -> str:
        """Allows the Architect to explicitly save a 'Learning'."""
        try:
            # We generate a unique ID based on content hash or timestamp
            path = metadata.get("path", f"fact_{abs(hash(text))}")
            self.rag.add_document(path, text, metadata)
            return "Stored in memory."
        except Exception as e:
            return f"Memory storage failed: {e}"


class DynamicSkillRegistry:
    """
    The Plugin System.
    It scans `orion_core/skills/dynamic` and registers valid skills
    so the Thinker (Lite) knows they exist.
    """
    def __init__(self):
        self.loaded_skills: Dict[str, Any] = {}

    def scan_and_load(self) -> Dict[str, str]:
        """
        Returns a manifest: { 'skill_name': 'docstring/description' }
        """
        self.loaded_skills.clear()
        manifest = {}
        
        if not os.path.exists(DYNAMIC_SKILLS_DIR):
            return manifest

        for filename in os.listdir(DYNAMIC_SKILLS_DIR):
            if filename.endswith(".py") and not filename.startswith("_"):
                module_name = filename[:-3]
                path = os.path.join(DYNAMIC_SKILLS_DIR, filename)
                
                try:
                    # 1. Dynamic Import
                    spec = importlib.util.spec_from_file_location(module_name, path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                        
                        # 2. Convention Check
                        # We look for a standalone function 'run' OR a class 'Skill'
                        if hasattr(module, "register"):
                            # Robust: Module defines how it fits in
                            meta = module.register()
                            self.loaded_skills[meta['name']] = module
                            manifest[meta['name']] = meta.get('description', 'Dynamic Skill')
                        elif hasattr(module, "run"):
                            # Simple Script
                            self.loaded_skills[module_name] = module
                            # Extract docstring as description
                            doc = module.__doc__ or "No description."
                            manifest[module_name] = doc.strip().split('\n')[0]
                            
                except Exception as e:
                    print(f"[Skills] ⚠️ Failed to load dynamic skill {filename}: {e}")
                    # We do NOT crash. We just skip the broken skill.
                    
        return manifest

    def get_skill(self, name: str) -> Optional[Any]:
        return self.loaded_skills.get(name)

# Compatibility alias for legacy imports
class Skills(Kernel):
    """Legacy alias to keep existing imports working."""
    pass