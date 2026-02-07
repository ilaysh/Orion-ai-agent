# orion_core/brain/executive.py
import asyncio
import os
import subprocess
import importlib.util
import sys
from typing import Dict, Any

# DEFINING THE PLAYGROUND
DYNAMIC_SKILLS_DIR = "orion_core/skills/dynamic"

class Executive:
    """
    The Hands. Executes Core Tools and Dynamic Skills.
    """
    def __init__(self, cortex, session_manager):
        self.cortex = cortex
        self.session = session_manager 
        self.dynamic_skills = {} 
        
        # Ensure the playground exists on boot
        os.makedirs(DYNAMIC_SKILLS_DIR, exist_ok=True)
        # Ensure Python can import from it
        if os.path.abspath(DYNAMIC_SKILLS_DIR) not in sys.path:
            sys.path.append(os.path.abspath(DYNAMIC_SKILLS_DIR))

    async def execute(self, tool: str, args: Dict[str, Any]) -> str:
        try:
            # 1. CORE TOOLS
            if tool == "cmd":
                return self._cmd(args.get("command", ""))
            elif tool == "fs":
                return self._fs(args.get("action"), args.get("path"), args.get("content"))
            elif tool == "system":
                return self._system(args.get("action"), args.get("module"))
            elif tool == "memory":
                # (Keep your existing memory logic here)
                return "Memory updated."

            # 2. DYNAMIC SKILLS (The Genesis Hook)
            # Format: "crypto.run" -> module: crypto, func: run
            if "." in tool:
                parts = tool.split(".")
                domain = parts[0] # e.g., "crypto"
                action = parts[1] # e.g., "run"
                
                if domain in self.dynamic_skills:
                    module = self.dynamic_skills[domain]
                    if hasattr(module, action):
                        func = getattr(module, action)
                        if asyncio.iscoroutinefunction(func): 
                            return await func(**args)
                        return str(func(**args))
                    return f"Skill '{domain}' has no function '{action}'."
            
            return f"Error: Unknown tool '{tool}'"
        except Exception as e:
            return f"Tool Failure: {e}"

    def _cmd(self, command: str) -> str:
        if not command: return "No command provided."
        print(f"[Executive] 💻 Executing: {command}")
        try:
            # Run in background if requested, otherwise wait
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
            return f"STDOUT: {result.stdout.strip()}\nSTDERR: {result.stderr.strip()}"
        except Exception as e: return str(e)

    def _fs(self, action: str, path: str, content: str = None) -> str:
        try:
            # Security: Prevent escaping project root if needed (optional)
            full_path = os.path.abspath(path)
            
            if action == "write":
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f: 
                    f.write(content or "")
                return f"File written: {path}"
            elif action == "read":
                if os.path.exists(full_path):
                    with open(full_path, "r", encoding="utf-8") as f: return f.read()
                return "File not found."
            elif action == "list":
                 if os.path.exists(full_path):
                     return str(os.listdir(full_path))
                 return "Directory not found."
            return f"Invalid FS action '{action}'."
        except Exception as e: return f"FS Error: {e}"
   
    def _reload_skills(self) -> str:
        """
        Scans `orion_core/skills/dynamic` and loads python files.
        Runs `test()` on them before approving.
        """
        loaded = []
        errors = []
        
        # Scan the Dynamic Directory
        for filename in os.listdir(DYNAMIC_SKILLS_DIR):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = filename[:-3]
                path = os.path.join(DYNAMIC_SKILLS_DIR, filename)
                
                try:
                    # 1. Load Spec
                    spec = importlib.util.spec_from_file_location(module_name, path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # 2. AUTO-TEST (The Vibe Check)
                    if hasattr(module, "test"):
                        try:
                            print(f"[Executive] 🧪 Testing skill: {module_name}")
                            module.test() # Must not raise exception
                            self.dynamic_skills[module_name] = module
                            loaded.append(module_name)
                        except Exception as test_err:
                            errors.append(f"{module_name} FAILED TEST: {test_err}")
                    else:
                        # Legacy/Simple skills without tests are allowed
                        self.dynamic_skills[module_name] = module
                        loaded.append(module_name)
                        
                except Exception as e: 
                    errors.append(f"{module_name} Load Error: {e}")
        
        return f"Loaded: {loaded}. Errors: {errors}" if errors else f"Systems Online. Loaded: {loaded}"

    def _system(self, action: str, module: str = None) -> str:
        if action == "reload" or action == "reload_skills":
            return self._reload_skills()
        return "Unknown system action."

    def get_loaded_skill_names(self) -> list:
        # Returns format for Prompt: ["crypto.run", "weather.check"]
        skills = []
        for name, mod in self.dynamic_skills.items():
            # Quick scan of callable functions not starting with _
            funcs = [f for f in dir(mod) if callable(getattr(mod, f)) and not f.startswith("_")]
            for f in funcs:
                skills.append(f"{name}.{f}")
        return skills