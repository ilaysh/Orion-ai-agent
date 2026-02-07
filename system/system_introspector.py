# orion_core/system/introspector.py
import importlib
import traceback
from typing import Dict, List
# Lazy import to avoid circular dependency; will be imported inside run_diagnostics
# The actual architecture we have built
CRITICAL_MODULES = {
    "Brain": "orion_core.brain.brain",
    "Thinker": "orion_core.brain.thinker",
    "Memory": "orion_core.brain.meta_bridge",
    "Scheduler": "orion_core.brain.scheduler",
    "Skills": "orion_core.skills.skills",
    "Personality": "orion_core.brain.personality"
}

class SystemIntrospector:
    def run_diagnostics(self) -> Dict:
        """
        Performs a 'System Check' on startup.
        Returns a status report.
        """
        status = {
            "healthy": True,
            "modules": {},
            "errors": []
        }

        print("[System] 🛠️ Running Diagnostics...")

        # 1. Check Module Imports
        for name, path in CRITICAL_MODULES.items():
            try:
                mod = importlib.import_module(path)
                status["modules"][name] = "ONLINE"
            except ImportError as e:
                status["healthy"] = False
                status["modules"][name] = "OFFLINE"
                status["errors"].append(f"Missing Critical Module: {name} ({e})")
            except Exception as e:
                status["healthy"] = False
                status["modules"][name] = "CRASHED"
                status["errors"].append(f"Error loading {name}: {e}")

        # 2. Check Skill Definitions (Prevent 'dict' errors)
        try:
            from orion_core.skills.skills import get_tools_metadata
            tools = get_tools_metadata()
            
            if not isinstance(tools, dict):
                 status["errors"].append("Skills metadata is not a dictionary.")
            else:
                 status["skill_count"] = len(tools)
                 # Validate Schema
                 for tool_name, meta in tools.items():
                     if "params" not in meta or "description" not in meta:
                         status["errors"].append(f"Skill '{tool_name}' has invalid metadata schema.")
        except Exception as e:
            status["healthy"] = False
            status["errors"].append(f"Skill Registry Failure: {e}")

        return status