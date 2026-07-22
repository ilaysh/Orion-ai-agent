# orion_core/brain/tool_registry.py
import os
import importlib
import importlib.util
from typing import List
from langchain_core.tools import BaseTool

from orion_core.brain.self_maintenance import get_self_maintenance_tools

# The module that holds Orion's built-in capabilities. Every @tool defined here is
# discovered automatically — see _discover_module_tools — so adding a capability is
# a single drop-in with NOTHING to hand-register. (Hand-listing was the original
# "tool missing from registry" bug, and it also blocked forge from adding tools.)
_CORE_TOOLS_MODULE = "orion_core.brain.orion_tools"


def _discover_module_tools(module_path: str) -> List[BaseTool]:
    """Return every LangChain @tool object defined in a module, by introspection."""
    mod = importlib.import_module(module_path)
    found = []
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, BaseTool):
            found.append(obj)
    return found


class ToolRegistry:
    """Single source of truth for Orion's tools.

    core = every @tool in the tools module (auto-discovered) + self-maintenance tools.
    skills = every @tool in the hot-reloaded skills dir (forge writes here).
    Both binding (what the model may call) and execution (what actually runs) read
    from get_all_tools(), so they can never drift out of sync.
    """

    def __init__(self, skills_dir: str = "orion_core/skills"):
        self.skills_dir = skills_dir
        self.core_tools = (
            _discover_module_tools(_CORE_TOOLS_MODULE)
            + list(get_self_maintenance_tools())
        )
        os.makedirs(self.skills_dir, exist_ok=True)

    def get_all_tools(self) -> List[BaseTool]:
        """Core tools + dynamically forged skills. Quarantine shield: a broken skill
        is logged and skipped so the system survives. The 'pending' staging dir and
        underscore-prefixed files are ignored — only promoted skills load."""
        active_tools = list(self.core_tools)

        for filename in os.listdir(self.skills_dir):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue
            full = os.path.join(self.skills_dir, filename)
            if os.path.isdir(full):
                continue
            name = filename[:-3]
            try:
                spec = importlib.util.spec_from_file_location(name, full)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    for attr_name in dir(mod):
                        attr = getattr(mod, attr_name)
                        if isinstance(attr, BaseTool):
                            active_tools.append(attr)
            except Exception as e:
                print(f"[Tool Registry] ⚠️ Quarantine: syntax/import error in {filename}: {e}")

        return active_tools
