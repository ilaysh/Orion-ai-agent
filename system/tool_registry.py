# orion_core/system/tool_registry.py
"""
The Official List of Orion's Hard Capabilities.
These map 1:1 to methods in the new `skills.py`.
"""

class ToolRegistry:
    def list_tools(self) -> dict:
        return {
            "web.search": {
                "description": "Search Google/DuckDuckGo for real-time info, reviews, or facts.",
                "params": ["query"]
            },
            "task.start": {"params": ["instructions"], "description": "CRITICAL: The Coder. Write files/code."},
            "cmd.run": {"params": ["command", "background"], "description": "Execute shell commands. Set 'background': true for long tasks."},
            "proc.list": {"params": [], "description": "Check running background processes."},
            "proc.kill": {"params": ["pid"], "description": "Stop a specific process by PID."},
            "fs.list": {"params": ["path"], "description": "List files."},
            "fs.read": {"params": ["path"], "description": "Read file."},
            "system.update_entity": {"params": ["entity_id", "updates"], "description": "Save memory/plans."},
            "system.refresh_capabilities": {"params": [], "description": "Update RAG."}
        }

# Singleton access (keeps compatibility with older modules if they import it)
_registry_instance = ToolRegistry()

def get_tool_registry():
    return _registry_instance.list_tools()