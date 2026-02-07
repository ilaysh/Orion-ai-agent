# orion_core/brain/intent_manager.py
from typing import Any, Dict
from dataclasses import dataclass
from system.tool_registry import ToolRegistry

class IntentValidationError(Exception):
    pass

@dataclass
class ValidatedDecision:
    action: str
    params: Dict[str, Any]
    needs_confirmation: bool

class IntentManager:
    """
    Validates actions to prevent crashes and hallucinations.
    """
    def __init__(self, tool_registry: ToolRegistry):
        self.registry = tool_registry

    def validate(self, thinker_result) -> ValidatedDecision:
        action = thinker_result.action
        params = thinker_result.params or {}
        
        # 1. Null Action -> Pass
        if not action:
            return ValidatedDecision(None, {}, thinker_result.needs_confirmation)

        # 2. Type Check (PREVENTS CRASH)
        if not isinstance(action, str):
            # If the model outputs a dict/object, reject it gracefully
            raise IntentValidationError(f"Invalid action format. Expected string (e.g. 'web.search'), got {type(action).__name__}.")

        # 3. Syntax Cleanup (PREVENTS HALLUCINATION)
        # If model outputs "task.start('instructions')", strip the params
        if "(" in action:
            action = action.split("(")[0].strip()

        # 4. Existence Check
        tools = self.registry.list_tools()
        if action not in tools:
            raise IntentValidationError(f"Unknown tool '{action}'. Valid tools: {list(tools.keys())}")

        return ValidatedDecision(action, params, thinker_result.needs_confirmation)