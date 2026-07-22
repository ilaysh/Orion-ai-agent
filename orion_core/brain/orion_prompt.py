# orion_core/brain/orion_prompt.py
import os
from langchain_core.messages import SystemMessage
from orion_core.brain.orion_state import OrionState

def get_orion_system_prompt(state: OrionState) -> SystemMessage:
    """Reads personality.txt and injects the live LangGraph state."""
    
    prompt_path = "orion_core/prompts/personality.txt"
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        raw_template = f.read()
        
    # --- Format State Variables ---
    blueprint = state.get("blueprint", "Awaiting instructions.")
    completed_steps = state.get("completed_steps", [])
    recent_errors = state.get("last_error_trace", "")
    
    ledger_text = "None"
    if completed_steps:
        ledger_text = "\n".join([f"- {step}" for step in completed_steps])
            
    error_context = ""
    if recent_errors:
        error_context = f"\n[RECENT SYSTEM ERROR TO CORRECT]:\n{recent_errors}\n"

    # --- Inject into Template ---
    formatted_prompt = raw_template.format(
        os_context=state.get("os_context", "Standard Constraints"),
        current_working_dir=state.get("current_working_dir", os.getcwd()),
        active_tool_directives=state.get("active_tool_directives", "None active."),
        blueprint=blueprint,
        ledger_text=ledger_text,
        error_context=error_context
    )

    return SystemMessage(content=formatted_prompt)