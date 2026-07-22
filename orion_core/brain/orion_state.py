# orion_core/brain/orion_state.py
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
import operator


class TaskManifest(TypedDict, total=False):
    """The immutable blueprint for a task."""
    task_id: str
    goal: str
    steps: List[str]
    current_step: int
    dependencies_met: bool
    requires_user_input: bool
    user_input_prompt: Optional[str]


class OrionState(TypedDict, total=False):
    """
    total=False is deliberate: every node returns a *partial* dict, and LangGraph
    merges it. Marking keys optional stops type-checkers complaining and lets
    .get() be the single access pattern everywhere.
    """
    # --- Conversation ---
    messages: Annotated[List[BaseMessage], operator.add]
    user_request: str
    final_response: Optional[str]

    # --- Routing / Planning ---
    intent_id: Optional[str]
    blueprint: Optional[str]
    active_model: str

    # --- Grounding / Context (built by session_manager, consumed by planner/alfred) ---
    os_context: Optional[str]
    current_working_dir: Optional[str]
    active_tool_directives: Optional[str]
    rag_context: Optional[str]

    # --- Identity / RBAC (ready for batch 2 enforcement) ---
    speaker: Optional[str]
    speaker_role: Optional[str]

    # --- Loop control ---
    loop_count: int
    strike_count: int
    exhausted: bool
    fatal_error: bool

    # --- Clarification gate ---
    awaiting_clarification: bool

    # --- Consent gate (Option 3): set True once the user approves a privileged
    # (root-daemon) action; reset every new request so consent never carries over. ---
    privilege_confirmed: bool

    # --- Repair ---
    repair_mode_active: bool
    last_error_trace: Optional[str]

    # --- Written by session_manager.build_initial_state ---
    allowed_tools: List[str]
    completed_steps: List[str]
    spoken_update: Optional[str]

    # --- Task manifest / scratch ---
    manifest: Optional[TaskManifest]
    workspace_data: dict