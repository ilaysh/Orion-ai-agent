from orion_core.brain.orion_state import OrionState

@timed("graph.audit_node")
async def audit_node(state: OrionState) -> dict:
    """The ONLY node authorized to interpret intent and define the TaskManifest."""
    request = state.get("user_request", "")
    
    # SYSTEM PROMPT: Define the deterministic flow
    audit_prompt = f"""
    Decompose the user request into a sequence of atomic steps.
    Request: {request}
    
    For each step, specify if it requires:
    1. Hardware Discovery (Local)
    2. Web Research (External)
    3. User Input (Wait for human)
    
    Output as JSON:
    {{
        "goal": "summarized goal",
        "steps": ["step 1", "step 2"],
        "requires_user_input": bool
    }}
    """
    # [Call LLM here to generate the manifest]
    manifest = await llm.ainvoke(audit_prompt)
    
    return {"manifest": manifest}