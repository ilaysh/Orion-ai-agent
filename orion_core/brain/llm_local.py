# orion_core/brain/llm_local.py

"""
Legacy LLM Compatibility Layer
------------------------------

This file exists so older Orion code can still call:

    generate_mistral()
    preload_default_models()
    unload_model()
    unload_all()
    select_model()

Internally, EVERYTHING routes to the new ChatEngine,
which runs Qwen2.5-14B as Orion's reasoning model.

This prevents breaking older modules like:
- DecisionManager
- Skills
- Planner prototypes
"""

import asyncio
from orion_core.brain.llm.chat_engine import get_chat_engine


# ---------------------------------------------------------
# MODEL SELECTION (legacy API)
# ---------------------------------------------------------
DEFAULT_CORTEX = "Qwen/Qwen2.5-Coder-14B-Instruct"
DEFAULT_CODER = DEFAULT_CORTEX


def select_model(kind: str | None = None) -> str:
    """Legacy: always return chat model."""
    return DEFAULT_CORTEX


# ---------------------------------------------------------
# MODEL PRELOAD (legacy API)
# ---------------------------------------------------------
def preload_default_models():
    """
    Load the ChatEngine model immediately on startup.
    Non-async wrappers are provided for compatibility.
    """
    engine = get_chat_engine()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(engine.load())
        else:
            loop.run_until_complete(engine.load())
    except RuntimeError:
        # If no loop exists (e.g., startup code), create one temporarily
        tmp_loop = asyncio.new_event_loop()
        tmp_loop.run_until_complete(engine.load())
        tmp_loop.close()

    print("[LLM Local] ✔ ChatEngine preloaded.")


# ---------------------------------------------------------
# UNLOAD (legacy API)
# ---------------------------------------------------------
def unload_model(name: str):
    """
    Legacy function — unload chat engine.
    """
    engine = get_chat_engine()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(engine.unload())
        else:
            loop.run_until_complete(engine.unload())
    except RuntimeError:
        tmp = asyncio.new_event_loop()
        tmp.run_until_complete(engine.unload())
        tmp.close()


def unload_all():
    unload_model(DEFAULT_CORTEX)


# ---------------------------------------------------------
# PRIMARY LEGACY GENERATE (maps to ChatEngine)
# ---------------------------------------------------------
def generate_mistral(
    prompt: str,
    system_prompt: str = "",
    context: str = "",
    max_new_tokens: int = 200,
    temperature: float = 0.4,
):
    """
    Legacy wrapper.

    All old calls like:

        generate_mistral("Hello", system_prompt="...", context="...")

    now funnel into ChatEngine.
    """
    engine = get_chat_engine()

    async def run():
        return await engine.generate_chat(
            prompt=prompt,
            system_prompt=system_prompt,
            context=context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(run(), loop).result()
        else:
            return loop.run_until_complete(run())
    except RuntimeError:
        # No loop — create temporary
        tmp = asyncio.new_event_loop()
        out = tmp.run_until_complete(run())
        tmp.close()
        return out
