# orion_core/brain/self_maintenance.py
"""
Orion's self-maintenance capabilities — the CTO managing his own body.

Three tools, three risk tiers:
  * clear_memory  — wipe memory stores by scope. Cheap scopes run immediately;
                    irreversible scopes (preferences/people/all) require confirm.
  * manage_skill  — create/update a SKILL (reusable capability), sandbox-verified,
                    then HOT-RELOADED live (skills are quarantined, low risk).
  * self_edit     — edit a CORE file. Verified (syntax+import), then STAGED for the
                    next restart (never hot-reloaded — a bad core edit must not be
                    able to crash the running process that would fix it).

The real quality gate is Orion's own self-review against self_review.md, since the
user (CEO) cannot audit Python. These tools enforce the MECHANICAL guarantees
(syntax, imports, staging, backups); the RUBRIC enforces the judgment guarantees.
Git + "upload to me for review" is the ultimate backstop.
"""
import os
import shutil
import asyncio
import importlib
import datetime
from typing import Optional
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field

# Core files Orion may edit but which are RESTART-TO-APPLY (never hot-reloaded).
# The self-maintenance module itself and the boot path are intentionally excluded
# from self_edit entirely — Orion must not be able to break the mechanism that
# fixes things, or the boot path that starts him.
_CORE_STAGE_ONLY = {
    "orion_core/brain/brain.py",
    "orion_core/brain/orion_graph.py",
    "orion_core/brain/orion_state.py",
    "orion_core/core.py",
    "orion_core/brain/orion_tools.py",
    "orion_core/brain/tool_registry.py",
}
_FORBIDDEN_SELF_EDIT = {
    "orion_core/brain/self_maintenance.py",  # the mechanism itself
    "runners/launcher.py",                    # boot path
    "runners/webview.py",
}
# Files/dirs safe to hot-reload live (prompts, skills, configs).
_HOT_RELOAD_OK_PREFIXES = ("orion_core/prompts/", "orion_core/skills/")

_BACKUP_DIR = "orion_core/memory/self_edit_backups"


# =========================================================================
# clear_memory
# =========================================================================
class ClearMemoryInput(BaseModel):
    scope: str = Field(
        ...,
        description="What to clear: 'conversation' (current+recent chat), "
                    "'episodic' (history of what happened), 'preferences' "
                    "(saved directives/SOPs), 'people' (the people tree), or "
                    "'all'. 'conversation' and 'episodic' are safe. "
                    "'preferences', 'people', and 'all' are IRREVERSIBLE and must "
                    "only be called after the user has explicitly confirmed.",
    )
    confirmed: bool = Field(
        False,
        description="Must be True to execute an irreversible scope "
                    "(preferences/people/all). Leave False and ask the user first.",
    )


@tool("clear_memory", args_schema=ClearMemoryInput)
async def clear_memory(scope: str, confirmed: bool = False) -> str:
    """Clear Orion's memory by scope. Use when the user asks Orion to forget /
    reset / clear his memory. Map plain language to a scope: 'clear our chat' →
    conversation; 'forget everything about my family' → people; 'clear all your
    memory' → default to conversation+episodic and CONFIRM before touching
    preferences or people (those are not easily undone)."""
    scope = (scope or "").strip().lower()
    irreversible = scope in ("preferences", "people", "all")
    if irreversible and not confirmed:
        return ("[CONFIRM REQUIRED] Clearing '%s' is irreversible (it erases saved "
                "preferences and/or the people you know). Ask the user to confirm, "
                "then call again with confirmed=True." % scope)

    done = []

    def _rm(path):
        try:
            if os.path.isfile(path):
                os.remove(path)
                return True
        except Exception:
            pass
        return False

    # conversation → the checkpoint DB (rolling thread state)
    if scope in ("conversation", "all"):
        if _rm("orion_core/memory/checkpoints.sqlite"):
            done.append("conversation history")
        else:
            done.append("conversation history (already clear)")

    # episodic / preferences / people live in ChromaDB collections.
    try:
        from orion_core.brain.rag_chroma import RAGMemory
        if scope in ("episodic", "all"):
            try:
                RAGMemory(collection_name="episodic_memory",
                          path="logs/rag_memory").delete(where={})
                done.append("episodic memory")
            except Exception as e:
                done.append(f"episodic (skipped: {e})")
        if scope in ("preferences", "all"):
            try:
                RAGMemory(collection_name="orion_directives").delete(where={})
                # also reset the seed marker so defaults re-seed on next boot
                _rm("data/.directives_seeded")
                done.append("saved preferences/directives")
            except Exception as e:
                done.append(f"preferences (skipped: {e})")
    except Exception as e:
        return f"[System Error: memory backend unavailable: {e}]"

    # people → the people tree JSON
    if scope in ("people", "all"):
        if _rm("data/people_tree.json"):
            done.append("people tree")
        else:
            done.append("people tree (already clear)")

    if not done:
        return f"[No matching memory scope: '{scope}']"
    return "Cleared: " + ", ".join(done) + ". (Takes full effect on next restart.)"


# =========================================================================
# self_edit  (verify → stage for restart)
# =========================================================================
class SelfEditInput(BaseModel):
    file_path: str = Field(..., description="Project-relative path of the CORE file "
                           "to edit (e.g. 'orion_core/brain/orion_graph.py').")
    new_content: str = Field(..., description="The COMPLETE new file contents (not a "
                             "diff). Self-reviewed against the rubric, no secrets, "
                             "no stubs, no duplicated logic, no keyword-branching.")
    reason: str = Field(..., description="What this change fixes/adds and why.")
    confirmed: bool = Field(False, description="Must be True — the user has been told "
                            "a core change is being staged and agreed. Ask first.")


@tool("self_edit", args_schema=SelfEditInput)
async def self_edit(file_path: str, new_content: str, reason: str,
                    confirmed: bool = False) -> str:
    """Edit one of Orion's own CORE files. The change is syntax+import verified,
    backed up, and STAGED — it takes effect on the next restart, never live (a bad
    core edit must not crash the running Orion). You MUST have self-reviewed against
    the rubric. Requires user confirmation (an awareness checkpoint) before staging."""
    fp = file_path.strip().lstrip("./")

    if fp in _FORBIDDEN_SELF_EDIT:
        return (f"[REFUSED] '{fp}' is the self-edit mechanism or boot path and "
                "cannot be self-edited — that risks an unrecoverable state. A human "
                "must change this file.")
    if fp not in _CORE_STAGE_ONLY and not fp.startswith(_HOT_RELOAD_OK_PREFIXES):
        return (f"[REFUSED] '{fp}' is not a recognised editable core file. If this "
                "is a new file or an unusual path, ask the user to add it manually.")

    if not os.path.exists(fp):
        return f"[REFUSED] '{fp}' does not exist. self_edit modifies existing files."

    if not confirmed:
        return (f"[CONFIRM REQUIRED] I want to change '{fp}' to: {reason}. This is a "
                "core change and will take effect on the next restart. Ask the user "
                "to confirm, then call again with confirmed=True.")

    # Security: refuse hardcoded secrets slipping into core.
    if any(k in new_content.lower() for k in ("api_key=\"", "api_key='", "secret=\"",
                                              "password=\"")) and "os.environ" not in new_content and "getenv" not in new_content:
        return ("[REJECTED] Proposed content appears to hardcode a secret. Keys must "
                "come from the environment.")

    # 1. Syntax check the new content
    try:
        compile(new_content, fp, "exec")
    except SyntaxError as e:
        return f"[REJECTED] Syntax error at line {e.lineno}: {e.msg}. Not applied."

    # 2. Backup the current file (timestamped) — recovery beyond git.
    os.makedirs(_BACKUP_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(_BACKUP_DIR, f"{os.path.basename(fp)}.{stamp}.bak")
    await asyncio.to_thread(shutil.copy2, fp, backup)

    # 3. Write to a staged sibling, import-test it in isolation before swapping.
    staged = fp + ".staged"
    await asyncio.to_thread(lambda: open(staged, "w").write(new_content))
    test_src = (
        "import importlib.util;"
        f"spec=importlib.util.spec_from_file_location('t',r'{staged}');"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)"
    )
    proc = await asyncio.create_subprocess_exec(
        "python3", "-c", test_src,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except asyncio.TimeoutError:
        proc.kill()
        os.remove(staged)
        return "[REJECTED] Import test timed out. Not applied."
    if proc.returncode != 0:
        os.remove(staged)
        # Note: import failure can be legitimate for files with heavy deps; report
        # it honestly rather than forcing through.
        return (f"[REJECTED] The edited file failed to import in isolation:\n"
                f"{stderr.decode(errors='ignore')[:500]}\nNot applied. Backup kept.")

    # 4. Swap staged → live. Takes effect on next restart (core files are read at
    # boot). We do NOT hot-reload core modules.
    await asyncio.to_thread(lambda: os.replace(staged, fp))
    return (f"Core file '{fp}' updated and staged (reason: {reason}). Verified: "
            f"syntax + isolated import passed. Backup at {backup}. "
            f"Change takes effect on the next restart.")


def get_self_maintenance_tools() -> list[BaseTool]:
    """Convenience accessor for the ToolRegistry.
    Note: skill creation is handled by forge_new_skill in orion_tools.py (single
    source of truth — no duplicate here)."""
    return [clear_memory, self_edit]
