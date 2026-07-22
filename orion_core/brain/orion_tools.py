# orion_core/brain/orion_tools.py
from orion_core.brain.mcp_manager import mcp_manager
import asyncio
import os
import glob
import base64
import json
import logging
import shlex
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

logger = logging.getLogger("orion.tools")

# Shared secret for the root daemon — read from env OR a project-relative file
# (must match orion_daemon.py). Env first (if a launch preserves it), else the
# file at <project_root>/.secrets/kernel_secret. Never hardcoded.
def _load_kernel_secret():
    env = os.environ.get("ORION_KERNEL_SECRET")
    if env:
        return env.strip()
    try:
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / ".secrets" / "kernel_secret"
        return p.read_text(encoding="utf-8").strip() or None
    except Exception:
        return None

_KERNEL_SECRET = _load_kernel_secret()

# =========================================================================
# PYDANTIC ARGUMENT SCHEMAS
# =========================================================================

class BashInput(BaseModel):
    command: str = Field(
        ...,
        description="A bash command run as the LOGGED-IN USER (not root) on Ubuntu "
                    "Wayland. Use for user-level work: reading files, listing/"
                    "discovering state (dpkg -l, which, gsettings get/set for the "
                    "user), running user apps. Do NOT use it for anything needing "
                    "root — installing/removing packages (apt), systemctl, editing "
                    "system files: those go through 'execute_python_payload' (the "
                    "root daemon), NOT sudo/pkexec here, which would pop a password "
                    "dialog and stall you. For product/price lookups use mcp_search.",
    )

class PythonPayloadInput(BaseModel):
    code: str = Field(
        ...,
        description="A complete Python script executed as ROOT by the daemon — the "
                    "correct tool for ANY privileged/system change: installing or "
                    "removing packages (wrap apt in subprocess, e.g. "
                    "subprocess.run(['apt-get','install','-y','<pkg>'])), systemctl, "
                    "writing system files. Use this instead of sudo/pkexec so the "
                    "action runs seamlessly without a password dialog. Also for "
                    "complex logic or math. Runs headless: no print()/input() for "
                    "control flow; return values via stdout.",
    )

class MCPSearchInput(BaseModel):
    query: str = Field(..., description="The precise technical or product query for the live web.")

class ForgeSkillInput(BaseModel):
    skill_name: str = Field(..., description="Short underscore_separated name (e.g., clipboard_manager).")
    description: str = Field(..., description="Concise docstring explaining what the tool does.")
    python_code: str = Field(..., description="Raw executable Python containing the @tool decorated logic.")

class SchedulerInput(BaseModel):
    task_type: str = Field(..., description="Category: 'reminder', 'monitor', 'insight'.")
    time_str: str = Field(..., description="'YYYY-MM-DD HH:MM' (or 'HH:MM' for today).")
    payload: str = Field(..., description="The reminder text or the instruction to run when due.")
    action: str = Field("notify", description="'notify' to just speak it; 'run' to feed it back through the brain.")
    repeat: str = Field(None, description="Optional: daily, weekly, hourly, twice_a_day.")

class AskExternalExpertInput(BaseModel):
    architecture_context: str = Field(..., description="The OS, hardware, and strict constraints.")
    failed_code: str = Field(..., description="The exact code that failed.")
    error_trace: str = Field(..., description="The raw stderr or stack trace.")
    specific_query: str = Field(..., description="The highly technical question for the Senior AI.")

class AskUserInput(BaseModel):
    question: str = Field(
        ...,
        description="One concise question to put to the user, PHRASED IN YOUR BUTLER "
                    "VOICE — refined and courteous ('May I ask who I have the pleasure "
                    "of addressing, Sir?'), never blunt ('What is your name?'). This "
                    "text is spoken to the user verbatim. Use this ONLY when a "
                    "fact is something ONLY the user could provide — a name they want, "
                    "content they intend, or a preference not on record and NOT "
                    "discoverable with another tool. Never use it for facts you can "
                    "find yourself (device models, file paths, prices, system state). "
                    "Do NOT use it for OFFERS or PERMISSION ('would you like me to…?', "
                    "'shall I start it?'). State an offer as a normal reply and let the "
                    "user respond in their own time — suspending on an offer would "
                    "swallow their next instruction.",
    )

# =========================================================================
# HUMAN-IN-THE-LOOP  (control-plane tool: suspends the task, awaits a reply)
# =========================================================================

@tool("ask_user", args_schema=AskUserInput)
async def ask_user(question: str) -> str:
    """Pause the current task and ask the user one clarifying question, then wait
    for their reply. Use ONLY for facts that cannot be discovered with any other
    tool, and NEVER for offers or permission ('shall I…?', 'would you like…?') —
    those are ordinary replies, not reasons to suspend. Calling this SUSPENDS
    execution; the user's answer is handed back to you so you can continue the
    SAME task — you are not restarting from scratch."""
    # The real suspend/resume is performed by the graph's human_input_node, which
    # owns the interrupt() so resume re-runs a trivial node rather than replaying
    # sibling tool calls. This body is only a fallback if the tool is ever invoked
    # directly inside a node context.
    return interrupt({"question": question})


# =========================================================================
# CORE EXECUTION TOOLS
# =========================================================================

@tool("execute_bash", args_schema=BashInput)
async def execute_bash(command: str) -> str:
    """Executes raw bash commands natively on the Ubuntu Wayland host system."""
    if not command.strip():
        return "[System Error: No command provided.]"

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        output = ""
        if stdout:
            output += stdout.decode(errors="ignore").strip()
        if stderr:
            output += f"\n[STDERR]: {stderr.decode(errors='ignore').strip()}"

        if process.returncode == 0:
            return output or "Execution complete. Exit Code (0)."
        return f"Error: Command failed with Exit Code {process.returncode}.\nOutput:\n{output}"
    except Exception as e:
        return f"BASH_EXCEPTION: {str(e)}"


@tool("execute_python_payload", args_schema=PythonPayloadInput)
async def execute_python_payload(code: str) -> str:
    """Dispatches a Python payload to the high-privilege root daemon socket."""
    if not _KERNEL_SECRET:
        return ("[SYSTEM ERROR] ORION_KERNEL_SECRET is not set in this process, so "
                "I cannot authenticate to the root daemon. Set it and restart.")
    encoded = base64.b64encode(code.encode("utf-8")).decode("utf-8")
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", 65432)
        writer.write(json.dumps({"auth": _KERNEL_SECRET, "code_b64": encoded}).encode())
        await writer.drain()

        resp = await reader.read(65536)
        writer.close()
        await writer.wait_closed()

        data = json.loads(resp.decode())
        return data.get("output", "Execution complete. No output returned.")
    except (ConnectionRefusedError, OSError):
        return "[SYSTEM ERROR] Root daemon offline. Run 'pkexec python3 runners/orion_daemon.py'."


@tool("mcp_search", args_schema=MCPSearchInput)
async def mcp_search(query: str) -> str:
    """Use for live external web data (weather, news, product specs, prices)."""
    try:
        text = await mcp_manager.call_search(query)
        if not text.strip():
            return "[System Error: the search returned no results.]"
        return f"[Source: Live Web Search]\n{text}"
    except Exception as e:
        return f"[System Error: Research Gateway failed: {str(e)}]"


# =========================================================================
# SCHEDULING TOOLS  (wraps system/scheduler.py)
# =========================================================================

@tool("schedule_task", args_schema=SchedulerInput)
async def schedule_task(task_type: str, time_str: str, payload: str,
                        action: str = "notify", repeat: str = None) -> str:
    """Creates a persistent scheduled task, reminder, or recurring monitor."""
    from system.scheduler import Scheduler
    def _add():
        return Scheduler().add_task(task_type, time_str, action, payload, repeat)
    return await asyncio.to_thread(_add)


@tool("list_scheduled_tasks")
async def list_scheduled_tasks() -> str:
    """Lists all pending scheduled tasks and reminders."""
    from system.scheduler import Scheduler
    return await asyncio.to_thread(lambda: Scheduler().get_tasks())


# =========================================================================
# FORGE — now with sandbox → test → promote staging
# =========================================================================

@tool("forge_new_skill", args_schema=ForgeSkillInput)
async def forge_new_skill(skill_name: str, description: str, python_code: str) -> str:
    """
    Permanently create a new capability. The code is staged, syntax-checked, and
    import-tested in an isolated subprocess BEFORE it is promoted to the live
    registry. Bad code never reaches the active skills directory.
    """
    safe = "".join(c for c in skill_name.lower() if c.isalnum() or c == "_")
    if not safe:
        return "[FORGE REJECTED] Invalid skill name."

    pending_dir = "orion_core/skills/pending"
    live_dir = "orion_core/skills"
    os.makedirs(pending_dir, exist_ok=True)
    os.makedirs(live_dir, exist_ok=True)
    pending_path = os.path.join(pending_dir, f"{safe}.py")
    live_path = os.path.join(live_dir, f"{safe}.py")

    # 1. Stage to pending
    await asyncio.to_thread(lambda: open(pending_path, "w").write(python_code))

    # 2. Syntax gate
    try:
        compile(python_code, pending_path, "exec")
    except SyntaxError as e:
        return f"[FORGE REJECTED] Syntax error line {e.lineno}: {e.msg}"

    # 3. Import test in an isolated subprocess with a hard timeout
    test_src = (
        "import importlib.util,sys;"
        f"spec=importlib.util.spec_from_file_location('t',r'{pending_path}');"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
        # 4. Verify at least one LangChain @tool is exposed
        "from langchain_core.tools import BaseTool;"
        "assert any(isinstance(getattr(m,a),BaseTool) for a in dir(m)),'no @tool found'"
    )
    proc = await asyncio.create_subprocess_exec(
        "python3", "-c", test_src,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=25)
    except asyncio.TimeoutError:
        proc.kill()
        return "[FORGE REJECTED] Import test timed out — code may block or loop at import time."

    if proc.returncode != 0:
        err = stderr.decode(errors="ignore")[:500]
        return f"[FORGE REJECTED] Sandbox import failed:\n{err}"

    # 5. Promote
    await asyncio.to_thread(lambda: os.replace(pending_path, live_path))
    return (f"Skill '{safe}' passed syntax, import, and @tool-presence checks "
            f"and was promoted to the live registry.")


class SaveDirectiveInput(BaseModel):
    rule: str = Field(..., description="The standing rule to remember, phrased as an instruction.")
    scope: str = Field(
        ...,
        description="'prime' for an always-on rule that applies to EVERY request "
                    "(e.g. how to address the user, safety rules); 'contextual' for "
                    "a situational preference that only applies sometimes "
                    "(e.g. preferred shops when buying, formatting when writing code).",
    )
    applies_when: str = Field(
        "",
        description="REQUIRED for contextual scope: a short natural-language "
                    "description of the situations this rule applies to "
                    "(e.g. 'buying, shopping, comparing prices'). Leave empty for prime.",
    )


@tool("save_directive", args_schema=SaveDirectiveInput)
async def save_directive(rule: str, scope: str, applies_when: str = "") -> str:
    """
    Persist a standing user preference so it survives across sessions. Use when the
    user expresses a lasting rule ('remember to always...', 'from now on...',
    'never...'). Choose 'prime' for universal rules, 'contextual' for situational
    ones. Confirm the choice with the user in your spoken reply.
    """
    from orion_core.brain.directives import DirectivesManager
    dm = DirectivesManager()

    def _persist():
        if scope.strip().lower() == "prime":
            dm.add_prime(rule)
            return f"Saved as an always-on directive: '{rule}'."
        # contextual
        if not applies_when.strip():
            return ("[NEEDS applies_when] A contextual directive requires a short "
                    "description of when it applies. Re-call with applies_when set.")
        sop_id = "sop_" + "".join(c for c in rule.lower() if c.isalnum())[:32]
        dm.add_sop(sop_id, applies_when, rule)
        return f"Saved as a contextual directive (applies when: {applies_when}): '{rule}'."

    return await asyncio.to_thread(_persist)


@tool("check_background_jobs")
async def check_background_jobs() -> str:
    """Checks the status of active background jobs in the spool directory."""
    def _read_jobs():
        os.makedirs("orion_core/spool", exist_ok=True)
        jobs = []
        for filepath in glob.glob("orion_core/spool/*.json"):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    jobs.append(
                        f"- {data.get('job_id', 'Unknown')} "
                        f"({data.get('type', 'Task')}): "
                        f"{data.get('status', 'Unknown')} - {data.get('progress', '0%')}"
                    )
            except Exception:
                continue
        return "\n".join(jobs) if jobs else "There are currently no active background tasks."
    return await asyncio.to_thread(_read_jobs)


# =========================================================================
# ENVIRONMENT / SECRET PROVISIONING
# =========================================================================
# "set an env var" is really four places depending on scope; picking the right
# file is the skill. And a SECRET must never travel through the model — so when
# no value is supplied, we capture it via a GUI dialog straight into the file.

def _upsert_export(path: str, name: str, quoted_value: str) -> None:
    """Append (or replace) `export NAME=value` in a file, idempotently."""
    lines = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            lines = [ln for ln in fh.readlines()
                     if not ln.strip().startswith(f"export {name}=")]
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    lines.append(f"export {name}={quoted_value}\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


def _ensure_line(path: str, line: str) -> None:
    """Ensure `line` is present in `path` (create the file if needed)."""
    existing = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            existing = fh.read()
    if line.strip() not in existing:
        with open(path, "a", encoding="utf-8") as fh:
            if existing and not existing.endswith("\n"):
                fh.write("\n")
            fh.write(line if line.endswith("\n") else line + "\n")


class SetEnvVarInput(BaseModel):
    var_name: str = Field(
        ...,
        description="The variable NAME only, e.g. GOOGLE_API_KEY or EDITOR. "
                    "NEVER put the value here.",
    )
    scope: str = Field(
        ...,
        description="'session' = this virtualenv only (reload by reactivating the "
                    "venv); 'user' = persists for your login and is readable by "
                    "other apps (reload by starting a new login shell). Ask the user "
                    "which they want if it is not clear — it is a choice only they "
                    "can make.",
    )
    value: str = Field(
        None,
        description="OMIT this for anything sensitive — API keys, tokens, passwords, "
                    "credentials. When omitted, a secure desktop dialog captures the "
                    "value so the secret NEVER enters this conversation, the logs, or "
                    "memory. Only pass a value here for NON-sensitive settings "
                    "(e.g. EDITOR=vim).",
    )


@tool("set_env_variable", args_schema=SetEnvVarInput)
async def set_env_variable(var_name: str, scope: str, value: str = None) -> str:
    """Set an environment variable in the correct file for its scope. For SECRETS
    (API keys, tokens, passwords) OMIT `value` — a secure GUI dialog captures it so
    the secret never passes through the model, chat, voice, logs, or memory. The
    stored value is never echoed back. Tell the user the reload step afterwards."""
    import re as _re
    name = var_name.strip()
    if not _re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        return f"[SYSTEM ERROR] '{var_name}' is not a valid environment variable name."
    scope = (scope or "").strip().lower()
    if scope not in ("session", "user"):
        return "[SYSTEM ERROR] scope must be 'session' or 'user'."

    # Secret path: no value supplied → capture via GUI so it bypasses the model.
    captured_secretly = False
    if value is None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "zenity", "--entry", "--hide-text",
                "--title", "Orion — secure value",
                "--text", f"Enter the value for {name}\n(kept private — not stored in chat or memory):",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            out, _ = await proc.communicate()
            if proc.returncode != 0:
                return "[SYSTEM ERROR] The secure input dialog was cancelled; nothing was stored."
            value = out.decode(errors="ignore").rstrip("\n")
            captured_secretly = True
        except FileNotFoundError:
            return ("[SYSTEM ERROR] zenity is not installed, so I cannot capture the "
                    "value securely. Install zenity (apt) or provide a non-secret value.")
    if value == "":
        return "[SYSTEM ERROR] No value was entered; nothing was stored."

    quoted = shlex.quote(value)

    def _write():
        if scope == "session":
            venv = os.environ.get("VIRTUAL_ENV")
            if not venv:
                return None, None
            target = os.path.join(venv, "bin", "activate")
            _upsert_export(target, name, quoted)
            return target, "deactivate and reactivate the virtualenv"
        # user scope: dedicated 600 file, sourced from ~/.profile so apps see it too
        cfg_dir = os.path.expanduser("~/.config/orion")
        os.makedirs(cfg_dir, exist_ok=True)
        target = os.path.join(cfg_dir, "env")
        _upsert_export(target, name, quoted)
        os.chmod(target, 0o600)
        _ensure_line(os.path.expanduser("~/.profile"),
                     '[ -f "$HOME/.config/orion/env" ] && . "$HOME/.config/orion/env"')
        return target, "start a new login shell (or run: source ~/.config/orion/env)"

    target, reload_hint = await asyncio.to_thread(_write)
    if target is None:
        return "[SYSTEM ERROR] scope 'session' requires an active virtualenv (VIRTUAL_ENV is unset)."

    how = "captured privately via a secure dialog" if captured_secretly else "set"
    return (f"{name} {how} at {scope} scope (written to {target}). "
            f"To load it: {reload_hint}. The value itself was written directly and "
            f"is not recorded in this conversation.")


def _owner_profile(pm):
    """The owner's profile, or None if no owner is truly established.

    'Established' means the pointer AND the profile it names both exist. A dangling
    pointer (owner set to a profile that was deleted) previously deadlocked the
    system: set_owner refused ("already established"), while every lookup returned
    "Unknown" ("no such person"). Treating a dangling pointer as NOT established
    makes that contradiction unrepresentable.
    """
    key = pm.tree.get("owner")
    if not key:
        return None
    return pm.tree.get("profiles", {}).get(key)


class SetOwnerInput(BaseModel):
    name: str = Field(..., description="The name of the person you've confirmed is "
                                       "the system owner.")


@tool("set_owner", args_schema=SetOwnerInput)
async def set_owner(name: str) -> str:
    """Establish the system OWNER, with full authority. Call this ONCE — only after
    the speaker has confirmed (in their own words, which YOU interpret) that they are
    the owner and told you their name. Do not call it if an owner already exists."""
    from orion_core.brain.people_manager import PeopleManager

    def _do():
        pm = PeopleManager()
        existing = _owner_profile(pm)
        if existing:
            return (f"An owner is already established ({existing.get('name')}); "
                    f"I won't change it.")
        clean = (name or "").strip()
        if not clean:
            return "[Cannot set owner: no name was given.]"
        # No real owner (none set, or a dangling pointer) → establish/heal it.
        return pm.register_person(clean, role="Owner")

    return await asyncio.to_thread(_do)


class RememberPersonInput(BaseModel):
    person: str = Field(
        ...,
        description="The person's name, EXACTLY as it appears in the [PEOPLE] list in "
                    "your context. Everyone you know is listed there — read it and use "
                    "the real name (the owner is marked). Never invent a name from a "
                    "username or hostname, and never use a placeholder.",
    )
    key: str = Field(
        ...,
        description="The attribute name. Use 'home' and 'work' for places/addresses "
                    "— these are what location-aware features (weather, 'near me', "
                    "regional search) read. Otherwise use whatever attribute name "
                    "fits the fact naturally.",
    )
    value: str = Field(..., description="The value to store (a place, a date, a preference, etc.).")


@tool("remember_about_person", args_schema=RememberPersonInput)
async def remember_about_person(person: str, key: str, value: str) -> str:
    """Durably remember a stable FACT about a person — the owner or someone they
    know — so it survives across sessions and is available to future requests. Use
    for lasting facts: where someone lives ('home'), works ('work'), a date, a
    preference. For the owner, pass person='me'. Places stored as 'home'/'work'
    become the owner's location context. This is for FACTS ABOUT PEOPLE, not
    standing rules (use save_directive for rules) and not objects."""
    from orion_core.brain.people_manager import PeopleManager

    def _do():
        pm = PeopleManager()
        target = (person or "").strip()
        if not target:
            return "[Cannot store: no person named.]"
        if pm.get_person(target):
            return pm.update_attribute(target, key, value)
        # Not on record. Do NOT guess who they are or how they're related.
        return (f"[UNKNOWN PERSON] '{person}' is not in the people list I can see. "
                f"Use the exact name as listed in [PEOPLE]. If they are genuinely new, "
                f"ask the user (ask_user) their name and how they're related first.")

    return await asyncio.to_thread(_do)


@tool("ask_expert", args_schema=AskExternalExpertInput)
async def ask_expert(architecture_context: str, failed_code: str,
                     error_trace: str, specific_query: str) -> str:
    """ESCALATION: escalates to Groq Llama-3.3-70B for expert debugging."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    prompt = (
        f"You are a Senior AI Architect. Diagnose and fix this.\n"
        f"Architecture: {architecture_context[:400]}\n"
        f"Failing code:\n{failed_code[:2000]}\n"
        f"Error trace:\n{error_trace[:2000]}\n"
        f"Task: {specific_query}"
    )
    response = await llm.ainvoke([("human", prompt)])
    return f"[EXPERT RESPONSE]:\n{response.content}"


# =========================================================================
# ENVIRONMENT / SECRET PROVISIONING
# =========================================================================
# Two scopes Orion can actually persist (a child process cannot mutate the parent
# shell, so "this shell only" is not settable by Orion):
#   • user  → ~/.config/orion/env  (chmod 600), sourced from ~/.bashrc so shells
#             and the apps they launch see it.
#   • venv  → <VIRTUAL_ENV>/bin/activate  (only inside Orion's virtualenv).
# SECURITY RULE: a secret VALUE must never pass through the model. set_env_var is
# for non-secret values the model already has; provision_secret takes NO value —
# the user types it into a GUI dialog, and it goes straight to disk.

def _scope_target(scope: str):
    s = (scope or "user").strip().lower()
    if s == "venv":
        venv = os.environ.get("VIRTUAL_ENV")
        if not venv:
            return None, None, "No active virtualenv (VIRTUAL_ENV unset); cannot use venv scope."
        return Path(venv) / "bin" / "activate", "venv", None
    if s == "user":
        return Path.home() / ".config" / "orion" / "env", "user", None
    return None, None, f"Unknown scope '{scope}'. Use 'user' or 'venv'."

def _write_export(path: Path, name: str, value: str):
    """Append/replace an `export NAME="value"` line and lock perms to 600."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    prefix = f"export {name}="
    lines = [ln for ln in lines if not ln.strip().startswith(prefix)]
    lines.append(f'export {name}={shlex.quote(value)}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass

def _ensure_user_sourced():
    """Make ~/.bashrc source the Orion env file once, so user-scope vars load."""
    bashrc = Path.home() / ".bashrc"
    marker = "/.config/orion/env"
    try:
        existing = bashrc.read_text(encoding="utf-8") if bashrc.exists() else ""
        if marker not in existing:
            with bashrc.open("a", encoding="utf-8") as f:
                f.write("\n# Orion-managed environment\n"
                        "[ -f ~/.config/orion/env ] && . ~/.config/orion/env\n")
    except Exception:
        pass

def _reload_hint(scope: str) -> str:
    return ("open a new terminal or run 'source ~/.bashrc'" if scope == "user"
            else "deactivate and reactivate the virtualenv")


class SetEnvVarInput(BaseModel):
    name: str = Field(..., description="Variable name, e.g. EDITOR.")
    value: str = Field(
        ...,
        description="The value — use this tool ONLY for NON-secret values you "
                    "already have (e.g. EDITOR=nano, LANG=en_US.UTF-8). For API keys, "
                    "tokens, passwords, or ANY value the user must supply, use "
                    "'provision_secret' instead so the value never enters this "
                    "conversation.",
    )
    scope: str = Field("user", description="'user' (persistent, all shells/apps) or "
                                           "'venv' (only inside Orion's virtualenv).")


@tool("set_env_var", args_schema=SetEnvVarInput)
async def set_env_var(name: str, value: str, scope: str = "user") -> str:
    """Persist a NON-secret environment variable to the chosen scope. Do NOT use for
    secrets — use provision_secret for anything sensitive."""
    def _do():
        path, resolved, err = _scope_target(scope)
        if err:
            return f"[SYSTEM ERROR] {err}"
        _write_export(path, name, value)
        if resolved == "user":
            _ensure_user_sourced()
        return f"Set {name} at {resolved} scope ({path}). To load it, {_reload_hint(resolved)}."
    return await asyncio.to_thread(_do)


class ProvisionSecretInput(BaseModel):
    name: str = Field(..., description="Variable name to store, e.g. GOOGLE_API_KEY.")
    scope: str = Field("user", description="'user' or 'venv' (see set_env_var).")
    hidden: bool = Field(True, description="True (default) shows a masked password "
                                           "dialog. Set False only for non-sensitive "
                                           "text the user must type.")
    label: str = Field("", description="Optional short label for the dialog, e.g. "
                                       "'Google API key'.")


@tool("provision_secret", args_schema=ProvisionSecretInput)
async def provision_secret(name: str, scope: str = "user",
                           hidden: bool = True, label: str = "") -> str:
    """Ask the USER for a value through a secure GUI dialog and store it as an env
    var. The value NEVER enters this conversation, is never spoken, logged, or sent
    to any model — you supply only the NAME and SCOPE; the user types the value.
    Use for API keys, tokens, passwords, or any value only the user holds. Prefer
    this over ask_user for anything sensitive."""
    title = f"Orion: {label or name}"
    if hidden:
        args = ["zenity", "--password", f"--title={title}"]
    else:
        args = ["zenity", "--entry", f"--title={title}", f"--text=Enter value for {name}:"]

    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    except FileNotFoundError:
        return ("[SYSTEM ERROR] zenity is not installed, so I cannot prompt for the "
                "value securely. Install zenity (apt) and try again — I will not ask "
                "for a secret through the conversation.")
    try:
        # Hard time limit: the dialog can never hang the assistant forever (this was
        # the 6-minute freeze). If the user doesn't answer, we give up cleanly.
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=180)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return ("[SYSTEM ERROR] The secure input dialog wasn't answered within 3 "
                "minutes, so I stopped waiting and stored nothing. If no window "
                "appeared, it may have opened behind another — tell me and I'll retry.")
    if proc.returncode != 0:
        return "[SYSTEM ERROR] The secure input dialog was cancelled; nothing was stored."
    secret_value = out.decode(errors="ignore").rstrip("\n")
    if not secret_value:
        return "[SYSTEM ERROR] Empty value; nothing was stored."

    def _store():
        path, resolved, err = _scope_target(scope)
        if err:
            return None, f"[SYSTEM ERROR] {err}"
        _write_export(path, name, secret_value)
        if resolved == "user":
            _ensure_user_sourced()
        return resolved, None
    resolved, err = await asyncio.to_thread(_store)
    if err:
        return err
    return (f"Stored {name} securely at {resolved} scope — value hidden, never seen "
            f"by me. To load it, {_reload_hint(resolved)}. I've recorded that {name} "
            f"is now provisioned.")


class CheckEnvVarInput(BaseModel):
    name: str = Field(..., description="Variable name to check.")


@tool("check_env_var", args_schema=CheckEnvVarInput)
async def check_env_var(name: str) -> str:
    """Report WHETHER an environment variable is set, and in which scope, WITHOUT
    revealing its value. Use to check if a key is already configured before asking
    the user for it."""
    def _check():
        found = []
        if os.environ.get(name):
            found.append("the running process")
        for scope in ("user", "venv"):
            path, resolved, err = _scope_target(scope)
            if err or not path or not path.exists():
                continue
            try:
                if any(ln.strip().startswith(f"export {name}=")
                       for ln in path.read_text(encoding="utf-8").splitlines()):
                    found.append(f"{resolved} scope")
            except Exception:
                continue
        if found:
            return f"{name} is set in: " + ", ".join(found) + ". (Value not shown.)"
        return f"{name} is not set in any scope I manage."
    return await asyncio.to_thread(_check)