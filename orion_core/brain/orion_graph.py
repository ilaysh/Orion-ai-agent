# orion_core/brain/orion_graph.py
import os
import re
import json
import time
from datetime import datetime
from pathlib import Path
from orion_core.brain.tool_registry import ToolRegistry
from system.telemetry.telemetry import timed
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq  # Free-tier escalation (preferred over paid Sonnet)

# Control-plane tools (ask_user, provision_secret, remember_about_person, …) are no
# longer hand-listed here — the ToolRegistry auto-discovers every @tool in the tools
# module, so binding and execution share one source of truth. ask_user is routed to
# human_input_node upstream; execute_python_payload is gated by confirm_action_node.
# The graph refers to those by name (strings) where routing needs them.

# Project Imports
from orion_core.brain.orion_state import OrionState
from system.project_mapper import ProjectMapper
from orion_core.brain.personality import Personality
from orion_core.brain.directives import DirectivesManager
from orion_core.brain.rag_chroma import RAGMemory

personality = Personality()

LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"
LOCAL_MODEL = "gemma-4-12b"
# Escalation target. Groq's free tier, OpenAI-compatible, supports tool calling —
# so the escalated model can still USE tools, unlike the text-only ask_expert path.
ESCALATION_MODEL = "llama-3.3-70b-versatile"

# Escalate to the cloud reasoner only after the LOCAL model has had a real chance
# to see an error and re-plan — NOT on the first failed command. One diagnosis
# turn locally first. (Was: escalate on any single errored tool = hair-trigger.)
ESCALATION_STRIKE_THRESHOLD = 2

# No single tool result may exceed this many characters (~1.5k tokens). Prevents a
# giant `ls`/`cat` from overflowing the context window and defeating the trimmer.
_MAX_TOOL_OUTPUT_CHARS = 6000

# A tool result beginning with any of these is treated as a failure (a strike).
ERROR_PREFIXES = ("[FORGE REJECTED]", "[SYSTEM ERROR]", "[SYSTEM BLOCKED]",
                  "[CONFIRM REQUIRED]", "Error:", "BASH_EXCEPTION", "[FATAL EXCEPTION]")

# --- Always-on tool-execution audit (bug #2) ---
# Every tool invocation is appended here as one JSON line, success or failure,
# so destructive actions always leave a trace independent of the chat log.
_AUDIT_PATH = Path("logs") / "tool_audit.jsonl"

def _audit_tool(name: str, args: dict, ok: bool, result_str: str) -> None:
    try:
        _AUDIT_PATH.parent.mkdir(exist_ok=True)
        with _AUDIT_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "tool": name,
                "args": {k: str(v)[:300] for k, v in (args or {}).items()},
                "ok": ok,
                "result": result_str[:500],
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass  # auditing must never break execution

# --- Lightweight module singletons (avoid re-loading the embedder per call) ---
_directives = None
_project_rag = None
_bridge = None


def _get_directives() -> DirectivesManager:
    global _directives
    if _directives is None:
        _directives = DirectivesManager()
    return _directives


def _get_project_rag() -> RAGMemory:
    global _project_rag
    if _project_rag is None:
        _project_rag = RAGMemory(collection_name="orion_rag")
    return _project_rag


def _get_bridge():
    global _bridge
    if _bridge is None:
        from orion_core.brain.meta_bridge import MetaBridge
        _bridge = MetaBridge()
    return _bridge


def _local_llm(temperature: float = 0.0, max_tokens: int = 1536,
               thinking: bool = False, **kw) -> ChatOpenAI:
    # Two hard-won defaults:
    #  • max_tokens is ALWAYS set — an uncapped call let Gemma 4's reasoning run for
    #    13 MINUTES on the planner (the 802s stall). Nothing is unbounded now.
    #  • thinking is OFF by default. Gemma 4's reasoning is on under --jinja and,
    #    uncapped, generates a huge trace before its answer. The llama.cpp-supported
    #    way to disable it per request is chat_template_kwargs.enable_thinking=false.
    #    Simple nodes (router/planner/validator) never need it; pass thinking=True
    #    on a node only if it genuinely needs step-by-step reasoning.
    extra_body = None if thinking else {"chat_template_kwargs": {"enable_thinking": False}}
    return ChatOpenAI(base_url=LOCAL_BASE_URL, api_key="EMPTY",
                      model=LOCAL_MODEL, temperature=temperature,
                      max_tokens=max_tokens, extra_body=extra_body, **kw)


async def narrate(situation: str, fallback: str) -> str:
    """Generate ONE short, natural, butler-voiced spoken line for a UX moment, so
    Orion never repeats a canned string. `situation` describes what to convey (the
    model phrases it freshly); `fallback` is used ONLY if generation itself fails —
    a genuine last resort, never the normal path. Kept tiny and thinking-off so it
    is cheap and cannot stall, and it uses a fresh minimal prompt so it works even
    when the main context has overflowed."""
    try:
        llm = _local_llm(temperature=0.6, max_tokens=70)
        resp = await llm.ainvoke([HumanMessage(content=(
            "You are Alfred — a refined, concise British butler. Reply with ONE "
            "short, natural spoken sentence (no lists, no quotes, no stage "
            f"directions) that conveys: {situation}. Address the user as Sir, and "
            "vary your wording."))])
        line = (getattr(resp, "content", "") or "").strip().strip('"')
        return line or fallback
    except Exception:
        return fallback


def _log_io(node: str, sent_messages, response=None, error=None):
    """Uniform request/response logging for every model-calling node.
    Gated by LOG_LLM_IO (default on). Set LOG_LLM_IO=0 to silence."""
    if os.environ.get("LOG_LLM_IO", "1") != "1":
        return
    if response is None and error is None:
        print(f"┌── [{node}] LLM REQUEST " + "─" * 30)
        for m in sent_messages:
            role = getattr(m, "type", "?")
            body = str(getattr(m, "content", m))
            print(f"│ [{role}] {body[:500]}{'…' if len(body) > 500 else ''}")
        print("└" + "─" * 52)
    elif error is not None:
        print(f"┌── [{node}] LLM ERROR " + "─" * 32)
        print(f"│ {str(error)[:500]}")
        print("└" + "─" * 52)
    else:
        print(f"┌── [{node}] LLM RESPONSE " + "─" * 29)
        print(f"│ content: {str(getattr(response, 'content', response))[:700]}")
        if getattr(response, "tool_calls", None):
            print(f"│ tool_calls: {[tc['name'] for tc in response.tool_calls]}")
        print("└" + "─" * 52)


# =========================================================================
# GROUNDING
# =========================================================================
# =========================================================================
# CLARIFICATION  (the infra answer to ambiguous requests — not a bigger model)
# =========================================================================
@timed("graph.clarification_node")
async def clarification_node(state: OrionState) -> dict:
    # If we asked a question last turn, the user is now answering — proceed.
    if state.get("awaiting_clarification"):
        return {"awaiting_clarification": False}

    request = state.get("user_request", "")
    llm = _local_llm()
    prompt = (
        "You gate autonomous execution. Decide if this request is specified "
        "enough to act on WITHOUT guessing the user's intent. Unknown facts that "
        "you can DISCOVER yourself (device models, file paths, prices) do NOT "
        "count as ambiguity.\n"
        "If actionable, reply exactly: PROCEED\n"
        "If genuinely ambiguous, reply: CLARIFY: <one short question>\n\n"
        f"Request: {request}"
    )
    resp = await llm.ainvoke([HumanMessage(content=prompt)])
    text = resp.content.strip()
    if text.upper().startswith("CLARIFY"):
        question = text.split(":", 1)[1].strip() if ":" in text else \
            "Could you clarify what you'd like, Sir?"
        return {"awaiting_clarification": True, "final_response": question}
    return {"awaiting_clarification": False}


# =========================================================================
# PLANNER  (now grounded: directives + project RAG + live OS context)
# =========================================================================
@timed("graph.planner_node")
async def planner_node(state: OrionState) -> dict:
    """GROUNDING STEP (formerly the LLM planner).

    It no longer generates a speculative blueprint. That plan was written BEFORE any
    discovery, cost ~8s per request, and Alfred correctly ignored it (it planned to
    probe ~/.ssh and gpg for what was a one-call task). Best practice for an
    unpredictable, discover-as-you-go environment is ReAct: Alfred decides each step
    from what actually happened, not from a plan made in the dark. So this node now
    only does fast, LLM-free grounding (contextual directives + project RAG) and
    hands straight to Alfred, who plans and acts in one loop.
    """
    request = state.get("user_request", "")

    # 1. Data-driven directives — matched SEMANTICALLY, no keyword code. Best-effort:
    # if the module is old/missing or errors, inject nothing rather than crash.
    directive_block = ""
    try:
        dm = _get_directives()
        getter = getattr(dm, "get_contextual", None)
        if callable(getter):
            directive_block = getter(request) or ""
    except Exception as e:
        print(f"[Grounding] ⚠️ directive grounding skipped: {e}")

    # 2. Retrieval-augmented grounding from indexed project/docs (safe if empty).
    try:
        hits = _get_project_rag().retrieve(request, top_k=3)
        rag_context = "\n---\n".join(h["text"][:800] for h in hits) if hits else ""
    except Exception:
        rag_context = ""

    return {
        "intent_id": "system_engineer",
        "rag_context": rag_context,
        "active_tool_directives": directive_block or state.get("active_tool_directives"),
        "active_model": LOCAL_MODEL,
        "strike_count": 0,
        "loop_count": 0,
    }


# =========================================================================
# ALFRED  (hub: plans → tool calls → synthesises)
# =========================================================================
@timed("graph.alfred_node")
async def alfred_node(state: OrionState) -> dict:
    current_loops = state.get("loop_count", 0)
    if current_loops >= 30:
        text = await narrate(
            "you've reached your effort limit on this task without completing it, so "
            "you're honestly pausing rather than pressing on blindly",
            "Sir, I've reached my limit on this task without completing it — I've "
            "paused rather than press on blindly.")
        msg = AIMessage(content=text)
        # Force termination: flag it so routing goes straight to END, and clear
        # repair so validation can't drag it back into the loop.
        return {"messages": [msg], "final_response": msg.content,
                "repair_mode_active": False, "exhausted": True}

    current_model = state.get("active_model", LOCAL_MODEL)
    strike_count = state.get("strike_count", 0)
    cwd = state.get("current_working_dir", None)
    messages = state["messages"]
    last_msg = messages[-1] if messages else None

    # Error-driven escalation to the free Groq reasoner — but only as a LAST
    # resort. The local model gets to SEE the error and re-plan first; we escalate
    # only once strikes cross the threshold. Escalating on the first failure both
    # masked the local model's real ability and didn't help (the 70B failed the
    # same way when it inherited a blind-batch task).
    if strike_count >= ESCALATION_STRIKE_THRESHOLD:
        current_model = ESCALATION_MODEL

    # Detect the phase and pick the matching prompt.
    last_tool = None
    if last_msg is not None and getattr(last_msg, "tool_calls", None):
        last_tool = last_msg.tool_calls[0]["name"]

    # No pre-classification gate. Tools are ALWAYS available and the MODEL decides
    # whether to answer directly (conversation) or act (tools). This removes the
    # SIMPLE/COMPLEX router, whose misclassification silently stripped tools from
    # "remember I live in …" and let it claim success without doing anything.
    if last_tool == "forge_new_skill":
        # Forging a capability — use the specialised builder prompt.
        system_prompt = personality.get_skill_forge_prompt(cwd=cwd)
    elif last_msg is not None and last_msg.type == "human":
        # Turn 1: butler persona + architect discipline + grounded context. The model
        # replies directly to conversation, or discovers-and-acts if work is needed.
        def _cap(s, n):
            return (s[:n] + "…") if s and len(s) > n else (s or "")
        grounding = ""
        if state.get("active_tool_directives"):
            grounding += f"\n\n{_cap(state['active_tool_directives'], 1500)}"
        if state.get("os_context"):
            grounding += f"\n\n[LIVE SYSTEM CONTEXT]\n{_cap(state['os_context'], 1200)}"
        if state.get("rag_context"):
            grounding += f"\n\n[RETRIEVED KNOWLEDGE]\n{_cap(state['rag_context'], 2000)}"
        system_prompt = (
            f"{personality.kernel_text}"
            f"{grounding}\n\n"
            "[CONVERSATION OR ACTION — YOUR JUDGEMENT]\n"
            "If the request is casual conversation, a greeting, or something you can "
            "answer directly from the context above, simply reply as the butler — do "
            "NOT call a tool. If it needs discovery, a change to the system, looking "
            "something up, or remembering/recalling a fact, then act using tools, one "
            "step at a time.\n"
            "When the user TELLS you a lasting fact about themselves or someone they "
            "know, that is a request to REMEMBER it — store it with a tool, even "
            "though it is phrased as a passing remark. NEVER say you have noted, "
            "remembered, or recorded something unless you actually called the tool "
            "that stores it and saw it succeed.\n\n"
            "[HOW TO HANDLE MISSING INFORMATION]\n"
            "Before asking the user anything, resolve it yourself in this order:\n"
            "1. If the fact is already in the context above (system state, speaker, "
            "directives), USE it — never ask for something you were already told.\n"
            "2. If the fact is discoverable with a tool (inspect hardware, read the "
            "filesystem, search the web), CALL the tool to find it.\n"
            "3. Only if a fact is something ONLY the user could know (a name they "
            "want, content they intend, a preference not on record) — call the "
            "'ask_user' tool with ONE concise question, informed by what you "
            "already found. Do NOT write the question as a normal reply; use the "
            "tool, which pauses the task and waits for their answer so you can "
            "continue exactly where you left off. Do NOT use 'ask_user' to OFFER "
            "or ask PERMISSION ('shall I start it?') — say that as a normal reply.\n"
            "Prefer acting and discovering over asking.\n\n"
            "[MODIFYING EXISTING WORK — READ BEFORE YOU WRITE]\n"
            "When the request builds on something that already exists ('add a "
            "button', 'polish the UI', 'now add…', 'change it to…'), the current "
            "artifact ON DISK is the source of truth, NOT your memory of earlier "
            "turns. FIRST read the current file(s) with a tool, THEN make the "
            "smallest change that satisfies the request. NEVER regenerate a file "
            "from scratch or from recollection — you will silently erase work done "
            "in earlier steps. If you are unsure which file, discover it "
            "(list the working directory) before editing.\n\n"
            "[CONVERSATION HISTORY IS REFERENCE, NOT A TO-DO LIST]\n"
            "Earlier messages in this conversation are BACKGROUND to help you "
            "interpret the user's CURRENT message (resolving 'yes', 'that one', "
            "'open it'). Respond ONLY to the current message. Do NOT resume, retry, "
            "or complete an earlier request unless the current message explicitly "
            "asks you to. A task that already finished or failed is closed — leave "
            "it closed.\n\n"
            f"{personality.soul_text}"
        ) + personality._get_system_context(cwd)
    elif last_msg is not None and (getattr(last_msg, "tool_calls", None) or last_msg.type == "tool"):
        # Post-tool phase. Alfred may call another tool OR conclude to the user.
        # Butler voice (so any spoken conclusion is refined, no raw tool output)
        # + the single source of execution discipline (worker_system.md). The
        # rules used to live orphaned in worker_system.md AND duplicated inline;
        # now there is ONE source, loaded where execution actually happens.
        system_prompt = (
            personality.get_butler_prompt(cwd=cwd)
            + "\n\n" + personality.worker_text
        )
    else:
        # Final summary phase — butler voice.
        system_prompt = personality.get_butler_prompt(cwd=cwd)

    # Engine selection.
    if current_model == ESCALATION_MODEL:
        # Groq: free tier, tool-calling capable, ~70B reasoning for hard repairs.
        llm_engine = ChatGroq(model=ESCALATION_MODEL, temperature=0.1, max_tokens=1024)
    else:
        # No manual `stop` list. Under llama-server --jinja the Gemma 4 template
        # owns termination; a client-side stop can fire INSIDE a native tool token
        # (<|tool_call>…<tool_call|>) or thought channel and truncate the call,
        # which surfaces as a "malformed tool call". Let the server end the turn.
        llm_engine = _local_llm(temperature=0.1, max_tokens=1024)

    registry = ToolRegistry()
    # Tools are ALWAYS bound — the model, not a pre-classifier, decides whether to
    # use one. One source of truth: everything callable comes from the registry.
    brain_with_skills = llm_engine.bind_tools(registry.get_all_tools())

    # --- CONTEXT BUDGET GUARD ---
    # The checkpointed thread accumulates every past message, tool result, and
    # repair injection. Left unchecked it overflows the 16K window (the "who are
    # you? = 18852 tokens" bug). Keep only the most recent messages.
    # NOTE: we deliberately do NOT pin messages[0]. On a rolling multi-turn thread
    # that is the first human turn of the whole CONVERSATION — often a stale, done
    # task — and pinning it fed the "re-run the old request" bug. The user's
    # CURRENT request is the most recent human message, always inside the window.
    def _approx_tokens(msgs):
        return sum(len(str(getattr(m, "content", "")))
                   for m in msgs) // 4  # ~4 chars/token

    trimmed = list(messages)
    # Reserve ~4k for the system prompt + generation headroom against a 16k ctx.
    budget = 9000  # tokens of history we allow
    while len(trimmed) > 1 and _approx_tokens(trimmed) > budget:
        trimmed.pop(0)  # drop oldest first; recent turns (incl. current) survive
    # Front-trimming can sever an assistant tool_call from its tool result. A
    # leading orphan tool message (or a tool_call whose results were cut) is an
    # INVALID sequence that strict endpoints (Groq, on escalation) reject with 400 —
    # local llama-server tolerates it, which is why it only bit on the cloud path.
    # Drop any leading partial tool-exchange so the window starts on a clean turn.
    while len(trimmed) > 1 and (
        getattr(trimmed[0], "type", "") == "tool"
        or getattr(trimmed[0], "tool_calls", None)
    ):
        trimmed.pop(0)

    # Boundary repair. Front-popping can leave the window starting on an orphaned
    # tool result (a ToolMessage whose parent tool_call was trimmed) or a dangling
    # assistant tool_call. Strict backends (Groq on the escalation path) reject that
    # with a 400 even though local llama-server tolerates it. Drop any leading
    # tool-exchange fragment so the history starts on a clean boundary.
    while len(trimmed) > 1 and (
        getattr(trimmed[0], "type", "") == "tool"
        or getattr(trimmed[0], "tool_calls", None)
    ):
        trimmed.pop(0)

    invocation = [SystemMessage(content=system_prompt)] + trimmed
    sys_tokens = len(system_prompt) // 4
    hist_tokens = _approx_tokens(trimmed)
    print(f"[Alfred] Model={current_model} Loop={current_loops + 1} "
          f"Strikes={strike_count} HistMsgs={len(trimmed)}/{len(messages)} "
          f"~Tokens(sys={sys_tokens}, hist={hist_tokens}, total={sys_tokens + hist_tokens})")

    # --- FULL I/O LOGGING (set LOG_LLM_IO=0 in env to silence) ---
    if os.environ.get("LOG_LLM_IO", "1") == "1":
        print("┌── LLM REQUEST " + "─" * 45)
        print(f"│ SYSTEM PROMPT ({len(system_prompt)} chars):")
        print(system_prompt)
        print("│ MESSAGES:")
        for m in trimmed:
            role = getattr(m, "type", "?")
            body = str(getattr(m, "content", ""))
            print(f"│  [{role}] {body[:600]}{'…' if len(body) > 600 else ''}")
        print("└" + "─" * 60)

    try:
        response = await brain_with_skills.ainvoke(invocation)
        if os.environ.get("LOG_LLM_IO", "1") == "1":
            print("┌── LLM RESPONSE " + "─" * 44)
            print(f"│ content: {str(response.content)[:1000]}")
            if getattr(response, "tool_calls", None):
                print(f"│ tool_calls: {[tc['name'] for tc in response.tool_calls]}")
            print("└" + "─" * 60)
    except Exception as e:
        err = str(e)
        print(f"[Alfred] 💥 Engine error: {err}")
        # FATAL errors can never be fixed by retrying the same call — bail out of
        # the graph immediately instead of feeding the repair loop 20 times.
        fatal = any(s in err.lower() for s in (
            "exceed_context_size", "context size", "context length",
            "400", "401", "403", "invalid_api_key", "authentication",
        ))
        if fatal:
            msg = await narrate(
                "the request was too large for you to process before you could reason "
                "about it; explain that honestly and suggest starting it fresh or "
                "shortening it",
                "Sir, that request grew too large for me to handle — might we start "
                "it fresh, or pare it down?")
            return {
                "messages": [AIMessage(content=msg)],
                "loop_count": current_loops + 1,
                "active_model": current_model,
                "final_response": msg,
                "repair_mode_active": False,   # do NOT trigger repair
                "fatal_error": True,           # signals validation to end now
            }
        # Non-fatal (transient parse/format) — allow one repair pass.
        fb_text = await narrate(
            "you had a brief formatting hiccup and are reviewing it",
            "Sir, I had a brief hiccup with that — reviewing it now.")
        fb = AIMessage(content=fb_text)
        return {"messages": [fb], "loop_count": current_loops + 1,
                "active_model": current_model, "final_response": fb.content}

    final_text = response.content if not response.tool_calls else None
    return {
        "messages": [response],
        "loop_count": current_loops + 1,
        "active_model": current_model,
        "final_response": final_text,
    }


# =========================================================================
# TOOL EXECUTION  (keyword guardrail REMOVED — generalized per your principle)
# =========================================================================
@timed("graph.tool_execution_node")
async def tool_execution_node(state: OrionState) -> dict:
    last_message = state["messages"][-1]
    registry = ToolRegistry()
    # Same single source as binding — what the model may call and what actually runs
    # are identical, so a tool can never be callable-but-unexecutable. ask_user is
    # present but never reached here (routed to human_input_node upstream).
    tool_map = {t.name: t for t in registry.get_all_tools()}

    tool_results = []
    current_strikes = state.get("strike_count", 0)
    repair_active = False
    last_err = None
    halted = False

    # STOP-ON-ERROR. The model sometimes emits several interdependent commands in
    # one turn (install → cp → sed → bind). Running the whole batch blind is what
    # let a failed step cascade into a fake success. So: execute in order, and the
    # moment one fails, STOP — respond to the remaining calls with a "not executed"
    # note (keeps the tool/response pairing valid) and hand the failure back so the
    # next turn re-plans against what actually happened.
    for tool_call in last_message.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]

        if halted:
            tool_results.append(ToolMessage(
                content="[NOT EXECUTED] A previous command in this batch failed. "
                        "Re-plan from the error above; do not assume this ran.",
                tool_call_id=tool_call["id"]))
            continue

        try:
            func = tool_map[name]  # KeyError if the model invents a tool
            result = await func.ainvoke(args)
            result_str = str(result)
            # A single oversized tool result (e.g. a recursive `ls` of the project =
            # 18k tokens) can blow the context by itself and can't be trimmed away
            # (it's one message). Cap it here so no single result can flood context;
            # the model can re-query more specifically (head/tail/grep) if it needs more.
            if len(result_str) > _MAX_TOOL_OUTPUT_CHARS:
                result_str = (result_str[:_MAX_TOOL_OUTPUT_CHARS]
                              + f"\n…[output truncated: {len(result_str) - _MAX_TOOL_OUTPUT_CHARS}"
                              f" more chars. Re-run more specifically — head/tail/grep — if you need the rest.]")
            is_error = result_str.startswith(ERROR_PREFIXES)
            if is_error:
                current_strikes += 1
                repair_active = True
                last_err = result_str
                halted = True  # stop the rest of the batch
            _audit_tool(name, args, ok=not is_error, result_str=result_str)
        except Exception as e:
            result_str = f"[FATAL EXCEPTION] {name}: {str(e)}"
            current_strikes += 1
            repair_active = True
            last_err = result_str
            halted = True
            _audit_tool(name, args, ok=False, result_str=result_str)

        tool_results.append(ToolMessage(content=result_str, tool_call_id=tool_call["id"]))

    return {
        "messages": tool_results,
        "strike_count": current_strikes,
        "loop_count": state.get("loop_count", 0) + 1,
        "repair_mode_active": repair_active,
        "last_error_trace": last_err,
    }


# =========================================================================
# VALIDATION  (now runs ONLY on Alfred's final text, never on tool output)
# =========================================================================
@timed("graph.validation_node")
async def semantic_validation_node(state: OrionState) -> dict:
    # A fatal engine error or an exhausted run already produced an honest terminal
    # message — do not re-judge it.
    if state.get("fatal_error") or state.get("exhausted"):
        return {"repair_mode_active": False}

    last_msg = state["messages"][-1]
    request = state.get("user_request", "")
    content = getattr(last_msg, "content", "") or ""

    # Structural check — an empty/near-empty final answer is a non-answer. Send it
    # back to repair; the global loop cap (exhaustion) bounds any runaway and ends
    # with an HONEST message, so we no longer auto-accept just because strikes are
    # high (that was the bug: the more a task failed, the LESS it got checked).
    if len(content.strip()) < 5:
        return {"repair_mode_active": True,
                "strike_count": state.get("strike_count", 0) + 1,
                "last_error_trace": "Final response lacked substantive content."}

    # Gather the tool evidence produced during THIS turn (ToolMessages since the
    # last human message) so we can judge the answer against what actually
    # happened — not against its own confident prose.
    evidence = []
    for m in reversed(state["messages"][:-1]):
        if getattr(m, "type", "") == "human":
            break
        if getattr(m, "type", "") == "tool":
            evidence.append(str(getattr(m, "content", ""))[:400])
    evidence.reverse()
    evidence_block = "\n".join(f"- {e}" for e in evidence) if evidence else "(no tools were run)"
    had_error = state.get("last_error_trace") is not None

    llm = _local_llm()
    # Evidence-grounded HONESTY gate. Lenient about style/completeness, strict
    # about truthfulness: an answer that claims success the evidence does not
    # support is INVALID — that is the Law of Honesty, enforced.
    prompt = (
        "You check a butler's REPLY against the TOOL EVIDENCE from THIS turn.\n"
        "Reply ONLY 'VALID' or 'INVALID'.\n"
        "INVALID if the REPLY says it has done something — stored, noted, updated, "
        "recorded, installed, configured, sent, verified — but the EVIDENCE shows no "
        "tool that actually did it. A claim of action without matching evidence is a "
        "falsehood no matter how plausible or well-mannered it sounds. This is the "
        "single most important thing you catch.\n"
        "INVALID if the EVIDENCE shows an unresolved error but the REPLY presents "
        "success. INVALID if the REPLY is empty, a refusal, or about a different "
        "topic than the REQUEST.\n"
        "VALID if the REPLY merely STATES what it knows — facts about people, the "
        "system, or an answer to a question — without claiming to have performed an "
        "action. Answering from knowledge needs no evidence.\n"
        "VALID if the REPLY honestly reports partial success or failure.\n"
        f"REQUEST: {request}\n"
        f"TOOL EVIDENCE:\n{evidence_block}\n"
        f"REPLY: {content}"
    )
    verdict = await llm.ainvoke([HumanMessage(content=prompt)])

    if "INVALID" in verdict.content.upper():
        note = ("Your reply claimed an outcome the tool evidence does not support. "
                "Either VERIFY the end state with a tool, or report honestly what "
                "actually succeeded and what did not — do not claim success you "
                "cannot show.") if had_error else \
               "Response did not address the request; reconcile it with the evidence."
        return {"repair_mode_active": True,
                "strike_count": state.get("strike_count", 0) + 1,
                "last_error_trace": note}

    # Valid → commit to episodic memory so Orion actually remembers.
    try:
        _get_bridge().record_action(
            source="alfred",
            description=f"Q: {request} | A: {content[:300]}",
            meta={"type": "episodic"},
        )
    except Exception as e:
        print(f"[Validation] ⚠️ memory write failed: {e}")

    return {"repair_mode_active": False}


# =========================================================================
# REPAIR
# =========================================================================
@timed("graph.repair_node")
async def repair_node(state: OrionState) -> dict:
    error_trace = state.get("last_error_trace", "Unknown error context.")
    cwd = state.get("current_working_dir", None)
    print(f"[Repair] Analyzing: {str(error_trace)[:60]}...")

    project_map = ProjectMapper().get_live_map_summary()
    # Hard cap: this map was ballooning the prompt past the context limit each loop.
    if project_map and len(project_map) > 2000:
        project_map = project_map[:2000] + "\n...[map truncated]"
    qa_directive = personality.get_qa_prompt(error_trace=error_trace, cwd=cwd)
    qa_directive += f"\n\n[HOST FILE ARCHITECTURE MAP]\n{project_map}"
    qa_directive += (
        "\n\n[REPAIR DIRECTIVE]: Your previous action failed. Do NOT use "
        "'forge_new_skill' to work around it. Select an existing tool suited to "
        "the actual task, or escalate with 'ask_expert' if the constraint is real."
    )

    repair_msg = ToolMessage(content=qa_directive,
                             tool_call_id="repair_node_override",
                             name="system_repair")
    # Do NOT reset strike_count here — that was defeating the loop cap and letting
    # validation → repair → alfred cycle forever. Strikes must accumulate so the
    # validator can give up after a bounded number of attempts.
    return {"messages": [repair_msg], "repair_mode_active": False}


# =========================================================================
# HUMAN INPUT  (the interrupt owner — suspends the graph, resumes with the answer)
# =========================================================================
@timed("graph.human_input_node")
async def human_input_node(state: OrionState) -> dict:
    """Alfred asked the user something via the ask_user tool. Suspend the thread
    here and wait. On resume (brain passes Command(resume=<answer>)), interrupt()
    returns the answer; we hand it back as the tool result for EVERY pending
    tool_call so the message list stays OpenAI-valid, then return to Alfred with
    all task state (loop_count, blueprint, strikes) intact — no restart."""
    last = state["messages"][-1]
    tool_calls = list(getattr(last, "tool_calls", None) or [])

    ask_tc = next((tc for tc in tool_calls if tc["name"] == "ask_user"), None)
    question = (ask_tc["args"].get("question") if ask_tc else None) \
        or "Could you clarify what you'd like, Sir?"

    # Suspend. This is the ONLY line that pauses; on resume it returns the answer.
    answer = interrupt({"question": question})

    # Respond to the ask_user call with the answer, and to any sibling tool_calls
    # with a deferral note (they were not executed while we waited).
    out = []
    for tc in tool_calls:
        if tc is ask_tc:
            out.append(ToolMessage(content=str(answer), tool_call_id=tc["id"]))
        else:
            out.append(ToolMessage(
                content="[Deferred: was awaiting the user's answer to a question.]",
                tool_call_id=tc["id"]))
    if not out:  # defensive: ask_user with no resolvable id
        out.append(ToolMessage(content=str(answer), tool_call_id="ask_user"))

    return {"messages": out, "awaiting_clarification": False}


# =========================================================================
# CONSENT GATE  (Option 3: privileged/root actions require a spoken 'yes')
# =========================================================================
# Affirmative/negative word sets for reading a yes/no consent reply. This is
# parsing a direct yes/no answer — NOT branching task behaviour on request
# keywords — and it DEFAULTS TO DENY: a root action proceeds only on a clear yes.
_AFFIRMATIVE = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "proceed",
                "confirm", "confirmed", "affirmative", "aye", "go", "do", "please"}
_NEGATIVE = {"no", "dont", "stop", "cancel", "wait", "nope", "never", "not", "hold"}


@timed("graph.confirm_action_node")
async def confirm_action_node(state: OrionState) -> dict:
    """A privileged (root-daemon) action is pending. Suspend and ask the user to
    confirm before it runs. On yes: mark the task consented and let the pending
    tool calls execute. On no: decline every pending call and hand back to Alfred.
    The gate is on the CAPABILITY (root access), not on parsing the command."""
    last = state["messages"][-1]
    tool_calls = list(getattr(last, "tool_calls", None) or [])
    priv = [tc for tc in tool_calls if tc["name"] == "execute_python_payload"]
    preview = str(priv[0]["args"].get("code", ""))[:200] if priv else "a system action"

    answer = interrupt({
        "question": (f"This requires root access to your system, Sir:\n{preview}\n"
                     "Shall I proceed?"),
        "requires_confirmation": True,
    })

    tokens = set(re.findall(r"[a-z']+", str(answer).lower()))
    approved = bool(tokens & _AFFIRMATIVE) and not (tokens & _NEGATIVE)

    if approved:
        # Consent granted for THIS task; pending calls now execute, and later
        # privileged calls in the same task won't re-prompt (reset per request).
        return {"privilege_confirmed": True}

    # Declined — answer every pending call so the message list stays valid.
    declined = [ToolMessage(
        content="[DECLINED BY USER] You did not approve this privileged action. Do "
                "not retry it; find a non-privileged path or report that you cannot "
                "proceed without permission.",
        tool_call_id=tc["id"]) for tc in tool_calls]
    return {"messages": declined, "privilege_confirmed": False}


def route_after_confirm(state: OrionState) -> Literal["tool_execution_node", "alfred_node"]:
    # Approved → run the pending (now-consented) tool calls. Declined → Alfred
    # already has the decline messages and reports/replans.
    return "tool_execution_node" if state.get("privilege_confirmed") else "alfred_node"


# =========================================================================
# ROUTING
# =========================================================================
def route_after_alfred(state: OrionState) -> Literal["tool_execution_node", "human_input_node", "confirm_action_node", "semantic_validation_node", "__end__"]:
    if state.get("exhausted"):
        return END
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        # Control-plane: asking the user suspends the graph (human-in-the-loop);
        # it is NOT a capability execution. This is infra routing on a system tool,
        # not use-case keyword branching.
        if any(tc["name"] == "ask_user" for tc in tool_calls):
            return "human_input_node"
        # Consent gate: any root-daemon (privileged) call needs a spoken yes first,
        # once per task. Gated on the capability, not on parsing the command.
        if not state.get("privilege_confirmed") and any(
                tc["name"] == "execute_python_payload" for tc in tool_calls):
            return "confirm_action_node"
        return "tool_execution_node"
    # Final answer → ALWAYS validate. Do NOT skip validation when no tools ran:
    # that is precisely when a false claim is most likely ("I've made a note of
    # that, Sir" with nothing saved). The validator compares the claim against the
    # turn's tool evidence — and "no tools were run" IS evidence, the damning kind.
    # A genuinely conversational reply claims nothing and passes trivially.
    return "semantic_validation_node"


def route_after_validation(state: OrionState) -> str:
    return "repair_node" if state.get("repair_mode_active") else END


# =========================================================================
# BUILD
# =========================================================================
def build_orion_graph(checkpointer=None):
    """Compile the Orion graph.

    checkpointer MUST be supplied for the rolling conversation thread and for
    ask_user interrupts (suspend/resume) to work. It is a parameter — not set as
    an attribute after the fact — so compilation wires it in properly.
    """
    b = StateGraph(OrionState)
    b.add_node("planner_node", planner_node)
    b.add_node("alfred_node", alfred_node)
    b.add_node("tool_execution_node", tool_execution_node)
    b.add_node("human_input_node", human_input_node)
    b.add_node("confirm_action_node", confirm_action_node)
    b.add_node("semantic_validation_node", semantic_validation_node)
    b.add_node("repair_node", repair_node)

    # Every request enters at grounding (fast, LLM-free) then Alfred, who decides
    # reply-vs-act himself with tools always bound. No SIMPLE/COMPLEX router — that
    # gate misclassified memory writes and stripped their tools.
    b.add_edge(START, "planner_node")
    b.add_edge("planner_node", "alfred_node")

    # Alfred is the hub: he discovers (tools), asks (human_input), seeks consent
    # for privileged actions (confirm_action), or answers. Tools, the user's
    # answer, and approved actions all return to Alfred; his final text is validated.
    b.add_conditional_edges("alfred_node", route_after_alfred)
    b.add_edge("tool_execution_node", "alfred_node")
    b.add_edge("human_input_node", "alfred_node")
    b.add_conditional_edges("confirm_action_node", route_after_confirm)
    b.add_conditional_edges("semantic_validation_node", route_after_validation)
    b.add_edge("repair_node", "alfred_node")

    return b.compile(checkpointer=checkpointer)