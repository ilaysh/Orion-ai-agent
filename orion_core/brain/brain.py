# orion_core/brain/brain.py
from system.telemetry.telemetry import timed
import os
import asyncio
from typing import Optional, Callable, Any
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from langchain_core.messages import HumanMessage
import uuid

# Project Imports
from orion_core.brain.orion_graph import build_orion_graph, narrate
from orion_core.brain.session_manager import SessionManager
from orion_core.brain.meta_bridge import MetaBridge
from orion_core.base_component import BaseComponent, ComponentState

class Brain(BaseComponent):
    name: str = "brain"

    def __init__(self, bubble_thought: Optional[Callable[[str], Any]] = None) -> None:
        super().__init__()
        self.bubble = bubble_thought
        self.bridge = MetaBridge()
        self.session = SessionManager(self.bridge)
        
        self.orion_graph = None
        self.checkpointer: Optional[AsyncSqliteSaver] = None
        self._checkpointer_cm = None
        # Fresh thread per boot. The checkpointer is SHORT-TERM conversation memory
        # (so "yes"/"that one" resolve within a session); it must NOT be an eternal
        # thread that replays weeks of stale turns on every reboot (the 193-msg bug).
        # Durable memory is the MetaBridge/RAG episodic store, not this checkpoint.
        self.current_thread_id = f"session_{uuid.uuid4().hex[:8]}"
        self._last_active_model = "gemma-4-12b"

    async def init(self) -> str:
        print("[Brain] 🧠 Awakening cognitive infrastructure...")
        os.makedirs("orion_core/memory", exist_ok=True)
        self._checkpointer_cm = AsyncSqliteSaver.from_conn_string("orion_core/memory/checkpoints.sqlite")
        self.checkpointer = await self._checkpointer_cm.__aenter__()
        
        # Compile with the checkpointer wired in (required for the rolling thread
        # AND for ask_user interrupts to suspend/resume correctly).
        self.orion_graph = build_orion_graph(self.checkpointer)

        self.state = ComponentState.READY
        return "Supervisor Kernel online."

    async def _has_pending_interrupt(self, config: dict) -> bool:
        """True if the thread is paused on an ask_user interrupt — i.e. the next
        user utterance is an ANSWER to resume, not a new request. If we can't read
        the state for any reason, fall back to treating input as a new request."""
        try:
            snap = await self.orion_graph.aget_state(config)
            if not snap or not snap.next:
                return False
            for task in getattr(snap, "tasks", ()) or ():
                if getattr(task, "interrupts", None):
                    return True
            return False
        except Exception as e:
            print(f"[Brain] ⚠️ interrupt-state check failed, treating as new: {e}")
            return False

    async def _handle_vram_swapping(self, active_model: str):
        if active_model != self._last_active_model:
            print(f"[Brain] 💾 Memory Event: Swapping engine from {self._last_active_model} -> {active_model}")
            # Hook to your model load/unload process here
            self._last_active_model = active_model

   
    @timed("brain.think")
    async def think(self, user_text: str) -> str:
        if self.state != ComponentState.READY:
            return "[System Fault: Brain is not initialized.]"

        print("\n" + "="*60)
        print(f"[BRAIN KERNEL] 📥 Inbound User Request: '{user_text}'")
        print("="*60)

        # ROLLING CONVERSATION THREAD (short-term memory): one thread per boot
        # carries recent turns so follow-ups like "yes"/"that one" resolve. Two
        # distinct entry paths share it:
        #   • RESUME  — the graph is paused on an ask_user interrupt and this input
        #     is the user's ANSWER. We resume the SAME task in place (no reset, no
        #     re-plan) via Command(resume=...). This is what makes
        #     "…shall I use X?" → "yes" continue the original task.
        #   • NEW REQUEST — no pending interrupt. We append the new human message
        #     and HARD-RESET all transient execution state, so a prior turn's
        #     strikes/exhaustion/blueprint/error never bleed in (the "retry the
        #     failed ink task" bug). Durable conversation stays in `messages`.
        config = {"configurable": {"thread_id": self.current_thread_id}}

        is_resume = await self._has_pending_interrupt(config)

        if is_resume:
            print("[BRAIN KERNEL] ↩️ Resuming a paused task with the user's answer.")
            stream_input = Command(resume=user_text)
        else:
            fresh_env = self.session.build_initial_state(user_text)
            stream_input = {
                # operator.add → appends; rolling context already in the thread.
                "messages": [HumanMessage(content=user_text)],
                "user_request": user_text,
                # Volatile grounding, refreshed each turn.
                "os_context": fresh_env["os_context"],
                "current_working_dir": fresh_env["current_working_dir"],
                "active_tool_directives": fresh_env["active_tool_directives"],
                # FULL transient reset — every ephemeral execution field, not a
                # partial subset. `messages` and speaker identity are preserved.
                "intent_id": "pending",
                "blueprint": None,
                "loop_count": 0,
                "strike_count": 0,
                "repair_mode_active": False,
                "exhausted": False,
                "fatal_error": False,
                "awaiting_clarification": False,
                "privilege_confirmed": False,
                "last_error_trace": None,
                "manifest": None,
                "workspace_data": {},
                "spoken_update": None,
                "completed_steps": [],
                "final_response": None,
            }

        final_response = ""
        loop_idx = 0

        # --- STATUS MILESTONES ---
        # Natural-language updates derived from graph phase changes (NOT from the
        # worker prompt, so the lean loop + ask_expert stay uncluttered). We only
        # speak when the *phase* changes, so a 25-loop task yields ~4 updates, not 25.
        last_phase = None
        first_tool_seen = False
        spoke = {"any": False}  # did anything reach the user's ears this turn?

        async def _status(text: str):
            spoke["any"] = True
            if self.bubble:
                await self.bubble(text)

        # LIVENESS WATCHDOG (not a debounce). Speech is proportionate to task length:
        #   • fast task  → nothing extra; the final answer is the only utterance, and
        #     it also confirms the request was understood.
        #   • slow task  → after a short silence, ONE reassurance that we're on it
        #     (only if nothing else — a repair note, a consent prompt — already spoke).
        #   • long task  → occasional "still working" so it never feels dead.
        # It generates a line only when silence has actually occurred, so nothing is
        # produced-then-cancelled.
        async def _watchdog():
            try:
                await asyncio.sleep(4.0)
                if not spoke["any"]:
                    await _status(await narrate(
                        "reassure the user you have understood and are working on their "
                        "request, and it will take a moment",
                        "I'm looking into that for you now, Sir."))
                while True:
                    await asyncio.sleep(30.0)
                    await _status(await narrate(
                        "reassure the user you are still working on the task",
                        "Still on it, Sir — this is taking a little longer."))
            except asyncio.CancelledError:
                pass

        watchdog = asyncio.create_task(_watchdog())

        try:
            print("[BRAIN KERNEL] 🚀 Invoking LangGraph execution pipeline...")
            async_stream = self.orion_graph.astream(stream_input, config, stream_mode="updates")
            
            async for compile_delta in async_stream:
                loop_idx += 1
                print(f"\n⚡ [GRAPH LOOP #{loop_idx}] Processing state mutation delta...")

                # ask_user fired: the graph is now suspended. Surface the question
                # as the response; the NEXT think() call resumes with the answer.
                if "__interrupt__" in compile_delta:
                    intr = compile_delta["__interrupt__"]
                    payload = intr[0].value if intr else {}
                    question = payload.get("question") if isinstance(payload, dict) else str(payload)
                    final_response = question or "Could you clarify what you'd like, Sir?"
                    print(f"  ❓ Awaiting user input: {final_response[:120]}")
                    continue

                for node_name, node_outputs in compile_delta.items():
                    print(f"  📍 Active Node: '{node_name}'")
                    print(f"  📦 Node Outputs Payload Keys: {list(node_outputs.keys())}")
                    
                    if "active_model" in node_outputs:
                        print(f"  ⚙️ Node reported target inference engine: {node_outputs['active_model']}")
                        await self._handle_vram_swapping(node_outputs["active_model"])

                    # ----- PHASE-BASED STATUS (speaks only on phase change) -----
                    # Every line is generated fresh (natural, varied); the canned
                    # text is a last-resort fallback only if generation fails.
                    # NOTE: no bubble on the grounding step — it fires before Orion
                    # knows what he's doing, so it could only say something vague, and
                    # the first-tool bubble (which knows the actual action) followed
                    # ~3s later saying the same thing twice. One informative
                    # acknowledgement beats two vague ones; we are spoken aloud.
                    if node_name == "repair_node" and last_phase != "repair":
                        last_phase = "repair"
                        await _status(await narrate(
                            "your first approach didn't work, so you're trying a "
                            "different route",
                            "That approach didn't hold, Sir — trying another route."))

                    elif node_name == "alfred_node":
                        if "messages" in node_outputs and node_outputs["messages"]:
                            last_ai_msg = node_outputs["messages"][-1]

                            if getattr(last_ai_msg, "tool_calls", None):
                                tool_name = last_ai_msg.tool_calls[0]["name"]
                                print(f"  🛠️ Node requested tool: {tool_name}")
                                # A secure input dialog is about to open and BLOCK — the
                                # user must be told to look for the window; the phrasing
                                # is generated but MUST carry those key points.
                                if tool_name == "provision_secret" and last_phase != "awaiting_secret":
                                    first_tool_seen = True
                                    last_phase = "awaiting_secret"
                                    await _status(await narrate(
                                        "a secure input window has opened; the user "
                                        "should type the value there, you will not see "
                                        "it yourself, and they should look behind the "
                                        "current window if they don't see it",
                                        "I've opened a secure window, Sir — please type "
                                        "it there; I shan't see it myself. Check behind "
                                        "this window if you don't see it."))
                                elif not first_tool_seen:
                                    # First tool call. No automatic bubble here — a
                                    # fast task should speak only its answer, and the
                                    # liveness watchdog covers the case where the task
                                    # turns out to be slow. This is what removes the
                                    # "I'll note it" / "I noted it" double-narration.
                                    first_tool_seen = True
                                elif tool_name == "ask_expert" and last_phase != "consulting":
                                    last_phase = "consulting"
                                    await _status(await narrate(
                                        "this is intricate, so you're consulting a "
                                        "specialist for expert help",
                                        "This is intricate, Sir — consulting a specialist."))

                            elif getattr(last_ai_msg, "content", None):
                                print(f"  ✍️ Content ({len(last_ai_msg.content)} chars): {last_ai_msg.content[:120]}")
                                final_response = last_ai_msg.content

                    # Explicit final_response override in state payload
                    if node_outputs.get("final_response"):
                        print(f"  🎯 final_response: {node_outputs['final_response'][:120]}")
                        final_response = node_outputs["final_response"]

            # =========================================================================
            # FIXED: POST-STREAM RECOVERY GATEWAY
            # =========================================================================
            print("\n[BRAIN KERNEL] 🔎 Stream complete. Fetching final state snapshot...")
            post_run_state = await self.orion_graph.aget_state(config)
            state_values = post_run_state.values or {}
            
            # 1. Capture dynamic final_response updates if present
            if not final_response:
                final_response = state_values.get("final_response", "")
            
            # 2. Fallback to extracting textual content from the final message block
            if not final_response and "messages" in state_values and state_values["messages"]:
                last_msg = state_values["messages"][-1]
                if hasattr(last_msg, "content") and last_msg.content:
                    final_response = last_msg.content

            print("\n" + "="*60)
            if not final_response:
                print("[BRAIN KERNEL] ⚠️ Diagnostic Alert: Execution path finished with an empty response object.")
                final_response = await narrate(
                    "you finished but have nothing to report; apologize briefly and "
                    "invite the user to rephrase",
                    "Sir, I completed the cycle but have nothing to report — might you rephrase?")
            else:
                print(f"[BRAIN KERNEL] 🗣️ Final Response Captured: '{final_response}'")
            print("="*60 + "\n")

            return final_response

        except Exception as e:
            print(f"\n💥 [CRITICAL RUNTIME EXCEPTION]: {str(e)}\n")
            # Even here, try to speak naturally; fall back only if that also fails.
            return await narrate(
                "something went wrong on your end and you couldn't complete the "
                "request; apologize briefly, without technical jargon",
                "My apologies, Sir — something went awry on my end just then.")
        finally:
            # The turn is over: stop the liveness watchdog so no stale "still working"
            # line trails a finished answer.
            watchdog.cancel()

    async def close(self):
        if self._checkpointer_cm:
            print("[Brain] 💾 Committing cached state updates to SQLite disk...")
            await self._checkpointer_cm.__aexit__(None, None, None)