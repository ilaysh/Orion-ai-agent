# orion_core/brain/thinker.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable

from orion_core.brain.personality import Personality
from system.telemetry.telemetry import timed
from orion_core.brain.session_manager import SessionManager
from orion_core.brain.meta_bridge import MetaBridge
from orion_core.brain.llm.chat_engine import ChatEngine, get_chat_engine


@dataclass
class IntentDecision:
    kind: str  # "small_chat", "needs_clarification", "task"
    confidence: float

@dataclass
class ThinkerResult:
    classification: str
    immediate_text: Optional[str]
    anticipation_question: Optional[str]


class Thinker:
    """
    The State Manager & Gatekeeper.
    Swaps Persona based on System Integrity.
    """
    
    def __init__(
        self,
        *,
        session: SessionManager,
        meta_bridge: MetaBridge,
        bubble_thought: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.session: SessionManager = session
        self.meta_bridge: MetaBridge = meta_bridge
        self.bubble_thought: Optional[Callable[[str], Any]] = bubble_thought
        self.engine: ChatEngine = get_chat_engine()
        self.personality = Personality()

    async def preload(self) -> None:
        await self.engine.load()
        await self.engine.generate_chat(
            prompt="hi", system_prompt="JSON", context="", max_new_tokens=1, temperature=0.0
        )

    @timed("thinker.intent_pass")
    async def intent_pass(self, user_text: str) -> IntentDecision:
        """
        Step 1: The Triage.
        """
        # Note: We rely on the Bootstrap Prompt in 'reflect' to handle 
        # conversation flow, so we keep the router standard here.
        router_system = self.personality.get_thinker_prompt()

        # router_system = (
        #     "CLASSIFY INPUT:\n"
        #     "1. 'small_chat': Greetings, identity, emotions, casual talk.\n"
        #     "2. 'needs_clarification': Request is vague.\n"
        #     "3. 'task': Actionable commands, Registration, Fixes.\n"
        #     "OUTPUT JSON: { \"classification\": \"...\" }"
        # )
        
        user_prompt = f"INPUT: {user_text}\nJSON:"
        
        raw = await self.engine.generate_chat(
            prompt=user_prompt, system_prompt=router_system, context="", max_new_tokens=128, temperature=0.0
        )
        
        data = self._try_parse_json(raw) or {}
        kind = data.get("classification", "small_chat")
        
        if kind not in ("task", "small_chat", "needs_clarification"):
            kind = "small_chat"
        
        return IntentDecision(kind=kind, confidence=1.0)

    @timed("thinker.reflect")
    async def reflect(
        self,
        *,
        user_text: str,
        mode: str = "lite", 
    ) -> ThinkerResult:
        """
        Step 2: The Response.
        Swaps 'Soul' based on System Integrity Checks.
        """
        # 1. DIAGNOSE SYSTEM HEALTH
        alerts = []#self.session.check_system_integrity()
        
        # 2. SELECT PERSONA
        if alerts:
            # STATE: MAINTENANCE / BOOTSTRAP
            sys_msg = self.personality.get_bootstrap_prompt()
            
            # Inject alerts into context so LLM knows what is broken
            alert_str = "\n".join(alerts)
            world_state = self.session.get_world_state() # Includes [SYSTEM DIAGNOSTICS] block
            
            # Specialized Context for Maintenance
            context = (
                f"{world_state}\n\n"
                f"*** ACTIVE ALERTS ***\n{alert_str}\n"
                "*********************\n\n"
                f"USER INPUT:\n{user_text}\n\n"
                "INSTRUCTION: Reply in valid JSON. Prioritize fixing alerts."
            )
            
        else:
            # STATE: PRIME / SERVICE
            sys_msg = self.personality.get_thinker_prompt()
            
            world_state = self.session.get_world_state()
            history = self.session.format_history(limit=10) 
            
            context = (
                f"{world_state}\n\n"
                f"CHAT HISTORY:\n{history}\n\n"
                f"USER INPUT:\n{user_text}\n\n"
                "--------------------------------------------------\n"
                "INSTRUCTION: Reply in valid JSON following the schema."
            )

        # 3. GENERATE
        raw = await self.engine.generate_chat(
            prompt=context,
            system_prompt=sys_msg, 
            context="",
            max_new_tokens=256,
            temperature=0.7, # Allow natural phrasing
        )

        # 4. PARSE & CLEAN
        data = self._try_parse_json(raw)
        
        if not data:
            data = await self._repair_json(
                system_prompt="Output valid JSON only.", 
                raw_output=raw, 
                max_new_tokens=128
            )

        if not isinstance(data, dict):
            return ThinkerResult("small_chat", raw[:100], None)

        cls = data.get("classification", "small_chat")
        text = data.get("immediate_text")
        
        # Clean Prompt Leaks (if model repeats "System:")
        if text:
            text = text.replace("System:", "").replace("Orion:", "").strip()

        # Failsafe for silent turns
        if cls == "needs_clarification" and not text:
            text = "Could you be more specific, Sir?"
            
        return ThinkerResult(
            classification=cls,
            immediate_text=text,
            anticipation_question=data.get("anticipation_question"),
        )

    def _try_parse_json(self, raw: Optional[str]) -> Optional[Dict[str, Any]]:
        """Robust JSON extraction."""
        if not raw: return None
        s = raw.strip()
        
        pattern = r"```(?:json|JSON)?\s*(\{.*?\})\s*```"
        matches = list(re.finditer(pattern, s, re.DOTALL))
        if matches:
            for match in reversed(matches):
                try: return json.loads(match.group(1))
                except: continue
        
        end = s.rfind("}")
        if end != -1:
            balance = 0
            for i in range(end, -1, -1):
                if s[i] == "}": balance += 1
                elif s[i] == "{":
                    balance -= 1
                    if balance == 0:
                        try: return json.loads(s[i:end+1])
                        except: pass
                        break
        
        if s.startswith("{") and s.endswith("}"):
             try: return json.loads(s)
             except: pass
            
        return None

    async def _repair_json(self, *, system_prompt: str, raw_output: str, max_new_tokens: int) -> Optional[Dict[str, Any]]:
        truncated = raw_output[-500:] 
        repair_prompt = f"Fix this broken JSON:\n...{truncated}"
        repaired = await self.engine.generate_chat(prompt=repair_prompt, system_prompt=system_prompt, context="", max_new_tokens=max_new_tokens, temperature=0.0)
        return self._try_parse_json(repaired)