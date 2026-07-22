# orion_core/brain/router.py
from dataclasses import dataclass
from typing import Dict, List, Any
import json
from orion_core.brain.llm.chat_engine import get_chat_engine

@dataclass
class ExecutionProfile:
    intent_id: str
    target_role: str
    allowed_tools: List[str]
    system_override_prompt: str
    max_loop_turns: int

class SemanticRouter:
    """
    Classifies user intent using a fast micro-pass with the main LLM.
    Ensures strict RBAC and context isolation.
    """
    def __init__(self, executive_layer: Any) -> None:
        self.executive = executive_layer
        self.engine = get_chat_engine()
        self.profiles: Dict[str, ExecutionProfile] = {}
        self._initialize_profiles()

    def _initialize_profiles(self) -> None:
        # Profile 1: The Researcher
        self.profiles["research_analyst"] = ExecutionProfile(
            intent_id="research_analyst",
            target_role="System Owner",
            allowed_tools=["mcp_hub", "execute_bash", "response"], 
            system_override_prompt="You are in RESEARCH mode. Gather info, cross-reference data, and verify hardware (if needed via bash).",
            max_loop_turns=5
        )

        # Profile 2: The Engineer
        self.profiles["system_engineer"] = ExecutionProfile(
            intent_id="system_engineer",
            target_role="System Owner",
            allowed_tools=["fs", "execute_bash", "execute_python_payload", "upsert_skill", "execute_skill", "knowledge", "response"],
            system_override_prompt="You are in SYSTEM ENGINEERING mode. Write code, interact with the OS, debug, and self-correct.",
            max_loop_turns=12
        )

        # Profile 3: The Butler
        self.profiles["butler"] = ExecutionProfile(
            intent_id="butler",
            target_role="Family",
            allowed_tools=["response", "memory", "schedule"],
            system_override_prompt="You are Orion. Respond deferentially to casual conversation or memory retrieval tasks.",
            max_loop_turns=2
        )

    async def route(self, user_text: str) -> dict:
        import re
        # --- THE API CONTRACT UPDATE ---
        # We force the LLM to explicitly declare the state transition as a boolean.
        prompt = (
            "You are the Orion Pre-Flight Architect. Analyze the user's request.\n"
            "1. Identify hidden dependencies (e.g., do we need to discover local hardware/state before acting?).\n"
            "2. Select the required execution profile: 'research_analyst', 'system_engineer', or 'butler'.\n"
            "3. Draft a strict 1-2 sentence execution blueprint for the loop to follow.\n\n"
            f"USER REQUEST: {user_text}\n\n"
            "Output strictly valid JSON:\n"
            "{\n"
            '  "reasoning": "...",\n'
            '  "requires_local_recon": true_or_false,\n'
            '  "intent": "...",\n'
            '  "blueprint": "..."\n'
            "}"
        )
        
        system = "You are the Pre-Flight routing engine. You must output JSON."
        
        try:
            raw_response = await self.engine.generate_chat(prompt=prompt, system_prompt=system, thinking=True)
            
            start = raw_response.find('{')
            end = raw_response.rfind('}')
            
            if start != -1 and end != -1 and start <= end:
                json_str = raw_response[start:end+1]
                data = json.loads(json_str)
            else:
                raise ValueError("No valid JSON structure found in raw output.")
                
            intent = data.get("intent", "system_engineer")
            blueprint = data.get("blueprint", "Execute the user's request.")
            requires_recon = bool(data.get("requires_local_recon", False)) # Explicit Boolean
            
            if intent not in self.profiles:
                intent = "system_engineer"
                
        except Exception as e:
            print(f"[Architect] ⚠️ Deep reasoning failed ({e}). Defaulting to System Engineer.")
            intent, blueprint, requires_recon = "system_engineer", "Execute safely.", False

        print(f"[Architect] 🎯 Profile: {intent} | Recon Required: {requires_recon} | Blueprint: {blueprint}")
        
        return {
            "profile": self.profiles[intent],
            "blueprint": blueprint,
            "requires_recon": requires_recon
        }