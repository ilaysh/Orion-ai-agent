# orion_core/brain/brain.py
import asyncio
import json
import re
import time
from typing import Optional

from orion_core.base_component import BaseComponent, ComponentState
from orion_core.brain.cortex import Cortex
from orion_core.brain.executive import Executive
from orion_core.brain.session_manager import SessionManager
from orion_core.brain.personality import Personality

class Brain(BaseComponent):
    name: str = "brain"

    def __init__(self, bubble_thought=None) -> None:
        super().__init__()
        self.bubble = bubble_thought
        self.cortex = Cortex(bubble_thought=bubble_thought) 
        self.session = SessionManager()
        self.personality = Personality()
        self.executive = Executive(self.cortex, self.session)
    
    async def init(self) -> None:
        self.state = ComponentState.READY
        await self.cortex.ensure_chat_mode()
        self.personality.load_cache()

    async def think(self, user_text: str) -> str:
        if not user_text.strip(): return ""
        await self.cortex.ensure_chat_mode()
        
        # 1. Gather Context
        active_skills = self.executive.get_loaded_skill_names()
        system_prompt = self.personality.get_system_prompt(active_skills)
        world_state = self.session.get_world_state()
        history = self.session.format_history(limit=6)
        
        full_context = f"{world_state}\n\nCHAT HISTORY:\n{history}\n\nUSER: {user_text}"
        print(f"[Brain] Full Context: {full_context}")
        print(f"[Brain] System Prompt: {system_prompt}")
        print(f"[Brain] User Text: {user_text}")
        try:
            final_response = await self._execute_react_loop(full_context, system_prompt, user_text)
            
            print(f"[Brain] Final Response: {final_response}")
            self.session.add_turn("user", user_text)
            self.session.add_turn("assistant", final_response)
            
            return final_response
            
        except Exception as e:
            print(f"[Brain] 💥 Critical Logic Failure: {e}")
            # FINAL FAILSAFE (Only if LLM itself crashes)
            return "Sir, my cognitive systems have encountered a critical error."

    async def _execute_react_loop(self, current_input, system_prompt, original_user_text):
        turn_history = ""
        max_turns = 5
        
        for i in range(max_turns):
            # A. THINK
            response = await self.cortex.generate_chat(
                prompt=current_input,
                system_prompt=system_prompt,
                max_new_tokens=1024,
                temperature=0.1
            )
            
            # B. PARSE
            tool_call = self._extract_json(response)
            thought_text = self._remove_json_string(response)

            # C. BUBBLE
            if self.bubble and thought_text:
                clean_thought = thought_text.replace("Thought:", "").strip()
                if len(clean_thought) > 3:
                    await self.bubble(clean_thought)

            # D. ACT
            if tool_call and "tool" in tool_call:
                tool_name = tool_call["tool"]
                args = tool_call.get("args", {})
                
                # Executive returns RAW DATA (e.g. "STDOUT: ...")
                result = await self.executive.execute(tool_name, args)
                
                # Feed RAW DATA back to Brain context
                turn_history += f"\n\nASSISTANT: {response}\nSYSTEM: {result}\n"
                current_input = f"{current_input}{turn_history}"
                
                continue
            
            # E. RESPOND (The Natural Layer)
            else:
                final_text = thought_text.strip()
                
                # DYNAMIC FALLBACK: If LLM did the work but forgot to speak
                if not final_text:
                    final_text = await self.cortex.generate_chat(
                        prompt=f"User Request: {original_user_text}\nStatus: The tool executed successfully.\nTask: Confirm completion naturally.",
                        system_prompt="You are a refined British Butler. Be concise.",
                        max_new_tokens=64
                    )

                # VOICE FILTER: Final polish
                final_text = await self._enforce_butler_voice(final_text, original_user_text)
                return final_text

        # LOOP EXHAUSTION FALLBACK (Dynamic)
        return await self.cortex.generate_chat(
            prompt="I have tried 5 times to solve this but failed. Apologize to the Master.",
            system_prompt="You are a refined British Butler.",
            max_new_tokens=64
        )

    async def _enforce_butler_voice(self, text: str, user_query: str) -> str:
        """
        The Voice Filter.
        Catches: Underscores, Paths (/), Brackets {}, 'Success', 'Error'.
        """
        is_robotic = any(x in text for x in ["_", "{", "}", "success:", "error:", "/", "\\"])
        
        if is_robotic:
            rephrase_prompt = (
                f"Original Query: {user_query}\n"
                f"Technical Result: {text}\n\n"
                f"TASK: Rewrite the 'Technical Result' as a refined British Butler. "
                f"Do NOT mention code, paths, or error codes. Speak only to the outcome."
            )
            return await self.cortex.generate_chat(
                prompt=rephrase_prompt, 
                system_prompt="You are a Voice Filter. Output ONLY spoken text.",
                max_new_tokens=128
            )
        return text

    def _extract_json(self, text: str) -> Optional[dict]:
        try:
            if "```json" in text:
                clean = text.split("```json")[1].split("```")[0]
                return json.loads(clean)
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                return json.loads(match.group(0))
            return None
        except: return None

    def _remove_json_string(self, text: str) -> str:
        text = re.sub(r"```json.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"\{[\s\S]*\}", "", text)
        return text.strip()