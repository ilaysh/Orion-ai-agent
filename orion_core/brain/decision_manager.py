# orion_core/brain/decision_manager.py

from dataclasses import dataclass, field
from typing import Dict, Any
import json

from system.telemetry.telemetry import timed
from orion_core.brain.llm_local import generate_mistral
from orion_core.brain.meta_bridge import MetaBridge


@dataclass
class DecisionResult:
    action: str = "normal_chat"  # default: just answer naturally
    params: Dict[str, Any] = field(default_factory=dict)


class DecisionManager:
    """
    Orion's Decision Manager.
    Receives user text + context and returns the best single 'intent' JSON.

    This version supports:
    - dynamic capabilities (skill/tools/actions) injected from VectorDB
    - clean natural language → action reasoning
    - safe fallback to normal_chat if uncertain
    """

    def __init__(self, personality, memory=None):
        self.personality = personality
        self.memory = memory  # vectorDB / RAG memory manager
        self.bridge = MetaBridge()

    # ----------------------------------------------------------
    # Main Decision Function
    # ----------------------------------------------------------
    @timed("decision_manager decide")
    def decide(
        self,
        user_text: str,
        context: str = "",
        has_active_plan: bool = False,
    ) -> DecisionResult:

        text = (user_text or "").strip()
        if not text:
            return DecisionResult()

        # ========================================
        # Load dynamic capability map from DB (if exists)
        # ========================================
        capability_block = ""
        if self.memory:
            cap = self.memory.load_capabilities()
            if cap:
                try:
                    cap_json = json.dumps(cap, ensure_ascii=False)
                    capability_block = (
                        "\n[CAPABILITIES]\n" + cap_json + "\n[/CAPABILITIES]\n"
                    )
                except Exception:
                    capability_block = ""

        # ========================================
        # SYSTEM PROMPT (FULL AGENT-ACTION LOGIC)
        # ========================================
        system_prompt = (
            "You are Orion's internal DecisionManager.\n"
            "Your job:\n"
            "- Read the user's message and recent conversation.\n"
            "- Use Orion's CAPABILITIES (if provided) to choose what action to trigger.\n"
            "- Produce ONLY a single JSON of the form:\n"
            "{\n"
            "  \"intent\": {\n"
            "    \"action\": \"<string>\",\n"
            "    \"params\": { ... }\n"
            "  }\n"
            "}\n"
            "\n"
            "The CAPABILITIES block (if present) describes exactly what Orion can do.\n"
            "You MUST NOT invent actions or skills that are not listed.\n"
            "If no CAPABILITIES are available, you may choose from actions referenced in the conversation.\n"
            "\n"
            "Rules:\n"
            "1) If no specific tool/skill/action is required, use:\n"
            "     \"action\": \"normal_chat\"\n"
            "\n"
            "2) If the user explicitly asks for an ability that exists in CAPABILITIES,\n"
            "   choose that action and fill params.\n"
            "\n"
            "3) Planner actions (if present):\n"
            "   - plan_task\n"
            "   - execute_plan\n"
            "   - modify_active_plan\n"
            "   - query_progress\n"
            "   - pause_plan / resume_plan\n"
            "\n"
            "4) Skill actions (if present):\n"
            "   Example: time.now, dashboard.open, image.generate, file.write, system.inspect, etc.\n"
            "\n"
            "5) Never include ANY natural language in the JSON.\n"
            "   Only \"action\" and \"params\".\n"
            "\n"
            "6) If uncertain → normal_chat.\n"
            "\n"
            "Return ONLY the JSON. Do not explain.\n"
            + capability_block  # Insert capabilities dynamically here
        )

        # ========================================
        # USER PROMPT
        # ========================================
        plan_hint = "yes" if has_active_plan else "no"
        user_prompt = (
            f"User message:\n{text}\n\n"
            f"Recent context:\n{context}\n\n"
            f"Is there an active plan running? {plan_hint}\n\n"
            "Identify the best 'intent'. Return only JSON."
        )

        # Call the model
        raw = generate_mistral(
            prompt=user_prompt,
            system_prompt=system_prompt,
            context=""
        )

        # ========================================
        # JSON Extraction (Strict)
        # ========================================
        def _extract_intent_json(raw_text: str) -> str:
            n = len(raw_text)
            for start in range(n):
                if raw_text[start] != "{":
                    continue
                depth = 0
                for end in range(start, n):
                    ch = raw_text[end]
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            candidate = raw_text[start:end+1]
                            try:
                                data = json.loads(candidate)
                                if isinstance(data, dict) and "intent" in data:
                                    return candidate
                            except Exception:
                                pass
                            break
            raise ValueError("No valid JSON intent found in LLM output")

        try:
            json_str = _extract_intent_json(raw)
            natural_part = raw.replace(json_str, "").strip()
            data = json.loads(json_str)

            intent = data.get("intent") or {}
            action = str(intent.get("action") or "normal_chat").strip()
            params = intent.get("params") or {}

            # For Thinker (natural language layer)
            params["natural_response"] = natural_part

            result = DecisionResult(action=action, params=params)

        except Exception as e:
            print("DecisionManager: Failed to parse intent JSON:", raw)
            print("Reason:", e)

            result = DecisionResult(
                action="normal_chat",
                params={"natural_response": raw}
            )

        # ----------------------------------------
        # Telemetry + MetaBridge logging
        # ----------------------------------------
        meta = {}
        try:
            meta["params_json"] = json.dumps(
                params or {}, ensure_ascii=False)[:512]
        except Exception:
            meta["params_json"] = "<encode error>"

        self.bridge.record_action(
            "DecisionManager",
            f"Action={result.action}",
            meta=meta,
        )

        return result
