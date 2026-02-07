# orion_core/voice_layer.py

from typing import Optional, Dict

from orion_core.brain.llm_local import generate_mistral


class OrionVoice:
    """
    LLM-based transformation layer.
    Converts internal system output into one short, natural spoken sentence.
    """

    def speak(self, raw: str, context: Optional[Dict] = None) -> str:
        if not raw:
            return ""

        prompt = self._build_prompt(raw, context)
        return generate_mistral(prompt).strip()

    def _build_prompt(self, raw: str, context: Optional[Dict] = None) -> str:
        base = f"""
        You are ORION, a personal AI assistant.

        Answer in ONE short factual sentence and STOP immediately.

        Statement:
            {raw}
        """

        if context:
            base += f"\nContext: {context}\n"

        return base.strip()


if __name__ == "__main__":
    v = OrionVoice()

    print(v.speak("task completed successfully"))
    print(v.speak("error: permission denied"))
    print(v.speak("waiting for input"))
    print(v.speak("music paused", {"target": "music", "action": "pause"}))
