from orion_core.brain.intent_manager import IntentManager
from orion_core.brain.session_manager import SessionManager
from orion_core.brain.llm_local import generate as llm_generate
from orion_core.brain.rag_chroma import retrieve_context
from orion_core.skills.skills import Skills
from orion_core.brain.personality import Personality


class Brain:
    def __init__(self):
        self.intents = IntentManager()
        self.session = SessionManager()
        self.skills = Skills()
        self.personality = Personality()

        # Index RAG once
        try:
            from orion_core.brain.rag_chroma import RAGMemory
            rag = RAGMemory()
            rag.ingest_sessions()
            print("[Brain] 🧠 RAG memory indexed.")
        except Exception as e:
            print(f"[Brain] ⚠️ RAG init failed: {e}")

    async def think(self, text: str) -> str:
        """High-level reasoning pipeline."""
        # Log user input
        self.session.add_turn("user", text)

        # 1️⃣ Intents → Skills
        intent = self.intents.parse(text)
        if intent:
            result = self.skills.handle(intent.name, **intent.slots)
            if result:
                self.session.add_turn("orion", result)
                return result

        # 2️⃣ Retrieve context from session + RAG
        context = self.session.get_context(6)
        rag_context = retrieve_context(text)
        combined_context = f"{context}\n{rag_context}".strip()

        # 3️⃣ Construct system prompt with personality
        sys_prompt = self.personality.system_prompt()
        prompt = f"Context:\n{combined_context}\n\nUser: {text}\n{self.personality.name}:"

        # 4️⃣ Generate reasoning output
        prompt_full = f"{sys_prompt.strip()}\n\nUser: {prompt.strip()}\nAssistant:"
        reply = llm_generate(prompt_full)
        print(f"[Brain] 🧠 Prompt sent to LLM ({len(prompt_full)} chars)")
        if not reply or not reply.strip():
            reply = self.personality.fallback_reply(text)

        # Log assistant reply
        self.session.add_turn("orion", reply)
        return reply

    def remember(self, role: str, text: str):
        self.session.add_turn(role, text)
