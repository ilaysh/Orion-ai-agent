# orion_core/brain/meta_bridge.py
"""
Orion Meta-Bridge:
Connects the Coder and Thinker halves of Orion's mind
so that both share awareness, context and knowledge.
"""

import os
import json
from datetime import datetime
from orion_core.brain.rag_chroma import RAGMemory
from typing import Dict, Any

ROOT_ID = "root_user_profile"


class MetaBridge:
    def __init__(self, rag_dir: str = "logs/rag_memory"):
        self.log_dir = "logs/meta_bridge"
        os.makedirs(self.log_dir, exist_ok=True)
        self.rag = RAGMemory(path=rag_dir)

    # ------------------------------------------------------------------
    def record_action(self, source: str, description: str, meta: dict = None):
        """
        Log an action or insight from either Coder or Thinker.
        This simultaneously adds it to RAG memory for later recall.
        """
        meta = meta or {}
        entry = {
            "source": source,
            "description": description.strip(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "meta": meta,
        }
        # Save locally
        path = os.path.join(self.log_dir, f"{datetime.now():%Y%m%d}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Also push to RAG
        text = f"{source}: {description}"
        self.rag.collection.add(
            documents=[text],
            metadatas=[{"source": source, **meta}],
            ids=[f"{source}_{datetime.now().timestamp()}"],
        )
        print(
            f"[MetaBridge] 🧩 Recorded action from {source}: {description[:60]}")

    # ------------------------------------------------------------------
    def recall_recent(self, n: int = 5) -> list[str]:
        """
        Fetch recent context from both halves of Orion.
        """
        path = os.path.join(self.log_dir, f"{datetime.now():%Y%m%d}.jsonl")
        if not os.path.exists(path):
            return []
        lines = open(path, encoding="utf-8").read().splitlines()[-n:]
        return [json.loads(l)["description"] for l in lines]

    # ------------------------------------------------------------------
    def query_memory(self, query: str, top_k: int = 4) -> list[str]:
        """
        Query shared memory (RAG) for references to a concept or action.
        """
        return self.rag.retrieve(query, top_k=top_k)

    def get_entity(self, entity_id: str) -> Dict[str, Any]:
        """Loads a profile or entity by ID from RAG."""
        try:
            result = self.rag.collection.get(ids=[entity_id])
            if result and result['documents']:
                return json.loads(result['documents'][0])
        except Exception:
            pass
        return {"id": entity_id, "type": "generic", "name": "Unknown", "attributes": {}, "relationships": []}

    def update_entity(self, entity_id: str, data: Dict[str, Any]):
        """Persists an entity (User, Device, etc.) to the DB."""
        data["id"] = entity_id
        self.rag.collection.upsert(
            ids=[entity_id],
            documents=[json.dumps(data, ensure_ascii=False)],
            metadatas=[
                {"type": "entity", "updated": datetime.now().isoformat()}]
        )

    def load_root_profile(self) -> Dict[str, Any]:
        return self.get_entity(ROOT_ID)
