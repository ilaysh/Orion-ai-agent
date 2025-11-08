# orion_core/brain/rag_chroma.py
"""
Offline high-performance RAG using ChromaDB + MiniLM embeddings.
Persists vectors locally for instant recall.
"""
import os
import json
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# One global persistent client reused across Orion
_client = None
_collection = None


def _get_collection(persist_dir="logs/rag_memory", collection="orion_memory"):
    global _client, _collection
    if _collection:
        return _collection
    os.makedirs(persist_dir, exist_ok=True)
    _client = chromadb.PersistentClient(path=persist_dir)

    # --- Fast local embedding model (offline, cached) ---
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    _collection = _client.get_or_create_collection(
        name=collection,
        embedding_function=emb_fn,
    )
    return _collection


class RAGMemory:
    """Local persistent semantic memory."""

    def __init__(self, persist_dir="logs/rag_memory", collection="orion_memory"):
        self.collection = _get_collection(persist_dir, collection)

    # ------------------ Ingest sessions ------------------
    def ingest_sessions(self, session_dir="logs/sessions"):
        files = [f for f in os.listdir(session_dir) if f.endswith(".json")]
        for f in files:
            path = os.path.join(session_dir, f)
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                for i, turn in enumerate(data):
                    uid = f"{f}_{i}"
                    text = f"{turn['role']}: {turn['text']}"
                    meta = {
                        "session": f,
                        "role": turn["role"],
                        "ts": turn["timestamp"],
                    }
                    self.collection.add(
                        documents=[text], metadatas=[meta], ids=[uid])
            except Exception as e:
                print(f"[RAG] ⚠️ Failed to ingest {f}: {e}")
        print(f"[RAG] ✅ Indexed {self.collection.count()} items total.")

    # ------------------ Retrieval ------------------
    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        if not self.collection.count():
            return []
        res = self.collection.query(query_texts=[query], n_results=top_k)
        return res["documents"][0]


# ------------------ Helper for Brain ------------------
def retrieve_context(query: str, top_k: int = 4) -> str:
    rag = RAGMemory()
    results = rag.retrieve(query, top_k)
    return "\n".join(results)
