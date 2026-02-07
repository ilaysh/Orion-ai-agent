# orion_core/brain/rag_chroma.py
"""
Offline high-performance RAG using ChromaDB + MiniLM embeddings.
Persists vectors locally for instant recall.
"""
import os
import json
import chromadb
from chromadb.utils import embedding_functions
# SentenceTransformer is not imported here - only used via embedding_functions
from system.telemetry.telemetry import timed
_client = None
_collection = None


def _get_collection(path: str = "data/rag"):

    global _client, _collection
    if _client is None:
        os.makedirs(path, exist_ok=True)
        _client = chromadb.PersistentClient(path=path)

    if _collection is None:
        # Use a small sentence-transformer for speed
        embed = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        _collection = _client.get_or_create_collection(
            name="orion_rag",
            embedding_function=embed
        )
    return _collection


class RAGMemory:
    """
    Simple project-wide RAG memory using ChromaDB.
    """

    def __init__(self, path: str = "data/rag"):
        self.path = path
        self.collection = _get_collection(path)

    # ------------------ Ingestion ------------------
    def ingest_file(self, file_path: str, kind: str = "code"):
        """Add a single file to the RAG index."""
        if not os.path.isfile(file_path):
            return
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            print(f"[RAG] ⚠️ Failed to read {file_path}: {e}")
            return

        doc_id = os.path.abspath(file_path)
        meta = {"path": doc_id, "type": kind}
        self.collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[meta],
        )
        print(f"[RAG] 📥 Ingested {file_path}")

    def ingest_project(self, root: str = "."):
        """
        Walks a project directory and indexes .py/.md/.txt files.
        """
        exts = {".py", ".md", ".txt"}
        count = 0
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in exts:
                    self.ingest_file(os.path.join(dirpath, fn), kind="code")
                    count += 1
        print(f"[RAG] ✅ Indexed {count} files from project.")

    # ------------------ Retrieval ------------------
    @timed("RAG retrieve")
    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        if not self.collection.count():
            return []
        res = self.collection.query(query_texts=[query], n_results=top_k)
        return res["documents"][0]

    @timed("RAG query")
    def query(self, query: str, top_k: int = 4) -> list[str]:
        """Alias for retrieve, kept for clarity + telemetry."""
        return self.retrieve(query, top_k)

    @timed("RAG query")
    def ingest_sessions(self, sessions: list[dict]):
        """
        Add chat history to RAG index.
        Each session entry is expected to be:
        {"role": "...", "content": "..."}
        """
        for i, entry in enumerate(sessions):
            text = entry.get("content") or ""
            doc_id = f"session_{i}"
            meta = {"type": "session", "role": entry.get("role")}
            try:
                self.collection.add(
                    ids=[doc_id],
                    documents=[text],
                    metadatas=[meta],
                )
            except Exception as e:
                print(f"[RAG] ⚠️ Failed to ingest session entry {i}: {e}")

    def add_document(self, path: str, text: str, metadata=None):
        """Add a single document to RAG memory."""
        try:
            abs_path = os.path.abspath(path)
            md = metadata or {}
            md["path"] = abs_path

            self.collection.add(
                ids=[abs_path],
                documents=[text],
                metadatas=[md],
            )
        except Exception as e:
            print(f"[RAGMemory] ⚠️ add_document failed: {e}")

# ------------------ Helper for Brain ------------------


def retrieve_context(query: str, top_k: int = 4) -> str:
    rag = RAGMemory()
    results = rag.retrieve(query, top_k)
    return "\n".join(results)
