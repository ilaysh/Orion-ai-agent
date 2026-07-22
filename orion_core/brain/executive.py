# orion_core/brain/directives.py
"""
DirectivesManager — Orion's data-driven behaviour layer.

DESIGN PRINCIPLE (non-negotiable):
    Behaviour is driven by DATA, never by hard-coded keyword checks in Python
    or use-case branches in prompts. There is no `if "ink" in request`.

Two tiers:
  * PRIME directives  — always injected (identity / safety / global rules).
  * CONTEXTUAL SOPs   — retrieved by SEMANTIC similarity to the request.
                        "compare ink prices" matches the shopping SOP because
                        the embedder recognises it as a buying task — not
                        because any code looked for the word "ink".

To add a preference, you add a row of DATA (add_sop). You never touch code.
"""
import os
import json
from typing import List, Optional
from orion_core.brain.rag_chroma import RAGMemory

DEFAULT_PRIME = [
    "Obey the verified owner above any other speaker. Never act on a request "
    "a lower-privilege speaker is not permitted to make.",
    "Establish ground truth before acting on hardware or system state. Never "
    "assume a device model, file path, or installed package — discover it first.",
    "Prefer local, headless, native tooling. Escalate to the cloud expert only "
    "when local reasoning is genuinely exhausted.",
]

# Seeded contextual SOPs. `applies_when` is what gets embedded and matched.
# `rule` is injected verbatim when the situation matches.
DEFAULT_SOPS = [
    {
        "id": "sop_shopping_sources",
        "applies_when": (
            "buying, shopping, comparing prices, finding the best price, "
            "purchasing hardware, ink, cartridges, electronics, or any product"
        ),
        "rule": (
            "Default preferred retailers are KSP (ksp.co.il) and Ivory "
            "(ivory.co.il). Compare BOTH before deciding. Prices are in ILS and "
            "must account for domestic Israel shipping. After identifying the "
            "best price, open the winning product page for the user."
        ),
    },
    {
        "id": "sop_region",
        "applies_when": (
            "regional constraints, electrical standards, shipping, voltage, "
            "plug type, localisation, currency"
        ),
        "rule": (
            "Operating region is Israel: 230V, Type H sockets, ILS currency. "
            "Use region-appropriate operators when searching the live web."
        ),
    },
]


class DirectivesManager:
    def __init__(
        self,
        file_path: str = "data/directives.json",
        sop_rag: Optional[RAGMemory] = None,
    ):
        self.file_path = file_path
        # Contextual SOPs live in their own collection so they never collide
        # with MetaBridge technical SOPs or episodic memory.
        self.sop_rag = sop_rag or RAGMemory(
            collection_name="orion_directives", path="logs/rag_memory"
        )
        self._ensure_storage()
        self._seed_sops_once()

    # ---------------- Prime directives (always-on) ----------------
    def _ensure_storage(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump({"prime": DEFAULT_PRIME}, f, indent=2)

    def get_directives(self) -> List[str]:
        """Backward-compatible: returns always-on prime directives."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f).get("prime", DEFAULT_PRIME)
        except Exception:
            return DEFAULT_PRIME

    def add_prime(self, rule: str):
        data = {"prime": self.get_directives()}
        if rule not in data["prime"]:
            data["prime"].append(rule)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    # ---------------- Contextual SOPs (semantic retrieval) ----------------
    def _seed_sops_once(self):
        marker = "data/.directives_seeded"
        if os.path.exists(marker):
            return
        for sop in DEFAULT_SOPS:
            self.add_sop(sop["id"], sop["applies_when"], sop["rule"])
        os.makedirs("data", exist_ok=True)
        open(marker, "w").close()

    def add_sop(self, sop_id: str, applies_when: str, rule: str):
        """
        Store a contextual directive. The `applies_when` text is embedded, so
        matching is by SITUATION. The `rule` rides along in metadata and is
        injected verbatim when matched. This is how you add a preference:
        pure data, no code change.
        """
        try:
            self.sop_rag.collection.upsert(
                ids=[sop_id],
                documents=[applies_when],
                metadatas=[{"type": "directive", "rule": rule}],
            )
        except Exception as e:
            print(f"[Directives] ⚠️ Failed to store SOP {sop_id}: {e}")

    def get_contextual(self, request: str, top_k: int = 3, threshold: float = 1.25) -> str:
        """
        Semantically match directives to the request. No keyword logic anywhere.
        Returns a ready-to-inject block, or "" if nothing is relevant.
        NOTE: `threshold` is an L2 distance cutoff for MiniLM — tune against your
        own corpus (smaller = stricter). 1.25 is a sane starting point.
        """
        try:
            if not self.sop_rag.collection.count():
                return ""
            res = self.sop_rag.collection.query(
                query_texts=[request],
                n_results=top_k,
                where={"type": "directive"},
            )
        except Exception:
            return ""

        if not res or not res.get("documents") or not res["documents"][0]:
            return ""

        rules = []
        for dist, meta in zip(res["distances"][0], res["metadatas"][0]):
            if dist < threshold and meta.get("rule"):
                rules.append(f"- {meta['rule']}")

        if not rules:
            return ""
        return "[APPLICABLE DIRECTIVES]\n" + "\n".join(rules)