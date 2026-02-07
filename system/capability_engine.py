# orion_core/system/capability_engine.py
"""
CapabilityEngine — builds a structured snapshot of Orion's capabilities.

- Uses SystemIntrospector to check core/optional modules, skills, planner state.
- Optionally runs ProjectMapper to map the codebase and index into RAG.
- Saves a JSON snapshot to logs/capabilities.json
- Indexes a natural-language summary into RAG for later recall.
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from system.telemetry.telemetry import timed
from orion_core.brain.rag_chroma import RAGMemory
from system.project_mapper import ProjectMapper
from system.system_introspector import SystemIntrospector


CAPABILITIES_SNAPSHOT_PATH = "logs/capabilities.json"
CAPABILITIES_DOC_ID = "orion_capabilities"


@dataclass
class CapabilitySnapshot:
    timestamp: float
    core_components_found: List[str]
    core_components_missing: List[str]
    optional_modules_found: List[str]
    optional_modules_missing: List[str]
    import_errors: List[str]
    unregistered_skills: List[str]
    unused_skill_files: List[str]
    planner_tools_missing: List[str]
    project_map_path: Optional[str] = None


class CapabilityEngine:
    """
    Central place to build and store Orion's capability map.
    """

    def __init__(
        self,
        root_dir: str = "orion_core",
        rag_dir: str = "logs/rag_memory",
        snapshot_path: str = CAPABILITIES_SNAPSHOT_PATH,
    ):
        self.root_dir = root_dir
        self.snapshot_path = snapshot_path
        self.rag = RAGMemory(path=rag_dir)
        self.introspector = SystemIntrospector()
        self.project_mapper = ProjectMapper(root_dir=root_dir, rag_dir=rag_dir)

    # ------------------------------------------------------------------
    @timed("capabilities_refresh")
    def refresh(self) -> Dict[str, Any]:
        """
        Run a full capability scan:

        - Introspector: core/optional modules, skills, planner tools, import issues
        - ProjectMapper: map project files and index structure into RAG
        - Save a structured snapshot to JSON
        - Store a natural-language summary in RAG under a fixed doc id
        """
        # 1) Deep system inspection
        report = self.introspector.inspect()

        snapshot = CapabilitySnapshot(
            timestamp=time.time(),
            core_components_found=report.get("core_components_found", []),
            core_components_missing=report.get("core_components_missing", []),
            optional_modules_found=report.get("optional_modules_found", []),
            optional_modules_missing=report.get(
                "optional_modules_missing", []),
            import_errors=report.get("import_errors", []),
            unregistered_skills=report.get("unregistered_skills", []),
            unused_skill_files=report.get("unused_skill_files", []),
            planner_tools_missing=report.get("planner_tools_missing", []),
            project_map_path=None,
        )

        # 2) Project map + RAG index (re-use existing ProjectMapper logic)
        file_map = self.project_mapper.scan_project()
        self.project_mapper.index_to_rag()
        snapshot.project_map_path = self.project_mapper.map_path

        # 3) Persist snapshot to JSON
        data = asdict(snapshot)
        os.makedirs(os.path.dirname(self.snapshot_path), exist_ok=True)
        with open(self.snapshot_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # 4) Build a natural summary + index into RAG
        summary = self._build_summary(data)
        self.rag.add_document(
            doc_id=CAPABILITIES_DOC_ID,
            text=summary,
            metadata={
                "type": "capabilities",
                "source": "capability_engine",
                "snapshot_path": self.snapshot_path,
            },
        )

        return data

    # ------------------------------------------------------------------
    def _build_summary(self, data: Dict[str, Any]) -> str:
        """
        Create a compact natural-language summary that Thinker/RAG can use later.
        """
        core_ok = len(data.get("core_components_found", []))
        core_missing = len(data.get("core_components_missing", []))
        opt_ok = len(data.get("optional_modules_found", []))
        opt_missing = len(data.get("optional_modules_missing", []))
        import_errs = len(data.get("import_errors", []))
        unreg = len(data.get("unregistered_skills", []))
        unused = len(data.get("unused_skill_files", []))
        planner_missing = len(data.get("planner_tools_missing", []))
        project_map_path = data.get("project_map_path") or "unknown"

        return (
            "Orion capabilities snapshot. "
            f"Core components present: {core_ok}, missing: {core_missing}. "
            f"Optional modules present: {opt_ok}, missing: {opt_missing}. "
            f"Import errors: {import_errs}. "
            f"Unregistered skill modules: {unreg}, unused skill files: {unused}. "
            f"Planner tools missing: {planner_missing}. "
            f"Project map stored at: {project_map_path}."
        )

    # ------------------------------------------------------------------
    def load_last_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Load previously saved capabilities snapshot from JSON, if exists.
        """
        if not os.path.exists(self.snapshot_path):
            return None
        try:
            with open(self.snapshot_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
