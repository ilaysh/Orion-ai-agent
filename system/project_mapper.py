# orion_core/brain/project_mapper.py
"""
PROJECT MAPPER — Orion's structural awareness.
Scans the entire Orion project, analyzes Python files, builds a dependency map,
and indexes this structure into the RAG for semantic recall.
"""

import os
import ast
import json
import time
from datetime import datetime
from system.message_bus import global_bus
from orion_core.brain.rag_chroma import RAGMemory


class ProjectMapper:
    def __init__(self, root_dir="orion_core", rag_dir="logs/rag_memory"):
        self.root_dir = root_dir
        self.map_path = "logs/project_map.json"
        self.rag = RAGMemory(path=rag_dir)
        self.file_map = {}

    # -------------------------------------------------------------
    def analyze_file(self, path: str):
        """Analyze a Python file — extract functions, classes, and imports."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                code = f.read()
            tree = ast.parse(code)

            funcs = [n.name for n in ast.walk(
                tree) if isinstance(n, ast.FunctionDef)]
            classes = [n.name for n in ast.walk(
                tree) if isinstance(n, ast.ClassDef)]

            imports = []
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    imports += [a.name for a in n.names]
                elif isinstance(n, ast.ImportFrom) and n.module:
                    imports.append(n.module)

            doc = ast.get_docstring(tree) or ""
            return {
                "functions": funcs,
                "classes": classes,
                "imports": list(set(imports)),
                "doc": doc.strip().replace("\n", " ")[:300],
            }
        except Exception as e:
            return {"error": str(e)}

    # -------------------------------------------------------------
    def scan_project(self):
        """Recursively scan project files and build map."""
        print(f"[ProjectMapper] 🧭 Scanning {self.root_dir}...")
        self.file_map = {}
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if not file.endswith(".py") or file.startswith("__"):
                    continue
                path = os.path.join(root, file)
                rel = os.path.relpath(path, self.root_dir)
                info = self.analyze_file(path)
                self.file_map[rel] = info

        with open(self.map_path, "w", encoding="utf-8") as f:
            json.dump(self.file_map, f, indent=2, ensure_ascii=False)

        print(f"[ProjectMapper] 🗺️ Map saved → {self.map_path}")
        return self.file_map

    def get_live_map_summary(self) -> str:
        """Returns a string summary of the project structure for the System Prompt."""
        if not self.file_map:
            self.scan_project()
        
        summary = ["### 🗺️ LIVE SYSTEM MAP (Self-Awareness)"]
        
        # Group by folder for readability
        tree = {}
        for path, data in self.file_map.items():
            folder = os.path.dirname(path)
            if folder not in tree: tree[folder] = []
            tree[folder].append(os.path.basename(path))

        for folder, files in tree.items():
            summary.append(f"- **{folder or 'root'}/**: {', '.join(files)}")
            
        return "\n".join(summary)
    # -------------------------------------------------------------
    def index_to_rag(self):
        """Index project structure into RAG for contextual recall."""
        print("[ProjectMapper] 🧠 Indexing project map into RAG...")
        for rel_path, data in self.file_map.items():
            summary = (
                f"File: {rel_path}. "
                f"Defines functions: {data.get('functions')}, "
                f"classes: {data.get('classes')}, "
                f"imports: {data.get('imports')}. "
                f"Doc: {data.get('doc')}"
            )
            self.rag.add_document(
                rel_path,
                summary,
                metadata={"type": "project_map", "file": rel_path},
            )
        print("[ProjectMapper] ✅ Indexed project map to RAG.")

    # -------------------------------------------------------------
    async def run(self):
        """Full mapping process — scan, index, and notify."""
        start = time.time()
        file_map = self.scan_project()
        self.index_to_rag()

        await global_bus.publish("event", {
            "type": "project_mapped",
            "summary": f"Scanned {len(file_map)} files in {time.time()-start:.1f}s",
        })
        print(
            f"[ProjectMapper] 🔁 Project mapping completed ({len(file_map)} files).")

    # -------------------------------------------------------------
    def query_map(self, keyword: str):
        """Search for files related to a keyword."""
        result = []
        for rel, info in self.file_map.items():
            if keyword.lower() in rel.lower() or keyword.lower() in str(info).lower():
                result.append(rel)
        return result

        # -------------------------------------------------------------

    def map_signatures(self):
        """
        Extract function signatures (name + parameter list) from all mapped files.
        Returns: [{"name": "...", "params": {"arg": "type"}}]
        Minimal version: all params typed as 'Any'.
        """
        signatures = []

        # Ensure self.file_map is populated
        if not self.file_map:
            self.scan_project()

        for rel_path, info in self.file_map.items():
            functions = info.get("functions", [])

            # Re-read file to get parameter lists
            full_path = os.path.join(self.root_dir, rel_path)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    code = f.read()
                tree = ast.parse(code)
            except:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    name = node.name
                    if name not in functions:
                        continue

                    params = {}
                    for arg in node.args.args:
                        arg_name = arg.arg
                        params[arg_name] = "Any"  # minimal type info

                    signatures.append({
                        "name": name,
                        "params": params,
                        "file": rel_path
                    })

        return signatures
