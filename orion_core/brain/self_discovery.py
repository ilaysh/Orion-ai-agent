# orion_core/brain/self_discovery.py
"""
SelfDiscovery — Orion's introspective awareness module.
Scans its own directories, analyzes new or modified files,
and publishes events to the MessageBus when changes occur.
"""

import os
import time
import ast
import hashlib
import asyncio
from system.message_bus import global_bus
from system.telemetry.telemetry import timed


class SelfDiscovery:
    def __init__(self, brain=None, base_dirs=None, scan_interval=20):
        self.brain = brain
        self.base_dirs = base_dirs or ["orion_core/skills", "orion_core/brain"]
        self.scan_interval = scan_interval
        self.file_hashes = {}

    # ------------------------------------------------------------------
    def _hash_file(self, path):
        """Return short hash of file contents."""
        try:
            with open(path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return None

    def _analyze_python(self, path):
        """Analyze a Python file: functions, imports, docstrings."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                code = f.read()
            tree = ast.parse(code)
            funcs = [n.name for n in ast.walk(
                tree) if isinstance(n, ast.FunctionDef)]
            imports = []
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    imports += [a.name for a in n.names]
                elif isinstance(n, ast.ImportFrom) and n.module:
                    imports.append(n.module)
            doc = ast.get_docstring(tree) or "No module docstring."
            return {
                "functions": funcs,
                "imports": list(set(imports)),
                "doc": doc.strip().replace("\n", " ")[:300],
            }
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    async def scan_once(self):
        """Scan directories for new or changed files."""
        for base in self.base_dirs:
            if not os.path.exists(base):
                continue
            for file in os.listdir(base):
                if not file.endswith(".py") or file.startswith("__"):
                    continue
                path = os.path.join(base, file)
                h = self._hash_file(path)
                if not h:
                    continue
                if path not in self.file_hashes or self.file_hashes[path] != h:
                    self.file_hashes[path] = h
                    info = self._analyze_python(path)
                    summary = (
                        f"File: {path}. Functions: {info.get('functions')}. "
                        f"Imports: {info.get('imports')}. Doc: {info.get('doc')}"
                    )
                    print(f"[SelfDiscovery] 🧩 New or updated: {path}")

                    # שליחת אירוע למערכת דרך ה-Bus
                    await global_bus.publish("self_discovery", {
                        "path": path,
                        "summary": summary,
                        "metadata": {"type": "code"}
                    })

    # ------------------------------------------------------------------
    @timed("self_discovery_run")
    async def run(self):
        """Continuous background self-scan loop."""
        print("[SelfDiscovery] 🧠 Monitoring started.")
        while True:
            try:
                await self.scan_once()
                await asyncio.sleep(self.scan_interval)
            except asyncio.CancelledError:
                print("[SelfDiscovery] 🔻 Monitoring stopped.")
                break
            except Exception as e:
                print(f"[SelfDiscovery] ⚠️ Error: {e}")
                await asyncio.sleep(self.scan_interval)
