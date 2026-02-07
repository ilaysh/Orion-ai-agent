"""
Skill: file.write
Writes a file to disk safely.
"""

import os
import json


def run(text: str = "", **kwargs) -> str:
    """
    Expected kwargs:
    path: full file path
    content: string content
    """
    path = kwargs.get("path")
    content = kwargs.get("content", "")

    if not path:
        return "Error: missing path"

    if content is None:
        content = ""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
            return {
                "skill": "file.write",
                "result": {
                    "path": path,
                    "ok": True
                }
            }
    except Exception as e:
        return {
            "skill": "file.write",
            "result": {
                "path": path,
                "ok": False,
                "error": str(e)
            }
        }
