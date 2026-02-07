# orion_core/brain/problem_detector.py

from __future__ import annotations
from dataclasses import dataclass
import json


@dataclass
class ProblemReport:
    is_issue: bool
    summary: str = ""
    area: str | None = None


def build_problem_prompt(user_text: str, recent_context: str = "") -> str:
    """
    Builds a pure prompt for the LLM to classify if the user is reporting
    a real malfunction or just asking a normal question or hypothetical scenario.
    """
    user_text = (user_text or "").strip()
    recent_context = recent_context.strip() if recent_context else ""

    return (
        "You are a binary classifier inside Orion.\n"
        "Your ONLY job is to determine whether the user is reporting a REAL malfunction "
        "in Orion or the computer environment *right now*.\n\n"
        "If the user is only asking a question, giving a hypothetical example, or speaking "
        "in general terms, then it is NOT a real issue.\n\n"
        "Return STRICT JSON ONLY:\n"
        "{\"issue\": true/false, \"summary\": \"short summary\", \"area\": \"optional\"}\n\n"
        f"User message:\n{user_text}\n\n"
        f"Recent context:\n{recent_context}\n"
    )


def parse_problem_response(raw: str) -> ProblemReport:
    """
    Parses the JSON returned by Thinker into a ProblemReport.
    """
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(raw[start:end + 1])
        else:
            data = {}
    except Exception:
        data = {}

    return ProblemReport(
        is_issue=bool(data.get("issue", False)),
        summary=str(data.get("summary", "")),
        area=data.get("area"),
    )
