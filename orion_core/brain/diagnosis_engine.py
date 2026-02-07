# orion_core/brain/diagnosis_agent.py

from __future__ import annotations
from orion_core.brain.problem_detector import ProblemReport


def build_diagnosis_prompt(
    user_text: str,
    recent_context: str,
    report: ProblemReport,
    project_structure: str = "",
    telemetry_summary: str = "",
) -> str:
    """
    Builds a pure, LLM-friendly diagnosis prompt describing:
    - what the user said
    - the detected issue summary
    - any structural hints from project mapper
    - telemetry summary

    Thinker receives this prompt, not this function.
    """

    return (
        "You are Orion's internal diagnostic engine.\n"
        "Your job: explain clearly and concisely what is MOST LIKELY wrong, "
        "based on the user report, context, detected issue summary, and any "
        "available project structure or telemetry.\n\n"
        "Address the user as 'אדוני'.\n"
        "Do NOT output JSON — produce a clear explanation only.\n\n"
        f"User message:\n{user_text}\n\n"
        f"Detected issue summary:\n{report.summary}\n\n"
        f"Recent context:\n{recent_context}\n\n"
        f"Project structure (if any):\n{project_structure}\n\n"
        f"Telemetry (if any):\n{telemetry_summary}\n"
    )
