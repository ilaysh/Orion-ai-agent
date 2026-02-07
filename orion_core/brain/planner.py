# orion_core/brain/planner.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable

@dataclass
class PlanStep:
    index: int
    description: str
    tool: str  # "fs", "cmd", "system", "memory"
    args: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # "pending" | "done" | "failed"
    result: Optional[str] = None

@dataclass
class Plan:
    goal: str
    steps: List[PlanStep] = field(default_factory=list)

class Planner:
    """
    The Zero Method Executor.
    It executes the Architect's JSON plan deterministically.
    It is 'mute' - it returns logs to the Brain but does not broadcast thoughts.
    """
    def __init__(self):
        self.active_plan: Optional[Plan] = None

    def json_to_plan(self, data: dict) -> Plan:
        """Parses the Heavy Model's JSON output into a Plan object."""
        goal = data.get("goal", "Unknown Task")
        raw_steps = data.get("steps", [])
        
        parsed_steps = []
        for s in raw_steps:
            parsed_steps.append(PlanStep(
                index=int(s.get("index", 0)),
                description=s.get("description", "No description"),
                tool=s.get("tool", "unknown"),
                args=s.get("args", {})
            ))
        
        return Plan(goal=goal, steps=sorted(parsed_steps, key=lambda x: x.index))

    async def execute_plan(self, tool_dispatch: Callable, auto_confirm: bool = True) -> str:
        """
        Executes the plan step-by-step.
        
        Args:
            tool_dispatch: A callback that accepts (tool_name, args) and returns a string.
            auto_confirm: If True, executes destructive commands without pause.
        
        Returns:
            A full execution log string for the Architect/Brain to analyze.
        """
        if not self.active_plan:
            return "No active plan."

        results_log = []
        
        for step in self.active_plan.steps:
            if step.status == "done": continue

            # Execution
            try:
                # We pass the unpacked args to the dispatcher
                output = await tool_dispatch(step.tool, step.args)
                
                step.result = output
                step.status = "done"
                
                # We log the result for the Brain, but truncate huge outputs (like cat file)
                log_entry = f"Step {step.index} [{step.tool}]: {str(output)[:500]}"
                results_log.append(log_entry)
                
            except Exception as e:
                step.status = "failed"
                step.result = str(e)
                results_log.append(f"Step {step.index} FAILED: {e}")
                # We stop execution on system error to prevent cascading damage
                return "\n".join(results_log) + "\n\nPlan execution halted due to error."

        return "\n".join(results_log)