⭐ orion_coding_reference.md

Internal Coding Rules & Code-Generation Workflow for Orion

1. Purpose

This document defines how Orion generates, patches, validates, reloads, and finalizes code using:

Thinker (light model)

Heavy Coder/Planner model

Coder module

System Introspector

Capability Engine

Project Mapper

Planner

Brain

It ensures Orion writes code safely, modularly, and according to the architecture, and most importantly:

👉 Orion verifies that the generated feature works before reporting success.

This is mandatory for autonomous operation.

2. Coding Responsibility Breakdown
2.1 Thinker (Light Model)

Thinker does NOT write code directly.

Thinker:

Interprets user request

Decides simple / vague / complex

If vague → asks clarifying question

If complex → issues "action": "plan_task"

Constructs heavy-model prompt for planning or coding

Ensures JSON formatting

Provides natural responses after heavy tasks

Verifies whether the final code matches the request

File: thinker.py

2.2 Heavy Model (Planner/Coder)

The heavy model is used for:

multi-step planning

deep code analysis

code generation

file creation

large context modifications

multi-file refactoring

architecture-level decisions

Heavy model is invoked by Thinker through Brain using handle_heavy_task().

Files:

llm_local.py

(model weights external)

2.3 Coder Module

The Coder module:

collects file context (reads relevant files)

communicates with Heavy Model (via Thinker)

applies patches safely

creates new files when instructed

ensures file structure follows Orion’s architecture

never calls LLM directly

Files:

coder.py

file_write.py

2.4 System Introspector

SystemIntrospector is the validation and reload tool.

It:

detects syntax errors

detects import failures

reloads modules

validates new code

ensures pipeline is intact after modification

provides error data back to Thinker

File: system_introspector.py

2.5 Capability Engine

After new code is written, Orion checks:

is the new ability present?

is the new skill registered?

does it appear in Orion’s capability map?

File: capability_engine.py

2.6 Project Mapper

Maps natural-language user requests → relevant files.

File: project_mapper.py

2.7 Planner

Stores plans produced by Heavy Model and executes them step-by-step.

Planner does not use LLM.

File: planner.py

2.8 Brain

Orchestrates:

Thinker first

heavy-model loading/unloading

plan execution

coder commands

diagnostics

final natural reply

File: brain.py

3. Coding Workflow (Correct Autonomous Behavior)

Here is the universal sequence Orion follows for ANY coding request:

Step 1 — Thinker interprets request

Example:
“Orion, add face recognition.”

Thinker outputs JSON:

{
  "immediate_text": "Very well, sir.",
  "action": "plan_task",
  "needs_confirmation": false
}


If request is vague:

{
  "action": null,
  "needs_confirmation": true,
  "anticipation_question": "What should this feature do, sir?"
}

Step 2 — Thinker prepares heavy-model coding or planning prompt

Thinker gathers:

project file tree (via ProjectMapper)

relevant modules

capabilities

diagnostics

telemetry summary

architecture rules

expected module locations

required wiring steps

Thinker then prepares a structured prompt for the heavy model.

Step 3 — Brain triggers Heavy Model (deep reasoning)

If GPU is tight → unload Thinker.
Load Heavy Model → run plan generation or code generation.

Using:

handle_heavy_task(reason="coding", estimate=X)

Step 4 — Heavy Model generates plan or code

A plan might look like this:

[
  {"index":1, "description":"Create face_detection.py", "tool":"coder"},
  {"index":2, "description":"Add register to skills.py", "tool":"coder"},
  {"index":3, "description":"Integrate into pipeline", "tool":"coder"},
  {"index":4, "description":"Reload modules", "tool":"reload"},
  {"index":5, "description":"Validate imports", "tool":"diagnostics"}
]


For code generation, heavy model returns valid diff or full file content.

Step 5 — Planner stores plan

Planner does not reason.
It stores step list and status.

Step 6 — Brain executes steps via tool_dispatch

Possible tools:

Tool	Meaning
diagnostics	system introspection
map	find relevant files
capabilities	update ability map
coder	write/patch files
reload	hot reload modules
skills	call existing skills
files	direct file operations

Each step is executed sequentially.

Step 7 — Coder writes code

Coder:

builds context

sends it to heavy model (via Thinker)

receives diff / patch / files

applies them

verifies file presence

notifies Brain

Coder obeys your architecture:

✔ Skills go in their own file

Example:
skills/face_detection.py

✔ Skills are registered in skills.py

e.g.:

register("vision.face", FaceDetectionSkill)

✔ File structure must be respected

Folders used:

/skills

/orion_core

/modules

/abilities

/project specific folders

✔ All new modules must have predictable names

THinker ensures these rules are given to heavy model.

Step 8 — SystemIntrospector validates result

After patch or new file:

reload module

check errors

report import failures

confirm correct initialization

ensure skill is discoverable

ensure pipeline stability

If failure → Planner automatically generates next steps.

Step 9 — Thinker verifies result matches request

Thinker checks:

is the new ability discoverable?

is it correctly registered?

does Introspector report clean state?

does the module import successfully?

if applicable → can Orion run it?

does the output match user’s intent?

If not → Thinker triggers another plan_task.

Step 10 — Thinker returns final natural summary

Example:

“Sir, face recognition is implemented and operational. Shall I enable it now?”

Or:

“Sir, the patch resolved the error. STT is working again.”

4. Coding Structure & Rules
✔ Always create new abilities in separate files

e.g. /skills/face_detect.py

✔ Always register skills in skills.py

e.g.:

from skills.face_detect import FaceDetector
register("vision.face", FaceDetector)

✔ Heavy model must follow these rules exactly

Thinker must embed architecture rules in its heavy-model prompt.

✔ Coder must apply diffs safely

Coder must:

avoid overwriting unrelated code

handle indentation

use atomic writes

backup file if needed

✔ Orion must ALWAYS reload modules after patch

SystemIntrospector handles this.

✔ Orion must log changes for future debugging
5. New Feature Workflow (Example)
User:

“Orion, create a screenshot recognition ability.”

Thinker:

ask clarifying question if needed

else output "action": "plan_task"

Heavy Model Plan:

create file vision/screenshot.py

create file vision/analyze.py

add registry entries in skills.py

integrate with capability engine

reload modules

run validation (try calling the ability)

finalize

Thinker:

“Sir, the screenshot recognition feature is operational.”

6. Bug Fix Workflow (Example)

User:

“Fix STT”

Thinker:
action: "plan_task"

Heavy Model plan:

diagnostics

add logs

run STT

analyze telemetry

code patch

reload

re-test

finalize

Thinker:

“Sir, STT has been repaired.”

7. Safety & Validation Rules

Orion must ALWAYS ensure:

generated code runs

no syntax errors

no import errors

module loads

feature is callable

capability engine sees new ability

tool dispatch works

planner step did not fail

If failure:

Thinker creates a new plan to fix the failure itself
(“self-correction loop”)

8. File Responsibilities Summary
File	Purpose
thinker.py	Interpret requests, JSON decisions, heavy-model prompts
brain.py	Orchestrates everything
planner.py	Stores/execution state of plans
coder.py	Applies patches, writes files
skills.py	Registers abilities
capability_engine.py	Ability discovery
system_introspector.py	Diagnostics & reload
project_mapper.py	Map requests → file paths
llm_local.py	Model loader (Thinker + Heavy model)
file_write.py	Helpers for Coder
telemetry.py	Execution logs