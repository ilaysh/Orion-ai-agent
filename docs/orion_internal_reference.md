1. Overview

Orion is a local-first autonomous AI assistant capable of:

reasoning

planning

debugging itself

generating code

repairing pipelines

creating new features

building full apps

managing multi-step workflows

interacting naturally (Jarvis-like)

controlling and reloading its own modules

Orion is not a chatbot.
It is a modular autonomous agent with:

a lightweight “Thinker brain”

a heavy reasoning/coding model

an execution planner

a code patcher

capability discovery

dynamic module loading

telemetry-based diagnostics

This document describes Orion’s true architecture,
so any future chat or developer (or Orion itself) can operate with full understanding.

2. Core Pipeline
Flow:

User → Brain → Thinker → (Planner / Heavy Model / Tools / Coder) → Brain → User

User speaks or types

Brain logs input + builds context

Thinker interprets request, outputs structured JSON:

{
  "immediate_text": "...",
  "action": "...",
  "needs_confirmation": true/false,
  "anticipation_question": "..."
}


Brain routes based on "action"

If complex → "plan_task"

Brain calls Heavy Model to generate plan

Planner stores plan

Brain executes steps via tool_dispatch

Coder applies patches

SystemIntrospector reloads modules

Thinker gives the final natural response

3. Models Used
3.1 Thinker (light model)

Always loaded.

Responsibilities:

interpret user intent

decide simple vs. complex

ask clarifying questions

initiate planning ("action": "plan_task")

provide short natural replies

build prompts for Heavy Model

output strict JSON to Brain

File:

thinker.py

3.2 Heavy Model (planner & coder)

Loaded on demand via Brain’s handle_heavy_task().

Examples:

Qwen2.5-Coder-14B

DeepSeek Coder V2 Lite

Responsibilities:

multi-step planning

code generation / patching

architecture analysis

debugging

feature creation

application creation

Files:

llm_local.py (model loader & registry)

brain.py (switch logic)

4. Components & Responsibilities
4.1 Brain (the orchestrator)

The central nervous system of Orion.

Responsibilities:

call Thinker first for every request

route action: normal, skill, plan, diagnostics, coding

handle model switching (Thinker ↔ Heavy model)

maintain session context

execute Planner steps

run tool_dispatch

manage TTS/STT states

integrate telemetry

return final result to UI

publish thoughts (bubbles)

hot reload modules

Files:

brain.py

session_manager.py

router.py

Brain does not think, does not plan, does not generate code.
Brain coordinates everything.

4.2 Thinker (the mind)

The only LLM that interprets user requests.

Responsibilities:

produce structured JSON

handle clarification & vague requests

decide when planning is needed

prepare prompts for Heavy Model

never break JSON rules

never reveal reasoning

maintain Orion personality

Files:

thinker.py

personality.py

orion_personality.md

4.3 Planner (state container)

A simple container for autonomous tasks.

Responsibilities:

store plan JSON

track steps (pending/done/failed)

expose next step for execution

call tool_dispatch through Brain

Planner does not:

generate plans

call a model

think

Files:

planner.py

4.4 Coder (executor for code patches)

Hands of the system.

Responsibilities:

gather relevant code context

request code patches from Thinker → Heavy Model

apply patch to disk

handle file creation

help integrate new features

notify SystemIntrospector to reload

Files:

coder.py

file_write.py

Coder never calls LLM directly — Thinker always builds the prompt for Heavy Model.

4.5 Capability Engine

Discovers available features, modules, skills, and abilities.

Used by Thinker to detect whether a request requires new code or uses an existing ability.

Files:

capability_engine.py

skills.py

4.6 System Introspector

Detects system issues and reloads modules.

Responsibilities:

diagnose Orion’s systems

detect STT/VAD/audio/import failures

detect pipeline crashes

reload single modules or entire subsystems

verify code after patch application

Files:

system_introspector.py

4.7 Project Mapper

Maps natural-language requests → source files.

Used when fixing bugs or adding features.

Files:

project_mapper.py

4.8 Skills

Explicit predefined actions.
Used only when Thinker recognizes a simple intent.

Files:

skills.py

5. Autonomy Loop (ACTUAL behavior)
1. Interpret

Thinker interprets the request and decides:

simple → direct reply

vague → ask clarification

complex → start planning
(action: "plan_task")

2. Plan

For complex tasks:

Thinker prepares heavy-model prompt

Brain loads heavy model

Heavy model produces plan steps

Planner stores them

3. Execute

Brain performs:

diagnostics

capabilities scan

file mapping

code generation

patch application

hot reload

testing

4. Validate

SystemIntrospector checks for errors.

5. Report

Thinker returns the final natural summary.

6. Behavior Examples
6.1 “Orion, create a new feature”

Not enough information.

Thinker:

{
  "immediate_text": null,
  "action": null,
  "needs_confirmation": true,
  "anticipation_question": "What should this feature do, sir?"
}

6.2 “Add face recognition”

Clear task:

Thinker:

{
  "immediate_text": "Very well, sir.",
  "action": "plan_task",
  "needs_confirmation": false,
  "anticipation_question": null
}


Heavy-model plan example:

create git branch

generate face_detection.py

integrate into pipeline

test

reload

finalize

6.3 “Fix STT”

Complex, ambiguous request.
Thinker starts planning.

Orion then:

scans code

injects logs if unclear

asks user to test

reads telemetry

patches code

reloads

retests

resolves issue or suggests alternative pipeline (server mic)

6.4 “Create a new app”

Vague → ask:

“What should this app do, sir?”

After specification:

plan_task

create folder

initialize git repo

generate main files

integrate necessary modules

test + run

PR

final confirmation

7. Personality Rules (Orion)

Always addresses user as “sir”

Speaks politely and concisely

Never explains LLM reasoning

Never outputs text outside JSON

Uses short confirmations

Anticipates next questions naturally

Uses “One moment” / “Very well, sir” / “Understood, sir”

Files:

orion_personality.md

personality.py

8. Heavy Model Switching Logic

Brain uses:

handle_heavy_task(reason, est_time)

MemoryManager to detect GPU load

unload Thinker if necessary

load heavy model

run planning or coding

unload heavy model

reload Thinker

Files:

brain.py

llm_local.py

9. Files to Upload When Debugging

If you open a new session or need help debugging, upload the following:

Core logic:

brain.py

thinker.py

planner.py

coder.py

skills.py

System:

system_introspector.py

capability_engine.py

project_mapper.py

session_manager.py

LLM:

llm_local.py

personality.py

orion_personality.md

UI / Router:

router.py

app.js (if audio/websockets are involved)

speech engine or listener code (STT-related)

Diagnostics:

telemetry.jsonl

recent log files

10. Design Philosophy

Orion follows these principles:

🧠 Thinker = intelligence
🧮 Heavy Model = deep reasoning + coding
🧵 Brain = coordination & execution
📋 Planner = state container
✍️ Coder = file manipulator
🔍 System Introspector = diagnostics & reload
🗺 Project Mapper = code locator
🧩 Capability Engine = ability map
