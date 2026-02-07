1. Debugging Philosophy

Orion does not guess the cause of a problem.

Instead, he follows a consistent diagnostic sequence:

Collect data

Instrument missing logs if needed

Trigger the failing process and observe behavior

Analyze telemetry + introspection

Identify root cause

Patch code if needed

Reload modules

Validate

Finalize

This ensures Orion can handle:

STT failure

VAD silence

missing audio

race conditions

async errors

missing imports

broken modules

wrong folder structure

performance degradation

browser permission issues

missing dependencies

coder mistakes

2. Key Tools for Debugging
✔ System Introspector

Files:

system_introspector.py

Used for:

scanning imports

checking reload errors

identifying initialization failures

detecting stuck tasks

module dependency issues

recording exceptions

✔ Telemetry

Files:

telemetry.py

telemetry.jsonl

Used for:

timing analysis

STT/VAD warnings

exceptions

execution time anomalies

memory spikes

function-level profiling

✔ Capability Engine

Files:

capability_engine.py

Used for:

detecting missing features

disabled modules

incorrect wiring

✔ Project Mapper

Files:

project_mapper.py

Used to:

identify which files relate to a bug

return lists of file paths for Thinker to analyze

map vague user requests to specific code areas

✔ Coder

Files:

coder.py

Used to:

generate patches

insert debug logs

adjust settings (e.g., VAD threshold, buffer size)

refactor functions

improve stability

3. Debugging Flow (Actual Behavior)

This is the standard debugging loop:

User → Thinker → plan_task → Heavy Model → Planner → Brain → Tools → Coder → Introspector → Thinker


Sequence:

Step 1 — Thinker receives request (e.g., “Fix STT”)

Detects request is complex

Sets "action": "plan_task"

Returns short message: “One moment.”

Step 2 — Brain loads heavy model

Using:

handle_heavy_task(reason, estimate)

Step 3 — Heavy model generates plan

Example (the model generates this, not hard-coded):

1. Run diagnostics on STT pipeline
2. Insert temporary debug logs
3. Ask user to activate microphone
4. Record telemetry and detect anomalies
5. Generate patch to fix root cause
6. Apply patch and hot reload
7. Validate STT response

Step 4 — Planner stores plan

Planner does not reason — only stores steps.

Step 5 — Brain executes steps via tool_dispatch

Tools used:

diagnostics

map

capabilities

coder

reload

Step 6 — If logs are missing

Orion automatically adds them:

VAD input levels

PCM buffer lengths

STT timestamps

dropped audio chunks

event order

async timing

device status

WebRTC connection status

Step 7 — User triggers the system

Example: Clicking microphone once browser requires gesture.

Step 8 — Orion collects logs and telemetry

Checks:

Are PCM packets arriving?

Is VAD detecting?

Is STT receiving long enough chunks?

Is async loop blocking?

Is whisper model returning empty?

Timing drift?

Buffer underflow?

Browser constraints?

Step 9 — Orion identifies root cause

(e.g., buffer too small, browser audio blocked, incorrect async pipeline)

Step 10 — Coder patches code

Using heavy model to produce a fix.

Step 11 — SystemIntrospector reloads

Confirms the new code loads correctly.

Step 12 — Thinker returns summary

“Sir, STT has been repaired. Please test it.”

4. When Orion Cannot Fix the Issue

If the cause is external (browser blocking mic), Orion must:

Detect missing audio

Explain the reason

Offer alternative:

Server-side microphone

UI redesign (autostart mic on gesture)

Alternative input method

Orion never hallucinates fixes.

5. STT/VAD-Specific Rules

Orion knows this:

STT requires minimum PCM duration (≥ 300–500 ms)

VAD fails on:

tiny chunks

zero-signal

too-high threshold

browser autoplay restrictions

Whisper returns empty when:

sample rate mismatched

silent buffer

chunk < 16000 samples

Orion steps:

check audio size

check VAD result

check chunk timing

check browser permission

insert logs

analyze telemetry

fix code accordingly