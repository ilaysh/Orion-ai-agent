1. Planning Starts in Thinker

Thinker decides whether a user request requires planning.

Thinker conditions:

✔ Simple → reply

Examples:

“Who are you?”

“Stand by”

“What is the time?”

Thinker returns:

action: null
needs_confirmation: false

✔ Vague → ask question

Examples:

“Create a new feature”

“Improve this”

“Fix it”

“Make an app”

Thinker returns:

action: null
needs_confirmation: true
anticipation_question: "What should this feature do, sir?"

✔ Clear & complex → trigger plan

Examples:

“Add face recognition”

“Fix STT”

“Create an app that organizes images”

“Build a dashboard”

“Make a stock tracking agent”

“Create a Godot animation tool”

“Integrate OpenCV”

Thinker returns:

action: "plan_task"
immediate_text: "Very well, sir."


This triggers heavy-model reasoning.

2. Heavy Model Plans Everything

The heavy model generates full autonomous plans:

Each step includes:

index
description
tool   (diagnostics | map | coder | reload | skills | capabilities | files)


Planner NEVER generates steps.
Planner only stores them.

3. Planner Execution Rules

Planner executes steps in order:

diagnostics

map

capabilities

coder

reload

validation

reporting

Planner does not:

think

interpret

call LLM

modify tasks

4. Coder Integration Rules

Only heavy model generates code.

Coder:

gathers file context

passes to heavy model

receives patch

applies patch

reloads module

notifies Brain

Thinker never writes code by itself.

5. Brain Rules

Brain:

Always runs Thinker first

Handles “plan_task”

Switches models if needed

Ensures correct tool dispatch

Executes Planner steps

Provides logs for Thinker

Ensures pipeline stability

Brain cannot:

reason

generate code

plan

override Thinker decisions

6. Example Planning Conversations
Example: "Orion, create a new app"

Thinker:

needs_confirmation: true
anticipation_question: "What should this app do, sir?"


User: "A photo organizer."

Thinker:

action: "plan_task"


Heavy model plan (example):

create git branch

scaffold project folder

create main.py, ui.py

integrate plugin loader

write tests

build & run basic UI

return confirmation

Thinker:

“Sir, the app has been created.”

Example: "Add face recognition"

Same sequence.
After completion:

“Sir, I added face recognition support. Shall I activate it now?”

This universal planning works for most Orion tasks.