#!/usr/bin/env bash
# Run from the orion-v2 project root:  bash verify_deploy.sh
# Confirms the deployed files are the current versions. Any FAIL = stale file.
B=orion_core/brain
pass=0; fail=0
chk() { # chk <label> <expected: yes|no> <pattern> <file>
  local label="$1" want="$2" pat="$3" f="$4"
  if [ ! -f "$f" ]; then echo "  ?? MISSING FILE: $f"; fail=$((fail+1)); return; fi
  if grep -q -- "$pat" "$f"; then found=yes; else found=no; fi
  if [ "$found" = "$want" ]; then echo "  OK   $label"; pass=$((pass+1));
  else echo "  FAIL $label   ->  $f is STALE"; fail=$((fail+1)); fi
}

echo "== Orion deploy check =="
chk "orion_tools: recall_about_person removed"  no  "recall_about_person"      $B/orion_tools.py
chk "orion_tools: no 'me/i/myself' matching"    no  '"me", "i", "myself"'      $B/orion_tools.py
chk "orion_tools: set_owner present"            yes "def set_owner"            $B/orion_tools.py
chk "orion_tools: _owner_profile guard"         yes "_owner_profile"           $B/orion_tools.py
chk "session_manager: [PEOPLE] block"           yes "\[PEOPLE\]"               $B/session_manager.py
chk "session_manager: no [OWNER] block"         no  "\[OWNER\]"                $B/session_manager.py
chk "orion_graph: router removed"               yes "No pre-classification gate" $B/orion_graph.py
chk "orion_graph: narrate helper"               yes "async def narrate"        $B/orion_graph.py
chk "orion_graph: tool output cap"              yes "_MAX_TOOL_OUTPUT_CHARS"   $B/orion_graph.py
chk "tool_registry: auto-discovery"             yes "_discover_module_tools"   $B/tool_registry.py
chk "brain: no owner bootstrap"                 no  "_owner_bootstrap"         $B/brain.py

# prompts + data (adjust paths if yours differ)
K=$(find . -name kernel_system.md -not -path "*/node_modules/*" 2>/dev/null | head -1)
[ -n "$K" ] && chk "kernel_system: OWNER BOOTSTRAP removed" no "OWNER BOOTSTRAP" "$K"
[ -n "$K" ] && chk "kernel_system: personal names scrubbed" no "Leroy"          "$K"
P=$(find . -name people_tree.json -not -path "*/node_modules/*" 2>/dev/null | head -1)
if [ -n "$P" ]; then
  python3 - "$P" << 'PY'
import json,sys
d=json.load(open(sys.argv[1]))
k=d.get("owner"); prof=(d.get("profiles") or {}).get(k) if k else None
print(f"  {'OK  ' if prof else 'FAIL'} people_tree: owner resolves ({k!r} -> {'found' if prof else 'DANGLING/None'})")
PY
fi
D=$(find . -name directives.json -not -path "*/node_modules/*" 2>/dev/null | head -1)
[ -n "$D" ] && chk "directives: stale location directive gone" no "lives in Israel" "$D"

echo
echo "  passed: $pass   failed: $fail"
[ "$fail" -eq 0 ] && echo "  ALL GOOD — safe to test." || echo "  STALE FILES ABOVE — copy them, then: find . -name __pycache__ -exec rm -rf {} +"
