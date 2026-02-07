# orion_core/skills/skill_system_diagnostics.py
"""
SKILL: System Diagnostics
The "Self-Test" tool used by the Supervisor Layer.
"""
import uuid
import time
from typing import Dict, Any
from system.capability_engine import CapabilityEngine

# ------------------------------------------------------------------
# TOOL DEFINITION
# ------------------------------------------------------------------
def get_tools_metadata():
        """Teaches the Brain what tools exist."""
        return {
            "web.search": {
                "description": "Search online for real-time info, best practices, or library documentation.",
                "params": ["query"]
            },
            "task.start": {
                "description": "Write code or create files.",
                "params": ["instructions"]
            },
            "cmd.run": {
                "description": "Run terminal commands (use background=True for long tasks).",
                "params": ["command", "background"]
            },
            "fs.delete": {"description": "Delete a specific file.", "params": ["path"]},
            "fs.list": {"description": "List files in a folder.", "params": ["path"]},
            "proc.list": {"description": "List running processes.", "params": []},
            "proc.kill": {"description": "Kill a process by PID.", "params": ["pid"]},
            "system.update_entity": {"description": "Save facts to memory.", "params": ["entity_id", "updates"]},
        }
# ------------------------------------------------------------------
# IMPLEMENTATION
# ------------------------------------------------------------------
def run_diagnostics() -> str:
    """
    Executes the comprehensive self-test.
    1. Capability Scan (Code/Imports)
    2. Memory I/O Test (Database)
    """
    print("[Diagnostics] 🩺 Starting System Self-Test...")
    results = []
    
    # 1. CAPABILITY SCAN (Code Integrity)
    # Uses the user's existing capability_engine.py
    try:
        engine: CapabilityEngine = CapabilityEngine()
        cap_data: Dict[str, Any] = engine.refresh()
        
        missing_core = cap_data.get("core_components_missing", [])
        import_errors = cap_data.get("import_errors", [])
        
        if not missing_core and not import_errors:
            results.append("✅ CODE INTEGRITY: PASS")
        else:
            results.append(f"❌ CODE INTEGRITY: FAIL (Missing: {missing_core}, Errors: {len(import_errors)})")
    except Exception as e:
        results.append(f"❌ CODE INTEGRITY: CRITICAL ERROR ({str(e)})")

    # 2. MEMORY I/O TEST (Database Check)
    # This replaces the 'test_learning.py' concept
    try:
        bridge = MetaBridge()
        test_id = "diagnostic_temp_key"
        test_val = str(uuid.uuid4())
        
        # WRITE
        bridge.update_entity(test_id, {"test_val": test_val, "timestamp": time.time()})
        
        # READ
        data = bridge.get_entity(test_id)
        
        if data.get("test_val") == test_val:
            results.append("✅ MEMORY I/O: PASS")
            # Cleanup (Optional, keep if you want audit trail)
            # bridge.delete_entity(test_id) 
        else:
            results.append(f"❌ MEMORY I/O: FAIL (Read mismatch. Expected {test_val}, got {data.get('test_val')})")
            
    except Exception as e:
        results.append(f"❌ MEMORY I/O: CRITICAL ERROR ({str(e)})")

    # 3. CONSOLIDATE REPORT
    final_report = "\n".join(results)
    print(f"[Diagnostics] Report Generated:\n{final_report}")
    return final_report