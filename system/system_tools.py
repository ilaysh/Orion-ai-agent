from system.capability_engine import CapabilityEngine


def refresh_capabilities():
    """
    Machine-action only.
    Thinker decides whether to speak or stay silent.
    """
    engine = CapabilityEngine()
    data = engine.refresh()
    return {
        "capabilities_refreshed": True,
        "snapshot_path": engine.snapshot_path,
        "summary_doc_id": "orion_capabilities",
        "core_found": data.get("core_components_found", []),
        "core_missing": data.get("core_components_missing", []),
    }
