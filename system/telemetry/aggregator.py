
# aggregator.py
TELEMETRY_DATA = []

def telemetry_start():
    TELEMETRY_DATA.clear()

def telemetry_add(name, duration):
    TELEMETRY_DATA.append({"name": name, "time": duration})

def telemetry_finalize():
    if not TELEMETRY_DATA:
        return None, None
    total = sum(i["time"] for i in TELEMETRY_DATA)
    lines = ["========== ⏱️ ORION TELEMETRY =========="]
    for item in TELEMETRY_DATA:
        lines.append(f"• {item['name']:<32} {item['time']:.3f}s")
    lines.append("-------------------------------------------")
    lines.append(f"TOTAL PIPELINE TIME: {total:.3f}s")
    lines.append("===========================================\n")
    summary = "\n".join(lines)
    raw = [{**item, "total_pipeline": total} for item in TELEMETRY_DATA]
    TELEMETRY_DATA.clear()
    return summary, raw
