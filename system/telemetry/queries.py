# queries.py
from .storage import read_last_jsonl


def pipeline_timeline(max_lines=200):
    """Return (timestamp, total_pipeline) for timeline chart."""
    rows = read_last_jsonl(max_lines)
    out = []
    for row in rows:
        total = row.get("total_pipeline")
        ts = row.get("timestamp")
        if total is None or ts is None:
            continue
        out.append({"timestamp": ts, "total_pipeline": total})
    out.sort(key=lambda x: x["timestamp"])
    return out


def latest_raw(max_lines=200):
    """Return raw entries for tail display."""
    return read_last_jsonl(max_lines)


def stats(max_lines=500):
    """Aggregate metrics by avg/min/max/count."""
    rows = read_last_jsonl(max_lines)
    agg = {}

    for row in rows:
        name = row.get("metric") or row.get("name")
        t = row.get("time", 0.0)
        if not name:
            continue

        s = agg.setdefault(name, {
            "name": name,
            "count": 0,
            "total": 0.0,
            "min": None,
            "max": None
        })

        s["count"] += 1
        s["total"] += t
        s["min"] = t if (s["min"] is None or t < s["min"]) else s["min"]
        s["max"] = t if (s["max"] is None or t > s["max"]) else s["max"]

    # compute avg
    for s in agg.values():
        s["avg"] = s["total"] / s["count"] if s["count"] > 0 else 0.0

    # return sorted
    return sorted(agg.values(), key=lambda x: x["avg"], reverse=True)
