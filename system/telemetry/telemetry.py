
# telemetry.py
import time
from functools import wraps
from pathlib import Path
from .sampler import allow_sample
from .aggregator import telemetry_start, telemetry_add, telemetry_finalize
from .storage import write_text, write_jsonl


import inspect

def timed(tag=None):
    def decorator(func):
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not allow_sample():
                    return await func(*args, **kwargs)
                t0 = time.perf_counter()
                result = await func(*args, **kwargs)
                t1 = time.perf_counter()
                telemetry_add(tag or func.__name__, t1 - t0)
                return result
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not allow_sample():
                    return func(*args, **kwargs)
                t0 = time.perf_counter()
                result = func(*args, **kwargs)
                t1 = time.perf_counter()
                telemetry_add(tag or func.__name__, t1 - t0)
                return result
            return sync_wrapper
    return decorator


def telemetry_summary():
    summary, raw = telemetry_finalize()
    if not summary:
        return ""
    write_text(summary)
    for entry in raw:
        write_jsonl(entry)
    return summary
