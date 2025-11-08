# simple intent router (keyword-first; cheap & fast)
from dataclasses import dataclass


@dataclass
class Intent:
    name: str
    slots: dict


class IntentManager:
    def parse(self, text: str) -> Intent | None:
        t = text.lower().strip()
        if any(k in t for k in ["time", "what time"]):
            return Intent("time.now", {})
        if "open dashboard" in t or "dashboard" in t:
            return Intent("dashboard.open", {})
        if "image" in t or "generate" in t:
            return Intent("image.generate", {"prompt": t})
        if any(k in t for k in ["weather", "forecast"]):
            return Intent("weather.get", {"q": t})
        return None
