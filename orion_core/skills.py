import platform
import datetime
import sys
import os
import subprocess
import psutil  # pip install psutil


class Skills:
    def __init__(self):
        pass

    def handle(self, text: str) -> str:
        t = text.lower()

        # greetings
        if "hello" in t:
            return "Hello! I am Orion v2."

        # system info
        elif "system" in t:
            return f"System: {platform.system()} {platform.release()}"

        # time
        elif "time" in t:
            now = datetime.datetime.now().strftime("%H:%M:%S")
            return f"The current time is {now}"

        # math / calculator
        elif "calc" in t or "calculate" in t:
            expr = t.replace("calc", "").replace("calculate", "").strip()
            try:
                result = eval(expr, {"__builtins__": {}})
                return f"Result: {result}"
            except Exception:
                return "Sorry, I couldn't calculate that."

        # open dashboard
        elif "open dashboard" in t:
            proc_running = False
            for proc in psutil.process_iter(attrs=["name"]):
                if proc.info["name"] == "orion-dashboard":
                    proc_running = True
                    break

            if proc_running:
                # Try to raise existing window
                try:
                    subprocess.run(
                        ["wmctrl", -1, "-x", "-a", "orion-dashboard"],
                        check=False
                    )
                    return "Orion Dashboard is already open — bringing it to front."
                except Exception:
                    return "Orion Dashboard is already open."
            else:
                # Launch via desktop entry if possible
                try:
                    subprocess.Popen(["gtk-launch", "orion-dashboard"])
                    return "Opening Orion Dashboard..."
                except Exception:
                    # Fallback to direct binary path
                    subprocess.Popen(
                        ["/home/ilays/projects/dashboard/dist/orion-dashboard"],
                        cwd="/home/ilays/projects/dashboard/dist"
                    )
                    return "Opening Orion Dashboard..."

        # exit command
        elif "exit" in t or "stop" in t or "quit" in t:
            return "Exiting Orion..."

        # fallback
        return f"You said: {text}"
