import os
import shutil
import subprocess
import psutil  # pip install psutil


def handle_dashboard(text: str) -> str:
    """Handles dashboard-specific commands."""
    if "open" in text:
        return _open_dashboard()
    elif "restart" in text:
        return _restart_dashboard()
    elif "close" in text:
        return _close_dashboard()
    else:
        return "Dashboard command not recognized."


def _open_dashboard() -> str:
    """Open or focus Orion Dashboard."""
    proc_running = any(
        p.info["name"] == "orion-dashboard"
        for p in psutil.process_iter(attrs=["name"])
    )

    if proc_running:
        if shutil.which("wmctrl"):
            subprocess.run(["wmctrl", "-x", "-a", "orion-dashboard"], check=False)
            return "Orion Dashboard is already open — bringing it to front."
        return "Orion Dashboard is already open."
    else:
        _launch_dashboard()
        return "Opening Orion Dashboard..."


def _restart_dashboard() -> str:
    """Restart Orion Dashboard if running, otherwise open it."""
    killed = False
    for p in psutil.process_iter(attrs=["pid", "name"]):
        if p.info["name"] == "orion-dashboard":
            try:
                os.kill(p.info["pid"], 9)
                killed = True
            except Exception:
                pass
    _launch_dashboard()
    if killed:
        return "Restarted Orion Dashboard."
    else:
        return "Dashboard was not running, opening it now."


def _close_dashboard() -> str:
    """Close Orion Dashboard."""
    closed = False
    for p in psutil.process_iter(attrs=["pid", "name"]):
        if p.info["name"] == "orion-dashboard":
            try:
                os.kill(p.info["pid"], 9)
                closed = True
            except Exception:
                pass
    if closed:
        return "Closed Orion Dashboard."
    else:
        return "Dashboard is not running."


def _launch_dashboard():
    """Launch the dashboard either via gtk-launch or direct path."""
    if shutil.which("gtk-launch"):
        subprocess.Popen(["gtk-launch", "orion-dashboard"])
    else:
        subprocess.Popen(
            ["/home/ilays/projects/dashboard/dist/orion-dashboard"],
            cwd="/home/ilays/projects/dashboard/dist"
        )
