import subprocess
import sys
import time
import os
import signal
import psutil

TARGET_MODULE = "runners.webview" 
RESTART_CODE = 100

def cleanup_zombies():
    """Kill any lingering Orion processes from previous runs."""
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Look for python processes running 'launcher.py' that aren't US
            if proc.info['name'] == 'python3' or proc.info['name'] == 'python':
                if proc.info['cmdline'] and 'runners/launcher.py' in str(proc.info['cmdline']):
                    if proc.info['pid'] != current_pid:
                        print(f"[Launcher] 🧟 Killing zombie process: {proc.info['pid']}")
                        proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

# Run cleanup before starting
cleanup_zombies()

def run_orion():
    while True:
        print("\n[Launcher] 🚀 Starting Orion UI...")
        
        # Start the webview module with Python (similar to `python3 -m runners.webview`)
        proc = subprocess.Popen(
            ["python3", "-m", TARGET_MODULE],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            print("\n[Launcher] ⚠️ Ctrl+C detected. Killing all processes...")
            
            # Force kill the subprocess and all its children
            try:
                parent = psutil.Process(proc.pid)
                children = parent.children(recursive=True)
                
                # Kill children first
                for child in children:
                    try:
                        print(f"[Launcher] 💀 Killing child {child.pid}")
                        child.kill()
                    except:
                        pass
                
                # Kill parent
                proc.kill()
                print("[Launcher] ✅ All processes killed")
            except Exception as e:
                print(f"[Launcher] ⚠️ Kill failed: {e}")
                proc.kill()  # Force kill anyway
            
            # Exit immediately
            sys.exit(0)
        
        # If the process exits, check the code
        if code == RESTART_CODE:
            print("[Launcher] 🔄 Restart requested. Restarting...")
            continue
        
        # Otherwise, exit naturally
        print("[Launcher] 🔻 Orion exited.")
        break

if __name__ == "__main__":
    run_orion()