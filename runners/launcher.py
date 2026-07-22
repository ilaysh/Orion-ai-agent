# runners/launcher.py
from pathlib import Path
import subprocess
import sys
import time
import os
import signal
import psutil

TARGET_MODULE = "runners.webview" 
RESTART_CODE = 100
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
llama_log_file = open(log_dir / "llama_server.log", "w")

def get_free_vram():
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        return int(res.strip())
    except Exception:
        return 8000

def cleanup_zombies():
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

def wait_for_vllm():
    print("[Launcher] ⏳ Waiting for AI Engine to warm up...")
    while True:
        try:
            req = urllib.request.Request("http://127.0.0.1:8000/v1/models")
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    print("[Launcher] 🟢 AI Engine is ONLINE.")
                    break
        except urllib.error.URLError:
            pass
        time.sleep(2)

def run_orion():
    while True:
        free_vram = get_free_vram()
        n_ctx = "32768" # Safe baseline for your 16GB VRAM context allocation
        
        print(f"\n[Launcher] 🧠 Booting Gemma 4 Engine (VRAM: {free_vram}MB | n_ctx: {n_ctx})")
        
        vllm_cmd = [
            "/home/ilays/projects/llama.cpp/build/bin/llama-server",
            # --- UPDATED TO NEW GEMMA PATH ---
            "--model", "/home/ilays/projects/orion-v2/models/gemma-4-12b-it-Q4_K_M.gguf",
            "--host", "127.0.0.1",
            "--port", "8000",
            "--n-gpu-layers", "99",        
            "--threads", "8",
            # REQUIRED for Gemma 4 tool calling: uses the GGUF's embedded Jinja
            # template + llama-server's native PEG tool-parser. Without this,
            # llama-server falls back to the legacy built-in "gemma" template,
            # which does not emit/parse Gemma 4's native tool tokens.
            "--jinja",
            # NOTE: --log-disable removed on purpose. It was blanking
            # llama_server.log, so tool-parse warnings and the "thinking = N"
            # line never got captured. Needed for debugging bug #1 + audit (bug #2).
            "--ctx-size", n_ctx,            
            "--flash-attn", "on",
            # thinking-mode + temperature are handled PER-REQUEST from the
            # LangGraph client on tool turns (enable_thinking:false, temp 0-0.2),
            # NOT globally here, so conversational turns keep reasoning + normal temp.
            # Optimized KV cache layout for your hardware setup
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0"
        ]
        
        vllm_proc = subprocess.Popen(vllm_cmd, stdout=llama_log_file,
                                     stderr=llama_log_file,
                                     text=True)
        wait_for_vllm()

        print("\n[Launcher] 🚀 Starting Orion UI...")
        ui_proc = subprocess.Popen([sys.executable, "-m", TARGET_MODULE], stdout=sys.stdout, stderr=sys.stderr)
        
        try:
            code = ui_proc.wait()
        except KeyboardInterrupt:
            print("\n[Launcher] 🛑 Interrupted by user. Waiting for Orion to shut down gracefully...")
            try:
                ui_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                print("[Launcher] ⚠️ Orion did not shut down in time. Forcing kill.")
                ui_proc.kill()
            vllm_proc.kill()
            sys.exit(0)
        
        # If the process exits, check the code
        if code == RESTART_CODE:
            print("[Launcher] 🔄 Self-Reboot triggered by CTO. Cooling down VRAM...")
            vllm_proc.kill() 
            time.sleep(5)
            continue
        
        # Otherwise, exit naturally
        print("[Launcher] 🔻 Orion exited.")
        break

if __name__ == "__main__":
    run_orion()