# runners/webview.py
import webview, threading, time, requests
import uvicorn
import sys
import os

# Add project root to sys.path to find modules
sys.path.append(os.getcwd())

# CRITICAL FIX: Import from ui.router, NOT runners.web_server
from ui.router import app 

def run_server():
    # host="0.0.0.0" allows access from network if needed, but 127.0.0.1 is safer for local
    uvicorn.run(app, host="127.0.0.1", port=8080,reload=True)

if __name__ == "__main__":
    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    # Wait for server
    for _ in range(20):
        try:
            requests.get("http://127.0.0.1:8080/")
            break
        except:
            time.sleep(0.5)

    # Open Orion UI
    webview.create_window("Orion v2 Assistant", "http://127.0.0.1:8080/")
    webview.start(gui="qt")