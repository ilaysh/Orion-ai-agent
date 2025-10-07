import webview, threading, time, requests
import uvicorn
from runners.web_server import app
from orion_core.core import OrionCore

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8080)

if __name__ == "__main__":
    core = OrionCore()
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
