# ui/router.py
import asyncio
import base64
import contextlib
import os
import time
from datetime import datetime

from pathlib import Path
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import FileResponse
from orion_core.core import OrionCore

router = APIRouter()
orion = OrionCore()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start Orion once the event loop is alive (modern FastAPI style)."""
    print("[Router] 🔄 Starting Orion via lifespan hook...")
    asyncio.create_task(orion._on_init())
    yield
    print("[Router] 🔻 Orion shutting down...")
    await orion.shutdown()

# Create the app and attach router
app = FastAPI(lifespan=lifespan)
app.include_router(router)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@router.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


async def _pump_events(ws: WebSocket):
    """Background task to forward core events (state changes) to the UI."""
    try:
        while True:
            evt = await orion.events.get()
            await ws.send_json(evt)
    except Exception:
        # socket closed or task cancelled
        return


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    pump_task = asyncio.create_task(_pump_events(ws))

    try:
        # Send initial state to UI
        print(f"[router] state:{orion.state.name} at {datetime.now().strftime('%H:%M:%S')}")
        await ws.send_json({"type": "state", "state": orion.state.name})

        while True:
            msg = await ws.receive_json()
            msg_type = msg.get("type")


            # --- user typed text (manual input from UI) ---
            if msg_type == "user_text":
                user_text = msg.get("text", "")
                reply = await orion.handle_user_text(user_text)
                if reply and reply.get("type") == "orion_reply":
                    await ws.send_json({"type": "state", "state": "speaking"})
                if reply:
                    await ws.send_json(reply)

            # --- user streamed microphone audio ---
            elif msg_type == "user_audio":
                if orion.state.name.lower() == "speaking":
                    continue  # skip while speaking
                
                audio_chunk = base64.b64decode(msg["audio"])
                reply = await orion.handle_user_audio(audio_chunk)
                if reply and reply.get("type") == "orion_reply":
                    print(f"[router] return tts {datetime.now().strftime('%H:%M:%S')}")
                    await ws.send_json({"type": "state", "state": "speaking"})
                if reply:
                    await ws.send_json(reply)

            # --- playback finished ---
            elif msg_type == "playback_finished":
                orion.playback_finished()
                await ws.send_json({"type": "state", "state": "idle"})

            elif msg.get("type") == "diagnostic_audio":
                try:
                    b64 = msg.get("audio", "")
                    sr = int(msg.get("sampleRate", 16000))
                    raw = base64.b64decode(b64)
                    os.makedirs("logs", exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"logs/diag_js_{ts}.wav"
                    import struct
                    import wave
                    import io
                    buf = io.BytesIO()
                    with wave.open(buf, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(4)      # float32 = 4 bytes
                        wf.setframerate(sr)
                        wf.writeframes(raw)
                    data = buf.getvalue()

                    with open(fname, "wb") as f:
                        f.write(data)
                    # Acknowledge back (and optionally echo to play via blob URL client already set)
                    await ws.send_json({
                        "type": "diagnostic_saved",
                        "file": fname,
                        "sampleRate": sr,
                        "bytes": len(raw),
                    })
                except Exception as e:
                    await ws.send_json({
                        "type": "diagnostic_error",
                        "error": str(e),
                    })

    except WebSocketDisconnect:
        print("[Router] ⚠️ WebSocket disconnected")
    finally:
        pump_task.cancel()
        with contextlib.suppress(Exception):
            await pump_task
