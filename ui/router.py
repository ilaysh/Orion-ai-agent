# ui/router.py
import asyncio
import base64
import contextlib
import os
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import asynccontextmanager
from orion_core.core import OrionCore

router = APIRouter()
orion = OrionCore()
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Router] 🔄 Starting Orion via lifespan hook...")
    asyncio.create_task(orion._on_init())
    yield
    print("[Router] 🔻 Orion shutting down...")
    await orion.shutdown()

app = FastAPI(lifespan=lifespan)
app.include_router(router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@router.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


async def _pump_events(ws: WebSocket):
    try:
        while True:
            evt = await orion.events.get()
            if isinstance(evt, dict):
                await ws.send_json(evt)
            else:
                await ws.send_json({"type": "log", "text": str(evt)})
    except Exception:
        return


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    pump_task = asyncio.create_task(_pump_events(ws))
    try:
        await ws.send_json({"type": "state", "state": orion.state.name})
        while True:
            msg = await ws.receive_json()
            t = msg.get("type")

            if t == "user_text":
                await orion.handle_user_text(msg.get("text", ""))

            elif t == "user_audio":
                audio_chunk = base64.b64decode(msg["audio"])
                await orion.handle_user_audio(audio_chunk)

            elif t == "playback_finished":
                orion.playback_finished()
                await ws.send_json({"type": "state", "state": "idle"})

            elif t == "diagnostic_audio":
                try:
                    raw = base64.b64decode(msg["audio"])
                    sr = int(msg.get("sampleRate", 16000))
                    os.makedirs("logs", exist_ok=True)
                    fname = f"logs/diag_{datetime.now():%Y%m%d_%H%M%S}.wav"
                    import wave
                    with wave.open(fname, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sr)
                        wf.writeframes(raw)
                    await ws.send_json({"type": "diagnostic_saved", "file": fname})
                except Exception as e:
                    await ws.send_json({"type": "diagnostic_error", "error": str(e)})

    except WebSocketDisconnect:
        print("[Router] ⚠️ WebSocket disconnected")
    finally:
        pump_task.cancel()
        with contextlib.suppress(Exception):
            await pump_task
