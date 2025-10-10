# ui/router.py
import asyncio
import base64
import contextlib
from pathlib import Path
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect,FastAPI
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
    """
    Background task to forward core events (state changes) to the UI.
    """
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
    pump_task = None
    try:
        # start in idle; wake thread will flip to listening when fired
        await ws.send_json({"type": "state", "state": orion.state.name})
        pump_task = asyncio.create_task(_pump_events(ws))

        while True:
            msg = await ws.receive_json()

            # --- user typed text ---
            if msg.get("type") == "user_text":
                async for reply in orion.handle_user_text(msg.get("text", "")):
                    # when reply contains audio, switch UI to speaking
                    if reply.get("type") == "orion_reply":
                        await ws.send_json({"type": "state", "state": "speaking"})
                    await ws.send_json(reply)

            # --- user streamed audio ---
            elif msg.get("type") == "user_audio":
                # don’t let UI flood the server while speaking
                if orion.state == "speaking":
                    continue
                # base64 -> bytes
                audio_chunk = base64.b64decode(msg["audio"])
                async for reply in orion.handle_user_audio(audio_chunk):
                    if reply.get("type") == "orion_reply":
                        await ws.send_json({"type": "state", "state": "speaking"})
                    await ws.send_json(reply)

            # --- browser completed playback ---
            elif msg.get("type") == "playback_finished":
                orion.playback_finished()
                # the core will emit a state event, but we also push immediately
                await ws.send_json({"type": "state", "state": "idle"})

    except WebSocketDisconnect:
        pass
    finally:
        if pump_task:
            pump_task.cancel()
            with contextlib.suppress(Exception):
                await pump_task
