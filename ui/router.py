# ui/router.py
import asyncio
import base64
import contextlib
import os
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import asynccontextmanager
from orion_core.core import OrionCore

from system.telemetry import queries
from system.telemetry.aggregator import telemetry_start
from system.telemetry.telemetry import telemetry_summary

# 1. SETUP
TELEMETRY_DIR = Path("orion_core/system/telemetry")
TELEMETRY_LOG = TELEMETRY_DIR / "telemetry.log"
TELEMETRY_JSONL = TELEMETRY_DIR / "telemetry.jsonl"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# 2. INSTANCE
router = APIRouter()
orion = OrionCore()

# 3. LIFESPAN
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Router] 🔄 Starting Orion via lifespan hook...")
    await orion.start(run_init=True)
    yield
    print("[Router] 🔻 Orion shutting down...")
    await orion.shutdown()

# 4. APP DEFINITION
app = FastAPI(lifespan=lifespan)

# 5. ROUTES (Must be defined BEFORE include_router)
@router.get("/")
async def root():
    # Verify file exists to avoid 500 error
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Error: index.html not found in ui/static</h1>")
    return FileResponse(index_path)

@router.get("/telemetry")
async def telemetry_dashboard():
    # Fallback if dashboard.html missing
    dash_path = Path("orion_core/system/telemetry/dashboard.html")
    if not dash_path.exists():
         return HTMLResponse("<h1>Telemetry Dashboard Not Found</h1>")
    html = dash_path.read_text(encoding='utf-8')
    return HTMLResponse(html)

@router.get("/api/telemetry/pipeline")
async def get_pipeline():
    return JSONResponse(queries.pipeline_timeline())

@router.get("/api/telemetry/latest")
async def get_raw():
    return JSONResponse(queries.latest_raw())

@router.get("/api/telemetry/stats")
async def get_stats():
    return JSONResponse(queries.stats())

# WebSocket Logic
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
    
    # Track the active execution worker task
    active_core_task = None

    try:
        await ws.send_json({"type": "state", "state": orion.state.name})
        while True:
            msg = await ws.receive_json()
            t = msg.get("type")

            if t == "user_text":
                telemetry_start()
                text_input = msg.get("text", "")
                
                # Wrap execution in a non-blocking background task
                async def run_pipeline():
                    try:
                        await orion.handle_user_text(text_input)
                    except Exception as e:
                        print(f"[Router Fault] Pipeline error: {e}")
                    finally:
                        # Ensures telemetry logs dump immediately when execution finishes
                        summary = telemetry_summary()
                        if summary:
                            print(summary)

                active_core_task = asyncio.create_task(run_pipeline())

            elif t == "user_audio":
                telemetry_start()
                audio_chunk = base64.b64decode(msg["audio"])
                
                async def run_audio_pipeline():
                    await orion.handle_user_audio(audio_chunk)
                    print(telemetry_summary())
                    
                active_core_task = asyncio.create_task(run_audio_pipeline())

            elif t == "playback_finished":
                # The loop is awake and active, so this event can now be captured!
                orion.playback_finished()
                await ws.send_json({"type": "state", "state": "idle"})

    except WebSocketDisconnect:
        print("[Router] ⚠️ WebSocket disconnected")
    finally:
        pump_task.cancel()
        if active_core_task and not active_core_task.done():
            active_core_task.cancel()
        with contextlib.suppress(Exception):
            await pump_task

# 6. MOUNTS & INCLUDES (FINAL STEP)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(router)  # <--- CRITICAL: MUST BE LAST