
import asyncio
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from ui.router import router as web_router, orion

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[WebServer] 🔄 lifespan start")
    asyncio.create_task(orion._on_init())   # run init when loop is alive
    yield
    print("[WebServer] 🔻 lifespan stop")
    await orion.shutdown()

app = FastAPI(title="Orion v2", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="ui/static"), name="static")
app.include_router(web_router)

if __name__ == "__main__":
    uvicorn.run(
        "runners.web_server:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )
