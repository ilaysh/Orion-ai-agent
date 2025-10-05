
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from ui.router import router as web_router

app = FastAPI(title="Orion v2")

# ---- CORS settings ----
# during dev we allow everything, later we’ll restrict to localhost only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can replace with ["http://localhost:8080", "http://127.0.0.1:8080"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Static files ----
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

# ---- API + WebSocket router ----
app.include_router(web_router)

if __name__ == "__main__":
    uvicorn.run(
        "runners.web_server:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )
