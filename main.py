# main.py
import uvicorn
from fastapi import FastAPI
from ui import router

app = FastAPI()
app.include_router(router.router)

if __name__ == "__main__":
    print("🚀 Orion v2 server starting at http://localhost:8080")
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
