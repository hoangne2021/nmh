"""ASGI entrypoint: uvicorn backend.main:app --host 0.0.0.0 --port $PORT"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from sqlalchemy import text

from backend.config import FRONTEND_DIR
from backend.database import SessionLocal, init_database
from backend.routers import auth, iris, pages
from backend.services.iris_ml import get_state


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_database()
    get_state()
    yield


app = FastAPI(title="Botanical Lab Pro – Iris SVM API", version="10.0.0", lifespan=lifespan)
# Frontend and backend are served on the same origin: no unsafe wildcard CORS required.
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")
app.include_router(auth.router)
app.include_router(iris.router)


@app.get("/health", tags=["Operations"])
def health():
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception:
        return JSONResponse(status_code=503, content={"status": "database_unavailable"})


# Must be last: this is a frontend catch-all with an explicit allowlist.
app.include_router(pages.router)
