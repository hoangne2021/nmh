"""Cấu hình theo biến môi trường. Không lưu thông tin nhạy cảm trong Git."""
from pathlib import Path
import logging
import os
import secrets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(PROJECT_ROOT / "model" / "svm_model.pkl")))

ON_RENDER = bool(os.getenv("RENDER")) or bool(os.getenv("RENDER_SERVICE_ID"))

# Dữ liệu local nằm trong /data/ và sẽ được .gitignore loại khỏi repository.
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "history.db"
DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}").strip()
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL[len("postgres://"):]
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL[len("postgresql://"):]

SECRET_KEY = os.getenv("SECRET_KEY", "").strip()
if ON_RENDER and (not os.getenv("DATABASE_URL") or not SECRET_KEY):
    raise RuntimeError("Render cần DATABASE_URL (PostgreSQL) và SECRET_KEY trong Environment.")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_urlsafe(48)
    logging.warning("SECRET_KEY chưa đặt: tạo khóa tạm dùng tại máy local; JWT sẽ hết hiệu lực sau lần khởi động mới.")

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7
