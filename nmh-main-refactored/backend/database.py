"""SQLAlchemy: SQLite (phát triển), PostgreSQL (Render)."""
from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import DATABASE_URL


class Base(DeclarativeBase):
    pass


connect_args = {"check_same_thread": False, "timeout": 20} if DATABASE_URL.startswith("sqlite:") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    with SessionLocal() as db:
        yield db


def init_database() -> None:
    """Tạo bảng và bổ sung cột cần thiết cho schema SQLite cũ.

    Các lệnh nâng cấp này chỉ ADD COLUMN, không xóa tài khoản hoặc lịch sử.
    Với dự án lớn nên thay thế bằng Alembic migrations.
    """
    from backend.models import User, History  # noqa: F401 - đăng ký model

    Base.metadata.create_all(bind=engine)
    fields = {col["name"] for col in inspect(engine).get_columns("users")}
    with engine.begin() as conn:
        if "recovery_code_hash" not in fields:
            conn.execute(text("ALTER TABLE users ADD COLUMN recovery_code_hash VARCHAR(128)"))
        if "token_version" not in fields:
            conn.execute(text("ALTER TABLE users ADD COLUMN token_version INTEGER NOT NULL DEFAULT 0"))
