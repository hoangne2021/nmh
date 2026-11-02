"""Schema SQL dùng cho người dùng và lịch sử nhận diện."""
from sqlalchemy import Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(20), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    recovery_code_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    token_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")


class History(Base):
    __tablename__ = "history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Giữ username (không đổi thành user_id) để đọc được bảng lịch sử SQLite đời cũ.
    username: Mapped[str] = mapped_column(String(20), nullable=False)
    timestamp: Mapped[str] = mapped_column(String(25), nullable=False)
    sl: Mapped[float] = mapped_column(Float, nullable=False)
    sw: Mapped[float] = mapped_column(Float, nullable=False)
    pl: Mapped[float] = mapped_column(Float, nullable=False)
    pw: Mapped[float] = mapped_column(Float, nullable=False)
    species: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (Index("ix_history_username_id", "username", "id"),)
