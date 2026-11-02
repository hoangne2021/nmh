"""Chuyển users/history từ file history.db cũ vào DATABASE_URL hiện tại.

Ví dụ:
  DATABASE_URL='postgresql://...' SECRET_KEY='...' python scripts/import_sqlite.py --source ./history.db

Không xóa dữ liệu đích. Chạy khi đã sao lưu database.
"""
import argparse
from pathlib import Path
import sqlite3
import sys

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.database import SessionLocal, engine, init_database  # noqa: E402
from backend.models import History, User  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Nhập tài khoản/lịch sử từ file history.db đời cũ")
    parser.add_argument("--source", required=True, type=Path, help="Đường dẫn history.db cũ")
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_file():
        raise SystemExit(f"Không tồn tại: {source}")
    if engine.url.drivername.startswith("sqlite") and Path(engine.url.database).resolve() == source:
        raise SystemExit("Source và destination là một database! Không cần nhập.")
    init_database()
    old = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    old.row_factory = sqlite3.Row
    try:
        users = old.execute("SELECT username, password_hash FROM users ORDER BY id").fetchall()
        records = old.execute("SELECT username, timestamp, sl, sw, pl, pw, species, confidence FROM history ORDER BY id").fetchall()
    except sqlite3.DatabaseError as exc:
        raise SystemExit(f"Không đọc được schema nguồn: {exc}") from exc
    added_users = added_history = 0
    with SessionLocal() as db:
        for row in users:
            if db.scalar(select(User.id).where(User.username == row["username"])) is not None:
                continue
            db.add(User(username=row["username"], password_hash=row["password_hash"], token_version=0))
            added_users += 1
        for row in records:
            exists = db.scalar(select(History.id).where(
                History.username == row["username"],
                History.timestamp == row["timestamp"],
                History.sl == row["sl"], History.sw == row["sw"],
                History.pl == row["pl"], History.pw == row["pw"],
                History.species == row["species"],
            ))
            if exists is not None:
                continue
            db.add(History(**dict(row)))
            added_history += 1
        db.commit()
    old.close()
    print(f"Đã nhập {added_users} tài khoản và {added_history} bản ghi lịch sử.")
    print("Mật khẩu hash bcrypt cũ vẫn sử dụng được. Tài khoản cũ chưa có mã khôi phục:")
    print("Dùng scripts/issue_recovery_code.py để phát mã riêng (chỉ admin chạy).")


if __name__ == "__main__":
    main()
