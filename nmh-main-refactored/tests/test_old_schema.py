"""Schema migration SQLite v1: file app.py cũ chỉ có 2 cột users."""
import sqlite3
from pathlib import Path
import subprocess
import sys
import tempfile
import os


def test_upgrade_existing_sqlite_without_deleting_users():
    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "old.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL)")
            conn.execute("CREATE TABLE history (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, timestamp TEXT, sl REAL, sw REAL, pl REAL, pw REAL, species TEXT, confidence REAL)")
            conn.execute("INSERT INTO users (username, password_hash) VALUES ('legacy_user', '$2b$somebcrypt')")
        env = {**os.environ, "DATABASE_URL": f"sqlite:///{db}", "SECRET_KEY": "test-legacy-secret"}
        res = subprocess.run(
            [sys.executable, "-c", "from backend.database import init_database; init_database()"],
            cwd=Path(__file__).resolve().parent.parent, env=env, capture_output=True, text=True,
        )
        assert res.returncode == 0, res.stderr
        with sqlite3.connect(db) as conn:
            cols = [row[1] for row in conn.execute("PRAGMA table_info(users)")]
            assert "recovery_code_hash" in cols
            assert "token_version" in cols
            assert conn.execute("SELECT username FROM users").fetchall() == [("legacy_user",)]
