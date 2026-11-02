"""Công cụ chuyển database cũ sang database mới không xóa/nhân đôi tài khoản."""
from pathlib import Path
import os
import sqlite3
import subprocess
import sys
import tempfile


def test_import_users_and_history_idempotent():
    project = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        old, new = path / 'old.db', path / 'new.db'
        with sqlite3.connect(old) as conn:
            conn.executescript('''
              CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL);
              CREATE TABLE history (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, timestamp TEXT, sl REAL, sw REAL, pl REAL, pw REAL, species TEXT, confidence REAL);
              INSERT INTO users (username, password_hash) VALUES ('olduser', '$2b$fake-bcrypt-hash');
              INSERT INTO history (username,timestamp,sl,sw,pl,pw,species,confidence)
                  VALUES ('olduser','2026-09-01 12:00:00',5.1,3.4,1.4,0.3,'Iris Setosa',94.0);
            ''')
        env = {**os.environ, 'DATABASE_URL': f'sqlite:///{new}', 'SECRET_KEY': 'test-migration-123'}
        for i in range(2):
            res = subprocess.run(
                [sys.executable, 'scripts/import_sqlite.py', '--source', str(old)],
                cwd=project, env=env, capture_output=True, text=True, timeout=25,
            )
            assert res.returncode == 0, res.stderr
            assert ('1 tài khoản' if i == 0 else '0 tài khoản') in res.stdout
        with sqlite3.connect(new) as conn:
            assert conn.execute('SELECT username, password_hash FROM users').fetchall() == [('olduser', '$2b$fake-bcrypt-hash')]
            assert conn.execute('SELECT count(*) FROM history').fetchone()[0] == 1
