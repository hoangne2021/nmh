-- Database local: app tự tạo bằng SQLAlchemy khi khởi động.
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(20) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    recovery_code_hash VARCHAR(128),
    token_version INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(20) NOT NULL,
    timestamp VARCHAR(25) NOT NULL,
    sl REAL NOT NULL,
    sw REAL NOT NULL,
    pl REAL NOT NULL,
    pw REAL NOT NULL,
    species VARCHAR(64) NOT NULL,
    confidence REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_history_username_id ON history (username, id);
