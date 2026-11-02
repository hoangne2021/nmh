-- Tham khảo: backend tự tạo bảng bằng SQLAlchemy khi server khởi động.
-- Không chạy nếu bảng đã có dữ liệu và bạn không hiểu cơ chế migration.
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(20) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    recovery_code_hash VARCHAR(128),
    token_version INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_users_username ON users (username);

CREATE TABLE IF NOT EXISTS history (
    id SERIAL PRIMARY KEY,
    username VARCHAR(20) NOT NULL,
    timestamp VARCHAR(25) NOT NULL,
    sl DOUBLE PRECISION NOT NULL,
    sw DOUBLE PRECISION NOT NULL,
    pl DOUBLE PRECISION NOT NULL,
    pw DOUBLE PRECISION NOT NULL,
    species VARCHAR(64) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_history_username_id ON history (username, id);
