"""JWT + băm mật khẩu, hỗ trợ kiểm tra hash bcrypt đời cũ."""
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config import ACCESS_TOKEN_EXPIRE_MINUTES, SECRET_KEY
from backend.database import get_db
from backend.models import User

security = HTTPBearer(auto_error=False)


def password_hash(password: str) -> str:
    # PBKDF2-HMAC-SHA256 có trong Python stdlib (600.000 vòng) và có salt độc lập.
    salt = secrets.token_bytes(16)
    iterations = 600_000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256${}${}${}".format(
        iterations,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, stored_hash: str) -> bool:
    if stored_hash.startswith("pbkdf2_sha256$"):
        try:
            _alg, n, salt, expected = stored_hash.split("$", 3)
            if not (100_000 <= int(n) <= 2_000_000):
                return False
            actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), base64.b64decode(salt), int(n))
            return hmac.compare_digest(actual, base64.b64decode(expected))
        except (ValueError, TypeError):
            return False
    if stored_hash.startswith(("$2a$", "$2b$", "$2y$")):
        # Tài khoản cũ từng được app.py hash bằng bcrypt.
        try:
            import bcrypt
            return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
        except (ImportError, ValueError):
            return False
    return False


def new_recovery_code() -> str:
    return secrets.token_urlsafe(24)


def hash_recovery_code(code: str) -> str:
    # Mã khôi phục sinh ngẫu nhiên 192 bit, SHA256 đủ khi DB bị lộ.
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def valid_recovery_code(code: str, hashed: str | None) -> bool:
    return bool(hashed) and hmac.compare_digest(hash_recovery_code(code), hashed)


def create_access_token(user: User) -> str:
    expires = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": str(user.id), "ver": user.token_version, "exp": expires, "iat": datetime.now(timezone.utc)},
        SECRET_KEY, algorithm="HS256",
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    error = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Phiên đăng nhập không hợp lệ hoặc đã hết hạn")
    if credentials is None:
        raise error
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        user_id = int(payload["sub"])
        version = payload.get("ver")
    except (jwt.PyJWTError, KeyError, ValueError, TypeError):
        raise error from None
    user = db.get(User, user_id)
    if not user or user.token_version != version:
        raise error
    return user
