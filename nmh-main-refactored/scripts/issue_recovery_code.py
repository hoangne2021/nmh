"""Chỉ quản trị viên DB dùng để cấp mã khôi phục cho tài khoản cũ.

    DATABASE_URL='...' SECRET_KEY='...' python scripts/issue_recovery_code.py --username ten_user

Mã hiện một lần trên terminal. Không đưa mã vào log hoặc GitHub.
"""
import argparse
from pathlib import Path
import sys

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.database import SessionLocal, init_database  # noqa: E402
from backend.models import User  # noqa: E402
from backend.security import new_recovery_code, hash_recovery_code  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--username", required=True)
    args = p.parse_args()
    init_database()
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.username == args.username))
        if not user:
            raise SystemExit("Không tìm thấy tài khoản")
        recovery_code = new_recovery_code()
        user.recovery_code_hash = hash_recovery_code(recovery_code)
        user.token_version += 1
        db.commit()
    print("Tên tài khoản:", args.username)
    print("Mã khôi phục mới (chỉ hiển thị 1 lần):", recovery_code)
    print("Mọi JWT của tài khoản đã được vô hiệu hóa.")


if __name__ == "__main__":
    main()
