"""Đăng ký, đăng nhập, khôi phục mật khẩu bằng mã một lần."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.schemas import LoginInput, RegisterInput, ResetPasswordInput
from backend.security import (
    create_access_token, get_current_user, hash_recovery_code,
    new_recovery_code, password_hash, valid_recovery_code, verify_password,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", status_code=201)
def register(payload: RegisterInput, db: Session = Depends(get_db)):
    username = payload.username.strip()
    if len(username) < 3:
        raise HTTPException(status_code=422, detail="Tên đăng nhập cần ít nhất 3 ký tự")
    if db.scalar(select(User).where(User.username == username)):
        raise HTTPException(status_code=409, detail="Tên đăng nhập đã tồn tại")
    recovery_code = new_recovery_code()
    user = User(
        username=username,
        password_hash=password_hash(payload.password),
        recovery_code_hash=hash_recovery_code(recovery_code),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Tên đăng nhập đã tồn tại") from None
    # Recovery code CHỈ hiển thị một lần và không được ghi vào log.
    return {"message": "Đăng ký thành công", "recovery_code": recovery_code}


@router.post("/login")
def login(payload: LoginInput, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.username == payload.username.strip()))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Sai tên đăng nhập hoặc mật khẩu")
    token = create_access_token(user)
    return {"token": token, "token_type": "bearer", "username": user.username}


@router.post("/forgot-password")
def reset_password(payload: ResetPasswordInput, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.username == payload.username.strip()))
    if not user or not valid_recovery_code(payload.recovery_code.strip(), user.recovery_code_hash):
        # Không dùng phép toán trình duyệt để thay mật khẩu tài khoản bất kỳ.
        raise HTTPException(status_code=401, detail="Tên đăng nhập hoặc mã khôi phục không đúng")
    if verify_password(payload.new_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Mật khẩu mới phải khác mật khẩu cũ")
    new_code = new_recovery_code()
    user.password_hash = password_hash(payload.new_password)
    user.recovery_code_hash = hash_recovery_code(new_code)
    user.token_version += 1   # vô hiệu hóa mọi token đăng nhập cũ
    db.commit()
    return {"message": "Đổi mật khẩu thành công", "recovery_code": new_code}


@router.get("/me")
def whoami(user: User = Depends(get_current_user)):
    return {"username": user.username}
