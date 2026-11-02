"""Kiểm tra request trước khi xử lý backend."""
from pydantic import BaseModel, Field, field_validator


class LoginInput(BaseModel):
    username: str = Field(min_length=3, max_length=20)
    password: str = Field(min_length=1)

    @field_validator("username")
    @classmethod
    def trim_username(cls, value: str) -> str:
        return value.strip()


class RegisterInput(LoginInput):
    password: str = Field(min_length=8)

    @field_validator("password")
    @classmethod
    def password_length_bytes(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Mật khẩu phải tối đa 72 byte UTF-8")
        return value


class ResetPasswordInput(BaseModel):
    username: str = Field(min_length=3, max_length=20)
    recovery_code: str = Field(min_length=20, max_length=128)
    new_password: str = Field(min_length=8)

    @field_validator("new_password")
    @classmethod
    def password_length_bytes(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Mật khẩu phải tối đa 72 byte UTF-8")
        return value


class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0, lt=100)
    sepal_width: float = Field(gt=0, lt=100)
    petal_length: float = Field(gt=0, lt=100)
    petal_width: float = Field(gt=0, lt=100)
