"""Server render các template HTML đã tách khỏi mã Python."""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from backend.config import FRONTEND_DIR

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=str(FRONTEND_DIR / "templates"))

PAGE_CONFIG = {
    "/": ("index.html", "Trang chủ"),
    "/login": ("login.html", "Đăng nhập"),
    "/dataset": ("dataset.html", "Giới thiệu dữ liệu"),
    "/lab": ("lab.html", "Phân tích"),
    "/vector3d": ("vector3d.html", "Không gian 3D"),
    "/kernels": ("kernels.html", "Kernels"),
}


@router.get("/{page:path}", response_class=HTMLResponse)
def page(request: Request, page: str):
    # Mapped routes; never interpolate an arbitrary path into the template directory.
    path = f"/{page}" if page else "/"
    if path not in PAGE_CONFIG:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Không tìm thấy trang")
    template, title = PAGE_CONFIG[path]
    return templates.TemplateResponse(request=request, name=template, context={"title": title})
