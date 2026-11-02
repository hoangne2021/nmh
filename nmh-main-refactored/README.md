# Botanical Lab Pro – Iris SVM (đã tách frontend / backend)

Dự án phân loại hoa Iris bằng **Support Vector Machine**. Phiên bản này giữ nguyên các trang giao diện cơ bản nhưng tách độc lập HTML, CSS, JavaScript, Python API và tầng SQL.

## 1. Cấu trúc thư mục

```text
nmh-main-refactored/
├── frontend/
│   ├── templates/              # login, index, dataset, lab, vector3d, kernels, base.html
│   └── static/
│       ├── css/                # base.css, login.css, particles.css
│       ├── js/                 # common, login, dataset, lab, vector3d, kernels...
│       └── img/                # ảnh các loài Iris gốc
├── backend/
│   ├── main.py                 # FastAPI entrypoint, mount static/templates
│   ├── config.py               # DATABASE_URL, SECRET_KEY
│   ├── database.py             # SQLAlchemy + tạo bảng/nâng cấp cột
│   ├── models.py               # Model SQL: users, history
│   ├── schemas.py              # Pydantic input validation
│   ├── security.py             # Password hash, JWT, recovery code
│   ├── routers/
│   │   ├── auth.py             # Đăng ký, đăng nhập, khôi phục mật khẩu
│   │   ├── iris.py             # Predict, history, PCA, kernel
│   │   └── pages.py            # Render giao diện Jinja2
│   └── services/iris_ml.py     # Model SVM, thống kê, kernel
├── database/                   # SQL tham khảo cho PostgreSQL và SQLite
├── model/svm_model.pkl         # Model pickle bản gốc (tự train lại nếu phiên bản sklearn khác)
├── scripts/                   # Nhập DB cũ; cấp mã khôi phục user cũ
├── tests/                     # Test API + migration SQLite
├── data/                      # SQLite local, file DB đã bị Git bỏ qua
├── docs/da1.pdf               # File báo cáo PDF ban đầu
├── app.py                     # Shim tương thích lệnh uvicorn app:app cũ
├── train.py                   # Train lại SVM
├── requirements.txt
├── render.yaml                # Blueprint Render gồm Web + PostgreSQL
└── Procfile                   # Start command tùy chọn
```

**Frontend và backend đã tách file, nhưng vẫn dùng chung một Render Web Service/URL**. FastAPI phục vụ HTML/CSS/JS và API dưới cùng domain, nên không cần CORS `*`, không phải mua hoặc tạo thêm dịch vụ frontend. Nếu sau này dùng React/Vite hoặc đưa frontend lên GitHub Pages, cần bổ sung `API_BASE_URL` và cấu hình CORS theo origin cụ thể.

## 2. Chạy trên máy local

Yêu cầu Python 3.11–3.13 (khuyến nghị Python 3.12).

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -c "import secrets; print(secrets.token_urlsafe(48))"  # tạo khóa JWT
export SECRET_KEY='MA_VUA_TAO'      # Windows PowerShell: $env:SECRET_KEY='MA_VUA_TAO'
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Mở `http://127.0.0.1:8000/login` để đăng ký. Không khai báo `DATABASE_URL` thì database local nằm tại `data/history.db` (SQLite). File `.db` không được commit lên Git.

**API kiểm tra:** `/docs` (Swagger), `/health` (database check). Các API khác giữ URL cũ: `/auth/register`, `/auth/login`, `/auth/forgot-password`, `/auth/me`, `/predict`, `/api/history`, `/api/compare_kernels`, `/api/dataset`, `/api/pca`, `/api/kernel_scores`.

> Lưu ý: cần internet trên trình duyệt cho Tailwind CDN, Chart.js, Plotly và Google Fonts như phiên bản gốc.

## 3. Đưa lên GitHub và chạy Render

### A. Render Blueprint (dự án Render mới)

1. Giải nén dự án. **Đưa toàn bộ nội dung nằm trong thư mục `nmh-main-refactored` lên thư mục gốc repository GitHub**, giữ nguyên các folder `frontend`, `backend`, `model`, `database` và file `render.yaml`.
2. Trong Render Dashboard chọn **New → Blueprint**, kết nối GitHub repo, xác nhận các tài nguyên mà `render.yaml` mô tả.
3. Blueprint tự tạo web service `botanical-iris-svm`, database PostgreSQL `botanical-iris-postgres`, liên kết `DATABASE_URL` từ DB và tạo `SECRET_KEY` ngẫu nhiên.
4. Sau khi web service ở trạng thái Live, truy cập `https://<ten-service>.onrender.com/login`, đăng ký tài khoản, và kiểm tra `/health` trả `{"status":"ok"}`.

### B. Đã có sẵn Render Web Service gắn với GitHub (trường hợp bạn đang dùng)

Không cần tạo service thứ hai. Thay mã nguồn trong **repo hiện đang kết nối** bằng bộ mã này (để nguyên root repo), vào **Render → Web Service → Settings** và chỉnh:

| Cài đặt | Giá trị |
|---|---|
| Language / Runtime | Python 3 |
| Root Directory | Để trống nếu đã đưa nội dung lên gốc repo |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/health` |

Trong **Render → Environment**, tạo:

- `SECRET_KEY`: mã ngẫu nhiên đủ dài (chạy `python -c "import secrets; print(secrets.token_urlsafe(48))"`). Đừng thay mã này thường xuyên vì sẽ đăng xuất các tài khoản đang đăng nhập.
- `DATABASE_URL`: URL kết nối PostgreSQL **Internal Database URL** của Render PostgreSQL (nếu database cùng vùng) hoặc URL PostgreSQL từ nhà cung cấp khác. URL có dạng `postgresql://user:password@host/database`, driver được app tự chuẩn hóa.
- `PYTHON_VERSION` (không bắt buộc): `3.12.9`.

**Tuyệt đối không bỏ qua DATABASE_URL trên Render**: dự án sẽ cố ý báo lỗi cấu hình thay vì tự lưu tài khoản vào SQLite tạm rồi mất dữ liệu. Sau khi deploy lại, kiểm tra `/health`, thử đăng ký/đăng nhập, và thử restart để xác nhận tài khoản còn tồn tại.

### C. Giới hạn dữ liệu trên gói Free

Render Free Web Service dùng filesystem tạm: file SQLite không giữ được khi restart/spin down/redeploy. PostgreSQL **Free** ở Render cũng **hết hạn sau 30 ngày**; cần nâng cấp/cài DB lâu dài và sao lưu dữ liệu nếu muốn hệ thống sử dụng lâu dài. Không dùng gói miễn phí cho thông tin tài khoản quan trọng mà không có kế hoạch sao lưu/di chuyển.

## 4. Đăng ký / SQL / khôi phục mật khẩu

- Bảng `users`: id, username (unique), password_hash, recovery_code_hash, token_version.
- Bảng `history`: các lần dự đoán; giới hạn 50 bản ghi truy vấn gần nhất, phân tách theo username.
- Mật khẩu đăng ký mới dùng **PBKDF2-HMAC-SHA256** với salt ngẫu nhiên 16 byte và 600.000 vòng; tương thích việc kiểm tra **bcrypt** cho tài khoản tạo từ app.py cũ (đã có phụ thuộc `bcrypt`).
- JWT ký bằng `SECRET_KEY` lấy từ biến môi trường; API xác thực mỗi request dựa trên DB. Khi reset password sẽ tăng `token_version` để thu hồi JWT cũ.
- Không còn **đặt lại mật khẩu bằng phép cộng** do người dùng tự gửi kết quả. Thay vào đó hệ thống tạo **mã khôi phục dùng riêng cho tài khoản**, chỉ hiển thị **một lần** sau đăng ký/đổi mật khẩu, mã hash lưu trong database. Mã cũ bị vô hiệu hóa sau khi dùng. Người dùng phải lưu mã; nếu mất cả mật khẩu và mã thì không thể tự phục hồi bằng chức năng hiện tại (chưa tích hợp gửi email).

### Dữ liệu/tài khoản từ SQLite cũ

ZIP bạn gửi **không có `history.db`**, nên không thể lấy lại tài khoản cũ từ ZIP này. Nếu còn file DB thật trên máy/server, sao lưu lại rồi chạy:

```bash
export DATABASE_URL='postgresql://USER:PASSWORD@HOST:5432/DB'
export SECRET_KEY='KHOA_CUA_BAN'
python scripts/import_sqlite.py --source /path/to/history.db
```

Script nhập username, hash mật khẩu và lịch sử, không cần biết mật khẩu gốc. Để cấp mã khôi phục cho tài khoản đời cũ (chưa có mã), quản trị viên DB chạy:

```bash
python scripts/issue_recovery_code.py --username ten_tai_khoan_cu
```

Sao chép mã mới chỉ hiển thị trên terminal, gửi riêng qua kênh đã xác minh cho chủ tài khoản. Hai script này cần truy cập DB và chỉ dành cho quản trị viên. Không chèn username/password thật vào code/README hoặc log.

## 5. Test

```bash
pip install -r requirements-dev.txt
pytest -q
```

Test gồm đăng ký, đăng nhập, bảo vệ API, dự đoán Iris, lịch sử, reset mã bảo mật, vô hiệu hóa JWT cũ và nâng cấp bảng SQLite từ phiên bản trước.

## 6. Vận hành và lưu ý

- Mọi trang HTML phục vụ từ `frontend/templates` với HTML/CSS/JS đã tách riêng. Các file JS trong `frontend/static/js/`; backend không chứa HTML/JS/CSS nội tuyến.
- API `POST /predict` vẫn trả mảng `confidences` như cũ, nhưng giá trị là **điểm tương đồng heuristic theo tâm cụm**, không phải xác suất thống kê của SVM. Kết quả so sánh kernel dùng `predict_proba` của các SVC riêng.
- Model gốc `.pkl` được giữ lại; nếu phiên bản scikit-learn không tương thích, ứng dụng tự huấn luyện lại SVC linear trên `load_iris()` để tránh nạp pickle khác phiên bản.
- Đây là phiên bản phù hợp dự án học tập/demo. Khi mở đăng ký công khai quy mô lớn, nên bổ sung hạn chế số lượt đăng nhập (rate limiting), bảo vệ chống spam và giải pháp khôi phục tài khoản đã xác minh bằng email.

Tài liệu tham khảo Render: [FastAPI deployment](https://render.com/docs/deploy-fastapi), [Blueprint spec](https://render.com/docs/blueprint-spec), [Free-tier limitations](https://render.com/docs/free).
