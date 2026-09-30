import os
import math
import sqlite3
import bcrypt
import jwt
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import joblib

# ==========================================
# 1. CẤU HÌNH BẢO MẬT & JWT
# ==========================================
SECRET_KEY = "iris-pro-super-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # Token sống 7 ngày

security = HTTPBearer()

def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise HTTPException(status_code=401, detail="Token không hợp lệ")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Phiên đăng nhập đã hết hạn")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Xác thực thất bại")

# ==========================================
# 2. DB LỊCH SỬ, NGƯỜI DÙNG & MÔ HÌNH
# ==========================================
DB_PATH = "history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT, timestamp TEXT, sl REAL, sw REAL, pl REAL, pw REAL,
                  species TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

init_db()

MODEL_PATH = "svm_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# Dữ liệu Iris thật & Kernel 
KERNEL_NAMES = ["linear", "rbf", "poly", "sigmoid"]
KERNEL_MODELS, KERNEL_SCORES, PCA_POINTS, DATASET_STATS = {}, {}, None, None
try:
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.decomposition import PCA
    _iris = load_iris()
    _X, _y = _iris.data, _iris.target
    for _k in KERNEL_NAMES:
        KERNEL_MODELS[_k] = SVC(kernel=_k, probability=True, random_state=42).fit(_X, _y)
        KERNEL_SCORES[_k] = round(float(cross_val_score(SVC(kernel=_k), _X, _y, cv=5).mean()) * 100, 1)
    _p = PCA(n_components=3).fit(_X)
    _c = _p.transform(_X)
    PCA_POINTS = {"points": [{"x": float(a), "y": float(b), "z": float(c), "cls": int(t)} for (a, b, c), t in zip(_c, _y)],
                  "variance": [round(float(v) * 100, 1) for v in _p.explained_variance_ratio_]}
    DATASET_STATS = {"features": ["Sepal L", "Sepal W", "Petal L", "Petal W"],
                     "species": [{"name": n, "count": int((_y == i).sum()), "mean": [round(float(v), 2) for v in _X[_y == i].mean(axis=0)], "min": [round(float(v), 1) for v in _X[_y == i].min(axis=0)], "max": [round(float(v), 1) for v in _X[_y == i].max(axis=0)]} for i, n in enumerate(["Setosa", "Versicolor", "Virginica"])]}
except ImportError: 
    pass

app = FastAPI(title="Iris AI Classifier Pro", version="9.5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ==========================================
# 3. SCHEMA DỮ LIỆU
# ==========================================
class UserAuth(BaseModel):
    username: str = Field(..., min_length=3, max_length=20)
    password: str = Field(..., min_length=6)

class ForgotPassAuth(BaseModel):
    username: str
    num1: int
    num2: int
    answer: int
    new_password: str = Field(..., min_length=6)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

SPECIES_METADATA = {0: {"name": "Iris Setosa", "color": "#ea580c"}, 1: {"name": "Iris Versicolor", "color": "#7c3aed"}, 2: {"name": "Iris Virginica", "color": "#0d9488"}}
CENTROIDS = [[5.006, 3.428, 1.462, 0.246], [5.936, 2.770, 4.260, 1.326], [6.588, 2.974, 5.552, 2.026]]

# ==========================================
# 4. AUTHENTICATION APIs
# ==========================================
@app.post("/auth/register")
def register(user: UserAuth):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ?", (user.username,))
    if cur.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Tên đăng nhập đã tồn tại!")
    
    cur.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                (user.username, get_password_hash(user.password)))
    conn.commit()
    conn.close()
    return {"message": "Đăng ký thành công!"}

@app.post("/auth/login")
def login(user: UserAuth):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username = ?", (user.username,))
    row = cur.fetchone()
    conn.close()

    if not row or not verify_password(user.password, row[0]):
        raise HTTPException(status_code=401, detail="Sai tên đăng nhập hoặc mật khẩu!")
    
    token = create_access_token({"sub": user.username})
    return {"token": token, "username": user.username}

@app.post("/auth/forgot-password")
def forgot_password(data: ForgotPassAuth):
    if data.num1 + data.num2 != data.answer:
        raise HTTPException(status_code=400, detail="Câu trả lời bảo mật không chính xác!")
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username = ?", (data.username,))
    row = cur.fetchone()
    
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Tài khoản không tồn tại!")
    
    if verify_password(data.new_password, row[0]):
        conn.close()
        raise HTTPException(status_code=400, detail="Mật khẩu mới không được trùng mật khẩu cũ!")
    
    cur.execute("UPDATE users SET password_hash = ? WHERE username = ?", 
                (get_password_hash(data.new_password), data.username))
    conn.commit()
    conn.close()
    return {"message": "Đổi mật khẩu thành công!"}

@app.get("/auth/me")
def verify_me(username: str = Depends(get_current_user)):
    return {"username": username}

# ==========================================
# 5. IRIS APIs (ĐƯỢC BẢO VỆ BỞI JWT)
# ==========================================
@app.post("/predict")
def predict(data: IrisInput, username: str = Depends(get_current_user)):
    if model is None: raise HTTPException(status_code=500, detail="SVM Model missing.")
    feat = [data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]
    pred = int(model.predict([feat])[0])
    
    dists = [math.sqrt(sum((feat[i] - CENTROIDS[c][i]) ** 2 for i in range(4))) for c in range(3)]
    inv_dists = [1.0 / (d + 1e-5) for d in dists]
    inv_dists[pred] *= 1.8
    total = sum(inv_dists)
    conf = [round((v / total) * 100, 1) for v in inv_dists]
    meta = SPECIES_METADATA[pred]

    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO history (username, timestamp, sl, sw, pl, pw, species, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                 (username, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), *feat, meta["name"], conf[pred]))
    conn.commit()
    conn.close()
    return {"prediction": meta["name"], "confidences": conf, "features": feat, "color": meta["color"]}

@app.get("/api/history")
def get_history(username: str = Depends(get_current_user)):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT timestamp, sl, sw, pl, pw, species, confidence FROM history WHERE username=? ORDER BY id DESC LIMIT 50", (username,)).fetchall()
    conn.close()
    return [{"timestamp": r[0], "sl": r[1], "sw": r[2], "pl": r[3], "pw": r[4], "species": r[5], "confidence": r[6]} for r in rows]

@app.delete("/api/history")
def clear_history(username: str = Depends(get_current_user)):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM history WHERE username=?", (username,))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.post("/api/compare_kernels")
def compare_kernels(data: IrisInput, username: str = Depends(get_current_user)):
    feat = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    out = []
    for k in KERNEL_NAMES:
        probs = KERNEL_MODELS[k].predict_proba(feat)[0]
        idx = int(probs.argmax())
        out.append({"kernel": k, "prediction": SPECIES_METADATA[idx]["name"], "color": SPECIES_METADATA[idx]["color"], "confidence": round(float(probs[idx]) * 100, 1), "accuracy": KERNEL_SCORES[k]})
    return out

@app.get("/api/dataset"); def dataset_stats(): return DATASET_STATS
@app.get("/api/pca"); def pca_points(): return PCA_POINTS
@app.get("/api/kernel_scores"); def kernel_scores(): return KERNEL_SCORES


# ==========================================
# JS PARTICLES CONFIG (Sử dụng chung)
# ==========================================
PARTICLES_JS = """
<script src="https://cdn.jsdelivr.net/npm/tsparticles@2.12.0/tsparticles.bundle.min.js"></script>
<script>
    function initParticles() {
        tsParticles.load("tsparticles", {
            background: { color: { value: "transparent" } },
            particles: {
                number: { value: 60, density: { enable: true, value_area: 800 } },
                color: { value: ["#ea580c", "#7c3aed", "#0d9488"] },
                shape: { type: "circle" },
                opacity: { value: 0.6, random: true },
                size: { value: 4, random: true },
                links: { enable: true, distance: 150, color: "#a39b91", opacity: 0.3, width: 1 },
                move: { enable: true, speed: 1.2, direction: "none", random: true, straight: false, outModes: "bounce" }
            },
            interactivity: {
                events: { onHover: { enable: true, mode: "grab" }, onClick: { enable: true, mode: "push" } },
                modes: { grab: { distance: 140, links: { opacity: 0.8 } }, push: { quantity: 3 } }
            },
            detectRetina: true
        });
    }
    window.addEventListener('DOMContentLoaded', initParticles);
</script>
"""

# ==========================================
# 6. GIAO DIỆN AUTH (TRANG ĐĂNG NHẬP) - ĐÃ FIX LỖI UX/UI
# ==========================================
AUTH_HTML = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Đăng nhập | Botanical Lab Pro</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<link href="https://unpkg.com/aos@2.3.1/dist/aos.css" rel="stylesheet">
<script>
    tailwind.config = {{ darkMode: 'class', theme: {{ extend: {{ colors: {{ accent: '#ea580c', surface: '#ffffff', 'surface-dark': '#1a1715' }}, fontFamily: {{ sans: ['Plus Jakarta Sans','sans-serif'] }} }} }} }};
    if (localStorage.getItem('theme') === 'dark' || (!localStorage.getItem('theme') && matchMedia('(prefers-color-scheme: dark)').matches)) document.documentElement.classList.add('dark');
    if (localStorage.getItem('token')) window.location.href = '/';
</script>
<style>
    body {{ font-family: 'Plus Jakarta Sans', sans-serif; overflow-x: hidden; }}
    /* Fix lỗi chồng chéo bằng CSS Grid Stacking */
    .form-wrapper {{ display: grid; grid-template-columns: 1fr; grid-template-rows: 1fr; align-items: start; }}
    .form-wrapper > form {{ grid-area: 1 / 1; transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); }}
    
    .card {{ background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(12px); border: 1px solid #e5e7eb; box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.1); }}
    .dark .card {{ background: rgba(26, 23, 21, 0.85); border-color: #2e2925; box-shadow: 0 20px 40px -10px rgba(0,0,0, 0.5); }}
    .input-field {{ width: 100%; padding: 0.75rem 1rem; border-radius: 0.75rem; border: 1px solid #d1d5db; outline: none; transition: all 0.2s; background: transparent; }}
    .dark .input-field {{ border-color: #3f3f46; color: white; }}
    .input-field:focus {{ border-color: #ea580c; box-shadow: 0 0 0 3px rgba(234, 88, 12, 0.15); }}
    .btn-primary {{ background: #ea580c; color: white; padding: 0.75rem; border-radius: 0.75rem; font-weight: 700; width: 100%; transition: all 0.3s; }}
    .btn-primary:hover {{ transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(234, 88, 12, 0.4); }}
</style>
</head>
<body class="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-[#0f0d0c] text-gray-900 dark:text-gray-100 p-4 relative">
    
    <!-- Background Particles -->
    <div id="tsparticles" class="absolute inset-0 z-0 pointer-events-none"></div>

    <!-- Toast Container -->
    <div id="toast" class="fixed top-5 left-1/2 -translate-x-1/2 px-6 py-3 rounded-full text-sm font-bold text-white opacity-0 transition-all duration-300 pointer-events-none z-50 transform translate-y-[-20px]"></div>

    <div class="card p-8 rounded-3xl w-full max-w-md relative z-10 overflow-hidden" data-aos="zoom-in" data-aos-duration="800">
        <div class="text-center mb-8">
            <div class="w-14 h-14 rounded-2xl bg-orange-600 flex items-center justify-center text-3xl mx-auto mb-4 shadow-lg shadow-orange-600/30">🌿</div>
            <h2 class="text-2xl font-extrabold tracking-tight">Botanical Lab Pro</h2>
            <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">Đăng nhập để sử dụng AI phân tích</p>
        </div>

        <!-- CSS Grid Stacking Container (Khắc phục lỗi ghi đè absolute) -->
        <div class="form-wrapper">
            <!-- Form Login -->
            <form id="form-login" class="space-y-4 opacity-100 transform translate-x-0 pointer-events-auto">
                <div><label class="text-sm font-bold mb-1 block">Tên đăng nhập</label><input type="text" id="l-user" class="input-field" required></div>
                <div><label class="text-sm font-bold mb-1 block">Mật khẩu</label><input type="password" id="l-pass" class="input-field" required></div>
                <div class="flex justify-between items-center text-sm">
                    <label class="flex items-center gap-2 cursor-pointer"><input type="checkbox" class="accent-orange-600"> <span>Nhớ tài khoản</span></label>
                    <button type="button" onclick="switchTab('forgot')" class="text-orange-600 hover:underline font-semibold">Quên mật khẩu?</button>
                </div>
                <button type="submit" class="btn-primary mt-4 group">
                    Đăng nhập ngay <span class="inline-block transition-transform duration-300 group-hover:translate-x-1">➔</span>
                </button>
                <p class="text-center text-sm mt-4 text-gray-500">Chưa có tài khoản? <button type="button" onclick="switchTab('register')" class="text-orange-600 font-bold hover:underline">Tạo tài khoản</button></p>
            </form>

            <!-- Form Register -->
            <form id="form-register" class="space-y-4 opacity-0 transform translate-x-10 pointer-events-none">
                <div><label class="text-sm font-bold mb-1 block">Tên đăng nhập mới</label><input type="text" id="r-user" class="input-field" minlength="3" required></div>
                <div><label class="text-sm font-bold mb-1 block">Mật khẩu (Tối thiểu 6 ký tự)</label><input type="password" id="r-pass" class="input-field" minlength="6" required></div>
                <button type="submit" class="btn-primary mt-4 group">
                    Đăng ký tài khoản <span class="inline-block transition-transform duration-300 group-hover:translate-x-1">➔</span>
                </button>
                <p class="text-center text-sm mt-4 text-gray-500">Đã có tài khoản? <button type="button" onclick="switchTab('login')" class="text-orange-600 font-bold hover:underline">Đăng nhập</button></p>
            </form>

            <!-- Form Forgot Pass -->
            <form id="form-forgot" class="space-y-4 opacity-0 transform -translate-x-10 pointer-events-none">
                <div><label class="text-sm font-bold mb-1 block">Tên đăng nhập</label><input type="text" id="f-user" class="input-field" required></div>
                <div>
                    <label class="text-sm font-bold mb-1 block">Câu hỏi bảo mật</label>
                    <div class="flex gap-3">
                        <div id="math-question" class="bg-gray-100 dark:bg-gray-800 px-4 py-3 rounded-xl font-bold font-mono flex-shrink-0 w-24 text-center border border-gray-200 dark:border-gray-700">? + ?</div>
                        <input type="number" id="f-answer" class="input-field w-full" placeholder="Nhập kết quả" required>
                    </div>
                </div>
                <div><label class="text-sm font-bold mb-1 block">Mật khẩu mới</label><input type="password" id="f-pass" class="input-field" minlength="6" required></div>
                <button type="submit" class="btn-primary mt-4 bg-gray-900 dark:bg-gray-100 dark:text-gray-900 group">
                    Đặt lại mật khẩu <span class="inline-block transition-transform duration-300 group-hover:translate-x-1">➔</span>
                </button>
                <button type="button" onclick="switchTab('login')" class="w-full text-center text-sm mt-4 text-gray-500 hover:text-gray-900 dark:hover:text-white font-semibold">⬅ Quay lại Đăng nhập</button>
            </form>
        </div>
    </div>

<script src="https://unpkg.com/aos@2.3.1/dist/aos.js"></script>
{PARTICLES_JS}
<script>
    AOS.init({{ once: true }});

    let n1, n2;
    function genMath() {{ n1 = Math.floor(Math.random()*10)+1; n2 = Math.floor(Math.random()*10)+1; document.getElementById('math-question').textContent = `${{n1}} + ${{n2}}`; }}
    
    function showToast(msg, isSuccess=true) {{
        const t = document.getElementById('toast');
        t.textContent = msg; 
        t.className = `fixed top-5 left-1/2 -translate-x-1/2 px-6 py-3 rounded-full text-sm font-bold text-white transition-all duration-300 pointer-events-none z-50 ${{isSuccess?'bg-green-500':'bg-red-500'}} opacity-100 translate-y-0 shadow-lg`;
        setTimeout(() => {{ t.classList.remove('opacity-100', 'translate-y-0'); t.classList.add('opacity-0', 'translate-y-[-20px]'); }}, 3000);
    }}

    const forms = {{ login: document.getElementById('form-login'), register: document.getElementById('form-register'), forgot: document.getElementById('form-forgot') }};
    
    function switchTab(tab) {{
        // Reset all
        Object.values(forms).forEach(f => {{ 
            f.style.opacity = '0'; 
            f.style.pointerEvents = 'none'; 
            f.style.transform = 'translateX(20px)'; // default hidden state
        }});
        
        // Active logic
        forms.login.style.transform = tab === 'login' ? 'translateX(0)' : (tab === 'register' ? 'translateX(-20px)' : 'translateX(20px)');
        forms.register.style.transform = tab === 'register' ? 'translateX(0)' : 'translateX(20px)';
        forms.forgot.style.transform = tab === 'forgot' ? 'translateX(0)' : 'translateX(-20px)';
        
        setTimeout(() => {{ 
            forms[tab].style.opacity = '1'; 
            forms[tab].style.pointerEvents = 'auto'; 
        }}, 50);

        if (tab === 'forgot') genMath();
    }}

    // Đăng ký
    document.getElementById('form-register').onsubmit = async (e) => {{
        e.preventDefault(); const btn = e.target.querySelector('button'); btn.disabled = true; btn.innerHTML = "Đang xử lý...";
        const u = document.getElementById('r-user').value, p = document.getElementById('r-pass').value;
        const res = await fetch('/auth/register', {{ method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{username: u, password: p}})}});
        if (res.ok) {{ showToast('Đăng ký thành công!'); switchTab('login'); document.getElementById('l-user').value = u; document.getElementById('l-pass').focus(); }} 
        else {{ const d = await res.json(); showToast(d.detail, false); }}
        btn.disabled = false; btn.innerHTML = `Đăng ký tài khoản <span class="inline-block transition-transform duration-300 group-hover:translate-x-1">➔</span>`;
    }};

    // Đăng nhập
    document.getElementById('form-login').onsubmit = async (e) => {{
        e.preventDefault(); const btn = e.target.querySelector('button'); btn.disabled = true; btn.innerHTML = "Đang xử lý...";
        const res = await fetch('/auth/login', {{ method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{username: document.getElementById('l-user').value, password: document.getElementById('l-pass').value}})}});
        if (res.ok) {{ const d = await res.json(); localStorage.setItem('token', d.token); localStorage.setItem('username', d.username); window.location.href = '/'; }} 
        else {{ const d = await res.json(); showToast(d.detail, false); }}
        btn.disabled = false; btn.innerHTML = `Đăng nhập ngay <span class="inline-block transition-transform duration-300 group-hover:translate-x-1">➔</span>`;
    }};

    // Quên mật khẩu
    document.getElementById('form-forgot').onsubmit = async (e) => {{
        e.preventDefault(); const btn = e.target.querySelector('button'); btn.disabled = true; btn.innerHTML = "Đang xử lý...";
        const payload = {{ username: document.getElementById('f-user').value, num1: n1, num2: n2, answer: parseInt(document.getElementById('f-answer').value), new_password: document.getElementById('f-pass').value }};
        const res = await fetch('/auth/forgot-password', {{ method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(payload)}});
        if (res.ok) {{ showToast('Đổi mật khẩu thành công!'); switchTab('login'); document.getElementById('l-user').value = payload.username; }} 
        else {{ const d = await res.json(); showToast(d.detail, false); genMath(); document.getElementById('f-answer').value=''; }}
        btn.disabled = false; btn.innerHTML = `Đặt lại mật khẩu <span class="inline-block transition-transform duration-300 group-hover:translate-x-1">➔</span>`;
    }};
</script>
</body>
</html>"""

# ==========================================
# 7. KHUNG GIAO DIỆN CHÍNH (BASE_HTML) - THÊM HIỆU ỨNG
# ==========================================
BASE_HTML = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__ | Botanical Lab Pro</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link href="https://unpkg.com/aos@2.3.1/dist/aos.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
  if (!localStorage.getItem('token')) {{ window.location.href = '/login'; }}
  tailwind.config = {{ darkMode: 'class', theme: {{ extend: {{ fontFamily: {{ sans: ['Plus Jakarta Sans','sans-serif'], mono: ['JetBrains Mono','monospace'] }} }} }} }};
  (function(){{ const t = localStorage.getItem('theme'); if (t === 'dark' || (!t && matchMedia('(prefers-color-scheme: dark)').matches)) document.documentElement.classList.add('dark'); }})();
  
  async function authFetch(url, options = {{}}) {{
      const token = localStorage.getItem('token');
      if (!options.headers) options.headers = {{}};
      options.headers['Authorization'] = 'Bearer ' + token;
      const res = await fetch(url, options);
      if (res.status === 401) {{ localStorage.removeItem('token'); window.location.href = '/login'; }}
      return res;
  }}
</script>
<style>
  :root {{ --bg:#f7f5f2; --surface:rgba(255,255,255,0.85); --surface-2:#f1eee9; --text:#1c1917; --muted:#6b645c; --border:#e4dfd8; --accent:#ea580c; --accent-soft:rgba(234,88,12,.10); --c-setosa:#ea580c; --c-versicolor:#7c3aed; --c-virginica:#0d9488; --shadow:0 4px 6px -1px rgba(0, 0, 0, 0.05); }}
  .dark {{ --bg:#0f0d0c; --surface:rgba(26,23,21,0.85); --surface-2:#231f1c; --text:#f5f2ee; --muted:#a39b91; --border:#2e2925; --accent:#fb923c; --accent-soft:rgba(251,146,60,.14); --c-setosa:#fb923c; --c-versicolor:#a78bfa; --c-virginica:#2dd4bf; --shadow:0 10px 15px -3px rgba(0, 0, 0, 0.3); }}
  html {{ scroll-behavior:smooth; }} body {{ background:var(--bg); color:var(--text); font-family:'Plus Jakarta Sans',sans-serif; transition:background .3s,color .3s; overflow-x:hidden; }}
  
  /* Hiệu ứng thẻ Card Hover nổi lên + Glow viền */
  .card {{ background:var(--surface); backdrop-filter:blur(12px); border:1px solid var(--border); box-shadow:var(--shadow); border-radius:1.25rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); z-index: 10; position:relative; }} 
  .card-hover:hover {{ transform:translateY(-6px); box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 0 15px var(--accent-soft); border-color: var(--accent); }}
  .dark .card-hover:hover {{ box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.6), 0 0 20px rgba(234, 88, 12, 0.25); }}
  
  .soft {{ background:var(--surface-2); border:1px solid var(--border); border-radius:1rem; }} .accent {{ color:var(--accent); }}
  
  /* Nút bấm Hover Arrow / Sóng nhẹ */
  .btn {{ display:inline-flex; align-items:center; justify-content:center; gap:.5rem; padding:.75rem 1.25rem; border-radius:.85rem; font-weight:700; font-size:.9rem; transition:all 0.3s; cursor:pointer; }}
  .btn:hover:not(:disabled) {{ transform:translateY(-2px); box-shadow: 0 10px 15px -3px var(--accent-soft); }} .btn:active:not(:disabled) {{ transform:translateY(0) scale(.98); }} .btn:disabled {{ opacity:.45; cursor:not-allowed; }}
  .btn-primary {{ background:var(--accent); color:#fff; box-shadow:0 8px 20px -8px var(--accent); }} .btn-dark {{ background:var(--text); color:var(--bg); }} .btn-ghost {{ background:var(--surface-2); border:1px solid var(--border); color:var(--text); }} .btn-danger {{ background:rgba(220,38,38,.1); color:#dc2626; border:1px solid rgba(220,38,38,.25); }}
  
  input[type=range] {{ accent-color:var(--accent); }}
  .nav-link {{ position:relative; padding:.4rem .1rem; font-weight:600; font-size:.9rem; color:var(--muted); transition:color .2s; white-space:nowrap; }} .nav-link:hover, .nav-link.active {{ color:var(--text); }} .nav-link::after {{ content:''; position:absolute; left:0; bottom:-2px; height:2px; width:100%; background:var(--accent); transform:scaleX(0); transform-origin:left; transition:transform .3s; }} .nav-link.active::after, .nav-link:hover::after {{ transform:scaleX(1); }}
  .sw-c {{ color:var(--c-setosa); }} .ve-c {{ color:var(--c-versicolor); }} .vi-c {{ color:var(--c-virginica); }}
  
  /* Skeleton & Utility */
  @keyframes pop {{ from {{ opacity:0; transform:scale(.94); }} to {{ opacity:1; transform:none; }} }} .pop {{ animation:pop .35s ease both; }}
  .skel {{ background:linear-gradient(90deg,var(--surface-2),var(--border),var(--surface-2)); background-size:200% 100%; animation:shimmer 1.3s infinite linear; border-radius:.5rem; }} @keyframes shimmer {{ to {{ background-position:-200% 0; }} }}
  .typing::after {{ content:'|'; animation:blink 1s step-end infinite; }} @keyframes blink {{ 50% {{ opacity:0; }} }}
  
  /* Dropdown & Modal */
  .menu {{ position:absolute; right:0; top:calc(100% + .5rem); min-width:14rem; opacity:0; transform:translateY(-6px) scale(.98); pointer-events:none; transition:all 0.2s; z-index:60; }} .menu.open {{ opacity:1; transform:none; pointer-events:auto; }}
  #modal-bg {{ position:fixed; inset:0; background:rgba(0,0,0,.5); backdrop-filter:blur(4px); display:flex; align-items:center; justify-content:center; z-index:100; opacity:0; pointer-events:none; transition:opacity .2s; }} #modal-bg.open {{ opacity:1; pointer-events:auto; }} #modal-box {{ transform:scale(.94); transition:transform .25s; }} #modal-bg.open #modal-box {{ transform:none; }}
  #toasts {{ position:fixed; bottom:1.25rem; right:1.25rem; z-index:120; display:flex; flex-direction:column; gap:.5rem; }} .toast {{ background:var(--text); color:var(--bg); padding:.75rem 1.1rem; border-radius:.8rem; font-weight:600; font-size:.875rem; box-shadow:var(--shadow); animation:pop .3s ease both; }} .toast.out {{ opacity:0; transform:translateY(8px); transition:.3s; }}
  ::-webkit-scrollbar {{ width:6px; height:6px; }} ::-webkit-scrollbar-thumb {{ background:var(--border); border-radius:4px; }}
</style>
</head>
<body class="min-h-screen flex flex-col relative">

<!-- Background Particles -->
<div id="tsparticles" class="fixed inset-0 z-0 pointer-events-none"></div>

<header class="sticky top-0 z-50 border-b backdrop-blur-md" style="background:color-mix(in srgb,var(--bg) 80%,transparent);border-color:var(--border)">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between gap-4">
    <a href="/" class="flex items-center gap-3 shrink-0">
      <div class="w-10 h-10 rounded-xl flex items-center justify-center text-xl text-white" style="background:var(--accent)">🌿</div>
      <div><div class="font-extrabold tracking-tight leading-tight">Iris Pro</div><div class="text-[11px] muted font-mono">SVM Classifier</div></div>
    </a>
    <nav id="nav" class="hidden md:flex items-center gap-5 overflow-x-auto">
      <a href="/" class="nav-link">Trang chủ</a>
      <a href="/dataset" class="nav-link">Dữ liệu</a>
      <a href="/lab" class="nav-link">Phân tích</a>
      <a href="/vector3d" class="nav-link">Không gian 3D</a>
      <a href="/kernels" class="nav-link">Kernels</a>
    </nav>
    
    <!-- User Dropdown Menu -->
    <div class="relative shrink-0 flex items-center gap-2">
      <button onclick="toggleTheme()" class="w-9 h-9 rounded-full soft hover:bg-[var(--border)] transition flex items-center justify-center">🌓</button>
      <button id="user-btn" class="btn btn-ghost !py-2 !px-3 !rounded-full">
         <div class="w-6 h-6 rounded-full bg-orange-600 text-white text-xs flex items-center justify-center" id="avatar-initial">U</div>
         <span id="display-username" class="font-bold">User</span> ▾
      </button>
      <div id="user-menu" class="menu card p-2 mt-2">
        <div class="px-3 py-2 border-b mb-1" style="border-color:var(--border)">
            <p class="text-xs muted">Đăng nhập dưới tên</p>
            <p class="font-bold truncate" id="menu-username"></p>
        </div>
        <button onclick="alert('Chức năng cài đặt đang được phát triển')" class="w-full text-left p-2 text-sm font-semibold hover:bg-[var(--surface-2)] rounded-lg transition">⚙️ Cài đặt tài khoản</button>
        <button onclick="logout()" class="w-full text-left p-2 text-sm font-semibold text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition mt-1">🚪 Đăng xuất</button>
      </div>
    </div>
  </div>
</header>

<!-- Main content bọc thêm relative và z-index để đứng trên Particles -->
<main class="w-full flex-grow flex flex-col relative z-10">__CONTENT__</main>

<div id="modal-bg" class="z-50"><div id="modal-box" class="card p-6 max-w-sm w-[90%]"><h3 id="modal-title" class="font-bold text-lg mb-2"></h3><p id="modal-msg" class="text-sm muted mb-6"></p><div class="flex gap-3 justify-end"><button id="modal-no" class="btn btn-ghost">Hủy</button><button id="modal-yes" class="btn btn-danger">Xác nhận</button></div></div></div>
<div id="toasts"></div>

<script src="https://unpkg.com/aos@2.3.1/dist/aos.js"></script>
{PARTICLES_JS}
<script>
  AOS.init({{ once: true, offset: 50, duration: 600 }});

  // UI Setup (Username)
  const storedUser = localStorage.getItem('username') || 'User';
  document.getElementById('display-username').textContent = storedUser;
  document.getElementById('menu-username').textContent = storedUser;
  document.getElementById('avatar-initial').textContent = storedUser.charAt(0).toUpperCase();

  // Dropdown logic
  const menu = document.getElementById('user-menu');
  document.getElementById('user-btn').addEventListener('click', e => {{ e.stopPropagation(); menu.classList.toggle('open'); }});
  document.addEventListener('click', e => {{ if (!menu.contains(e.target)) menu.classList.remove('open'); }});
  
  // Theme logic
  function syncThemeUI() {{
    if (window.Chart) {{ Chart.defaults.color = getComputedStyle(document.documentElement).getPropertyValue('--muted').trim(); Chart.defaults.borderColor = getComputedStyle(document.documentElement).getPropertyValue('--border').trim(); Object.values(Chart.instances).forEach(c => c.update()); }}
    if (typeof onThemeChange === 'function') onThemeChange(document.documentElement.classList.contains('dark'));
  }}
  function toggleTheme() {{
    const dark = !document.documentElement.classList.contains('dark');
    document.documentElement.classList.toggle('dark', dark);
    localStorage.setItem('theme', dark ? 'dark' : 'light'); syncThemeUI();
  }}

  // Logout
  function logout() {{ localStorage.removeItem('token'); localStorage.removeItem('username'); window.location.href = '/login'; }}

  // Utils
  document.querySelectorAll('#nav a').forEach(a => {{
    if (a.getAttribute('href') === location.pathname) a.classList.add('active');
    a.addEventListener('click', e => {{ if (a.getAttribute('href') === location.pathname) return; e.preventDefault(); document.body.classList.add('leaving'); setTimeout(() => location.href = a.getAttribute('href'), 180); }});
  }});
  function toast(msg) {{ const el = document.createElement('div'); el.className = 'toast'; el.textContent = msg; document.getElementById('toasts').appendChild(el); setTimeout(() => {{ el.classList.add('out'); setTimeout(() => el.remove(), 300); }}, 2400); }}
  function confirmModal(title, msg) {{ return new Promise(resolve => {{ const bg = document.getElementById('modal-bg'); document.getElementById('modal-title').textContent = title; document.getElementById('modal-msg').textContent = msg; bg.classList.add('open'); const done = v => {{ bg.classList.remove('open'); yes.onclick = no.onclick = null; resolve(v); }}; const yes = document.getElementById('modal-yes'), no = document.getElementById('modal-no'); yes.onclick = () => done(true); no.onclick = () => done(false); bg.onclick = e => {{ if (e.target === bg) done(false); }}; }}); }}
  
  window.addEventListener('DOMContentLoaded', syncThemeUI);
</script>
</body>
</html>"""

def get_base_html(title, content): 
    return BASE_HTML.replace("__TITLE__", title).replace("__CONTENT__", content)

# ==========================================
# 8. CÁC TRANG RENDER (ROUTER)
# ==========================================
@app.get("/login", response_class=HTMLResponse)
def login_page(): 
    return AUTH_HTML

@app.get("/", response_class=HTMLResponse)
def hero_page():
    content = """
    <section class="border-b" style="border-color:var(--border)">
      <div class="max-w-7xl mx-auto px-6 py-16 lg:py-28 grid lg:grid-cols-2 gap-12 items-center">
        <div data-aos="fade-right">
          <span class="px-3 py-1 text-xs font-bold rounded-full inline-block mb-2" style="background:var(--accent-soft);color:var(--accent)">Khoa học dữ liệu</span>
          <h2 class="text-4xl lg:text-6xl font-extrabold leading-[1.1] tracking-tight">Nhận dạng hoa Diên Vĩ bằng SVM</h2>
          <p class="mt-6 text-lg muted leading-relaxed max-w-lg">Nhập bốn số đo của một bông hoa, mô hình sẽ cho biết đó là loài nào. Hệ thống được bảo mật mã hóa đầu cuối.</p>
          <div class="mt-8 flex flex-wrap gap-3">
            <!-- Nút có hiệu ứng dịch mũi tên -->
            <a href="/lab" class="btn btn-primary group">
              Khởi động Lab <span class="inline-block transition-transform duration-300 group-hover:translate-x-1">➔</span>
            </a>
          </div>
        </div>
        <div class="grid grid-cols-3 gap-4" data-aos="fade-left" data-aos-delay="200">
          <div class="card card-hover p-5 text-center"><div class="text-4xl mb-3 transition-transform duration-300 hover:scale-110 cursor-default">🌸</div><div class="font-bold sw-c">Setosa</div></div>
          <div class="card card-hover p-5 text-center" style="animation-delay: 100ms"><div class="text-4xl mb-3 transition-transform duration-300 hover:scale-110 cursor-default">🌺</div><div class="font-bold ve-c">Versicolor</div></div>
          <div class="card card-hover p-5 text-center" style="animation-delay: 200ms"><div class="text-4xl mb-3 transition-transform duration-300 hover:scale-110 cursor-default">🌷</div><div class="font-bold vi-c">Virginica</div></div>
        </div>
      </div>
    </section>"""
    return get_base_html("Trang chủ", content)

@app.get("/dataset", response_class=HTMLResponse)
def dataset_page():
    content = """
    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-8 w-full">
      <h2 class="text-2xl font-bold mb-6" data-aos="fade-up">Thống kê dữ liệu</h2>
      <div class="grid lg:grid-cols-3 gap-6 mb-6">
        <div class="card p-6 card-hover" data-aos="fade-up" data-aos-delay="100"><div class="h-56"><canvas id="pie"></canvas></div></div>
        <div class="card p-6 card-hover lg:col-span-2" data-aos="fade-up" data-aos-delay="200"><div class="h-56"><canvas id="bar"></canvas></div></div>
      </div>
    </div>
    <script>
      let D;
      const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
      const cols = () => [cssv('--c-setosa'), cssv('--c-versicolor'), cssv('--c-virginica')];
      function paint() {
        Object.values(Chart.instances).forEach(c => c.destroy()); const c = cols();
        new Chart(document.getElementById('pie'), { type:'doughnut', data:{ labels:D.species.map(s=>s.name), datasets:[{ data:D.species.map(s=>s.count), backgroundColor:c, borderWidth:0 }] }, options:{ maintainAspectRatio:false, plugins:{ legend:{ position:'bottom' } } } });
        new Chart(document.getElementById('bar'), { type:'bar', data:{ labels:D.features, datasets:D.species.map((s,i)=>({ label:s.name, data:s.mean, backgroundColor:c[i], borderRadius:6 })) }, options:{ maintainAspectRatio:false, plugins:{ legend:{ position:'bottom' } } } });
      }
      function onThemeChange() { if (D) paint(); }
      window.addEventListener('DOMContentLoaded', async () => { try { D = await (await authFetch('/api/dataset')).json(); paint(); } catch (e) { toast('Lỗi kết nối.'); } });
    </script>"""
    return get_base_html("Giới thiệu dữ liệu", content)

@app.get("/lab", response_class=HTMLResponse)
def lab_page():
    content = """
    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-8 w-full">
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start mb-6">
        <div class="lg:col-span-4 card p-6" data-aos="fade-right">
          <div class="flex justify-between items-center mb-6"><h2 class="font-bold text-lg">Tham số</h2><button onclick="randomizeInputs()" class="text-xl hover:rotate-180 transition duration-300">🎲</button></div>
          <div class="space-y-6">
            <div><div class="flex justify-between text-xs font-bold mb-2"><span>Sepal L</span><span id="txt-sl" class="accent">5.1</span></div><input type="range" id="sl" min="4.0" max="8.0" step="0.1" value="5.1" class="w-full" oninput="syncVal('sl','txt-sl')"></div>
            <div><div class="flex justify-between text-xs font-bold mb-2"><span>Sepal W</span><span id="txt-sw" class="accent">3.5</span></div><input type="range" id="sw" min="2.0" max="4.5" step="0.1" value="3.5" class="w-full" oninput="syncVal('sw','txt-sw')"></div>
            <div><div class="flex justify-between text-xs font-bold mb-2"><span>Petal L</span><span id="txt-pl" class="accent">1.4</span></div><input type="range" id="pl" min="1.0" max="7.0" step="0.1" value="1.4" class="w-full" oninput="syncVal('pl','txt-pl')"></div>
            <div><div class="flex justify-between text-xs font-bold mb-2"><span>Petal W</span><span id="txt-pw" class="accent">0.2</span></div><input type="range" id="pw" min="0.1" max="2.6" step="0.1" value="0.2" class="w-full" oninput="syncVal('pw','txt-pw')"></div>
          </div>
          <button id="btn-predict" onclick="executeInference()" class="btn btn-dark w-full mt-8 group">
            Phân tích ngay <span class="inline-block transition-transform duration-300 group-hover:translate-x-1">➔</span>
          </button>
          <button id="btn-compare" onclick="compareKernels()" disabled class="btn btn-primary w-full mt-3 hidden group">
            So sánh Kernels <span class="inline-block transition-transform duration-300 group-hover:translate-x-1">➔</span>
          </button>
        </div>
        <div class="lg:col-span-8 flex flex-col gap-6" data-aos="fade-left" data-aos-delay="200">
          <div class="card p-6 sm:p-8 relative overflow-hidden min-h-[140px] flex items-center">
            <div id="result-view" class="flex items-center justify-between w-full transition-opacity duration-300">
              <div><p class="text-xs font-bold muted">Kết quả dự đoán</p><h3 id="specimen-title" class="text-3xl font-black mt-1">Chưa có kết quả</h3></div>
              <div class="text-right"><p class="text-xs font-bold muted">Độ tin cậy</p><div class="text-4xl font-black font-mono accent" id="main-conf">–</div></div>
            </div>
          </div>
          <div id="compare-panel" class="hidden card p-6"><div id="compare-grid" class="grid sm:grid-cols-2 gap-4"></div></div>
          <div class="card p-6 h-72 flex flex-col card-hover"><div class="flex-grow w-full relative"><canvas id="scatterChart"></canvas></div></div>
        </div>
      </div>
      <div class="card p-6 card-hover" data-aos="fade-up">
        <div class="flex flex-wrap justify-between items-center gap-3 mb-4"><h2 class="font-bold text-lg">Lịch sử cá nhân</h2><button onclick="clearHistory()" class="btn btn-danger !py-2">🗑 Xóa</button></div>
        <div class="overflow-x-auto"><table class="w-full text-left text-sm whitespace-nowrap">
          <thead><tr class="muted text-xs border-b" style="border-color:var(--border)"><th class="py-3 px-4">Thời gian</th><th class="py-3 px-4">P.L</th><th class="py-3 px-4">P.W</th><th class="py-3 px-4">Dự đoán</th></tr></thead>
          <tbody id="history-tbody"></tbody></table></div>
      </div>
    </div>
    <script>
      let scatterChart, lastPayload = null; const KNAME = { linear:'Linear', rbf:'RBF', poly:'Polynomial', sigmoid:'Sigmoid' };
      const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
      function initCharts() { const c = [cssv('--c-setosa'), cssv('--c-versicolor'), cssv('--c-virginica')]; scatterChart = new Chart(document.getElementById('scatterChart').getContext('2d'), { type:'scatter', data:{ datasets:[ { label:'Setosa', data:[{x:1.4,y:0.2}], backgroundColor:c[0]+'99' }, { label:'Versicolor', data:[{x:4.7,y:1.4}], backgroundColor:c[1]+'99' }, { label:'Virginica', data:[{x:6.0,y:2.5}], backgroundColor:c[2]+'99' }, { label:'Mẫu mới', data:[], backgroundColor:cssv('--text'), pointRadius:8 } ] }, options:{ responsive:true, maintainAspectRatio:false } }); }
      function syncVal(s, t) { document.getElementById(t).textContent = document.getElementById(s).value; }
      function randomizeInputs() { const r = (a, b) => (Math.random() * (b - a) + a).toFixed(1); ['sl','sw','pl','pw'].forEach(id => { document.getElementById(id).value = r(id.includes('s')?2:0.1, id.includes('s')?8:7); syncVal(id, 'txt-' + id); }); }
      function renderHistory(data) { document.getElementById('history-tbody').innerHTML = data.map(r => `<tr class="border-b" style="border-color:var(--border)"><td class="py-2 px-4 text-xs font-mono muted">${r.timestamp}</td><td class="py-2 px-4">${r.pl}</td><td class="py-2 px-4">${r.pw}</td><td class="py-2 px-4 font-bold accent">${r.species} (${r.confidence}%)</td></tr>`).join(''); }
      async function fetchHistory() { const res = await authFetch('/api/history'); if (res && res.ok) renderHistory(await res.json()); }
      async function clearHistory() { if (await confirmModal('Xóa lịch sử?', 'Chắc chắn xóa lịch sử của bạn?')) { await authFetch('/api/history', { method:'DELETE' }); fetchHistory(); toast('Đã xóa'); } }
      async function executeInference() {
        const payload = { sepal_length:parseFloat(document.getElementById('sl').value), sepal_width:parseFloat(document.getElementById('sw').value), petal_length:parseFloat(document.getElementById('pl').value), petal_width:parseFloat(document.getElementById('pw').value) };
        const res = await authFetch('/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
        if (res && res.ok) { const d = await res.json(); document.getElementById('specimen-title').textContent = d.prediction; document.getElementById('specimen-title').style.color = d.color; document.getElementById('main-conf').textContent = Math.max(...d.confidences) + '%'; scatterChart.data.datasets[3].data = [{ x:payload.petal_length, y:payload.petal_width }]; scatterChart.update(); lastPayload = payload; document.getElementById('btn-compare').classList.remove('hidden'); document.getElementById('btn-compare').disabled = false; fetchHistory(); }
      }
      async function compareKernels() {
        const res = await authFetch('/api/compare_kernels', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(lastPayload) });
        if (res && res.ok) { const rows = await res.json(); document.getElementById('compare-panel').classList.remove('hidden'); document.getElementById('compare-grid').innerHTML = rows.map((r, i) => `<div class="soft p-4 pop"><div class="text-xs muted mb-1">${KNAME[r.kernel]}</div><div class="font-bold" style="color:${r.color}">${r.prediction}</div><div class="font-mono text-2xl font-black mt-1">${r.confidence}%</div></div>`).join(''); }
      }
      window.addEventListener('DOMContentLoaded', () => { initCharts(); fetchHistory(); });
    </script>"""
    return get_base_html("Phân tích", content)

@app.get("/vector3d", response_class=HTMLResponse)
def vector3d_page(): 
    return get_base_html("Không gian 3D", "<div class='p-8 flex justify-center h-[500px] card card-hover mx-6 mt-6' data-aos='fade-up'><div id='plot3d' class='w-full'></div></div><script>window.addEventListener('DOMContentLoaded', async () => { try { const PTS = await (await authFetch('/api/pca')).json(); const c = [getComputedStyle(document.body).getPropertyValue('--c-setosa'), getComputedStyle(document.body).getPropertyValue('--c-versicolor'), getComputedStyle(document.body).getPropertyValue('--c-virginica')]; const traces = [0,1,2].map(i => { const p = PTS.points.filter(q => q.cls === i); return { type:'scatter3d', mode:'markers', name:['Setosa','Versicolor','Virginica'][i], x:p.map(q=>q.x), y:p.map(q=>q.y), z:p.map(q=>q.z), marker:{ size:5, color:c[i] } }; }); Plotly.newPlot('plot3d', traces, {margin:{t:0,b:0,l:0,r:0}, paper_bgcolor:'rgba(0,0,0,0)'}); } catch (e) {} });</script>")

@app.get("/kernels", response_class=HTMLResponse)
def kernels_page(): 
    return get_base_html("Kernels", "<div class='p-8'><h2 class='text-2xl font-bold mb-4' data-aos='fade-right'>Độ chính xác Kernel</h2><div class='h-64 card p-4 card-hover' data-aos='fade-up'><canvas id='accChart'></canvas></div></div><script>window.addEventListener('DOMContentLoaded', async () => { const scores = await (await authFetch('/api/kernel_scores')).json(); new Chart(document.getElementById('accChart'), { type:'bar', data:{ labels:['Linear','RBF','Poly','Sigmoid'], datasets:[{ data:['linear','rbf','poly','sigmoid'].map(k=>scores[k]), backgroundColor:'#ea580c' }] }, options:{ maintainAspectRatio:false } }); });</script>")
