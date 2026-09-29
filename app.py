import os
import math
import logging
import sqlite3
import json
import uuid
import hashlib
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# ==========================================
# 1. DATABASE & AUTHENTICATION (SQLITE)
# ==========================================
DB_PATH = "botanical_app.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Users Table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE, password TEXT, role TEXT, api_key TEXT)''')
    # Sessions
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, user_id INTEGER)''')
    # History Table
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, timestamp TEXT, sl REAL, sw REAL, pl REAL, pw REAL, species TEXT, confidence REAL)''')
    # Configs Table
    c.execute('''CREATE TABLE IF NOT EXISTS configs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, name TEXT, kernel TEXT, c_val REAL, gamma REAL)''')
    
    # Create Default Admin
    c.execute("SELECT id FROM users WHERE email='admin@botanical.com'")
    if not c.fetchone():
        hashed_pw = hashlib.sha256("admin123".encode()).hexdigest()
        c.execute("INSERT INTO users (email, password, role, api_key) VALUES (?, ?, 'admin', 'sk-admin-master-key')", 
                  ("admin@botanical.com", hashed_pw))
    conn.commit()
    conn.close()

init_db()

def get_current_user(req: Request):
    token = req.cookies.get("session_token")
    if not token: return None
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT u.id, u.email, u.role, u.api_key FROM users u JOIN sessions s ON u.id = s.user_id WHERE s.token=?", (token,))
    user = c.fetchone()
    conn.close()
    if user: return {"id": user[0], "email": user[1], "role": user[2], "api_key": user[3]}
    return None

# ==========================================
# 2. MODEL & APP INIT
# ==========================================
MODEL_PATH = "svm_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

app = FastAPI(title="Iris API - By Nguyen Minh Hoang", version="7.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

CENTROIDS = [[5.006, 3.428, 1.462, 0.246], [5.936, 2.770, 4.260, 1.326], [6.588, 2.974, 5.552, 2.026]]
SPECIES = {
    0: {"name": "Iris Setosa", "color": "#f97316"}, 
    1: {"name": "Iris Versicolor", "color": "#fb923c"}, 
    2: {"name": "Iris Virginica", "color": "#fdba74"}
}

# ==========================================
# 3. API ENDPOINTS
# ==========================================
class AuthModel(BaseModel): email: str; password: str
class PredictModel(BaseModel): sepal_length: float; sepal_width: float; petal_length: float; petal_width: float
class ConfigModel(BaseModel): name: str; kernel: str; c_val: float; gamma: float

@app.post("/api/auth/login")
def login(data: AuthModel, res: Response):
    hashed_pw = hashlib.sha256(data.password.encode()).hexdigest()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, role FROM users WHERE email=? AND password=?", (data.email, hashed_pw))
    user = c.fetchone()
    if not user:
        conn.close(); raise HTTPException(status_code=400, detail="Sai tài khoản hoặc mật khẩu")
    
    token = str(uuid.uuid4())
    c.execute("INSERT INTO sessions (token, user_id) VALUES (?, ?)", (token, user[0]))
    conn.commit(); conn.close()
    res.set_cookie(key="session_token", value=token, httponly=True)
    return {"message": "Thành công", "role": user[1]}

@app.post("/api/auth/register")
def register(data: AuthModel):
    hashed_pw = hashlib.sha256(data.password.encode()).hexdigest()
    api_key = "sk-" + str(uuid.uuid4().hex)[:16]
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (email, password, role, api_key) VALUES (?, ?, 'user', ?)", (data.email, hashed_pw, api_key))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close(); raise HTTPException(status_code=400, detail="Email đã tồn tại")
    conn.close()
    return {"message": "Đăng ký thành công"}

@app.post("/api/auth/logout")
def logout(req: Request, res: Response):
    token = req.cookies.get("session_token")
    if token:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM sessions WHERE token=?", (token,))
        conn.commit(); conn.close()
    res.delete_cookie("session_token")
    return {"message": "Đã đăng xuất"}

@app.get("/api/me")
def get_me(req: Request):
    user = get_current_user(req)
    if not user: raise HTTPException(status_code=401)
    return user

@app.post("/api/predict")
def predict(data: PredictModel, req: Request):
    user = get_current_user(req)
    user_id = user["id"] if user else 0 # 0 for anonymous
    
    feat = [data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]
    pred = int(model.predict([feat])[0]) if model else 0
    
    # Calculate confidence distance (Mock for visual)
    dists = [math.sqrt(sum((feat[i] - CENTROIDS[c][i]) ** 2 for i in range(4))) for c in range(3)]
    inv_dists = [1.0 / (d + 1e-5) for d in dists]; inv_dists[pred] *= 1.8
    conf = [round((v / sum(inv_dists)) * 100, 1) for v in inv_dists]
    meta = SPECIES[pred]
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO history (user_id, timestamp, sl, sw, pl, pw, species, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
                 (user_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), *feat, meta["name"], conf[pred]))
    conn.commit(); conn.close()

    return {"prediction": meta["name"], "confidences": conf, "color": meta["color"]}

@app.get("/api/history")
def get_user_history(req: Request):
    user = get_current_user(req)
    user_id = user["id"] if user else 0
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp, sl, sw, pl, pw, species, confidence FROM history WHERE user_id=? ORDER BY id DESC LIMIT 20", (user_id,))
    rows = c.fetchall()
    conn.close()
    return [{"time": r[0], "sl": r[1], "sw": r[2], "pl": r[3], "pw": r[4], "species": r[5], "conf": r[6]} for r in rows]

@app.get("/api/admin/users")
def admin_get_users(req: Request):
    user = get_current_user(req)
    if not user or user["role"] != "admin": raise HTTPException(status_code=403)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, email, role, api_key FROM users")
    users = [{"id": r[0], "email": r[1], "role": r[2], "api_key": r[3]} for r in c.fetchall()]
    conn.close()
    return users

@app.post("/api/configs")
def save_config(data: ConfigModel, req: Request):
    user = get_current_user(req)
    if not user: raise HTTPException(status_code=401)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO configs (user_id, name, kernel, c_val, gamma) VALUES (?, ?, ?, ?, ?)", 
                 (user["id"], data.name, data.kernel, data.c_val, data.gamma))
    conn.commit(); conn.close()
    return {"message": "Đã lưu cấu hình mô hình"}

# ==========================================
# 4. SINGLE PAGE APPLICATION (SPA HTML)
# ==========================================
@app.get("/", response_class=HTMLResponse)
def serve_app():
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Botanical Lab Pro | Nguyễn Minh Hoàng</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <!-- AOS Animation -->
        <link href="https://unpkg.com/aos@2.3.1/dist/aos.css" rel="stylesheet">
        <script src="https://unpkg.com/aos@2.3.1/dist/aos.js"></script>
        <!-- tsparticles -->
        <script src="https://cdn.jsdelivr.net/npm/tsparticles-preset-links@2/tsparticles.preset.links.bundle.min.js"></script>
        
        <script>
            tailwind.config = {
                darkMode: 'class',
                theme: { 
                    extend: { 
                        fontFamily: { sans: ['Plus Jakarta Sans', 'sans-serif'], mono: ['JetBrains Mono', 'monospace'] },
                        colors: { orange: { 400: '#fb923c', 500: '#f97316' }, zinc: { 900: '#18181b', 950: '#09090b' } }
                    } 
                }
            }

            // Dark Mode with ViewTransitions API (Clip-path effect)
            function initTheme() {
                if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
                    document.documentElement.classList.add('dark');
                } else { document.documentElement.classList.remove('dark'); }
            }
            initTheme();

            function toggleTheme(event) {
                const isDark = document.documentElement.classList.contains('dark');
                
                if (!document.startViewTransition) {
                    executeThemeSwitch(!isDark);
                    return;
                }
                
                const x = event?.clientX ?? innerWidth / 2;
                const y = event?.clientY ?? innerHeight / 2;
                const endRadius = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));

                const transition = document.startViewTransition(() => { executeThemeSwitch(!isDark); });

                transition.ready.then(() => {
                    const clipPath = [`circle(0px at ${x}px ${y}px)`, `circle(${endRadius}px at ${x}px ${y}px)`];
                    document.documentElement.animate(
                        { clipPath: isDark ? [...clipPath].reverse() : clipPath },
                        { duration: 500, easing: "ease-in-out", pseudoElement: isDark ? "::view-transition-old(root)" : "::view-transition-new(root)" }
                    );
                });
            }

            function executeThemeSwitch(toDark) {
                if(toDark) { document.documentElement.classList.add('dark'); localStorage.theme = 'dark'; }
                else { document.documentElement.classList.remove('dark'); localStorage.theme = 'light'; }
            }
        </script>
        <style>
            ::view-transition-old(root), ::view-transition-new(root) { animation: none; mix-blend-mode: normal; }
            ::view-transition-old(root) { z-index: 1; }
            ::view-transition-new(root) { z-index: 2; }
            .dark::view-transition-old(root) { z-index: 2; }
            .dark::view-transition-new(root) { z-index: 1; }
            
            body { transition: background-color 0.3s, color 0.3s; }
            .card-panel { background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px); border: 1px solid #e4e4e7; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
            .dark .card-panel { background: rgba(24, 24, 27, 0.85); backdrop-filter: blur(10px); border: 1px solid #27272a; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5); }
            input[type=range] { accent-color: #f97316; }
            
            #tsparticles { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; }
            
            /* Skeleton & Typing */
            .typing-effect::after { content: '|'; animation: blink 1s step-end infinite; }
            @keyframes blink { 50% { opacity: 0; } }
            .loader-bar { background: linear-gradient(90deg, transparent, rgba(249, 115, 22, 0.5), transparent); background-size: 200% 100%; animation: shimmer 1.5s infinite linear; }
            @keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
        </style>
    </head>
    <body class="min-h-screen bg-slate-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100 flex flex-col relative selection:bg-orange-500 selection:text-white">
        
        <!-- Particles Background -->
        <div id="tsparticles"></div>

        <!-- Header -->
        <header class="sticky top-0 z-50 bg-white/80 dark:bg-zinc-950/80 backdrop-blur-md border-b border-zinc-200 dark:border-zinc-800">
            <div class="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
                <a href="#" onclick="showPage('home')" class="flex items-center gap-3 group">
                    <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-orange-600 to-orange-400 flex items-center justify-center text-xl text-white shadow-lg">🌿</div>
                    <div>
                        <h1 class="font-extrabold text-lg tracking-tight">IRIS PRO</h1>
                        <p class="text-[10px] text-zinc-500 font-mono">Nguyễn Minh Hoàng</p>
                    </div>
                </a>
                
                <nav class="hidden md:flex items-center gap-6 text-sm font-semibold">
                    <button onclick="showPage('lab')" class="hover:text-orange-500 transition">Phân tích</button>
                    <button onclick="showPage('dashboard')" class="hover:text-orange-500 transition" id="nav-dash" style="display:none;">Dashboard</button>
                    <button onclick="showPage('admin')" class="hover:text-orange-500 transition text-red-500" id="nav-admin" style="display:none;">Admin</button>
                </nav>

                <div class="flex items-center gap-3">
                    <button onclick="toggleTheme(event)" class="w-10 h-10 rounded-xl bg-zinc-100 dark:bg-zinc-900 flex items-center justify-center border border-zinc-200 dark:border-zinc-800">🌗</button>
                    <button id="btn-login-modal" onclick="document.getElementById('auth-modal').classList.remove('hidden')" class="px-4 py-2 bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 text-sm font-bold rounded-xl">Đăng nhập</button>
                    <button id="btn-logout" onclick="logout()" class="hidden px-4 py-2 bg-red-50 dark:bg-red-900/20 text-red-600 text-sm font-bold rounded-xl">Đăng xuất</button>
                </div>
            </div>
        </header>

        <!-- Main Content Wrapper -->
        <main class="w-full flex-grow flex flex-col relative z-10" id="app-content">
            <!-- PAGE: HOME -->
            <div id="page-home" class="page-section">
                <section class="max-w-7xl mx-auto px-6 py-20 lg:py-32 grid lg:grid-cols-2 gap-12 items-center">
                    <div data-aos="fade-up">
                        <span class="px-3 py-1 text-xs font-bold uppercase tracking-wider text-orange-600 bg-orange-100 dark:bg-orange-500/10 rounded-full border border-orange-200 dark:border-orange-500/20">Machine Learning App</span>
                        <h2 class="mt-6 text-4xl lg:text-6xl font-extrabold text-zinc-900 dark:text-white leading-tight">
                            Hệ thống AI nhận diện <br><span class="text-transparent bg-clip-text bg-gradient-to-r from-orange-500 to-orange-400">Thực vật học</span>
                        </h2>
                        <p class="mt-6 text-lg text-zinc-600 dark:text-zinc-400 leading-relaxed max-w-lg">
                            Tác giả: <b>Nguyễn Minh Hoàng</b><br>Sức mạnh của Support Vector Machine (SVM) được trực quan hóa. Tích hợp API Key và quản lý cấu hình.
                        </p>
                        <button onclick="showPage('lab')" class="mt-8 px-8 py-3.5 bg-orange-500 hover:bg-orange-600 text-white font-semibold rounded-xl shadow-lg shadow-orange-500/30 flex items-center gap-2">Vào phòng thí nghiệm ➔</button>
                    </div>
                </section>
                
                <section class="max-w-7xl mx-auto px-6 py-20">
                    <div data-aos="fade-up" class="grid md:grid-cols-3 gap-6 mb-12">
                        <div class="card-panel p-6 rounded-3xl text-center group">
                            <div class="w-16 h-16 mx-auto bg-orange-50 dark:bg-orange-500/10 rounded-2xl flex items-center justify-center text-3xl mb-4 text-[#f97316]">🌸</div>
                            <h4 class="font-bold text-lg text-[#f97316]">Iris Setosa</h4><p class="text-sm text-zinc-500">Phân tách tuyến tính tuyệt đối.</p>
                        </div>
                        <div class="card-panel p-6 rounded-3xl text-center group">
                            <div class="w-16 h-16 mx-auto bg-orange-50 dark:bg-orange-500/10 rounded-2xl flex items-center justify-center text-3xl mb-4 text-[#fb923c]">🌺</div>
                            <h4 class="font-bold text-lg text-[#fb923c]">Iris Versicolor</h4><p class="text-sm text-zinc-500">Hình thái trung gian, cân bằng.</p>
                        </div>
                        <div class="card-panel p-6 rounded-3xl text-center group">
                            <div class="w-16 h-16 mx-auto bg-orange-50 dark:bg-orange-500/10 rounded-2xl flex items-center justify-center text-3xl mb-4 text-[#fdba74]">🌷</div>
                            <h4 class="font-bold text-lg text-[#fdba74]">Iris Virginica</h4><p class="text-sm text-zinc-500">Kích thước lớn nhất, tráng lệ.</p>
                        </div>
                    </div>
                </section>
            </div>

            <!-- PAGE: LAB -->
            <div id="page-lab" class="page-section hidden max-w-7xl mx-auto px-4 sm:px-6 py-8 w-full">
                <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start" data-aos="fade-up">
                    <div class="lg:col-span-5 card-panel rounded-3xl p-6">
                        <h2 class="font-bold text-lg mb-6">Tham số dự đoán</h2>
                        <div class="space-y-6">
                            <div><div class="flex justify-between text-xs font-bold mb-2"><span>Sepal Length</span><span id="txt-sl" class="text-orange-500">5.1</span></div><input type="range" id="sl" min="4.0" max="8.0" step="0.1" value="5.1" class="w-full" oninput="document.getElementById('txt-sl').innerText=this.value"></div>
                            <div><div class="flex justify-between text-xs font-bold mb-2"><span>Sepal Width</span><span id="txt-sw" class="text-orange-500">3.5</span></div><input type="range" id="sw" min="2.0" max="4.5" step="0.1" value="3.5" class="w-full" oninput="document.getElementById('txt-sw').innerText=this.value"></div>
                            <div><div class="flex justify-between text-xs font-bold mb-2"><span>Petal Length</span><span id="txt-pl" class="text-orange-500">1.4</span></div><input type="range" id="pl" min="1.0" max="7.0" step="0.1" value="1.4" class="w-full" oninput="document.getElementById('txt-pl').innerText=this.value"></div>
                            <div><div class="flex justify-between text-xs font-bold mb-2"><span>Petal Width</span><span id="txt-pw" class="text-orange-500">0.2</span></div><input type="range" id="pw" min="0.1" max="2.6" step="0.1" value="0.2" class="w-full" oninput="document.getElementById('txt-pw').innerText=this.value"></div>
                        </div>
                        <button onclick="executePrediction()" class="w-full mt-8 py-3.5 bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 font-bold rounded-xl shadow-lg hover:scale-[1.02] transition">Phân Tích AI</button>
                    </div>

                    <div class="lg:col-span-7 flex flex-col gap-6">
                        <div class="card-panel rounded-3xl p-8 flex items-center justify-between relative overflow-hidden min-h-[160px]">
                            <!-- Normal View -->
                            <div id="result-view" class="w-full flex justify-between items-center z-10 transition-opacity duration-300">
                                <div>
                                    <p class="text-xs font-bold text-zinc-500 uppercase tracking-wider">Kết quả thuật toán</p>
                                    <h3 id="res-title" class="text-3xl font-black mt-1">Chưa phân tích</h3>
                                </div>
                                <div class="text-right">
                                    <p class="text-xs font-bold text-zinc-500 uppercase tracking-wider">Tin cậy</p>
                                    <div class="text-4xl font-black font-mono text-orange-500" id="res-conf">0%</div>
                                </div>
                            </div>
                            
                            <!-- Skeleton & Typing View -->
                            <div id="skeleton-view" class="hidden absolute inset-0 bg-white dark:bg-zinc-900 p-8 z-20 flex justify-between items-center">
                                <div class="w-2/3 space-y-4">
                                    <div class="text-sm font-mono text-orange-500 typing-effect">Model.predict([features])...</div>
                                    <div class="h-8 w-48 bg-zinc-200 dark:bg-zinc-800 rounded loader-bar"></div>
                                </div>
                                <div class="h-12 w-24 bg-zinc-200 dark:bg-zinc-800 rounded loader-bar"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- PAGE: DASHBOARD (User) -->
            <div id="page-dashboard" class="page-section hidden max-w-7xl mx-auto px-4 sm:px-6 py-8 w-full" data-aos="fade-up">
                <h2 class="text-3xl font-bold mb-8 text-orange-500">Không gian làm việc cá nhân</h2>
                
                <div class="grid md:grid-cols-2 gap-6 mb-8">
                    <div class="card-panel p-6 rounded-3xl">
                        <h3 class="font-bold text-lg mb-4 text-zinc-900 dark:text-white">API Key của bạn</h3>
                        <div class="bg-zinc-100 dark:bg-zinc-900 p-3 rounded-lg font-mono text-sm text-orange-500 break-all" id="dash-api-key">Đang tải...</div>
                        <p class="text-xs text-zinc-500 mt-2">Sử dụng API key này để gọi hệ thống SVM từ Code Python cá nhân.</p>
                    </div>
                    
                    <div class="card-panel p-6 rounded-3xl">
                        <h3 class="font-bold text-lg mb-4 text-zinc-900 dark:text-white">Lưu cấu hình mô hình (Hyperparameters)</h3>
                        <div class="flex gap-2">
                            <select id="conf-kernel" class="bg-zinc-100 dark:bg-zinc-900 p-2 rounded text-sm outline-none"><option value="Linear">Linear</option><option value="RBF">RBF</option></select>
                            <input type="number" id="conf-c" placeholder="C value (ex: 1.0)" class="bg-zinc-100 dark:bg-zinc-900 p-2 rounded text-sm w-full outline-none">
                            <button onclick="saveConfig()" class="px-4 py-2 bg-orange-500 text-white rounded font-bold text-sm whitespace-nowrap">Lưu cấu hình</button>
                        </div>
                    </div>
                </div>

                <div class="card-panel p-6 rounded-3xl">
                    <h3 class="font-bold text-lg mb-4 text-zinc-900 dark:text-white">Lịch sử phân tích cá nhân</h3>
                    <div class="overflow-x-auto">
                        <table class="w-full text-left text-sm whitespace-nowrap">
                            <thead><tr class="text-zinc-500 uppercase text-xs border-b border-zinc-200 dark:border-zinc-800"><th class="py-2">Thời gian</th><th class="py-2">S.L</th><th class="py-2">P.L</th><th class="py-2">Loài</th><th class="py-2">Tin cậy</th></tr></thead>
                            <tbody id="dash-history" class="divide-y divide-zinc-100 dark:divide-zinc-800 font-mono"></tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- PAGE: ADMIN -->
            <div id="page-admin" class="page-section hidden max-w-7xl mx-auto px-4 sm:px-6 py-8 w-full" data-aos="fade-up">
                <h2 class="text-3xl font-bold mb-8 text-red-500">Quản trị Hệ thống (Admin)</h2>
                <div class="card-panel p-6 rounded-3xl">
                    <h3 class="font-bold text-lg mb-4 text-zinc-900 dark:text-white">Danh sách Người dùng</h3>
                    <div class="overflow-x-auto">
                        <table class="w-full text-left text-sm whitespace-nowrap">
                            <thead><tr class="text-zinc-500 uppercase text-xs border-b border-zinc-200 dark:border-zinc-800"><th class="py-2">ID</th><th class="py-2">Email</th><th class="py-2">Role</th><th class="py-2">API Key</th></tr></thead>
                            <tbody id="admin-users" class="divide-y divide-zinc-100 dark:divide-zinc-800 font-mono"></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </main>

        <!-- Auth Modal -->
        <div id="auth-modal" class="hidden fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div class="card-panel p-8 rounded-3xl w-full max-w-md relative">
                <button onclick="document.getElementById('auth-modal').classList.add('hidden')" class="absolute top-4 right-4 text-zinc-500 hover:text-zinc-900">✕</button>
                <h2 class="text-2xl font-bold mb-6 text-center text-zinc-900 dark:text-white">Đăng nhập Lab</h2>
                
                <!-- Social Logins (Mock UI) -->
                <div class="grid grid-cols-2 gap-3 mb-6">
                    <button class="py-2 flex justify-center items-center gap-2 border border-zinc-200 dark:border-zinc-700 rounded-xl hover:bg-zinc-100 dark:hover:bg-zinc-800 transition font-bold text-sm"><img src="https://www.svgrepo.com/show/475656/google-color.svg" class="w-5 h-5"> Google</button>
                    <button class="py-2 flex justify-center items-center gap-2 border border-zinc-200 dark:border-zinc-700 rounded-xl hover:bg-zinc-100 dark:hover:bg-zinc-800 transition font-bold text-sm"><img src="https://www.svgrepo.com/show/512317/github-142.svg" class="w-5 h-5 dark:invert"> GitHub</button>
                </div>
                
                <div class="flex items-center gap-2 text-zinc-400 text-xs uppercase mb-6"><hr class="flex-grow border-zinc-200 dark:border-zinc-700">hoặc bằng Email<hr class="flex-grow border-zinc-200 dark:border-zinc-700"></div>

                <input type="email" id="auth-email" placeholder="Email" class="w-full mb-3 p-3 rounded-xl bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 outline-none focus:border-orange-500 transition">
                <input type="password" id="auth-pass" placeholder="Mật khẩu" class="w-full mb-6 p-3 rounded-xl bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 outline-none focus:border-orange-500 transition">
                
                <div class="flex gap-3">
                    <button onclick="handleAuth('/api/auth/login')" class="w-full py-3 bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 font-bold rounded-xl hover:opacity-90">Đăng nhập</button>
                    <button onclick="handleAuth('/api/auth/register')" class="w-full py-3 bg-orange-500 text-white font-bold rounded-xl hover:opacity-90">Đăng ký</button>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer class="border-t border-zinc-200 dark:border-zinc-800 bg-white/50 dark:bg-zinc-950/50 backdrop-blur mt-auto relative z-10">
            <div class="max-w-7xl mx-auto px-6 py-6 flex flex-col md:flex-row items-center justify-between text-xs text-zinc-500">
                <span>Tác giả: <b>Nguyễn Minh Hoàng</b> | Botanical AI System</span>
                <span>Support Vector Machine (SVM) Laboratory</span>
            </div>
        </footer>

        <script>
            // Init Particles
            tsParticles.load("tsparticles", {
                preset: "links",
                background: { color: "transparent" },
                particles: {
                    number: { value: 60, density: { enable: true, area: 800 } },
                    color: { value: ["#f97316", "#fb923c", "#fdba74"] },
                    links: { enable: true, distance: 150, color: "#a1a1aa", opacity: 0.2, width: 1 },
                    move: { enable: true, speed: 1, direction: "none", random: true, straight: false, outModes: "out" },
                    size: { value: { min: 2, max: 5 } }
                }
            });

            // Init AOS
            AOS.init({ duration: 800, once: true });

            // Navigation
            function showPage(pageId) {
                document.querySelectorAll('.page-section').forEach(el => el.classList.add('hidden'));
                document.getElementById('page-' + pageId).classList.remove('hidden');
                AOS.refresh();
                if(pageId === 'dashboard') loadDashboard();
                if(pageId === 'admin') loadAdmin();
            }

            // Auth Logic
            let currentUser = null;
            
            async function checkAuth() {
                try {
                    const res = await fetch('/api/me');
                    if(res.ok) {
                        currentUser = await res.json();
                        document.getElementById('btn-login-modal').classList.add('hidden');
                        document.getElementById('btn-logout').classList.remove('hidden');
                        document.getElementById('nav-dash').style.display = 'block';
                        if(currentUser.role === 'admin') document.getElementById('nav-admin').style.display = 'block';
                    }
                } catch(e) {}
            }

            async function handleAuth(endpoint) {
                const email = document.getElementById('auth-email').value;
                const password = document.getElementById('auth-pass').value;
                if(!email || !password) return alert("Vui lòng nhập đủ thông tin");
                
                const res = await fetch(endpoint, {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({email, password})
                });
                const data = await res.json();
                if(res.ok) {
                    alert(data.message);
                    document.getElementById('auth-modal').classList.add('hidden');
                    window.location.reload();
                } else {
                    alert(data.detail);
                }
            }

            async function logout() {
                await fetch('/api/auth/logout', {method: 'POST'});
                window.location.reload();
            }

            // AI Prediction
            async function executePrediction() {
                // UI Skeleton Effect
                const normalView = document.getElementById('result-view');
                const skeletonView = document.getElementById('skeleton-view');
                normalView.classList.add('opacity-0');
                setTimeout(() => { normalView.classList.add('hidden'); skeletonView.classList.remove('hidden'); }, 300);

                const payload = {
                    sepal_length: parseFloat(document.getElementById('sl').value),
                    sepal_width: parseFloat(document.getElementById('sw').value),
                    petal_length: parseFloat(document.getElementById('pl').value),
                    petal_width: parseFloat(document.getElementById('pw').value)
                };
                
                // Simulate processing delay for effect
                await new Promise(r => setTimeout(r, 1500)); 

                const res = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
                const data = await res.json();
                
                document.getElementById('res-title').textContent = data.prediction;
                document.getElementById('res-title').style.color = data.color;
                document.getElementById('res-conf').textContent = `${Math.max(...data.confidences)}%`;
                
                skeletonView.classList.add('hidden');
                normalView.classList.remove('hidden');
                setTimeout(() => normalView.classList.remove('opacity-0'), 50);
            }

            // Dashboard Logic
            async function loadDashboard() {
                document.getElementById('dash-api-key').textContent = currentUser.api_key;
                const res = await fetch('/api/history');
                const history = await res.json();
                document.getElementById('dash-history').innerHTML = history.map(r => `
                    <tr><td class="py-2 text-zinc-500">${r.time}</td><td>${r.sl}</td><td>${r.pl}</td><td class="font-bold text-orange-500">${r.species}</td><td>${r.conf}%</td></tr>
                `).join('');
            }

            async function saveConfig() {
                const kernel = document.getElementById('conf-kernel').value;
                const c_val = parseFloat(document.getElementById('conf-c').value) || 1.0;
                await fetch('/api/configs', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({name: 'Profile 1', kernel, c_val, gamma: 0.1}) });
                alert("Đã lưu cấu hình lên Server!");
            }

            // Admin Logic
            async function loadAdmin() {
                const res = await fetch('/api/admin/users');
                if(!res.ok) return;
                const users = await res.json();
                document.getElementById('admin-users').innerHTML = users.map(u => `
                    <tr><td class="py-2">${u.id}</td><td>${u.email}</td><td><span class="px-2 py-1 bg-zinc-200 dark:bg-zinc-800 rounded text-xs">${u.role}</span></td><td>${u.api_key}</td></tr>
                `).join('');
            }

            window.addEventListener('DOMContentLoaded', checkAuth);
        </script>
    </body>
    </html>
    """
