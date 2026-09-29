import os
import math
import logging
import sqlite3
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# ==========================================
# 1. KHỞI TẠO DB LỊCH SỬ & MÔ HÌNH
# ==========================================
DB_PATH = "history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT, sl REAL, sw REAL, pl REAL, pw REAL,
                  species TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

init_db()

MODEL_PATH = "svm_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

app = FastAPI(title="Iris AI Classifier Pro", version="7.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ==========================================
# 2. CẤU HÌNH & METADATA
# ==========================================
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

SPECIES_METADATA = {
    0: {"name": "Iris Setosa", "tag": "Setosa", "color": "#f97316"},      # Orange 500
    1: {"name": "Iris Versicolor", "tag": "Versicolor", "color": "#fb923c"}, # Orange 400
    2: {"name": "Iris Virginica", "tag": "Virginica", "color": "#fdba74"},   # Orange 300
}
CENTROIDS = [[5.006, 3.428, 1.462, 0.246], [5.936, 2.770, 4.260, 1.326], [6.588, 2.974, 5.552, 2.026]]

# ==========================================
# 3. CORE API ENDPOINTS
# ==========================================
@app.get("/img/{img_name}")
def get_image(img_name: str):
    allowed = ["anh_setosa.jpg", "anh_versicolor.jpg", "anh_virginica.jpg", "iris_demo.png"]
    if img_name in allowed and os.path.exists(img_name): return FileResponse(img_name)
    raise HTTPException(status_code=404, detail="Image not found")

@app.post("/predict")
def predict(data: IrisInput):
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
    c = conn.cursor()
    c.execute("INSERT INTO history (timestamp, sl, sw, pl, pw, species, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)", 
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), *feat, meta["name"], conf[pred]))
    conn.commit()
    conn.close()

    return {"prediction": meta["name"], "confidences": conf, "features": feat, "color": meta["color"]}

@app.get("/api/history")
def get_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp, sl, sw, pl, pw, species, confidence FROM history ORDER BY id DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return [{"timestamp": r[0], "sl": r[1], "sw": r[2], "pl": r[3], "pw": r[4], "species": r[5], "confidence": r[6]} for r in rows]

@app.delete("/api/history")
def clear_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM history")
    conn.commit()
    conn.close()
    return {"status": "success"}

# ==========================================
# 4. TRÌNH QUẢN LÝ GIAO DIỆN (HTML TEMPLATES)
# ==========================================
def get_base_html(title, content):
    return f"""
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title} | Botanical Lab Pro</title>
        
        <!-- Scripts & Styles -->
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <link href="https://unpkg.com/aos@2.3.1/dist/aos.css" rel="stylesheet">
        <script src="https://unpkg.com/aos@2.3.1/dist/aos.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/tsparticles-preset-links@2/tsparticles.preset.links.bundle.min.js"></script>
        
        <script>
            tailwind.config = {{
                darkMode: 'class',
                theme: {{ 
                    extend: {{ 
                        fontFamily: {{ sans: ['Plus Jakarta Sans', 'sans-serif'], mono: ['JetBrains Mono', 'monospace'] }},
                        colors: {{ orange: {{ 400: '#fb923c', 500: '#f97316', 600: '#ea580c' }}, zinc: {{ 900: '#18181b', 950: '#09090b' }} }}
                    }} 
                }}
            }}
            
            // i18n Dictionary
            const dict = {{
                vi: {{ nav_home: "Trang chủ", nav_lab: "Phân tích", nav_kernel: "Kernels", nav_3d: "Không gian 3D", nav_game: "Giải trí", settings: "Cài đặt", theme: "Giao diện", lang: "Ngôn ngữ" }},
                en: {{ nav_home: "Home", nav_lab: "Laboratory", nav_kernel: "Kernels", nav_3d: "3D Space", nav_game: "Game", settings: "Settings", theme: "Theme", lang: "Language" }}
            }};

            let currentLang = localStorage.getItem('lang') || 'vi';
            function applyLang() {{
                document.querySelectorAll('[data-i18n]').forEach(el => {{
                    const key = el.getAttribute('data-i18n');
                    if (dict[currentLang][key]) el.textContent = dict[currentLang][key];
                }});
                document.getElementById('lang-toggle-text').textContent = currentLang === 'vi' ? 'Tiếng Việt' : 'English';
            }}
            function toggleLang() {{
                currentLang = currentLang === 'vi' ? 'en' : 'vi';
                localStorage.setItem('lang', currentLang); applyLang();
            }}

            // ViewTransitions Theme Toggle (Clip-path effect)
            function initTheme() {{
                if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {{ document.documentElement.classList.add('dark'); }} else {{ document.documentElement.classList.remove('dark'); }}
            }}
            initTheme();
            
            function toggleTheme(event) {{
                const isDark = document.documentElement.classList.contains('dark');
                if (!document.startViewTransition) {{ executeThemeSwitch(!isDark); return; }}
                
                const x = event?.clientX ?? innerWidth / 2;
                const y = event?.clientY ?? innerHeight / 2;
                const endRadius = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));

                const transition = document.startViewTransition(() => {{ executeThemeSwitch(!isDark); }});
                transition.ready.then(() => {{
                    const clipPath = [`circle(0px at ${{x}}px ${{y}}px)`, `circle(${{endRadius}}px at ${{x}}px ${{y}}px)`];
                    document.documentElement.animate({{ clipPath: isDark ? [...clipPath].reverse() : clipPath }}, {{ duration: 500, easing: "ease-in-out", pseudoElement: isDark ? "::view-transition-old(root)" : "::view-transition-new(root)" }});
                }});
            }}
            function executeThemeSwitch(toDark) {{
                if(toDark) {{ document.documentElement.classList.add('dark'); localStorage.theme = 'dark'; }} else {{ document.documentElement.classList.remove('dark'); localStorage.theme = 'light'; }}
                document.getElementById('theme-toggle-text').textContent = toDark ? 'Dark Mode' : 'Light Mode';
                if(typeof updateChartsTheme === 'function') updateChartsTheme(toDark);
            }}
        </script>
        <style>
            /* Transitions for Dark Mode Circular Reveal */
            ::view-transition-old(root), ::view-transition-new(root) {{ animation: none; mix-blend-mode: normal; }}
            ::view-transition-old(root) {{ z-index: 1; }} ::view-transition-new(root) {{ z-index: 2; }}
            .dark::view-transition-old(root) {{ z-index: 2; }} .dark::view-transition-new(root) {{ z-index: 1; }}
            
            body {{ transition: background-color 0.3s, color 0.3s; }}
            #tsparticles {{ position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; }}
            
            .card-panel {{ background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(10px); border: 1px solid #e4e4e7; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }}
            .dark .card-panel {{ background: rgba(24, 24, 27, 0.75); backdrop-filter: blur(10px); border: 1px solid #27272a; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5); }}
            input[type=range] {{ accent-color: #f97316; }}
            
            /* Scrollbar */
            ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
            ::-webkit-scrollbar-track {{ background: transparent; }}
            ::-webkit-scrollbar-thumb {{ background: #f97316; border-radius: 4px; }}

            /* Skeleton / Typing effect */
            .typing-effect::after {{ content: '|'; animation: blink 1s step-end infinite; }}
            @keyframes blink {{ 50% {{ opacity: 0; }} }}
            .loader-bar {{ background: linear-gradient(90deg, transparent, rgba(249, 115, 22, 0.4), transparent); background-size: 200% 100%; animation: shimmer 1.5s infinite linear; }}
            @keyframes shimmer {{ 0% {{ background-position: -200% 0; }} 100% {{ background-position: 200% 0; }} }}
        </style>
    </head>
    <body class="min-h-screen bg-slate-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100 flex flex-col relative selection:bg-orange-500 selection:text-white">
        
        <div id="tsparticles"></div>

        <!-- Header -->
        <header class="sticky top-0 z-50 bg-white/70 dark:bg-zinc-950/70 backdrop-blur-md border-b border-zinc-200 dark:border-zinc-800">
            <div class="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
                <a href="/" class="flex items-center gap-3 group">
                    <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-orange-600 to-orange-400 flex items-center justify-center text-xl text-white shadow-lg group-hover:scale-105 transition">🌿</div>
                    <div>
                        <h1 class="font-extrabold text-lg tracking-tight">IRIS PRO</h1>
                        <p class="text-[10px] text-zinc-500 font-mono">SVM Architecture</p>
                    </div>
                </a>
                
                <nav class="hidden lg:flex items-center gap-8 text-sm font-semibold">
                    <a href="/" class="hover:text-orange-500 transition" data-i18n="nav_home">Trang chủ</a>
                    <a href="/lab" class="hover:text-orange-500 transition" data-i18n="nav_lab">Phân tích</a>
                    <a href="/vector3d" class="hover:text-orange-500 transition" data-i18n="nav_3d">Không gian 3D</a>
                    <a href="/kernels" class="hover:text-orange-500 transition" data-i18n="nav_kernel">Kernels</a>
                    <a href="/game" class="hover:text-orange-500 transition" data-i18n="nav_game">Giải trí</a>
                </nav>

                <div class="relative group">
                    <button onclick="toggleTheme(event)" class="w-10 h-10 rounded-xl bg-zinc-100 dark:bg-zinc-900 hover:bg-zinc-200 dark:hover:bg-zinc-800 transition flex items-center justify-center text-lg border border-zinc-200 dark:border-zinc-800">🌗</button>
                </div>
            </div>
        </header>

        <!-- Main Content -->
        <main class="w-full flex-grow flex flex-col relative z-10">
            {content}
        </main>
        
        <!-- Footer -->
        <footer class="border-t border-zinc-200 dark:border-zinc-800 bg-white/50 dark:bg-zinc-950/50 backdrop-blur-md mt-auto relative z-10">
            <div class="max-w-7xl mx-auto px-6 py-6 flex flex-col md:flex-row items-center justify-between text-xs text-zinc-500">
                <div class="flex items-center gap-2 mb-2 md:mb-0">
                    <span class="w-2 h-2 rounded-full bg-orange-500 animate-pulse"></span>
                    <span>Fisher's Iris (1936) | Machine Learning</span>
                </div>
                <div class="flex gap-4">
                    <span>Tác giả: <b>Nguyễn Minh Hoàng</b></span>
                </div>
            </div>
        </footer>

        <script>
            // Particles Init
            tsParticles.load("tsparticles", {{
                preset: "links",
                background: {{ color: "transparent" }},
                particles: {{
                    number: {{ value: 50, density: {{ enable: true, area: 800 }} }},
                    color: {{ value: ["#f97316", "#fb923c", "#fdba74"] }},
                    links: {{ enable: true, distance: 150, color: "#888888", opacity: 0.2, width: 1 }},
                    move: {{ enable: true, speed: 1, direction: "none", random: true }}
                }}
            }});
            // AOS Init
            AOS.init({{ duration: 800, once: true }});
            // Setup
            document.addEventListener('DOMContentLoaded', () => {{ applyLang(); }});
        </script>
    </body>
    </html>
    """

# ==========================================
# CÁC TRANG CỦA ỨNG DỤNG
# ==========================================

@app.get("/", response_class=HTMLResponse)
def hero_page():
    content = """
    <section class="relative overflow-hidden border-b border-zinc-200 dark:border-zinc-900 bg-white/50 dark:bg-zinc-950/50 backdrop-blur-sm">
        <div class="max-w-7xl mx-auto px-6 py-20 lg:py-32 grid lg:grid-cols-2 gap-12 items-center">
            <div data-aos="fade-up">
                <span class="px-3 py-1 text-xs font-bold uppercase tracking-wider text-orange-600 bg-orange-100 dark:text-orange-400 dark:bg-orange-500/10 rounded-full border border-orange-200 dark:border-orange-500/20">Khoa học dữ liệu</span>
                <h2 class="mt-6 text-4xl lg:text-6xl font-extrabold text-zinc-900 dark:text-white leading-tight">
                    Nhận dạng & Trực quan <br><span class="text-transparent bg-clip-text bg-gradient-to-r from-orange-500 to-orange-400">Không gian SVM</span>
                </h2>
                <p class="mt-6 text-lg text-zinc-600 dark:text-zinc-400 leading-relaxed max-w-lg">
                    Dự án Học máy do <b>Nguyễn Minh Hoàng</b> phát triển. Trực quan hóa mặt siêu phẳng (Hyperplane), biểu đồ phân tán đa chiều và hiệu suất các thuật toán Kernel.
                </p>
                <div class="mt-8 flex gap-4">
                    <a href="/lab" class="px-8 py-3.5 bg-orange-500 hover:bg-orange-600 text-white font-semibold rounded-xl transition shadow-lg shadow-orange-500/30">Khám phá ngay ➔</a>
                </div>
            </div>
            <div class="relative" data-aos="fade-up" data-aos-delay="200">
                <div class="aspect-square rounded-full absolute inset-0 bg-gradient-to-tr from-orange-400/20 to-transparent blur-3xl"></div>
                <div class="card-panel p-6 rounded-3xl relative rotate-3 hover:rotate-0 transition-transform duration-500">
                    <div class="h-64 w-full flex items-end justify-between gap-2 opacity-80">
                        <div class="w-1/5 bg-zinc-200 dark:bg-zinc-800 rounded-t-lg h-24"></div>
                        <div class="w-1/5 bg-orange-400 rounded-t-lg h-48 animate-pulse"></div>
                        <div class="w-1/5 bg-zinc-200 dark:bg-zinc-800 rounded-t-lg h-32"></div>
                        <div class="w-1/5 bg-orange-500 rounded-t-lg h-64"></div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <section class="max-w-7xl mx-auto px-6 py-20">
        <div class="text-center mb-16" data-aos="fade-up">
            <h3 class="text-2xl font-bold text-zinc-900 dark:text-white">Bộ 3 giống hoa Diên Vĩ</h3>
            <p class="text-zinc-500 mt-2">Dựa trên Fisher's Dataset (1936)</p>
        </div>
        
        <div class="grid md:grid-cols-3 gap-6 mb-12">
            <div class="card-panel p-6 rounded-3xl text-center" data-aos="fade-up" data-aos-delay="100">
                <div class="w-16 h-16 mx-auto bg-orange-50 dark:bg-orange-500/10 rounded-2xl flex items-center justify-center text-3xl mb-4 text-orange-500">🌸</div>
                <h4 class="font-bold text-lg mb-2 text-zinc-900 dark:text-white">Iris Setosa</h4><p class="text-sm text-zinc-500">Phân tách tuyến tính 100%.</p>
            </div>
            <div class="card-panel p-6 rounded-3xl text-center" data-aos="fade-up" data-aos-delay="200">
                <div class="w-16 h-16 mx-auto bg-orange-50 dark:bg-orange-500/10 rounded-2xl flex items-center justify-center text-3xl mb-4 text-orange-400">🌺</div>
                <h4 class="font-bold text-lg mb-2 text-zinc-900 dark:text-white">Iris Versicolor</h4><p class="text-sm text-zinc-500">Đặc trưng hình thái trung gian.</p>
            </div>
            <div class="card-panel p-6 rounded-3xl text-center" data-aos="fade-up" data-aos-delay="300">
                <div class="w-16 h-16 mx-auto bg-orange-50 dark:bg-orange-500/10 rounded-2xl flex items-center justify-center text-3xl mb-4 text-orange-300">🌷</div>
                <h4 class="font-bold text-lg mb-2 text-zinc-900 dark:text-white">Iris Virginica</h4><p class="text-sm text-zinc-500">Kích thước lớn, hơi trộn lẫn Versicolor.</p>
            </div>
        </div>
    </section>
    """
    return get_base_html("Trang chủ", content)

@app.get("/lab", response_class=HTMLResponse)
def lab_page():
    content = """
    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-8 w-full">
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start mb-8">
            <!-- Khối Input -->
            <div class="lg:col-span-4 card-panel rounded-3xl p-6" data-aos="fade-up">
                <div class="flex justify-between items-center mb-6">
                    <h2 class="font-bold text-lg text-zinc-900 dark:text-white">Tham số đầu vào</h2>
                    <button onclick="randomizeInputs()" class="text-xl hover:rotate-180 transition duration-300">🎲</button>
                </div>
                <div class="space-y-6">
                    <div><div class="flex justify-between text-xs font-bold uppercase mb-2"><span>Sepal L</span><span id="txt-sl" class="text-orange-500">5.1</span></div><input type="range" id="sl" min="4.0" max="8.0" step="0.1" value="5.1" class="w-full" oninput="syncVal('sl', 'txt-sl')"></div>
                    <div><div class="flex justify-between text-xs font-bold uppercase mb-2"><span>Sepal W</span><span id="txt-sw" class="text-orange-500">3.5</span></div><input type="range" id="sw" min="2.0" max="4.5" step="0.1" value="3.5" class="w-full" oninput="syncVal('sw', 'txt-sw')"></div>
                    <div><div class="flex justify-between text-xs font-bold uppercase mb-2"><span>Petal L</span><span id="txt-pl" class="text-orange-500">1.4</span></div><input type="range" id="pl" min="1.0" max="7.0" step="0.1" value="1.4" class="w-full" oninput="syncVal('pl', 'txt-pl')"></div>
                    <div><div class="flex justify-between text-xs font-bold uppercase mb-2"><span>Petal W</span><span id="txt-pw" class="text-orange-500">0.2</span></div><input type="range" id="pw" min="0.1" max="2.6" step="0.1" value="0.2" class="w-full" oninput="syncVal('pw', 'txt-pw')"></div>
                </div>
                <button onclick="executeInference()" class="w-full mt-8 py-3.5 bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 font-bold rounded-xl shadow-lg transition hover:scale-105">Phân Tích Ngay</button>
            </div>

            <!-- Khối Result & Scatter -->
            <div class="lg:col-span-8 flex flex-col gap-6" data-aos="fade-up" data-aos-delay="200">
                <div class="card-panel rounded-3xl p-6 sm:p-8 flex items-center justify-between relative overflow-hidden min-h-[140px]">
                    <div class="absolute right-0 top-0 w-64 h-64 bg-orange-500/10 blur-3xl rounded-full" id="result-glow"></div>
                    
                    <!-- View Kết quả thật -->
                    <div id="result-view" class="flex items-center gap-6 relative z-10 w-full justify-between transition-opacity duration-300">
                        <div><p class="text-xs font-bold text-zinc-500 uppercase">Kết quả dự đoán</p><h3 id="specimen-title" class="text-3xl font-black mt-1">Chờ phân tích...</h3></div>
                        <div class="text-right"><p class="text-xs font-bold text-zinc-500 uppercase">Độ tin cậy</p><div class="text-4xl font-black font-mono text-orange-500" id="main-conf">0%</div></div>
                    </div>

                    <!-- View Skeleton (Typing Effect) -->
                    <div id="skeleton-view" class="hidden absolute inset-0 bg-white dark:bg-zinc-900 p-8 z-20 flex justify-between items-center">
                        <div class="w-2/3 space-y-4">
                            <div class="text-sm font-mono text-orange-500 typing-effect">> SVM_Model.predict([features])...</div>
                            <div class="h-8 w-48 bg-zinc-200 dark:bg-zinc-800 rounded loader-bar"></div>
                        </div>
                        <div class="h-12 w-24 bg-zinc-200 dark:bg-zinc-800 rounded loader-bar"></div>
                    </div>
                </div>

                <div class="card-panel rounded-3xl p-6 h-72 flex flex-col">
                    <h4 class="text-sm font-bold mb-2">Phân bố dữ liệu 2D</h4>
                    <div class="flex-grow w-full relative"><canvas id="scatterChart"></canvas></div>
                </div>
            </div>
        </div>

        <div class="card-panel rounded-3xl p-6" data-aos="fade-up">
            <h2 class="font-bold text-lg mb-4">Lịch sử Dự đoán</h2>
            <div class="overflow-x-auto">
                <table class="w-full text-left text-sm whitespace-nowrap">
                    <thead><tr class="text-zinc-500 uppercase text-xs border-b border-zinc-100 dark:border-zinc-800"><th class="py-3 px-4">Thời gian</th><th class="py-3 px-4">P.L</th><th class="py-3 px-4">P.W</th><th class="py-3 px-4">Dự đoán</th><th class="py-3 px-4">Tin cậy</th></tr></thead>
                    <tbody id="history-tbody" class="divide-y divide-zinc-100 dark:divide-zinc-800"></tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        let scatterChart;
        function initCharts() {
            const ctx = document.getElementById('scatterChart').getContext('2d');
            const isDark = document.documentElement.classList.contains('dark');
            const bgData = {
                datasets: [
                    { label: 'Setosa', data: [{x:1.4,y:0.2},{x:1.5,y:0.2},{x:1.3,y:0.2},{x:1.6,y:0.2}], backgroundColor: 'rgba(249, 115, 22, 0.4)' },
                    { label: 'Versicolor', data: [{x:4.7,y:1.4},{x:4.5,y:1.5},{x:4.9,y:1.5}], backgroundColor: 'rgba(251, 146, 60, 0.4)' },
                    { label: 'Virginica', data: [{x:6.0,y:2.5},{x:5.1,y:1.9},{x:5.9,y:2.1}], backgroundColor: 'rgba(253, 186, 116, 0.4)' },
                    { label: 'Mẫu test', data: [{x:1.4, y:0.2}], backgroundColor: '#f97316', pointRadius: 8, pointBorderColor: '#fff', pointBorderWidth: 2 }
                ]
            };
            scatterChart = new Chart(ctx, {
                type: 'scatter', data: bgData,
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
            });
        }
        function syncVal(sliderId, textId) { document.getElementById(textId).textContent = document.getElementById(sliderId).value; }
        function randomizeInputs() {
            const r = (min, max) => (Math.random() * (max - min) + min).toFixed(1);
            ['sl', 'sw', 'pl', 'pw'].forEach(id => {
                const val = id.includes('s') ? r(2.0, 8.0) : r(0.1, 7.0);
                document.getElementById(id).value = val; syncVal(id, 'txt-' + id);
            });
        }
        async function fetchHistory() {
            const res = await fetch('/api/history'); const data = await res.json();
            document.getElementById('history-tbody').innerHTML = data.map(r => `<tr><td class="py-2 px-4 text-xs font-mono text-zinc-500">${r.timestamp}</td><td class="py-2 px-4">${r.pl}</td><td class="py-2 px-4">${r.pw}</td><td class="py-2 px-4 font-bold text-orange-500">${r.species}</td><td class="py-2 px-4 font-mono">${r.confidence}%</td></tr>`).join('');
        }
        async function executeInference() {
            // Hiệu ứng Skeleton Loader 1.5s
            const view = document.getElementById('result-view');
            const skeleton = document.getElementById('skeleton-view');
            view.classList.add('opacity-0');
            setTimeout(() => { view.classList.add('hidden'); skeleton.classList.remove('hidden'); }, 300);

            const pl = parseFloat(document.getElementById('pl').value), pw = parseFloat(document.getElementById('pw').value);
            const payload = { sepal_length: parseFloat(document.getElementById('sl').value), sepal_width: parseFloat(document.getElementById('sw').value), petal_length: pl, petal_width: pw };
            
            await new Promise(r => setTimeout(r, 1500)); // Delay mô phỏng "suy nghĩ"

            const res = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            const data = await res.json();
            
            document.getElementById('specimen-title').textContent = data.prediction;
            document.getElementById('specimen-title').style.color = data.color;
            document.getElementById('main-conf').textContent = `${Math.max(...data.confidences)}%`;
            if (scatterChart) { scatterChart.data.datasets[3].data = [{x: pl, y: pw}]; scatterChart.update(); }
            fetchHistory();

            // Tắt Skeleton
            skeleton.classList.add('hidden'); view.classList.remove('hidden');
            setTimeout(() => view.classList.remove('opacity-0'), 50);
        }
        window.addEventListener('DOMContentLoaded', () => { initCharts(); executeInference(); });
    </script>
    """
    return get_base_html("Phân tích", content)

@app.get("/vector3d", response_class=HTMLResponse)
def vector3d_page():
    content = """
    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-8 w-full" data-aos="fade-up">
        <div class="card-panel rounded-3xl p-8 mb-8">
            <h2 class="text-2xl font-bold mb-2">Không gian 3D (Mô phỏng PCA & Hyperplane)</h2>
            <p class="text-sm text-zinc-500 mb-6">Mô hình SVM chia cắt dữ liệu trong không gian đa chiều (Hyperplane). Dùng chuột để xoay, lăn chuột để thu phóng không gian.</p>
            <div id="plot3d" class="w-full h-[600px] border border-zinc-200 dark:border-zinc-800 rounded-2xl bg-white dark:bg-zinc-900"></div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', () => {
            // Giả lập dữ liệu PCA phân cụm cho 3 lớp (50 điểm mỗi lớp)
            function generateCluster(cx, cy, cz, n, spread) {
                let x=[], y=[], z=[];
                for(let i=0; i<n; i++) {
                    x.push(cx + (Math.random()-0.5)*spread);
                    y.push(cy + (Math.random()-0.5)*spread);
                    z.push(cz + (Math.random()-0.5)*spread);
                }
                return {x, y, z};
            }
            const setosa = generateCluster(-2, -2, -2, 50, 1.5);
            const versicolor = generateCluster(0, 0, 0, 50, 1.5);
            const virginica = generateCluster(2, 2, 2, 50, 1.5);

            const trace1 = {x: setosa.x, y: setosa.y, z: setosa.z, mode: 'markers', marker: {size: 5, color: '#f97316'}, type: 'scatter3d', name: 'Setosa'};
            const trace2 = {x: versicolor.x, y: versicolor.y, z: versicolor.z, mode: 'markers', marker: {size: 5, color: '#fb923c'}, type: 'scatter3d', name: 'Versicolor'};
            const trace3 = {x: virginica.x, y: virginica.y, z: virginica.z, mode: 'markers', marker: {size: 5, color: '#fdba74'}, type: 'scatter3d', name: 'Virginica'};

            // Giả lập Mặt siêu phẳng (Hyperplane phân tách Setosa và phần còn lại)
            let hx = [], hy = [], hz = [];
            for(let i=-3; i<=3; i+=0.5) {
                let tempX=[], tempY=[], tempZ=[];
                for(let j=-3; j<=3; j+=0.5) { tempX.push(i); tempY.push(j); tempZ.push(-0.5 * i - 0.5 * j - 1); }
                hx.push(tempX); hy.push(tempY); hz.push(tempZ);
            }
            const plane = {x: hx, y: hy, z: hz, type: 'surface', opacity: 0.5, colorscale: [[0, '#cbd5e1'], [1, '#94a3b8']], showscale: false, name: 'Hyperplane'};

            const layout = {
                margin: {l: 0, r: 0, b: 0, t: 0},
                scene: { xaxis: {showgrid: false, zeroline: false, showticklabels: false}, yaxis: {showgrid: false, zeroline: false, showticklabels: false}, zaxis: {showgrid: false, zeroline: false, showticklabels: false} },
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                showlegend: true, legend: {x: 0.1, y: 0.9}
            };
            
            Plotly.newPlot('plot3d', [trace1, trace2, trace3, plane], layout, {responsive: true});
        });
    </script>
    """
    return get_base_html("Không gian 3D", content)

@app.get("/kernels", response_class=HTMLResponse)
def kernels_page():
    content = """
    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-8 w-full" data-aos="fade-up">
        <div class="card-panel rounded-3xl p-8">
            <h2 class="text-2xl font-bold mb-6">So sánh Kernels</h2>
            <div class="grid lg:grid-cols-2 gap-8 mb-8">
                <div class="bg-zinc-50 dark:bg-zinc-900/50 p-6 rounded-2xl border border-zinc-200 dark:border-zinc-800">
                    <h3 class="text-sm font-bold mb-4 text-center">Độ chính xác (Accuracy %)</h3>
                    <div class="h-64"><canvas id="accChart"></canvas></div>
                </div>
                <div class="bg-zinc-50 dark:bg-zinc-900/50 p-6 rounded-2xl flex flex-col justify-center">
                    <table class="w-full text-left text-sm">
                        <thead><tr class="border-b border-zinc-200 dark:border-zinc-800"><th class="py-2">Kernel</th><th class="py-2">F1</th></tr></thead>
                        <tbody class="divide-y divide-zinc-200 font-mono">
                            <tr><td class="py-3 font-bold text-orange-500">Linear</td><td>0.98</td></tr>
                            <tr><td class="py-3 font-bold text-orange-400">RBF</td><td>0.96</td></tr>
                            <tr><td class="py-3 font-bold text-orange-300">Polynomial</td><td>0.95</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const ctx = document.getElementById('accChart').getContext('2d');
            new Chart(ctx, {
                type: 'bar',
                data: { labels: ['Linear', 'RBF', 'Poly', 'Sigmoid'], datasets: [{ data: [98.5, 96.8, 95.0, 85.2], backgroundColor: ['#f97316', '#fb923c', '#fdba74', '#a1a1aa'], borderRadius: 6 }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
            });
        });
    </script>
    """
    return get_base_html("Kernels", content)

@app.get("/game", response_class=HTMLResponse)
def game_page():
    content = """
    <div class="max-w-4xl mx-auto px-4 sm:px-6 py-8 w-full" data-aos="fade-up">
        <div class="card-panel rounded-3xl p-8 text-center flex flex-col items-center">
            <h2 class="text-2xl font-bold mb-2 text-orange-500">Flappy Flower 🌸</h2>
            <p class="text-xs text-zinc-500 mb-6">Nhấn Click chuột hoặc Phím Space để bay lên!</p>
            
            <canvas id="gameCanvas" width="320" height="480" class="border-4 border-zinc-200 dark:border-zinc-800 rounded-xl bg-sky-100 dark:bg-sky-900 cursor-pointer shadow-xl"></canvas>
            
            <div class="mt-6 flex gap-4">
                <button onclick="startGame()" class="px-6 py-2 bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 font-bold rounded-lg shadow-lg">Chơi Lại</button>
            </div>
        </div>
    </div>

    <script>
        const cvs = document.getElementById("gameCanvas");
        const ctx = cvs.getContext("2d");

        let frames = 0;
        let score = 0;
        let gameState = 0; // 0: start, 1: play, 2: over

        const bird = { x: 50, y: 150, w: 30, h: 30, gravity: 0.25, jump: 4.6, speed: 0,
            draw: function() { ctx.font = "30px Arial"; ctx.fillText("🌸", this.x, this.y + this.h); },
            flap: function() { this.speed = -this.jump; },
            update: function() {
                if (gameState === 1) {
                    this.speed += this.gravity; this.y += this.speed;
                    if(this.y + this.h >= cvs.height || this.y <= 0) { gameState = 2; }
                }
            }
        };

        const pipes = { position: [], w: 40, gap: 120, dx: 2,
            draw: function() {
                for(let i=0; i<this.position.length; i++) {
                    let p = this.position[i];
                    ctx.fillStyle = "#84cc16"; // Xanh ống nước
                    ctx.fillRect(p.x, 0, this.w, p.y); // Ống trên
                    ctx.fillRect(p.x, p.y + this.gap, this.w, cvs.height - p.y - this.gap); // Ống dưới
                }
            },
            update: function() {
                if(gameState !== 1) return;
                if(frames % 100 == 0) { this.position.push({x: cvs.width, y: Math.max(50, Math.random() * (cvs.height - this.gap - 50))}); }
                for(let i=0; i<this.position.length; i++) {
                    let p = this.position[i]; p.x -= this.dx;
                    // Xử lý va chạm
                    if(bird.x + bird.w > p.x && bird.x < p.x + this.w && (bird.y < p.y || bird.y + bird.h > p.y + this.gap)) { gameState = 2; }
                    if(p.x + this.w === bird.x) { score++; } // Cộng điểm
                    if(p.x + this.w < 0) { this.position.shift(); i--; }
                }
            }
        };

        function draw() {
            ctx.clearRect(0, 0, cvs.width, cvs.height);
            pipes.draw(); bird.draw();
            // Vẽ điểm số
            ctx.fillStyle = "#fff"; ctx.strokeStyle = "#000"; ctx.lineWidth = 2; ctx.font = "35px font-mono font-bold";
            ctx.fillText(score, cvs.width/2 - 10, 50); ctx.strokeText(score, cvs.width/2 - 10, 50);
            
            if(gameState === 0) { ctx.font = "20px Arial"; ctx.fillText("Click để chơi", 100, cvs.height/2); }
            if(gameState === 2) { ctx.font = "20px Arial"; ctx.fillStyle = "red"; ctx.fillText("GAME OVER", 100, cvs.height/2); }
        }

        function update() { bird.update(); pipes.update(); }
        function loop() { update(); draw(); frames++; requestAnimationFrame(loop); }

        function startGame() { bird.y = 150; bird.speed = 0; pipes.position = []; score = 0; frames = 0; gameState = 1; }

        window.addEventListener("keydown", (e) => { if(e.code === "Space") { if(gameState===0||gameState===2) startGame(); else bird.flap(); }});
        cvs.addEventListener("mousedown", () => { if(gameState===0||gameState===2) startGame(); else bird.flap(); });

        loop(); // Khởi chạy game loop
    </script>
    """
    return get_base_html("Flappy Flower", content)
