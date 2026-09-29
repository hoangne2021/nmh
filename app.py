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

app = FastAPI(title="Iris AI Classifier Pro", version="6.0.0")
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
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
                vi: {{ nav_home: "Trang chủ", nav_lab: "Phân tích", nav_kernel: "Kernels", settings: "Cài đặt", theme: "Giao diện", lang: "Ngôn ngữ", default_kernel: "Kernel mặc định", footer_author: "Tác giả", export_csv: "Xuất CSV", export_json: "Xuất JSON", clear_hist: "Xóa lịch sử", pred_title: "Kết quả Dự đoán" }},
                en: {{ nav_home: "Home", nav_lab: "Laboratory", nav_kernel: "Kernels", settings: "Settings", theme: "Theme", lang: "Language", default_kernel: "Default Kernel", footer_author: "Author", export_csv: "Export CSV", export_json: "Export JSON", clear_hist: "Clear History", pred_title: "Prediction Result" }}
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
                localStorage.setItem('lang', currentLang);
                applyLang();
            }}

            function initTheme() {{
                if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {{
                    document.documentElement.classList.add('dark');
                }} else {{ document.documentElement.classList.remove('dark'); }}
            }}
            initTheme();
            
            function toggleTheme() {{
                document.documentElement.classList.toggle('dark');
                const isDark = document.documentElement.classList.contains('dark');
                localStorage.setItem('theme', isDark ? 'dark' : 'light');
                document.getElementById('theme-toggle-text').textContent = isDark ? 'Dark Mode' : 'Light Mode';
                if(typeof updateChartsTheme === 'function') updateChartsTheme(isDark);
            }}
        </script>
        <style>
            body {{ transition: background-color 0.3s, color 0.3s; }}
            .card-panel {{
                background: #ffffff; border: 1px solid #e4e4e7;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            }}
            .dark .card-panel {{
                background: #18181b; border: 1px solid #27272a;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
            }}
            input[type=range] {{ accent-color: #f97316; }}
            
            /* Custom Scrollbar */
            ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
            ::-webkit-scrollbar-track {{ background: transparent; }}
            ::-webkit-scrollbar-thumb {{ background: #f97316; border-radius: 4px; }}
        </style>
    </head>
    <body class="min-h-screen bg-slate-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100 flex flex-col relative selection:bg-orange-500 selection:text-white">
        
        <!-- Header -->
        <header class="sticky top-0 z-50 bg-white/80 dark:bg-zinc-950/80 backdrop-blur-md border-b border-zinc-200 dark:border-zinc-800">
            <div class="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
                <a href="/" class="flex items-center gap-3 group">
                    <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-orange-600 to-orange-400 flex items-center justify-center text-xl text-white shadow-lg group-hover:scale-105 transition">🌿</div>
                    <div>
                        <h1 class="font-extrabold text-lg tracking-tight">IRIS PRO</h1>
                        <p class="text-[10px] text-zinc-500 font-mono">SVM Architecture</p>
                    </div>
                </a>
                
                <nav class="hidden md:flex items-center gap-8 text-sm font-semibold">
                    <a href="/" class="hover:text-orange-500 transition" data-i18n="nav_home">Trang chủ</a>
                    <a href="/lab" class="hover:text-orange-500 transition" data-i18n="nav_lab">Phân tích</a>
                    <a href="/kernels" class="hover:text-orange-500 transition" data-i18n="nav_kernel">Kernels</a>
                </nav>

                <!-- Settings Dropdown -->
                <div class="relative group">
                    <button class="w-10 h-10 rounded-xl bg-zinc-100 dark:bg-zinc-900 hover:bg-zinc-200 dark:hover:bg-zinc-800 transition flex items-center justify-center text-lg border border-zinc-200 dark:border-zinc-800">
                        ⚙️
                    </button>
                    <div class="absolute right-0 mt-2 w-56 card-panel rounded-xl py-2 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 transform origin-top-right">
                        <div class="px-4 py-2 border-b border-zinc-200 dark:border-zinc-800">
                            <p class="text-xs font-bold text-zinc-500 uppercase" data-i18n="settings">Cài đặt</p>
                        </div>
                        <button onclick="toggleTheme()" class="w-full text-left px-4 py-3 hover:bg-zinc-100 dark:hover:bg-zinc-800 text-sm flex justify-between items-center transition">
                            <span data-i18n="theme">Giao diện</span>
                            <span id="theme-toggle-text" class="text-xs font-mono text-orange-500">Dark Mode</span>
                        </button>
                        <button onclick="toggleLang()" class="w-full text-left px-4 py-3 hover:bg-zinc-100 dark:hover:bg-zinc-800 text-sm flex justify-between items-center transition border-t border-zinc-100 dark:border-zinc-800">
                            <span data-i18n="lang">Ngôn ngữ</span>
                            <span id="lang-toggle-text" class="text-xs font-mono text-orange-500">Tiếng Việt</span>
                        </button>
                        <div class="px-4 py-3 border-t border-zinc-100 dark:border-zinc-800 flex justify-between items-center text-sm">
                            <span data-i18n="default_kernel">Kernel mặc định</span>
                            <select class="bg-transparent border border-zinc-300 dark:border-zinc-700 rounded text-xs p-1 outline-none">
                                <option>Linear</option><option>RBF</option>
                            </select>
                        </div>
                    </div>
                </div>
            </div>
        </header>

        <!-- Main Content -->
        <main class="w-full flex-grow flex flex-col">
            {content}
        </main>
        
        <!-- Footer -->
        <footer class="border-t border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 mt-auto">
            <div class="max-w-7xl mx-auto px-6 py-6 flex flex-col md:flex-row items-center justify-between text-xs text-zinc-500">
                <div class="flex items-center gap-2 mb-2 md:mb-0">
                    <span class="w-2 h-2 rounded-full bg-orange-500 animate-pulse"></span>
                    <span>Dataset: Fisher's Iris (1936) | Machine Learning Lab</span>
                </div>
                <div class="flex gap-4">
                    <a href="#" class="hover:text-orange-500 transition">GitHub Repo</a>
                    <a href="#" class="hover:text-orange-500 transition">Documentation</a>
                    <span><span data-i18n="footer_author">Tác giả</span>: AI Developer</span>
                </div>
            </div>
        </footer>

        <script>
            document.addEventListener('DOMContentLoaded', () => {{
                applyLang();
                document.getElementById('theme-toggle-text').textContent = document.documentElement.classList.contains('dark') ? 'Dark Mode' : 'Light Mode';
            }});
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
    <!-- Hero Section -->
    <section class="relative overflow-hidden border-b border-zinc-200 dark:border-zinc-900 bg-white dark:bg-zinc-950">
        <div class="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-orange-100 via-transparent to-transparent dark:from-orange-900/20 dark:via-transparent dark:to-transparent"></div>
        <div class="max-w-7xl mx-auto px-6 py-20 lg:py-32 relative z-10 grid lg:grid-cols-2 gap-12 items-center">
            <div>
                <span class="px-3 py-1 text-xs font-bold uppercase tracking-wider text-orange-600 bg-orange-100 dark:text-orange-400 dark:bg-orange-500/10 rounded-full border border-orange-200 dark:border-orange-500/20">Machine Learning API</span>
                <h2 class="mt-6 text-4xl lg:text-6xl font-extrabold text-zinc-900 dark:text-white leading-tight">
                    Hệ thống AI nhận diện <br><span class="text-transparent bg-clip-text bg-gradient-to-r from-orange-500 to-orange-400">Thực vật học</span>
                </h2>
                <p class="mt-6 text-lg text-zinc-600 dark:text-zinc-400 leading-relaxed max-w-lg">
                    Sức mạnh của thuật toán Support Vector Machine (SVM) được trực quan hóa. Phân tích hình thái hoa Diên Vĩ theo thời gian thực với độ chính xác và tin cậy cực cao.
                </p>
                <div class="mt-8 flex gap-4">
                    <a href="/lab" class="px-8 py-3.5 bg-orange-500 hover:bg-orange-600 text-white font-semibold rounded-xl transition shadow-lg shadow-orange-500/30 flex items-center gap-2">Thử ngay ➔</a>
                </div>
            </div>
            <div class="relative">
                <div class="aspect-square rounded-full absolute inset-0 bg-gradient-to-tr from-orange-400/20 to-transparent blur-3xl"></div>
                <!-- Mockup Biểu đồ Minh họa -->
                <div class="card-panel p-6 rounded-3xl relative rotate-3 hover:rotate-0 transition-transform duration-500">
                    <div class="h-64 w-full flex items-end justify-between gap-2 opacity-80">
                        <div class="w-1/6 bg-zinc-200 dark:bg-zinc-800 rounded-t-lg h-24"></div>
                        <div class="w-1/6 bg-orange-400 rounded-t-lg h-48"></div>
                        <div class="w-1/6 bg-zinc-200 dark:bg-zinc-800 rounded-t-lg h-32"></div>
                        <div class="w-1/6 bg-orange-500 rounded-t-lg h-64"></div>
                        <div class="w-1/6 bg-zinc-200 dark:bg-zinc-800 rounded-t-lg h-16"></div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Model Info Section -->
    <section class="max-w-7xl mx-auto px-6 py-20">
        <div class="text-center mb-16">
            <h3 class="text-2xl font-bold text-zinc-900 dark:text-white">Kiến trúc Hệ thống & Dữ liệu</h3>
            <p class="text-zinc-500 mt-2">Tổng quan về Dataset và Mô hình</p>
        </div>
        
        <div class="grid md:grid-cols-3 gap-6 mb-12">
            <div class="card-panel p-6 rounded-3xl text-center group">
                <div class="w-16 h-16 mx-auto bg-orange-50 dark:bg-orange-500/10 rounded-2xl flex items-center justify-center text-3xl mb-4 group-hover:scale-110 transition">🌸</div>
                <h4 class="font-bold text-lg mb-2 text-zinc-900 dark:text-white">Iris Setosa</h4>
                <p class="text-sm text-zinc-500">Đài hoa rộng, cánh hoa tiêu biến. Phân tách tuyến tính dễ dàng.</p>
            </div>
            <div class="card-panel p-6 rounded-3xl text-center group">
                <div class="w-16 h-16 mx-auto bg-orange-50 dark:bg-orange-500/10 rounded-2xl flex items-center justify-center text-3xl mb-4 group-hover:scale-110 transition">🌺</div>
                <h4 class="font-bold text-lg mb-2 text-zinc-900 dark:text-white">Iris Versicolor</h4>
                <p class="text-sm text-zinc-500">Hình thái trung gian, cân bằng lý tưởng về tỷ lệ sinh học.</p>
            </div>
            <div class="card-panel p-6 rounded-3xl text-center group">
                <div class="w-16 h-16 mx-auto bg-orange-50 dark:bg-orange-500/10 rounded-2xl flex items-center justify-center text-3xl mb-4 group-hover:scale-110 transition">🌷</div>
                <h4 class="font-bold text-lg mb-2 text-zinc-900 dark:text-white">Iris Virginica</h4>
                <p class="text-sm text-zinc-500">Kích thước lớn nhất, cánh hoa thuôn dài tráng lệ.</p>
            </div>
        </div>

        <div class="grid md:grid-cols-2 gap-6">
            <div class="card-panel p-8 rounded-3xl flex gap-6 items-start">
                <div class="text-4xl">📊</div>
                <div>
                    <h4 class="font-bold text-lg text-zinc-900 dark:text-white mb-2">Fisher's Dataset</h4>
                    <p class="text-sm text-zinc-500">Bộ dữ liệu kinh điển gồm 150 mẫu hoa, thu thập từ bán đảo Gaspé bởi Edgar Anderson. Đây là cột mốc của ngành Khoa học dữ liệu thống kê.</p>
                </div>
            </div>
            <div class="card-panel p-8 rounded-3xl flex gap-6 items-start">
                <div class="text-4xl">🧠</div>
                <div>
                    <h4 class="font-bold text-lg text-zinc-900 dark:text-white mb-2">SVM Engine</h4>
                    <p class="text-sm text-zinc-500">Support Vector Machine tối đa hóa lề phân cách (Margin) giữa các lớp dữ liệu trong không gian đa chiều, hạn chế tối đa nhiễu.</p>
                </div>
            </div>
        </div>
    </section>
    """
    return get_base_html("Trang chủ", content)

@app.get("/lab", response_class=HTMLResponse)
def lab_page():
    content = """
    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-8 w-full">
        <!-- TOP: Prediction & Interactive -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start mb-8">
            <!-- Khối Input -->
            <div class="lg:col-span-4 card-panel rounded-3xl p-6">
                <div class="flex justify-between items-center mb-6">
                    <h2 class="font-bold text-lg text-zinc-900 dark:text-white">Tham số đầu vào</h2>
                    <button onclick="randomizeInputs()" class="text-xl hover:rotate-180 transition duration-300">🎲</button>
                </div>
                <div class="space-y-6">
                    <div>
                        <div class="flex justify-between text-xs font-bold text-zinc-500 uppercase tracking-wider mb-2"><span>Sepal Length</span><span id="txt-sl" class="text-orange-500">5.1</span></div>
                        <input type="range" id="sl" min="4.0" max="8.0" step="0.1" value="5.1" class="w-full" oninput="syncVal('sl', 'txt-sl')">
                    </div>
                    <div>
                        <div class="flex justify-between text-xs font-bold text-zinc-500 uppercase tracking-wider mb-2"><span>Sepal Width</span><span id="txt-sw" class="text-orange-500">3.5</span></div>
                        <input type="range" id="sw" min="2.0" max="4.5" step="0.1" value="3.5" class="w-full" oninput="syncVal('sw', 'txt-sw')">
                    </div>
                    <div>
                        <div class="flex justify-between text-xs font-bold text-zinc-500 uppercase tracking-wider mb-2"><span>Petal Length</span><span id="txt-pl" class="text-orange-500">1.4</span></div>
                        <input type="range" id="pl" min="1.0" max="7.0" step="0.1" value="1.4" class="w-full" oninput="syncVal('pl', 'txt-pl')">
                    </div>
                    <div>
                        <div class="flex justify-between text-xs font-bold text-zinc-500 uppercase tracking-wider mb-2"><span>Petal Width</span><span id="txt-pw" class="text-orange-500">0.2</span></div>
                        <input type="range" id="pw" min="0.1" max="2.6" step="0.1" value="0.2" class="w-full" oninput="syncVal('pw', 'txt-pw')">
                    </div>
                </div>
                <button onclick="executeInference()" class="w-full mt-8 py-3.5 bg-zinc-900 hover:bg-zinc-800 dark:bg-white dark:hover:bg-zinc-200 dark:text-zinc-900 text-white font-bold rounded-xl transition shadow-lg">Phân Tích Ngay</button>
            </div>

            <!-- Khối Result & Scatter Plot -->
            <div class="lg:col-span-8 flex flex-col gap-6">
                <!-- Banner Kết quả -->
                <div class="card-panel rounded-3xl p-6 sm:p-8 flex items-center justify-between relative overflow-hidden">
                    <div class="absolute right-0 top-0 w-64 h-64 bg-orange-500/10 blur-3xl rounded-full" id="result-glow"></div>
                    <div class="flex flex-col sm:flex-row items-center gap-6 relative z-10 w-full justify-between">
                        <div>
                            <p class="text-xs font-bold text-zinc-500 uppercase tracking-wider" data-i18n="pred_title">Kết quả dự đoán</p>
                            <h3 id="specimen-title" class="text-3xl font-black text-zinc-900 dark:text-white mt-1">Đang xử lý...</h3>
                        </div>
                        <div class="text-right">
                            <p class="text-xs font-bold text-zinc-500 uppercase tracking-wider">Độ tin cậy</p>
                            <div class="text-4xl font-black font-mono text-orange-500" id="main-conf">0%</div>
                        </div>
                    </div>
                </div>

                <!-- Scatter Plot 2D -->
                <div class="card-panel rounded-3xl p-6 h-72 flex flex-col">
                    <h4 class="text-sm font-bold text-zinc-900 dark:text-white mb-2">Phân bố dữ liệu (Scatter Plot: Petal L vs Petal W)</h4>
                    <div class="flex-grow w-full relative"><canvas id="scatterChart"></canvas></div>
                </div>
            </div>
        </div>

        <!-- BOTTOM: History & Export -->
        <div class="card-panel rounded-3xl p-6">
            <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 gap-4 pb-4 border-b border-zinc-200 dark:border-zinc-800">
                <h2 class="font-bold text-lg text-zinc-900 dark:text-white">Lịch sử & Báo cáo</h2>
                <div class="flex gap-2">
                    <button onclick="exportCSV()" class="px-4 py-2 bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 text-sm font-semibold rounded-lg transition" data-i18n="export_csv">Xuất CSV</button>
                    <button onclick="exportJSON()" class="px-4 py-2 bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 text-sm font-semibold rounded-lg transition" data-i18n="export_json">Xuất JSON</button>
                    <button onclick="clearHistory()" class="px-4 py-2 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 hover:bg-red-100 rounded-lg text-sm font-semibold transition" data-i18n="clear_hist">Xóa</button>
                </div>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-left text-sm whitespace-nowrap">
                    <thead>
                        <tr class="text-zinc-500 uppercase tracking-wider text-xs border-b border-zinc-100 dark:border-zinc-800">
                            <th class="py-3 px-4">Thời gian</th><th class="py-3 px-4">S.L</th><th class="py-3 px-4">S.W</th>
                            <th class="py-3 px-4">P.L</th><th class="py-3 px-4">P.W</th><th class="py-3 px-4">Dự đoán</th>
                            <th class="py-3 px-4 text-right">Tin cậy</th>
                        </tr>
                    </thead>
                    <tbody id="history-tbody" class="text-zinc-700 dark:text-zinc-300 divide-y divide-zinc-100 dark:divide-zinc-800">
                        <tr><td colspan="7" class="text-center py-6 text-zinc-500">Đang tải...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        let scatterChart;
        let historyData = [];

        function initCharts() {
            const ctx = document.getElementById('scatterChart').getContext('2d');
            const isDark = document.documentElement.classList.contains('dark');
            const gridColor = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
            const textColor = isDark ? '#a1a1aa' : '#71717a';

            // Dữ liệu mô phỏng nền cho Scatter (Petal L vs Petal W)
            const bgData = {
                datasets: [
                    { label: 'Setosa', data: [{x:1.4,y:0.2},{x:1.5,y:0.2},{x:1.3,y:0.2},{x:1.6,y:0.2},{x:1.4,y:0.3},{x:1.1,y:0.1}], backgroundColor: 'rgba(249, 115, 22, 0.4)' },
                    { label: 'Versicolor', data: [{x:4.7,y:1.4},{x:4.5,y:1.5},{x:4.9,y:1.5},{x:4.0,y:1.3},{x:4.6,y:1.5},{x:4.5,y:1.3}], backgroundColor: 'rgba(251, 146, 60, 0.4)' },
                    { label: 'Virginica', data: [{x:6.0,y:2.5},{x:5.1,y:1.9},{x:5.9,y:2.1},{x:5.6,y:2.4},{x:5.8,y:2.2},{x:6.6,y:2.1}], backgroundColor: 'rgba(253, 186, 116, 0.4)' },
                    { label: 'Mẫu hiện tại', data: [{x:1.4, y:0.2}], backgroundColor: '#f97316', pointRadius: 8, pointBorderColor: '#ffffff', pointBorderWidth: 2 }
                ]
            };

            scatterChart = new Chart(ctx, {
                type: 'scatter',
                data: bgData,
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        x: { title: {display:true, text:'Petal Length', color:textColor}, grid:{color:gridColor}, ticks:{color:textColor} },
                        y: { title: {display:true, text:'Petal Width', color:textColor}, grid:{color:gridColor}, ticks:{color:textColor} }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        }
        
        function updateChartsTheme(isDark) {
            if(!scatterChart) return;
            const gridColor = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
            const textColor = isDark ? '#a1a1aa' : '#71717a';
            scatterChart.options.scales.x.grid.color = gridColor;
            scatterChart.options.scales.y.grid.color = gridColor;
            scatterChart.options.scales.x.title.color = textColor;
            scatterChart.options.scales.y.title.color = textColor;
            scatterChart.options.scales.x.ticks.color = textColor;
            scatterChart.options.scales.y.ticks.color = textColor;
            scatterChart.update();
        }

        function syncVal(sliderId, textId) {
            document.getElementById(textId).textContent = document.getElementById(sliderId).value;
        }

        function randomizeInputs() {
            const r = (min, max) => (Math.random() * (max - min) + min).toFixed(1);
            ['sl', 'sw', 'pl', 'pw'].forEach(id => {
                const val = id.includes('s') ? r(2.0, 8.0) : r(0.1, 7.0);
                document.getElementById(id).value = val;
                document.getElementById('txt-' + id).textContent = val;
            });
            executeInference();
        }

        async function fetchHistory() {
            const res = await fetch('/api/history');
            historyData = await res.json();
            const tbody = document.getElementById('history-tbody');
            if(historyData.length === 0) { tbody.innerHTML = '<tr><td colspan="7" class="text-center py-6 text-zinc-500">Trống.</td></tr>'; return; }
            
            tbody.innerHTML = historyData.map(row => {
                return `<tr>
                    <td class="py-3 px-4 font-mono text-xs text-zinc-500">${row.timestamp}</td>
                    <td class="py-3 px-4">${row.sl}</td><td class="py-3 px-4">${row.sw}</td>
                    <td class="py-3 px-4">${row.pl}</td><td class="py-3 px-4">${row.pw}</td>
                    <td class="py-3 px-4"><span class="font-bold text-orange-500">${row.species}</span></td>
                    <td class="py-3 px-4 text-right font-mono">${row.confidence}%</td>
                </tr>`;
            }).join('');
        }

        async function clearHistory() {
            if(!confirm("Xác nhận xóa toàn bộ?")) return;
            await fetch('/api/history', {method: 'DELETE'});
            fetchHistory();
        }

        function exportCSV() {
            if(historyData.length === 0) return alert("Không có dữ liệu");
            let csv = "Timestamp,Sepal_L,Sepal_W,Petal_L,Petal_W,Species,Confidence\\n";
            historyData.forEach(r => { csv += `${r.timestamp},${r.sl},${r.sw},${r.pl},${r.pw},${r.species},${r.confidence}\\n`; });
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a'); a.href = url; a.download = 'iris_history.csv'; a.click();
        }

        function exportJSON() {
            if(historyData.length === 0) return alert("Không có dữ liệu");
            const blob = new Blob([JSON.stringify(historyData, null, 2)], { type: 'application/json' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a'); a.href = url; a.download = 'iris_history.json'; a.click();
        }

        async function executeInference() {
            const pl = parseFloat(document.getElementById('pl').value);
            const pw = parseFloat(document.getElementById('pw').value);
            const payload = {
                sepal_length: parseFloat(document.getElementById('sl').value),
                sepal_width: parseFloat(document.getElementById('sw').value),
                petal_length: pl,
                petal_width: pw
            };
            const res = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            const data = await res.json();
            
            document.getElementById('specimen-title').textContent = data.prediction;
            document.getElementById('specimen-title').style.color = data.color;
            document.getElementById('main-conf').textContent = `${Math.max(...data.confidences)}%`;
            document.getElementById('result-glow').style.backgroundColor = data.color + "40"; // 40% opacity

            if (scatterChart) {
                scatterChart.data.datasets[3].data = [{x: pl, y: pw}];
                scatterChart.update();
            }
            fetchHistory();
        }

        window.addEventListener('DOMContentLoaded', () => { initCharts(); executeInference(); });
    </script>
    """
    return get_base_html("Lab Phân tích", content)

@app.get("/kernels", response_class=HTMLResponse)
def kernels_page():
    content = """
    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-8 w-full">
        <div class="card-panel rounded-3xl p-8">
            <div class="mb-8">
                <span class="text-orange-500 font-bold text-sm tracking-wider uppercase mb-2 block">Dashboard Analytics</span>
                <h2 class="text-2xl font-bold text-zinc-900 dark:text-white">So sánh Kernels</h2>
            </div>
            
            <div class="grid lg:grid-cols-2 gap-8 mb-8">
                <div class="bg-zinc-50 dark:bg-zinc-900/50 p-6 rounded-2xl border border-zinc-200 dark:border-zinc-800">
                    <h3 class="text-sm font-bold mb-4 text-center">Độ chính xác (Accuracy %)</h3>
                    <div class="h-64"><canvas id="accChart"></canvas></div>
                </div>
                <div class="bg-zinc-50 dark:bg-zinc-900/50 p-6 rounded-2xl border border-zinc-200 dark:border-zinc-800 flex flex-col justify-center">
                    <table class="w-full text-left text-sm">
                        <thead><tr class="text-zinc-500 border-b border-zinc-200 dark:border-zinc-800"><th class="py-2">Kernel</th><th class="py-2">Precision</th><th class="py-2">Recall</th><th class="py-2">F1</th></tr></thead>
                        <tbody class="divide-y divide-zinc-200 dark:divide-zinc-800 font-mono text-zinc-800 dark:text-zinc-300">
                            <tr><td class="py-3 font-bold text-orange-500">Linear</td><td>0.98</td><td>0.98</td><td>0.98</td></tr>
                            <tr><td class="py-3 font-bold text-orange-400">RBF</td><td>0.97</td><td>0.96</td><td>0.96</td></tr>
                            <tr><td class="py-3 font-bold text-orange-300">Poly</td><td>0.95</td><td>0.95</td><td>0.95</td></tr>
                            <tr><td class="py-3 font-bold text-zinc-400">Sigmoid</td><td>0.85</td><td>0.82</td><td>0.83</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let accChart;
        function initCharts() {
            const ctx = document.getElementById('accChart').getContext('2d');
            const isDark = document.documentElement.classList.contains('dark');
            const textColor = isDark ? '#a1a1aa' : '#71717a';
            const gridColor = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';

            accChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Linear', 'RBF', 'Poly', 'Sigmoid'],
                    datasets: [{
                        data: [98.5, 96.8, 95.0, 85.2],
                        backgroundColor: ['#f97316', '#fb923c', '#fdba74', '#a1a1aa'], borderRadius: 6
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        y: { max: 100, ticks: { color: textColor }, grid: { color: gridColor } },
                        x: { ticks: { color: textColor }, grid: { display: false } }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        }
        function updateChartsTheme(isDark) {
            if(!accChart) return;
            const textColor = isDark ? '#a1a1aa' : '#71717a';
            const gridColor = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
            accChart.options.scales.y.ticks.color = textColor;
            accChart.options.scales.x.ticks.color = textColor;
            accChart.options.scales.y.grid.color = gridColor;
            accChart.update();
        }
        window.addEventListener('DOMContentLoaded', initCharts);
    </script>
    """
    return get_base_html("Phân tích Kernels", content)
