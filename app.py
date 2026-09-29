import os
import math
import logging
import sqlite3
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# ==========================================
# 1. KHỞI TẠO DB LỊCH SỬ & MÔ HÌNH SVM
# ==========================================
DB_PATH = "history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  sl REAL, sw REAL, pl REAL, pw REAL,
                  species TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

init_db()

MODEL_PATH = "svm_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    logging.info("Đã tải thành công mô hình SVM.")
else:
    model = None
    logging.warning(f"Không tìm thấy {MODEL_PATH}. Ứng dụng sẽ chạy ở chế độ Demo/No-Model.")

app = FastAPI(
    title="Iris Botanical Classifier",
    description="SVM Machine Learning Laboratory & Interactive Neural Dashboard",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. CẤU HÌNH & DỮ LIỆU CHUẨN
# ==========================================
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

SPECIES_METADATA = {
    0: {
        "name": "Iris Setosa", "author": "Pall. ex Link", "tag": "Setosa", "theme": "emerald", "accent": "#10b981",
        "badge": "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
        "desc": "Đặc trưng bởi đài hoa rộng nhưng cánh hoa tiêu biến cực nhỏ. Khả năng phân tách tuyến tính 100%.",
        "ecology": "Bắc bán cầu, khí hậu ôn đới lạnh, vùng đầm lầy ven biển.",
        "image": "/img/anh_setosa.jpg"
    },
    1: {
        "name": "Iris Versicolor", "author": "L.", "tag": "Versicolor", "theme": "cyan", "accent": "#06b6d4",
        "badge": "bg-cyan-500/10 text-cyan-300 border-cyan-500/30",
        "desc": "Mang hình thái trung gian, có sự cân bằng lý tưởng giữa tỷ lệ chiều dài cánh hoa và đài hoa.",
        "ecology": "Khu vực ẩm ướt Bắc Mỹ, ven hồ và đồng cỏ ngập nước ngọt.",
        "image": "/img/anh_versicolor.jpg"
    },
    2: {
        "name": "Iris Virginica", "author": "L.", "tag": "Virginica", "theme": "purple", "accent": "#a855f7",
        "badge": "bg-purple-500/10 text-purple-300 border-purple-500/30",
        "desc": "Loài hoa có kích thước lớn và cấu trúc tráng lệ nhất với cánh hoa thuôn dài, sắc tím đậm.",
        "ecology": "Đồng cỏ ẩm ven biển và đầm lầy phía Đông Bắc Mỹ.",
        "image": "/img/anh_virginica.jpg"
    },
}

CENTROIDS = [
    [5.006, 3.428, 1.462, 0.246],  # Setosa
    [5.936, 2.770, 4.260, 1.326],  # Versicolor
    [6.588, 2.974, 5.552, 2.026],  # Virginica
]

# ==========================================
# 3. CORE API ENDPOINTS
# ==========================================
@app.get("/img/{img_name}")
def get_image(img_name: str):
    allowed_images = ["anh_setosa.jpg", "anh_versicolor.jpg", "anh_virginica.jpg"]
    if img_name in allowed_images and os.path.exists(img_name):
        return FileResponse(img_name)
    raise HTTPException(status_code=404, detail="Không tìm thấy ảnh")

@app.get("/health")
def health():
    return {"status": "healthy", "engine": "SVM Linear Kernel", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: IrisInput):
    if model is None:
        raise HTTPException(status_code=500, detail="SVM Model không tồn tại.")
        
    feat = [data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]
    pred = int(model.predict([feat])[0])
    
    dists = [math.sqrt(sum((feat[i] - CENTROIDS[c][i]) ** 2 for i in range(4))) for c in range(3)]
    inv_dists = [1.0 / (d + 1e-5) for d in dists]
    inv_dists[pred] *= 1.8
    total = sum(inv_dists)
    confidences = [round((v / total) * 100, 1) for v in inv_dists]
    
    meta = SPECIES_METADATA[pred]
    
    # Lưu vào CSDL
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO history (timestamp, sl, sw, pl, pw, species, confidence) 
                 VALUES (?, ?, ?, ?, ?, ?, ?)""", 
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
               data.sepal_length, data.sepal_width, data.petal_length, data.petal_width, 
               meta["name"], confidences[pred]))
    conn.commit()
    conn.close()

    return {
        "class_id": pred,
        "prediction": meta["tag"].lower(),
        "species": meta,
        "confidences": {"setosa": confidences[0], "versicolor": confidences[1], "virginica": confidences[2]},
        "features": feat
    }

@app.get("/api/history")
def get_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp, sl, sw, pl, pw, species, confidence FROM history ORDER BY id DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return [{"timestamp": r[0], "sl": r[1], "sw": r[2], "pl": r[3], "pw": r[4], "species": r[5], "confidence": r[6]} for r in rows]

# ==========================================
# 4. GIAO DIỆN HTML (UI TEMPLATES)
# ==========================================
def get_base_html(title, active_tab, content):
    """Template nền chung cho mọi trang"""
    
    nav_links = {
        "dashboard": {"url": "/", "label": "Phân tích (Lab)"},
        "history": {"url": "/history", "label": "Lịch sử"},
        "about": {"url": "/about", "label": "Giới thiệu"}
    }
    
    nav_html = ""
    for key, info in nav_links.items():
        if key == active_tab:
            nav_html += f'<a href="{info["url"]}" class="text-emerald-400 border-b-2 border-emerald-400 pb-1">{info["label"]}</a>'
        else:
            nav_html += f'<a href="{info["url"]}" class="text-slate-400 hover:text-emerald-300 transition pb-1">{info["label"]}</a>'

    return f"""
    <!DOCTYPE html>
    <html lang="vi" class="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title} | Iris Intelligence</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            tailwind.config = {{
                darkMode: 'class',
                theme: {{
                    extend: {{
                        fontFamily: {{
                            sans: ['Plus Jakarta Sans', 'sans-serif'],
                            mono: ['JetBrains Mono', 'monospace'],
                        }},
                        keyframes: {{
                            kenburns: {{
                                '0%': {{ transform: 'scale(1) translate(0, 0)' }},
                                '50%': {{ transform: 'scale(1.05) translate(-1%, -1%)' }},
                                '100%': {{ transform: 'scale(1) translate(0, 0)' }},
                            }},
                            floatingfog: {{
                                '0%': {{ transform: 'translateX(-5%)', opacity: '0.4' }},
                                '50%': {{ transform: 'translateX(5%)', opacity: '0.6' }},
                                '100%': {{ transform: 'translateX(-5%)', opacity: '0.4' }},
                            }}
                        }},
                        animation: {{
                            'kenburns': 'kenburns 40s ease-in-out infinite',
                            'floatingfog': 'floatingfog 20s ease-in-out infinite',
                        }}
                    }}
                }}
            }}
        </script>
        <style>
            body {{ background-color: #020617; }}
            .glass-panel {{
                background: rgba(15, 23, 42, 0.75);
                backdrop-filter: blur(24px);
                -webkit-backdrop-filter: blur(24px);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            input[type=range]::-webkit-slider-thumb {{
                -webkit-appearance: none; height: 18px; width: 18px;
                border-radius: 50%; background: #818cf8; cursor: pointer;
                box-shadow: 0 0 10px rgba(129, 140, 248, 0.8); margin-top: -6px;
            }}
            input[type=range]::-webkit-slider-runnable-track {{
                width: 100%; height: 6px; cursor: pointer;
                background: rgba(255, 255, 255, 0.15); border-radius: 999px;
            }}
        </style>
    </head>
    <body class="min-h-screen text-slate-100 flex flex-col justify-between selection:bg-indigo-500 selection:text-white relative">
        <!-- Background -->
        <div class="fixed inset-0 z-[-1] overflow-hidden bg-slate-900">
            <div class="absolute inset-0 bg-cover bg-center bg-no-repeat animate-kenburns opacity-70" 
                 style="background-image: url('https://images.unsplash.com/photo-1472214103451-9374bd1c798e?q=80&w=2070&auto=format&fit=crop');">
            </div>
            <div class="absolute inset-0 bg-gradient-to-t from-slate-900 via-slate-900/60 to-transparent animate-floatingfog"></div>
            <div class="absolute inset-0 bg-slate-950/40 backdrop-blur-[2px]"></div>
        </div>

        <!-- Header / Navigation -->
        <nav class="glass-panel sticky top-0 z-50 border-b border-white/10 px-6 py-4">
            <div class="max-w-7xl mx-auto flex items-center justify-between">
                <div class="flex items-center gap-4">
                    <div class="relative flex items-center justify-center w-11 h-11 rounded-2xl bg-gradient-to-tr from-emerald-600 via-teal-600 to-cyan-600 shadow-lg shadow-emerald-500/30 overflow-hidden border border-white/20">
                        <span class="text-2xl absolute drop-shadow-md">🌿</span>
                    </div>
                    <div>
                        <div class="flex items-center gap-2">
                            <h1 class="font-extrabold text-lg tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white via-emerald-100 to-emerald-300">BOTANICAL LAB</h1>
                        </div>
                        <p class="text-xs text-slate-400">Hệ thống AI nhận diện thực vật</p>
                    </div>
                </div>

                <!-- Menu Desktop -->
                <div class="hidden md:flex items-center gap-8 text-sm font-semibold">
                    {nav_html}
                </div>

                <div class="flex items-center gap-4">
                    <button id="sound-btn" class="w-9 h-9 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center text-slate-300 hover:text-white hover:bg-white/10 transition">🔊</button>
                </div>
            </div>
        </nav>

        <!-- Main Content -->
        <main class="max-w-7xl mx-auto px-4 sm:px-6 py-8 w-full relative z-10 flex-grow">
            {content}
        </main>

        <!-- Footer -->
        <footer class="border-t border-white/10 glass-panel mt-auto py-5 px-6 text-center text-xs text-slate-400 flex flex-col sm:flex-row items-center justify-between max-w-7xl mx-auto w-full relative z-10">
            <div class="flex items-center gap-2">
                <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span>Hệ thống: <b>SVM Engine</b></span>
            </div>
            <div class="mt-2 sm:mt-0 font-mono text-slate-400/80">
                Powered by FastAPI & SQLite
            </div>
        </footer>
    </body>
    </html>
    """

# --- ROUTES HTML ---

@app.get("/", response_class=HTMLResponse)
def dashboard():
    content = """
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        <!-- Controls Column -->
        <div class="lg:col-span-5 flex flex-col gap-6">
            <div class="glass-panel rounded-3xl p-6 sm:p-7 shadow-2xl relative overflow-hidden">
                <div class="flex items-center justify-between mb-4">
                    <h2 class="text-base font-bold text-white flex items-center gap-2">
                        <span>📐</span> Thông số kích thước (cm)
                    </h2>
                    <button onclick="randomizeInputs()" class="text-xs text-slate-400 hover:text-emerald-400 font-mono transition flex items-center gap-1 bg-white/5 px-2 py-1 rounded-lg border border-white/5">
                        🎲 Ngẫu nhiên
                    </button>
                </div>

                <div class="grid grid-cols-3 gap-2 p-1 bg-black/40 rounded-2xl border border-white/10 mb-6">
                    <button onclick="applyPreset(5.0, 3.4, 1.5, 0.2)" class="py-2 px-1 text-center rounded-xl text-xs font-medium hover:bg-emerald-500/20 text-slate-300 hover:text-emerald-400 transition">Mẫu Setosa</button>
                    <button onclick="applyPreset(6.0, 2.8, 4.3, 1.3)" class="py-2 px-1 text-center rounded-xl text-xs font-medium hover:bg-cyan-500/20 text-slate-300 hover:text-cyan-400 transition">Mẫu Versicolor</button>
                    <button onclick="applyPreset(6.6, 3.0, 5.6, 2.1)" class="py-2 px-1 text-center rounded-xl text-xs font-medium hover:bg-purple-500/20 text-slate-300 hover:text-purple-400 transition">Mẫu Virginica</button>
                </div>

                <div class="space-y-5">
                    <div class="space-y-2">
                        <div class="flex justify-between items-center text-xs">
                            <span class="font-medium text-slate-300 flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-blue-400"></span> Chiều dài đài hoa</span>
                            <div class="font-mono text-emerald-300 bg-emerald-950/60 px-2 py-0.5 rounded-lg border border-emerald-800/50"><span id="txt-sl">5.1</span></div>
                        </div>
                        <input type="range" id="sl" min="4.0" max="8.0" step="0.1" value="5.1" class="w-full" oninput="syncVal('sl', 'txt-sl')">
                    </div>
                    <div class="space-y-2">
                        <div class="flex justify-between items-center text-xs">
                            <span class="font-medium text-slate-300 flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-sky-400"></span> Chiều rộng đài hoa</span>
                            <div class="font-mono text-emerald-300 bg-emerald-950/60 px-2 py-0.5 rounded-lg border border-emerald-800/50"><span id="txt-sw">3.5</span></div>
                        </div>
                        <input type="range" id="sw" min="2.0" max="4.5" step="0.1" value="3.5" class="w-full" oninput="syncVal('sw', 'txt-sw')">
                    </div>
                    <div class="space-y-2">
                        <div class="flex justify-between items-center text-xs">
                            <span class="font-medium text-slate-300 flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-purple-400"></span> Chiều dài cánh hoa</span>
                            <div class="font-mono text-emerald-300 bg-emerald-950/60 px-2 py-0.5 rounded-lg border border-emerald-800/50"><span id="txt-pl">1.4</span></div>
                        </div>
                        <input type="range" id="pl" min="1.0" max="7.0" step="0.1" value="1.4" class="w-full" oninput="syncVal('pl', 'txt-pl')">
                    </div>
                    <div class="space-y-2">
                        <div class="flex justify-between items-center text-xs">
                            <span class="font-medium text-slate-300 flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-pink-400"></span> Chiều rộng cánh hoa</span>
                            <div class="font-mono text-emerald-300 bg-emerald-950/60 px-2 py-0.5 rounded-lg border border-emerald-800/50"><span id="txt-pw">0.2</span></div>
                        </div>
                        <input type="range" id="pw" min="0.1" max="2.6" step="0.1" value="0.2" class="w-full" oninput="syncVal('pw', 'txt-pw')">
                    </div>
                </div>

                <div class="mt-4 flex items-center gap-2">
                    <input type="checkbox" id="live-toggle" checked class="w-4 h-4 text-emerald-500 rounded border-slate-600 bg-slate-700">
                    <label for="live-toggle" class="text-xs text-slate-400">Tự động phân tích khi kéo thanh trượt (Live Sync)</label>
                </div>

                <button onclick="executeInference()" id="predict-btn" class="w-full mt-5 py-3.5 px-4 bg-gradient-to-r from-emerald-600 to-teal-700 hover:opacity-95 text-white font-semibold rounded-2xl shadow-xl flex items-center justify-center gap-2">
                    <span id="btn-text">Phân tích sinh học & Lưu Lịch sử</span>
                    <div id="btn-spin" class="hidden w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin"></div>
                </button>
            </div>
        </div>

        <!-- Result Column -->
        <div class="lg:col-span-7 flex flex-col gap-6 relative">
            <div id="specimen-card" class="glass-panel rounded-3xl p-6 sm:p-8 relative overflow-hidden transition-all duration-500 border border-white/10 shadow-2xl">
                <div class="absolute -right-16 -top-16 w-64 h-64 rounded-full blur-3xl opacity-25 pointer-events-none" id="ambient-glow" style="background: #10b981;"></div>
                
                <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-5 pb-6 border-b border-white/10">
                    <div class="flex items-center gap-5">
                        <div class="relative shrink-0">
                            <img id="specimen-img" src="" alt="Iris Image" class="w-20 h-20 sm:w-24 sm:h-24 object-cover rounded-2xl border-2 border-white/20">
                        </div>
                        <div>
                            <span id="specimen-badge" class="px-2.5 py-0.5 rounded-full text-[11px] font-mono uppercase tracking-wider font-semibold border">---</span>
                            <h3 id="specimen-title" class="text-2xl sm:text-3xl font-extrabold text-white mt-1 mb-1">Đang chờ mẫu...</h3>
                            <p id="specimen-author" class="text-xs text-slate-400 font-mono">Taxonomy: ---</p>
                        </div>
                    </div>
                    <div class="text-right sm:self-center w-full sm:w-auto bg-black/40 px-4 py-3 rounded-2xl border border-white/10">
                        <span class="text-[10px] text-slate-400 uppercase tracking-wider font-mono">Độ tin cậy</span>
                        <div class="text-3xl font-black font-mono text-white" id="main-conf">0.0%</div>
                    </div>
                </div>

                <div class="py-5 space-y-3">
                    <p id="specimen-desc" class="text-sm text-slate-200 font-light"></p>
                    <div class="flex items-start gap-2 text-xs text-slate-400 bg-black/20 p-3 rounded-xl border border-white/5">
                        <span class="mt-0.5">📍</span> <span id="specimen-eco"></span>
                    </div>
                </div>

                <div class="space-y-3 pt-4 border-t border-white/10">
                    <span class="text-xs font-mono text-slate-400 uppercase tracking-wider">Mức độ tương đồng phân lớp</span>
                    <div class="space-y-2.5">
                        <div>
                            <div class="flex justify-between text-xs font-mono mb-1"><span class="text-emerald-400">Iris Setosa</span><span id="bar-val-0" class="text-slate-300">0%</span></div>
                            <div class="w-full bg-black/50 h-2 rounded-full overflow-hidden border border-white/5"><div id="bar-0" class="bg-emerald-500 h-full rounded-full transition-all duration-500" style="width: 0%"></div></div>
                        </div>
                        <div>
                            <div class="flex justify-between text-xs font-mono mb-1"><span class="text-cyan-400">Iris Versicolor</span><span id="bar-val-1" class="text-slate-300">0%</span></div>
                            <div class="w-full bg-black/50 h-2 rounded-full overflow-hidden border border-white/5"><div id="bar-1" class="bg-cyan-500 h-full rounded-full transition-all duration-500" style="width: 0%"></div></div>
                        </div>
                        <div>
                            <div class="flex justify-between text-xs font-mono mb-1"><span class="text-purple-400">Iris Virginica</span><span id="bar-val-2" class="text-slate-300">0%</span></div>
                            <div class="w-full bg-black/50 h-2 rounded-full overflow-hidden border border-white/5"><div id="bar-2" class="bg-purple-500 h-full rounded-full transition-all duration-500" style="width: 0%"></div></div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="glass-panel rounded-3xl p-6 border border-white/10">
                <div class="flex items-center justify-between mb-4">
                    <h4 class="text-sm font-bold text-white flex items-center gap-2"><span>📊</span> Radar Chart</h4>
                </div>
                <div class="h-56 w-full flex items-center justify-center">
                    <canvas id="radarChart"></canvas>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let radarChart;
        function initChart() {
            const ctx = document.getElementById('radarChart').getContext('2d');
            radarChart = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['S. Length', 'S. Width', 'P. Length', 'P. Width'],
                    datasets: [
                        { label: 'Mẫu phân tích', data: [5.1, 3.5, 1.4, 0.2], backgroundColor: 'rgba(16, 185, 129, 0.25)', borderColor: '#34d399', borderWidth: 2 },
                        { label: 'Trung bình loài', data: [5.0, 3.4, 1.5, 0.2], backgroundColor: 'transparent', borderColor: 'rgba(255, 255, 255, 0.3)', borderWidth: 1, borderDash: [4, 4] }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: { r: { angleLines: { color: 'rgba(255,255,255,0.1)' }, grid: { color: 'rgba(255,255,255,0.08)' }, pointLabels: { color: '#e2e8f0' }, ticks: { display: false } } },
                    plugins: { legend: { display: false } }
                }
            });
        }

        let debounceTimer = null;
        function syncVal(sliderId, textId) {
            document.getElementById(textId).textContent = document.getElementById(sliderId).value;
            if (document.getElementById('live-toggle').checked) {
                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(executeInference, 300);
            }
        }

        function applyPreset(sl, sw, pl, pw) {
            ['sl', 'sw', 'pl', 'pw'].forEach((id, idx) => {
                const val = [sl, sw, pl, pw][idx];
                document.getElementById(id).value = val;
                document.getElementById('txt-' + id).textContent = val;
            });
            executeInference();
        }

        function randomizeInputs() {
            const r = (min, max) => (Math.random() * (max - min) + min).toFixed(1);
            applyPreset(r(4.5, 7.5), r(2.2, 4.0), r(1.2, 6.5), r(0.2, 2.4));
        }

        async function executeInference() {
            const payload = {
                sepal_length: parseFloat(document.getElementById('sl').value),
                sepal_width: parseFloat(document.getElementById('sw').value),
                petal_length: parseFloat(document.getElementById('pl').value),
                petal_width: parseFloat(document.getElementById('pw').value)
            };
            try {
                const res = await fetch('/predict', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
                });
                const data = await res.json();
                const spec = data.species;
                
                document.getElementById('specimen-title').textContent = spec.name;
                document.getElementById('specimen-author').textContent = `Taxonomy: ${spec.author}`;
                document.getElementById('specimen-desc').textContent = spec.desc;
                document.getElementById('specimen-eco').textContent = spec.ecology;
                document.getElementById('main-conf').textContent = `${data.confidences[data.prediction]}%`;
                document.getElementById('specimen-img').src = spec.image;
                
                const badge = document.getElementById('specimen-badge');
                badge.textContent = spec.tag;
                badge.className = `px-2.5 py-0.5 rounded-full text-[11px] font-mono uppercase tracking-wider font-semibold border ${spec.badge}`;
                document.getElementById('ambient-glow').style.background = spec.accent;

                document.getElementById('bar-0').style.width = `${data.confidences.setosa}%`;
                document.getElementById('bar-val-0').textContent = `${data.confidences.setosa}%`;
                document.getElementById('bar-1').style.width = `${data.confidences.versicolor}%`;
                document.getElementById('bar-val-1').textContent = `${data.confidences.versicolor}%`;
                document.getElementById('bar-2').style.width = `${data.confidences.virginica}%`;
                document.getElementById('bar-val-2').textContent = `${data.confidences.virginica}%`;

                if (radarChart) {
                    radarChart.data.datasets[0].data = data.features;
                    radarChart.data.datasets[0].borderColor = spec.accent;
                    radarChart.data.datasets[0].pointBackgroundColor = spec.accent;
                    const rgb = spec.accent.match(/\w\w/g).map(x => parseInt(x, 16));
                    radarChart.data.datasets[0].backgroundColor = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.25)`;
                    radarChart.update();
                }
            } catch(e) { console.error(e); }
        }

        window.addEventListener('DOMContentLoaded', () => {
            initChart(); executeInference();
        });
    </script>
    """
    return get_base_html("Bảng điều khiển", "dashboard", content)

@app.get("/history", response_class=HTMLResponse)
def history_page():
    content = """
    <div class="glass-panel rounded-3xl p-6 sm:p-8 border border-white/10 shadow-2xl">
        <div class="flex items-center justify-between mb-8 border-b border-white/10 pb-4">
            <div>
                <h2 class="text-2xl font-bold text-white">Lịch sử phân tích</h2>
                <p class="text-sm text-slate-400 mt-1">Dữ liệu tự động được lưu trữ sau mỗi lần máy học (SVM) thực hiện dự đoán.</p>
            </div>
            <button onclick="fetchHistory()" class="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-xl text-sm transition">Làm mới 🔄</button>
        </div>
        
        <div class="overflow-x-auto">
            <table class="w-full text-left border-collapse min-w-[700px]">
                <thead>
                    <tr class="border-b border-white/10 text-slate-400 text-xs uppercase tracking-wider">
                        <th class="py-3 px-4 font-semibold">Thời gian</th>
                        <th class="py-3 px-4 font-semibold">S.L</th>
                        <th class="py-3 px-4 font-semibold">S.W</th>
                        <th class="py-3 px-4 font-semibold">P.L</th>
                        <th class="py-3 px-4 font-semibold">P.W</th>
                        <th class="py-3 px-4 font-semibold">Kết quả dự đoán</th>
                        <th class="py-3 px-4 font-semibold text-right">Độ tin cậy</th>
                    </tr>
                </thead>
                <tbody id="history-tbody" class="text-sm text-slate-300">
                    <tr><td colspan="7" class="text-center py-8 text-slate-500">Đang tải dữ liệu...</td></tr>
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        async function fetchHistory() {
            try {
                const res = await fetch('/api/history');
                const data = await res.json();
                const tbody = document.getElementById('history-tbody');
                
                if(data.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" class="text-center py-8 text-slate-500">Chưa có dữ liệu lịch sử nào.</td></tr>';
                    return;
                }
                
                tbody.innerHTML = '';
                data.forEach(row => {
                    let colorClass = row.species.includes('Setosa') ? 'text-emerald-400 bg-emerald-400/10' : 
                                     row.species.includes('Versicolor') ? 'text-cyan-400 bg-cyan-400/10' : 
                                     'text-purple-400 bg-purple-400/10';

                    tbody.innerHTML += `
                        <tr class="border-b border-white/5 hover:bg-white/5 transition">
                            <td class="py-3 px-4 font-mono text-xs text-slate-400">${row.timestamp}</td>
                            <td class="py-3 px-4">${row.sl}</td>
                            <td class="py-3 px-4">${row.sw}</td>
                            <td class="py-3 px-4">${row.pl}</td>
                            <td class="py-3 px-4">${row.pw}</td>
                            <td class="py-3 px-4">
                                <span class="px-2 py-1 rounded-lg text-xs font-semibold border border-white/5 ${colorClass}">
                                    ${row.species}
                                </span>
                            </td>
                            <td class="py-3 px-4 text-right font-mono">${row.confidence}%</td>
                        </tr>
                    `;
                });
            } catch(e) {
                console.error(e);
            }
        }
        window.addEventListener('DOMContentLoaded', fetchHistory);
    </script>
    """
    return get_base_html("Lịch sử", "history", content)

@app.get("/about", response_class=HTMLResponse)
def about_page():
    content = """
    <div class="max-w-4xl mx-auto space-y-8">
        <div class="glass-panel rounded-3xl p-8 border border-white/10 shadow-2xl relative overflow-hidden">
            <div class="absolute -right-20 -top-20 w-72 h-72 rounded-full blur-[80px] bg-emerald-500/20 pointer-events-none"></div>
            
            <h2 class="text-3xl font-extrabold text-white mb-4">Về dự án Botanical Lab</h2>
            <p class="text-slate-300 leading-relaxed text-lg">
                Hệ thống <b>AI Botanical Laboratory</b> được thiết kế để nhận dạng 3 giống hoa Diên Vĩ (Iris) dựa trên thuật toán Support Vector Machine (SVM).
            </p>
            
            <div class="grid md:grid-cols-2 gap-6 mt-8">
                <div class="bg-black/40 p-6 rounded-2xl border border-white/5">
                    <div class="text-3xl mb-3">🧬</div>
                    <h3 class="text-lg font-bold text-white mb-2">Tập dữ liệu (Dataset)</h3>
                    <p class="text-sm text-slate-400 leading-relaxed">
                        Sử dụng tập dữ liệu kinh điển <b>Fisher's Iris dataset (1936)</b>. Tập dữ liệu chứa 150 mẫu hoa thuộc 3 loài (Setosa, Virginica, Versicolor) với 4 đặc trưng hình học: chiều dài/rộng đài hoa và cánh hoa.
                    </p>
                </div>
                <div class="bg-black/40 p-6 rounded-2xl border border-white/5">
                    <div class="text-3xl mb-3">🧠</div>
                    <h3 class="text-lg font-bold text-white mb-2">Mô hình AI (SVM)</h3>
                    <p class="text-sm text-slate-400 leading-relaxed">
                        Mô hình Support Vector Machine tìm kiếm siêu phẳng (hyperplane) tối ưu trong không gian 4 chiều để phân tách các lớp thực vật với độ chính xác cao (thường >97%).
                    </p>
                </div>
                <div class="bg-black/40 p-6 rounded-2xl border border-white/5">
                    <div class="text-3xl mb-3">⚡</div>
                    <h3 class="text-lg font-bold text-white mb-2">Công nghệ Backend</h3>
                    <p class="text-sm text-slate-400 leading-relaxed">
                        Xây dựng trên <b>FastAPI</b> (Python) cho hiệu suất cao, xử lý inference thời gian thực. Tích hợp <b>SQLite</b> để lưu trữ toàn bộ lịch sử các mẫu phân tích.
                    </p>
                </div>
                <div class="bg-black/40 p-6 rounded-2xl border border-white/5">
                    <div class="text-3xl mb-3">🎨</div>
                    <h3 class="text-lg font-bold text-white mb-2">Giao diện (Frontend)</h3>
                    <p class="text-sm text-slate-400 leading-relaxed">
                        Sử dụng <b>Tailwind CSS</b> kết hợp phong cách Glassmorphism. Biểu đồ đa chiều (Radar Chart) render trực tiếp bằng Chart.js.
                    </p>
                </div>
            </div>
            
            <div class="mt-8 pt-8 border-t border-white/10 flex items-center justify-between">
                <p class="text-sm text-slate-400">Được phát triển và tối ưu hóa trải nghiệm người dùng.</p>
                <a href="/" class="px-6 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white font-medium rounded-xl transition">Thử nghiệm ngay</a>
            </div>
        </div>
    </div>
    """
    return get_base_html("Giới thiệu", "about", content)
