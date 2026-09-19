import os
import math
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# 1. Khởi tạo mô hình
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
    version="3.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# THÊM API ĐỂ TRẢ VỀ ẢNH TỪ THƯ MỤC GỐC MỘT CÁCH AN TOÀN
@app.get("/img/{img_name}")
def get_image(img_name: str):
    allowed_images = ["anh_setosa.jpg", "anh_versicolor.jpg", "anh_virginica.jpg"]
    if img_name in allowed_images and os.path.exists(img_name):
        return FileResponse(img_name)
    raise HTTPException(status_code=404, detail="Không tìm thấy ảnh")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# 2. Cập nhật Metadata với đường dẫn qua API mới
SPECIES_METADATA = {
    0: {
        "name": "Iris Setosa",
        "author": "Pall. ex Link",
        "tag": "Setosa",
        "theme": "emerald",
        "accent": "#10b981",
        "badge": "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
        "desc": "Đặc trưng bởi đài hoa rộng nhưng cánh hoa tiêu biến cực nhỏ. Loài hoa này có khả năng phân tách tuyến tính tuyệt đối.",
        "ecology": "Bắc bán cầu, khí hậu ôn đới lạnh, vùng đầm lầy ven biển.",
        "image": "/img/anh_setosa.jpg"  # Gọi qua API /img/
    },
    1: {
        "name": "Iris Versicolor",
        "author": "L.",
        "tag": "Versicolor",
        "theme": "cyan",
        "accent": "#06b6d4",
        "badge": "bg-cyan-500/10 text-cyan-300 border-cyan-500/30",
        "desc": "Mang hình thái trung gian, có sự cân bằng lý tưởng giữa tỷ lệ chiều dài cánh hoa và đài hoa.",
        "ecology": "Khu vực ẩm ướt Bắc Mỹ, ven hồ và đồng cỏ ngập nước ngọt.",
        "image": "/img/anh_versicolor.jpg" # Gọi qua API /img/
    },
    2: {
        "name": "Iris Virginica",
        "author": "L.",
        "tag": "Virginica",
        "theme": "purple",
        "accent": "#a855f7",
        "badge": "bg-purple-500/10 text-purple-300 border-purple-500/30",
        "desc": "Loài hoa có kích thước lớn và cấu trúc tráng lệ nhất với cánh hoa thuôn dài, sắc tím đậm đặc trưng.",
        "ecology": "Đồng cỏ ẩm ven biển và đầm lầy phía Đông Bắc Mỹ.",
        "image": "/img/anh_virginica.jpg" # Gọi qua API /img/
    },
}

CENTROIDS = [
    [5.006, 3.428, 1.462, 0.246],  # Setosa
    [5.936, 2.770, 4.260, 1.326],  # Versicolor
    [6.588, 2.974, 5.552, 2.026],  # Virginica
]

@app.get("/health")
def health():
    return {"status": "healthy", "engine": "SVM Linear Kernel", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: IrisInput):
    if model is None:
        raise HTTPException(status_code=500, detail="SVM Model không tồn tại. Vui lòng cung cấp file svm_model.pkl")
        
    feat = [data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]
    pred = int(model.predict([feat])[0])
    
    dists = [math.sqrt(sum((feat[i] - CENTROIDS[c][i]) ** 2 for i in range(4))) for c in range(3)]
    inv_dists = [1.0 / (d + 1e-5) for d in dists]
    inv_dists[pred] *= 1.8
    total = sum(inv_dists)
    confidences = [round((v / total) * 100, 1) for v in inv_dists]
    
    meta = SPECIES_METADATA[pred]
    return {
        "class_id": pred,
        "prediction": meta["tag"].lower(),
        "species": meta,
        "confidences": {
            "setosa": confidences[0],
            "versicolor": confidences[1],
            "virginica": confidences[2]
        },
        "features": feat
    }

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html lang="vi" class="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Iris Intelligence | AI Botanical Laboratory</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

        <script>
            tailwind.config = {
                darkMode: 'class',
                theme: {
                    extend: {
                        fontFamily: {
                            sans: ['Plus Jakarta Sans', 'sans-serif'],
                            mono: ['JetBrains Mono', 'monospace'],
                        },
                        keyframes: {
                            kenburns: {
                                '0%': { transform: 'scale(1) translate(0, 0)' },
                                '50%': { transform: 'scale(1.05) translate(-1%, -1%)' },
                                '100%': { transform: 'scale(1) translate(0, 0)' },
                            },
                            floatingfog: {
                                '0%': { transform: 'translateX(-5%)', opacity: '0.4' },
                                '50%': { transform: 'translateX(5%)', opacity: '0.6' },
                                '100%': { transform: 'translateX(-5%)', opacity: '0.4' },
                            }
                        },
                        animation: {
                            'kenburns': 'kenburns 40s ease-in-out infinite',
                            'floatingfog': 'floatingfog 20s ease-in-out infinite',
                        }
                    }
                }
            }
        </script>
        <style>
            body { background-color: #020617; }
            .glass-panel {
                background: rgba(15, 23, 42, 0.75);
                backdrop-filter: blur(24px);
                -webkit-backdrop-filter: blur(24px);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            input[type=range]::-webkit-slider-thumb {
                -webkit-appearance: none; height: 18px; width: 18px;
                border-radius: 50%; background: #818cf8; cursor: pointer;
                box-shadow: 0 0 10px rgba(129, 140, 248, 0.8); margin-top: -6px;
            }
            input[type=range]::-webkit-slider-runnable-track {
                width: 100%; height: 6px; cursor: pointer;
                background: rgba(255, 255, 255, 0.15); border-radius: 999px;
            }
        </style>
    </head>
    <body class="min-h-screen text-slate-100 flex flex-col justify-between selection:bg-indigo-500 selection:text-white relative">

        <div class="fixed inset-0 z-[-1] overflow-hidden bg-slate-900">
            <div class="absolute inset-0 bg-cover bg-center bg-no-repeat animate-kenburns opacity-70" 
                 style="background-image: url('https://images.unsplash.com/photo-1472214103451-9374bd1c798e?q=80&w=2070&auto=format&fit=crop');">
            </div>
            <div class="absolute inset-0 bg-gradient-to-t from-slate-900 via-slate-900/60 to-transparent animate-floatingfog"></div>
            <div class="absolute inset-0 bg-slate-950/40 backdrop-blur-[2px]"></div>
        </div>

        <nav class="glass-panel sticky top-0 z-50 border-b border-white/10 px-6 py-4">
            <div class="max-w-7xl mx-auto flex items-center justify-between">
                <div class="flex items-center gap-4">
                    <div class="relative flex items-center justify-center w-11 h-11 rounded-2xl bg-gradient-to-tr from-emerald-600 via-teal-600 to-cyan-600 shadow-lg shadow-emerald-500/30 overflow-hidden border border-white/20">
                        <span class="text-2xl absolute drop-shadow-md">🌿</span>
                    </div>
                    <div>
                        <div class="flex items-center gap-2">
                            <h1 class="font-extrabold text-lg tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white via-emerald-100 to-emerald-300">BOTANICAL LAB</h1>
                            <span class="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">SVM Engine</span>
                        </div>
                        <p class="text-xs text-slate-400">Hệ thống AI nhận diện thực vật</p>
                    </div>
                </div>

                <div class="flex items-center gap-4">
                    <label class="hidden sm:flex items-center gap-2.5 text-xs text-slate-300 cursor-pointer select-none bg-white/5 px-3 py-1.5 rounded-xl border border-white/10 hover:border-white/20 transition">
                        <span>Live Sync</span>
                        <input type="checkbox" id="live-toggle" checked class="sr-only peer">
                        <div class="w-8 h-4.5 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-3.5 after:w-3.5 after:transition-all peer-checked:bg-emerald-500"></div>
                    </label>
                    <button onclick="toggleAudio()" id="sound-btn" class="w-9 h-9 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center text-slate-300 hover:text-white hover:bg-white/10 transition">🔊</button>
                </div>
            </div>
        </nav>

        <main class="max-w-7xl mx-auto px-4 sm:px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8 items-start w-full relative z-10">
            
            <div class="lg:col-span-5 flex flex-col gap-6">
                <div class="glass-panel rounded-3xl p-6 sm:p-7 shadow-2xl relative overflow-hidden">
                    <div class="flex items-center justify-between mb-4">
                        <div>
                            <h2 class="text-base font-bold text-white flex items-center gap-2">
                                <span>📐</span> Thông số kích thước (cm)
                            </h2>
                        </div>
                        <button onclick="randomizeInputs()" class="text-xs text-slate-400 hover:text-emerald-400 font-mono transition flex items-center gap-1 bg-white/5 px-2 py-1 rounded-lg border border-white/5 hover:border-emerald-400/30">
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

                    <button onclick="executeInference()" id="predict-btn" class="w-full mt-7 py-3.5 px-4 bg-gradient-to-r from-emerald-600 to-teal-700 hover:opacity-95 active:scale-[0.99] text-white font-semibold rounded-2xl shadow-xl shadow-emerald-900/30 border border-emerald-400/20 transition flex items-center justify-center gap-2">
                        <span id="btn-text">Phân tích đặc điểm sinh học</span>
                        <div id="btn-spin" class="hidden w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin"></div>
                    </button>
                </div>
            </div>

            <div class="lg:col-span-7 flex flex-col gap-6 relative">
                
                <div id="specimen-card" class="glass-panel rounded-3xl p-6 sm:p-8 relative overflow-hidden transition-all duration-500 border border-white/10 shadow-2xl">
                    <div class="absolute -right-16 -top-16 w-64 h-64 rounded-full blur-3xl opacity-25 pointer-events-none transition-colors duration-1000" id="ambient-glow" style="background: #10b981;"></div>
                    
                    <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-5 pb-6 border-b border-white/10">
                        <div class="flex items-center gap-5">
                            <div class="relative shrink-0">
                                <img id="specimen-img" src="" alt="Iris Image" class="w-20 h-20 sm:w-24 sm:h-24 object-cover rounded-2xl shadow-xl border-2 border-white/20 transition-all duration-500 bg-black/50" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdib3g9IjAgMCAxMDAgMTAwIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0iIzMzMyIvPjx0ZXh0IHg9IjUwIiB5PSI1MCIgZmlsbD0iI2ZmZiIgZm9udC1zaXplPSIxMiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIEltYWdlPC90ZXh0Pjwvc3ZnPg=='">
                                <div class="absolute inset-0 rounded-2xl ring-1 ring-inset ring-white/10 pointer-events-none"></div>
                            </div>
                            
                            <div>
                                <span id="specimen-badge" class="px-2.5 py-0.5 rounded-full text-[11px] font-mono uppercase tracking-wider font-semibold border">SETOSA</span>
                                <h3 id="specimen-title" class="text-2xl sm:text-3xl font-extrabold text-white tracking-tight mt-1 mb-1">Iris Setosa</h3>
                                <p id="specimen-author" class="text-xs text-slate-400 font-mono">Taxonomy: Pall. ex Link</p>
                            </div>
                        </div>

                        <div class="text-right sm:self-center w-full sm:w-auto bg-black/40 backdrop-blur-md px-4 py-3 rounded-2xl border border-white/10 shadow-inner">
                            <span class="text-[10px] text-slate-400 uppercase tracking-wider font-mono">Độ tin cậy</span>
                            <div class="text-3xl font-black font-mono text-white tracking-tighter" id="main-conf">98.4%</div>
                        </div>
                    </div>

                    <div class="py-5 space-y-3">
                        <p id="specimen-desc" class="text-sm text-slate-200 leading-relaxed font-light"></p>
                        <div class="flex items-start gap-2 text-xs text-slate-400 bg-black/20 p-3 rounded-xl border border-white/5">
                            <span class="mt-0.5">📍</span> <span id="specimen-eco" class="leading-relaxed"></span>
                        </div>
                    </div>

                    <div class="space-y-3 pt-4 border-t border-white/10">
                        <span class="text-xs font-mono text-slate-400 uppercase tracking-wider">Mức độ tương đồng phân lớp</span>
                        <div class="space-y-2.5">
                            <div>
                                <div class="flex justify-between text-xs font-mono mb-1"><span class="text-emerald-400">Iris Setosa</span><span id="bar-val-0" class="text-slate-300">0%</span></div>
                                <div class="w-full bg-black/50 h-2 rounded-full overflow-hidden border border-white/5"><div id="bar-0" class="bg-emerald-500 h-full rounded-full transition-all duration-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]" style="width: 0%"></div></div>
                            </div>
                            <div>
                                <div class="flex justify-between text-xs font-mono mb-1"><span class="text-cyan-400">Iris Versicolor</span><span id="bar-val-1" class="text-slate-300">0%</span></div>
                                <div class="w-full bg-black/50 h-2 rounded-full overflow-hidden border border-white/5"><div id="bar-1" class="bg-cyan-500 h-full rounded-full transition-all duration-500 shadow-[0_0_10px_rgba(6,182,212,0.5)]" style="width: 0%"></div></div>
                            </div>
                            <div>
                                <div class="flex justify-between text-xs font-mono mb-1"><span class="text-purple-400">Iris Virginica</span><span id="bar-val-2" class="text-slate-300">0%</span></div>
                                <div class="w-full bg-black/50 h-2 rounded-full overflow-hidden border border-white/5"><div id="bar-2" class="bg-purple-500 h-full rounded-full transition-all duration-500 shadow-[0_0_10px_rgba(168,85,247,0.5)]" style="width: 0%"></div></div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="glass-panel rounded-3xl p-6 border border-white/10">
                    <div class="flex items-center justify-between mb-4">
                        <h4 class="text-sm font-bold text-white flex items-center gap-2"><span>📊</span> Phân tích đa chiều (Radar Chart)</h4>
                    </div>
                    <div class="h-56 w-full flex items-center justify-center">
                        <canvas id="radarChart"></canvas>
                    </div>
                </div>

            </div>
        </main>

        <footer class="border-t border-white/10 glass-panel mt-auto py-5 px-6 text-center text-xs text-slate-400 flex flex-col sm:flex-row items-center justify-between max-w-7xl mx-auto w-full relative z-10">
            <div class="flex items-center gap-2">
                <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span>Hệ thống suy luận: <b>SVM (Support Vector Machine)</b></span>
            </div>
            <div class="mt-2 sm:mt-0 font-mono text-slate-400/80">
                Powered by FastAPI
            </div>
        </footer>

        <script>
            let audioEnabled = true;
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            function playChime(freq = 600) {
                if (!audioEnabled || audioCtx.state === 'suspended') audioCtx.resume();
                if (!audioEnabled) return;
                try {
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.type = "sine";
                    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
                    gain.gain.setValueAtTime(0.03, audioCtx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.3);
                    osc.connect(gain);
                    gain.connect(audioCtx.destination);
                    osc.start();
                    osc.stop(audioCtx.currentTime + 0.3);
                } catch(e){}
            }

            function toggleAudio() {
                audioEnabled = !audioEnabled;
                document.getElementById('sound-btn').textContent = audioEnabled ? '🔊' : '🔇';
            }

            let radarChart;
            function initChart() {
                const ctx = document.getElementById('radarChart').getContext('2d');
                radarChart = new Chart(ctx, {
                    type: 'radar',
                    data: {
                        labels: ['S. Length', 'S. Width', 'P. Length', 'P. Width'],
                        datasets: [
                            {
                                label: 'Mẫu phân tích',
                                data: [5.1, 3.5, 1.4, 0.2],
                                backgroundColor: 'rgba(16, 185, 129, 0.25)',
                                borderColor: '#34d399',
                                pointBackgroundColor: '#34d399',
                                borderWidth: 2,
                            },
                            {
                                label: 'Chuẩn trung bình loài',
                                data: [5.0, 3.4, 1.5, 0.2],
                                backgroundColor: 'transparent',
                                borderColor: 'rgba(255, 255, 255, 0.3)',
                                borderWidth: 1,
                                borderDash: [4, 4],
                            }
                        ]
                    },
                    options: {
                        responsive: true, maintainAspectRatio: false,
                        scales: {
                            r: {
                                angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                                grid: { color: 'rgba(255, 255, 255, 0.08)' },
                                pointLabels: { color: '#e2e8f0', font: { family: 'Plus Jakarta Sans', size: 11, weight: '500' } },
                                ticks: { display: false, maxTicksLimit: 5 },
                                min: 0, max: 8
                            }
                        },
                        plugins: { legend: { display: false } }
                    }
                });
            }

            let debounceTimer = null;
            function syncVal(sliderId, textId) {
                const val = document.getElementById(sliderId).value;
                document.getElementById(textId).textContent = val;

                if (document.getElementById('live-toggle').checked) {
                    clearTimeout(debounceTimer);
                    debounceTimer = setTimeout(executeInference, 150);
                }
            }

            function applyPreset(sl, sw, pl, pw) {
                document.getElementById('sl').value = sl; document.getElementById('sw').value = sw;
                document.getElementById('pl').value = pl; document.getElementById('pw').value = pw;
                document.getElementById('txt-sl').textContent = sl; document.getElementById('txt-sw').textContent = sw;
                document.getElementById('txt-pl').textContent = pl; document.getElementById('txt-pw').textContent = pw;
                executeInference();
            }

            function randomizeInputs() {
                const r = (min, max) => (Math.random() * (max - min) + min).toFixed(1);
                applyPreset(r(4.5, 7.5), r(2.2, 4.0), r(1.2, 6.5), r(0.2, 2.4));
            }

            async function executeInference() {
                const btnSpin = document.getElementById('btn-spin');
                btnSpin.classList.remove('hidden');

                const payload = {
                    sepal_length: parseFloat(document.getElementById('sl').value),
                    sepal_width: parseFloat(document.getElementById('sw').value),
                    petal_length: parseFloat(document.getElementById('pl').value),
                    petal_width: parseFloat(document.getElementById('pw').value)
                };

                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    
                    if (!res.ok) return;
                    
                    const data = await res.json();
                    const spec = data.species;
                    
                    document.getElementById('specimen-title').textContent = spec.name;
                    document.getElementById('specimen-author').textContent = `Taxonomy: ${spec.author}`;
                    document.getElementById('specimen-desc').textContent = spec.desc;
                    document.getElementById('specimen-eco').textContent = spec.ecology;
                    document.getElementById('main-conf').textContent = `${data.confidences[data.prediction]}%`;
                    
                    const imgEl = document.getElementById('specimen-img');
                    
                    const currentImgPath = new URL(imgEl.src, window.location.origin).pathname;
                    if (currentImgPath !== spec.image) {
                        imgEl.style.opacity = 0;
                        setTimeout(() => {
                            imgEl.src = spec.image + "?t=" + new Date().getTime();
                            imgEl.style.opacity = 1;
                        }, 200);
                    }
                    
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

                    playChime(spec.tag === 'Setosa' ? 700 : (spec.tag === 'Versicolor' ? 880 : 1050));
                } catch(e) {
                    console.error(e);
                } finally {
                    btnSpin.classList.add('hidden');
                }
            }

            window.addEventListener('DOMContentLoaded', () => {
                initChart();
                executeInference();
            });
        </script>
    </body>
    </html>
    """

