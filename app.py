import os
import math
import sqlite3
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# ==========================================
# 1. DB LỊCH SỬ & MÔ HÌNH
# ==========================================
DB_PATH = "history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT, sl REAL, sw REAL, pl REAL, pw REAL,
                  species TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

init_db()

MODEL_PATH = "svm_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# Dữ liệu Iris thật + 4 mô hình theo 4 kernel (phục vụ so sánh, PCA 3D, thống kê)
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
    PCA_POINTS = {
        "points": [{"x": float(a), "y": float(b), "z": float(c), "cls": int(t)} for (a, b, c), t in zip(_c, _y)],
        "variance": [round(float(v) * 100, 1) for v in _p.explained_variance_ratio_],
    }
    DATASET_STATS = {
        "features": ["Sepal L", "Sepal W", "Petal L", "Petal W"],
        "species": [
            {"name": n, "count": int((_y == i).sum()),
             "mean": [round(float(v), 2) for v in _X[_y == i].mean(axis=0)],
             "min": [round(float(v), 1) for v in _X[_y == i].min(axis=0)],
             "max": [round(float(v), 1) for v in _X[_y == i].max(axis=0)]}
            for i, n in enumerate(["Setosa", "Versicolor", "Virginica"])
        ],
    }
except ImportError:
    pass

app = FastAPI(title="Iris AI Classifier Pro", version="8.1.0")
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
    0: {"name": "Iris Setosa", "tag": "Setosa", "color": "#ea580c"},
    1: {"name": "Iris Versicolor", "tag": "Versicolor", "color": "#7c3aed"},
    2: {"name": "Iris Virginica", "tag": "Virginica", "color": "#0d9488"},
}
CENTROIDS = [[5.006, 3.428, 1.462, 0.246], [5.936, 2.770, 4.260, 1.326], [6.588, 2.974, 5.552, 2.026]]

# ==========================================
# 3. API
# ==========================================
@app.get("/img/{img_name}")
def get_image(img_name: str):
    allowed = ["anh_setosa.jpg", "anh_versicolor.jpg", "anh_virginica.jpg", "iris_demo.png"]
    if img_name in allowed and os.path.exists(img_name):
        return FileResponse(img_name)
    raise HTTPException(status_code=404, detail="Image not found")


@app.post("/predict")
def predict(data: IrisInput):
    if model is None:
        raise HTTPException(status_code=500, detail="SVM Model missing.")
    feat = [data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]
    pred = int(model.predict([feat])[0])

    dists = [math.sqrt(sum((feat[i] - CENTROIDS[c][i]) ** 2 for i in range(4))) for c in range(3)]
    inv_dists = [1.0 / (d + 1e-5) for d in dists]
    inv_dists[pred] *= 1.8
    total = sum(inv_dists)
    conf = [round((v / total) * 100, 1) for v in inv_dists]
    meta = SPECIES_METADATA[pred]

    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO history (timestamp, sl, sw, pl, pw, species, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                 (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), *feat, meta["name"], conf[pred]))
    conn.commit()
    conn.close()
    return {"prediction": meta["name"], "confidences": conf, "features": feat, "color": meta["color"]}


@app.post("/api/compare_kernels")
def compare_kernels(data: IrisInput):
    if not KERNEL_MODELS:
        raise HTTPException(status_code=500, detail="Cần cài scikit-learn để so sánh kernel.")
    feat = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    out = []
    for k in KERNEL_NAMES:
        probs = KERNEL_MODELS[k].predict_proba(feat)[0]
        idx = int(probs.argmax())
        meta = SPECIES_METADATA[idx]
        out.append({"kernel": k, "prediction": meta["name"], "color": meta["color"],
                    "confidence": round(float(probs[idx]) * 100, 1), "accuracy": KERNEL_SCORES[k]})
    return out


@app.get("/api/kernel_scores")
def kernel_scores():
    return KERNEL_SCORES


@app.get("/api/pca")
def pca_points():
    if PCA_POINTS is None:
        raise HTTPException(status_code=500, detail="Cần cài scikit-learn.")
    return PCA_POINTS


@app.get("/api/dataset")
def dataset_stats():
    if DATASET_STATS is None:
        raise HTTPException(status_code=500, detail="Cần cài scikit-learn.")
    return DATASET_STATS


@app.get("/api/history")
def get_history():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT timestamp, sl, sw, pl, pw, species, confidence FROM history ORDER BY id DESC LIMIT 50").fetchall()
    conn.close()
    return [{"timestamp": r[0], "sl": r[1], "sw": r[2], "pl": r[3], "pw": r[4], "species": r[5], "confidence": r[6]} for r in rows]


@app.delete("/api/history")
def clear_history():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM history")
    conn.commit()
    conn.close()
    return {"status": "success"}


# ==========================================
# 4. KHUNG GIAO DIỆN CƠ BẢN (BASE_HTML)
# ==========================================
BASE_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__ | Botanical Lab Pro</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
  tailwind.config = { darkMode: 'class', theme: { extend: { fontFamily: { sans: ['Plus Jakarta Sans','sans-serif'], mono: ['JetBrains Mono','monospace'] } } } };
  // Khởi tạo theme sớm
  (function(){ const t = localStorage.getItem('theme');
    if (t === 'dark' || (!t && matchMedia('(prefers-color-scheme: dark)').matches)) document.documentElement.classList.add('dark'); })();
</script>
<style>
  /* ===== Biến màu ===== */
  :root {
    --bg:#f7f5f2; --surface:#ffffff; --surface-2:#f1eee9; --text:#1c1917; --muted:#6b645c;
    --border:#e4dfd8; --accent:#ea580c; --accent-soft:rgba(234,88,12,.10);
    --c-setosa:#ea580c; --c-versicolor:#7c3aed; --c-virginica:#0d9488;
    --shadow:0 1px 2px rgba(28,25,23,.06), 0 8px 24px -12px rgba(28,25,23,.12);
  }
  .dark {
    --bg:#0f0d0c; --surface:#1a1715; --surface-2:#231f1c; --text:#f5f2ee; --muted:#a39b91;
    --border:#2e2925; --accent:#fb923c; --accent-soft:rgba(251,146,60,.14);
    --c-setosa:#fb923c; --c-versicolor:#a78bfa; --c-virginica:#2dd4bf;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 12px 32px -12px rgba(0,0,0,.6);
  }
  html { scroll-behavior:smooth; }
  body { background:var(--bg); color:var(--text); font-family:'Plus Jakarta Sans',sans-serif; transition:background .3s,color .3s; }
  .muted { color:var(--muted); }
  .card { background:var(--surface); border:1px solid var(--border); box-shadow:var(--shadow); border-radius:1.25rem; transition:background .3s,border-color .3s,transform .25s,box-shadow .25s; }
  .card-hover:hover { transform:translateY(-3px); }
  .soft { background:var(--surface-2); border:1px solid var(--border); border-radius:1rem; }
  .accent { color:var(--accent); }
  .btn { display:inline-flex; align-items:center; justify-content:center; gap:.5rem; padding:.75rem 1.25rem; border-radius:.85rem; font-weight:700; font-size:.9rem; transition:transform .2s, box-shadow .2s, background .2s, opacity .2s; cursor:pointer; }
  .btn:hover:not(:disabled) { transform:translateY(-2px); }
  .btn:active:not(:disabled) { transform:translateY(0) scale(.98); }
  .btn:disabled { opacity:.45; cursor:not-allowed; }
  .btn-primary { background:var(--accent); color:#fff; box-shadow:0 8px 20px -8px var(--accent); }
  .btn-dark { background:var(--text); color:var(--bg); }
  .btn-ghost { background:var(--surface-2); border:1px solid var(--border); color:var(--text); }
  .btn-danger { background:rgba(220,38,38,.1); color:#dc2626; border:1px solid rgba(220,38,38,.25); }
  input[type=range] { accent-color:var(--accent); }
  .nav-link { position:relative; padding:.4rem .1rem; font-weight:600; font-size:.9rem; color:var(--muted); transition:color .2s; white-space:nowrap; }
  .nav-link:hover, .nav-link.active { color:var(--text); }
  .nav-link::after { content:''; position:absolute; left:0; bottom:-2px; height:2px; width:100%; background:var(--accent); transform:scaleX(0); transform-origin:left; transition:transform .3s; }
  .nav-link.active::after, .nav-link:hover::after { transform:scaleX(1); }
  .sw-c { color:var(--c-setosa); } .ve-c { color:var(--c-versicolor); } .vi-c { color:var(--c-virginica); }

  /* Animation */
  @keyframes pageIn { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:none; } }
  main { animation:pageIn .45s ease both; }
  body.leaving main { opacity:0; transform:translateY(-6px); transition:opacity .2s, transform .2s; }
  @keyframes pop { from { opacity:0; transform:scale(.94); } to { opacity:1; transform:none; } }
  .pop { animation:pop .35s ease both; }

  /* Skeleton */
  .skel { background:linear-gradient(90deg,var(--surface-2),var(--border),var(--surface-2)); background-size:200% 100%; animation:shimmer 1.3s infinite linear; border-radius:.5rem; }
  @keyframes shimmer { to { background-position:-200% 0; } }
  .typing::after { content:'|'; animation:blink 1s step-end infinite; }
  @keyframes blink { 50% { opacity:0; } }

  /* Menu Dropdown */
  .menu { position:absolute; right:0; top:calc(100% + .5rem); min-width:16rem; opacity:0; transform:translateY(-6px) scale(.98); pointer-events:none; transition:opacity .2s, transform .2s; z-index:60; }
  .menu.open { opacity:1; transform:none; pointer-events:auto; }
  .switch { width:2.6rem; height:1.5rem; border-radius:999px; background:var(--border); position:relative; transition:background .25s; flex-shrink:0; }
  .switch::after { content:''; position:absolute; top:.19rem; left:.19rem; width:1.12rem; height:1.12rem; border-radius:50%; background:#fff; transition:transform .25s; }
  .dark .switch { background:var(--accent); } .dark .switch::after { transform:translateX(1.1rem); }
  
  /* Modal / Toast */
  #modal-bg { position:fixed; inset:0; background:rgba(0,0,0,.5); backdrop-filter:blur(4px); display:flex; align-items:center; justify-content:center; z-index:100; opacity:0; pointer-events:none; transition:opacity .2s; }
  #modal-bg.open { opacity:1; pointer-events:auto; }
  #modal-box { transform:scale(.94); transition:transform .25s; }
  #modal-bg.open #modal-box { transform:none; }
  #toasts { position:fixed; bottom:1.25rem; right:1.25rem; z-index:120; display:flex; flex-direction:column; gap:.5rem; }
  .toast { background:var(--text); color:var(--bg); padding:.75rem 1.1rem; border-radius:.8rem; font-weight:600; font-size:.875rem; box-shadow:var(--shadow); animation:pageIn .3s ease both; }
  .toast.out { opacity:0; transform:translateY(8px); transition:.3s; }
  ::-webkit-scrollbar { width:6px; height:6px; } ::-webkit-scrollbar-thumb { background:var(--border); border-radius:4px; }
</style>
</head>
<body class="min-h-screen flex flex-col">
<header class="sticky top-0 z-50 border-b backdrop-blur-md" style="background:color-mix(in srgb,var(--bg) 80%,transparent);border-color:var(--border)">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between gap-4">
    <a href="/" class="flex items-center gap-3 shrink-0">
      <div class="w-10 h-10 rounded-xl flex items-center justify-center text-xl text-white" style="background:var(--accent)">🌿</div>
      <div><div class="font-extrabold tracking-tight leading-tight">Iris Pro</div><div class="text-[11px] muted font-mono">SVM Classifier</div></div>
    </a>
    <nav id="nav" class="flex items-center gap-5 overflow-x-auto">
      <a href="/" class="nav-link" data-i18n="nav_home">Trang chủ</a>
      <a href="/dataset" class="nav-link" data-i18n="nav_dataset">Dữ liệu</a>
      <a href="/lab" class="nav-link" data-i18n="nav_lab">Phân tích</a>
      <a href="/vector3d" class="nav-link" data-i18n="nav_3d">Không gian 3D</a>
      <a href="/kernels" class="nav-link" data-i18n="nav_kernels">Kernels</a>
      <a href="/game" class="nav-link" data-i18n="nav_game">Giải trí</a>
    </nav>
    <div class="relative shrink-0">
      <button id="settings-btn" class="btn btn-ghost !py-2" aria-haspopup="true" data-i18n="settings">⚙️ Cài đặt ▾</button>
      <div id="settings-menu" class="menu card p-2">
        <button onclick="toggleTheme()" class="w-full flex items-center justify-between gap-4 p-3 rounded-xl hover:bg-[var(--surface-2)] transition text-left">
          <span class="text-sm font-semibold"><span data-i18n="theme_dark">Giao diện tối</span> <span id="theme-label" class="block text-xs muted font-normal">Đang dùng chế độ sáng</span></span>
          <span class="switch"></span>
        </button>
        <button onclick="toggleLang()" class="w-full flex items-center justify-between gap-4 p-3 rounded-xl hover:bg-[var(--surface-2)] transition text-left mt-1 border-t" style="border-color:var(--border)">
          <span class="text-sm font-semibold" data-i18n="lang_label">Ngôn ngữ (Language)</span>
          <span id="lang-status" class="text-xs font-mono accent">Tiếng Việt</span>
        </button>
      </div>
    </div>
  </div>
</header>

<main class="w-full flex-grow flex flex-col">__CONTENT__</main>

<footer class="border-t mt-auto" style="border-color:var(--border)">
  <div class="max-w-7xl mx-auto px-6 py-5 flex flex-col md:flex-row items-center justify-between gap-2 text-xs muted">
    <span>Fisher's Iris (1936) · Machine Learning</span>
    <span><span data-i18n="author">Tác giả:</span> <b style="color:var(--text)">Nguyễn Minh Hoàng</b></span>
  </div>
</footer>

<div id="modal-bg"><div id="modal-box" class="card p-6 max-w-sm w-[90%]">
  <h3 id="modal-title" class="font-bold text-lg mb-2"></h3><p id="modal-msg" class="text-sm muted mb-6"></p>
  <div class="flex gap-3 justify-end"><button id="modal-no" class="btn btn-ghost">Hủy</button><button id="modal-yes" class="btn btn-danger">Xác nhận</button></div>
</div></div>
<div id="toasts"></div>

<script>
  // ---- Ngôn ngữ (i18n) ----
  const i18nDict = {
    vi: {
        nav_home: "Trang chủ", nav_dataset: "Dữ liệu", nav_lab: "Phân tích",
        nav_3d: "Không gian 3D", nav_kernels: "Kernels", nav_game: "Giải trí",
        settings: "⚙️ Cài đặt ▾", theme_dark: "Giao diện tối", lang_label: "Ngôn ngữ",
        lang_curr: "Tiếng Việt", author: "Tác giả:"
    },
    en: {
        nav_home: "Home", nav_dataset: "Dataset", nav_lab: "Lab",
        nav_3d: "3D Space", nav_kernels: "Kernels", nav_game: "Game",
        settings: "⚙️ Settings ▾", theme_dark: "Dark mode", lang_label: "Language",
        lang_curr: "English", author: "Author:"
    }
  };
  let currentLang = localStorage.getItem('lang') || 'vi';

  function applyLang() {
      document.querySelectorAll('[data-i18n]').forEach(el => {
          const key = el.getAttribute('data-i18n');
          if (i18nDict[currentLang] && i18nDict[currentLang][key]) {
              el.innerHTML = i18nDict[currentLang][key];
          }
      });
      document.getElementById('lang-status').textContent = i18nDict[currentLang]['lang_curr'];
      syncThemeUI(); // Cập nhật luôn chữ trạng thái theme
  }

  function toggleLang() {
      currentLang = currentLang === 'vi' ? 'en' : 'vi';
      localStorage.setItem('lang', currentLang);
      applyLang();
  }

  // ---- Theme ----
  function syncThemeUI() {
    const dark = document.documentElement.classList.contains('dark');
    const txtVi = dark ? 'Đang dùng chế độ tối' : 'Đang dùng chế độ sáng';
    const txtEn = dark ? 'Using dark mode' : 'Using light mode';
    document.getElementById('theme-label').textContent = currentLang === 'en' ? txtEn : txtVi;
    
    if (window.Chart) { 
        Chart.defaults.color = getComputedStyle(document.documentElement).getPropertyValue('--muted').trim();
        Chart.defaults.borderColor = getComputedStyle(document.documentElement).getPropertyValue('--border').trim();
        Object.values(Chart.instances).forEach(c => c.update()); 
    }
    if (typeof onThemeChange === 'function') onThemeChange(dark);
  }

  function toggleTheme() {
    const dark = !document.documentElement.classList.contains('dark');
    document.documentElement.classList.toggle('dark', dark);
    localStorage.setItem('theme', dark ? 'dark' : 'light');
    syncThemeUI();
  }

  // ---- Tương tác UI Khác ----
  const menu = document.getElementById('settings-menu');
  document.getElementById('settings-btn').addEventListener('click', e => { e.stopPropagation(); menu.classList.toggle('open'); });
  document.addEventListener('click', e => { if (!menu.contains(e.target)) menu.classList.remove('open'); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') menu.classList.remove('open'); });
  
  // Xử lý chuyển trang mượt
  document.querySelectorAll('#nav a').forEach(a => {
    if (a.getAttribute('href') === location.pathname) a.classList.add('active');
    a.addEventListener('click', e => { if (a.getAttribute('href') === location.pathname) return;
      e.preventDefault(); document.body.classList.add('leaving'); setTimeout(() => location.href = a.getAttribute('href'), 180); });
  });

  function toast(msg) {
    const el = document.createElement('div'); el.className = 'toast'; el.textContent = msg;
    document.getElementById('toasts').appendChild(el);
    setTimeout(() => { el.classList.add('out'); setTimeout(() => el.remove(), 300); }, 2400);
  }

  function confirmModal(title, msg) {
    return new Promise(resolve => {
      const bg = document.getElementById('modal-bg');
      document.getElementById('modal-title').textContent = title; document.getElementById('modal-msg').textContent = msg;
      bg.classList.add('open');
      const done = v => { bg.classList.remove('open'); yes.onclick = no.onclick = null; resolve(v); };
      const yes = document.getElementById('modal-yes'), no = document.getElementById('modal-no');
      yes.onclick = () => done(true); no.onclick = () => done(false);
      bg.onclick = e => { if (e.target === bg) done(false); };
    });
  }
  
  window.addEventListener('DOMContentLoaded', () => {
      applyLang();
  });
</script>
</body>
</html>"""


def get_base_html(title, content):
    return BASE_HTML.replace("__TITLE__", title).replace("__CONTENT__", content)


# ==========================================
# 5. CÁC TRANG
# ==========================================
@app.get("/", response_class=HTMLResponse)
def hero_page():
    content = """
    <section class="border-b" style="border-color:var(--border)">
      <div class="max-w-7xl mx-auto px-6 py-16 lg:py-28 grid lg:grid-cols-2 gap-12 items-center">
        <div>
          <span class="px-3 py-1 text-xs font-bold rounded-full" style="background:var(--accent-soft);color:var(--accent)">Khoa học dữ liệu</span>
          <h2 class="mt-6 text-4xl lg:text-6xl font-extrabold leading-[1.1] tracking-tight">Nhận dạng hoa Diên Vĩ bằng SVM</h2>
          <p class="mt-6 text-lg muted leading-relaxed max-w-lg">Nhập bốn số đo của một bông hoa, mô hình sẽ cho biết đó là loài nào. Bạn cũng có thể xem dữ liệu trong không gian 3D và so sánh bốn kernel.</p>
          <div class="mt-8 flex flex-wrap gap-3">
            <a href="/lab" class="btn btn-primary">Thử dự đoán</a>
            <a href="/dataset" class="btn btn-ghost">Xem bộ dữ liệu</a>
          </div>
        </div>
        <div class="grid grid-cols-3 gap-4">
          <div class="card card-hover p-5 text-center"><div class="text-4xl mb-3">🌸</div><div class="font-bold sw-c">Setosa</div><p class="text-xs muted mt-1">Tách biệt rõ ràng</p></div>
          <div class="card card-hover p-5 text-center"><div class="text-4xl mb-3">🌺</div><div class="font-bold ve-c">Versicolor</div><p class="text-xs muted mt-1">Kích thước trung bình</p></div>
          <div class="card card-hover p-5 text-center"><div class="text-4xl mb-3">🌷</div><div class="font-bold vi-c">Virginica</div><p class="text-xs muted mt-1">Lớn nhất, hơi giống Versicolor</p></div>
        </div>
      </div>
    </section>"""
    return get_base_html("Trang chủ", content)


@app.get("/dataset", response_class=HTMLResponse)
def dataset_page():
    content = """
    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-8 w-full">
      <h2 class="text-2xl font-bold">Giới thiệu dữ liệu</h2>
      <p class="muted text-sm mt-1 mb-6">Fisher's Iris (1936): số đo của 150 bông hoa, chia đều cho 3 loài.</p>

      <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div class="card card-hover p-5"><div class="text-3xl font-extrabold accent">150</div><div class="text-sm muted">Mẫu hoa</div></div>
        <div class="card card-hover p-5"><div class="text-3xl font-extrabold accent">4</div><div class="text-sm muted">Số đo (cm)</div></div>
        <div class="card card-hover p-5"><div class="text-3xl font-extrabold accent">3</div><div class="text-sm muted">Loài hoa</div></div>
        <div class="card card-hover p-5"><div class="text-3xl font-extrabold accent">0</div><div class="text-sm muted">Giá trị thiếu</div></div>
      </div>

      <div class="grid lg:grid-cols-3 gap-6 mb-6">
        <div class="card p-6"><h3 class="font-bold text-sm mb-3">Số mẫu mỗi loài</h3><div class="h-56"><canvas id="pie"></canvas></div></div>
        <div class="card p-6 lg:col-span-2"><h3 class="font-bold text-sm mb-3">Số đo trung bình theo loài (cm)</h3><div class="h-56"><canvas id="bar"></canvas></div></div>
      </div>

      <div class="grid md:grid-cols-4 gap-4 mb-6" id="feature-cards">
        <div class="soft p-4"><b>Sepal L</b><p class="text-xs muted mt-1">Chiều dài đài hoa, phần lá xanh bao ngoài cánh.</p></div>
        <div class="soft p-4"><b>Sepal W</b><p class="text-xs muted mt-1">Chiều rộng đài hoa.</p></div>
        <div class="soft p-4"><b>Petal L</b><p class="text-xs muted mt-1">Chiều dài cánh hoa. Số đo giúp phân biệt loài nhiều nhất.</p></div>
        <div class="soft p-4"><b>Petal W</b><p class="text-xs muted mt-1">Chiều rộng cánh hoa. Cũng phân biệt loài rất tốt.</p></div>
      </div>

      <div class="card p-6">
        <h3 class="font-bold text-sm mb-3">Bảng thống kê (nhỏ nhất – trung bình – lớn nhất)</h3>
        <div class="overflow-x-auto"><table class="w-full text-sm text-left whitespace-nowrap">
          <thead class="muted text-xs"><tr id="stat-head"><th class="py-2 pr-4">Loài</th></tr></thead>
          <tbody id="stat-body" class="font-mono"></tbody></table></div>
      </div>
    </div>
    <script>
      const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
      const cols = () => [cssv('--c-setosa'), cssv('--c-versicolor'), cssv('--c-virginica')];
      let D;
      function paint() {
        Object.values(Chart.instances).forEach(c => c.destroy());
        const c = cols();
        new Chart(document.getElementById('pie'), { type:'doughnut',
          data:{ labels:D.species.map(s=>s.name), datasets:[{ data:D.species.map(s=>s.count), backgroundColor:c, borderWidth:0 }] },
          options:{ maintainAspectRatio:false, cutout:'65%', plugins:{ legend:{ position:'bottom' } } } });
        new Chart(document.getElementById('bar'), { type:'bar',
          data:{ labels:D.features, datasets:D.species.map((s,i)=>({ label:s.name, data:s.mean, backgroundColor:c[i], borderRadius:6 })) },
          options:{ maintainAspectRatio:false, plugins:{ legend:{ position:'bottom' } } } });
      }
      function onThemeChange() { if (D) paint(); }
      window.addEventListener('DOMContentLoaded', async () => {
        try {
          D = await (await fetch('/api/dataset')).json();
          document.getElementById('stat-head').innerHTML += D.features.map(f=>`<th class="py-2 px-3">${f}</th>`).join('');
          document.getElementById('stat-body').innerHTML = D.species.map((s,i)=>`<tr class="border-t" style="border-color:var(--border)"><td class="py-3 pr-4 font-bold">${s.name}</td>${s.mean.map((m,j)=>`<td class="py-3 px-3">${s.min[j]} – <b>${m}</b> –${s.max[j]}</td>`).join('')}</tr>`).join('');
          paint();
        } catch (e) { toast('Không tải được dữ liệu. Hãy cài scikit-learn.'); }
      });
    </script>"""
    return get_base_html("Giới thiệu dữ liệu", content)


@app.get("/lab", response_class=HTMLResponse)
def lab_page():
    content = """
    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-8 w-full">
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start mb-6">
        <div class="lg:col-span-4 card p-6">
          <div class="flex justify-between items-center mb-6">
            <h2 class="font-bold text-lg">Tham số đầu vào</h2>
            <button onclick="randomizeInputs()" title="Ngẫu nhiên" class="text-xl hover:rotate-180 transition duration-300">🎲</button>
          </div>
          <div class="space-y-6">
            <div><div class="flex justify-between text-xs font-bold mb-2"><span>Sepal L</span><span id="txt-sl" class="accent">5.1</span></div><input type="range" id="sl" min="4.0" max="8.0" step="0.1" value="5.1" class="w-full" oninput="syncVal('sl','txt-sl')"></div>
            <div><div class="flex justify-between text-xs font-bold mb-2"><span>Sepal W</span><span id="txt-sw" class="accent">3.5</span></div><input type="range" id="sw" min="2.0" max="4.5" step="0.1" value="3.5" class="w-full" oninput="syncVal('sw','txt-sw')"></div>
            <div><div class="flex justify-between text-xs font-bold mb-2"><span>Petal L</span><span id="txt-pl" class="accent">1.4</span></div><input type="range" id="pl" min="1.0" max="7.0" step="0.1" value="1.4" class="w-full" oninput="syncVal('pl','txt-pl')"></div>
            <div><div class="flex justify-between text-xs font-bold mb-2"><span>Petal W</span><span id="txt-pw" class="accent">0.2</span></div><input type="range" id="pw" min="0.1" max="2.6" step="0.1" value="0.2" class="w-full" oninput="syncVal('pw','txt-pw')"></div>
          </div>
          <button id="btn-predict" onclick="executeInference()" class="btn btn-dark w-full mt-8">Phân tích ngay</button>
          <!-- Nút so sánh: ẩn cho tới khi có kết quả -->
          <button id="btn-compare" onclick="compareKernels()" disabled class="btn btn-primary w-full mt-3 hidden">So sánh 4 nhân (Kernels) của mô hình</button>
        </div>

        <div class="lg:col-span-8 flex flex-col gap-6">
          <div class="card p-6 sm:p-8 relative overflow-hidden min-h-[140px] flex items-center">
            <div id="result-view" class="flex items-center justify-between w-full transition-opacity duration-300">
              <div><p class="text-xs font-bold muted">Kết quả dự đoán</p><h3 id="specimen-title" class="text-3xl font-black mt-1">Chưa có kết quả</h3></div>
              <div class="text-right"><p class="text-xs font-bold muted">Độ tin cậy</p><div class="text-4xl font-black font-mono accent" id="main-conf">–</div></div>
            </div>
            <div id="skeleton-view" class="hidden absolute inset-0 p-8 z-10 flex justify-between items-center" style="background:var(--surface)">
              <div class="w-2/3 space-y-4">
                <div class="text-sm font-mono accent typing">Đang xử lý...</div>
                <div class="skel h-8 w-48"></div>
              </div>
              <div class="skel h-12 w-24"></div>
            </div>
          </div>

          <div id="compare-panel" class="hidden card p-6">
            <h4 class="font-bold mb-4">Kết quả của 4 kernel với cùng số đo</h4>
            <div id="compare-grid" class="grid sm:grid-cols-2 xl:grid-cols-4 gap-4"></div>
            <p class="text-xs muted mt-4">Độ chính xác lấy từ kiểm định chéo 5 lần trên toàn bộ dữ liệu Iris.</p>
          </div>

          <div class="card p-6 h-72 flex flex-col">
            <h4 class="text-sm font-bold mb-2">Phân bố dữ liệu 2D (Petal L – Petal W)</h4>
            <div class="flex-grow w-full relative"><canvas id="scatterChart"></canvas></div>
          </div>
        </div>
      </div>

      <div class="card p-6">
        <div class="flex flex-wrap justify-between items-center gap-3 mb-4">
          <h2 class="font-bold text-lg">Lịch sử dự đoán</h2>
          <button onclick="clearHistory()" class="btn btn-danger !py-2">🗑 Xóa lịch sử</button>
        </div>
        <div class="overflow-x-auto"><table class="w-full text-left text-sm whitespace-nowrap">
          <thead><tr class="muted text-xs border-b" style="border-color:var(--border)"><th class="py-3 px-4">Thời gian</th><th class="py-3 px-4">P.L</th><th class="py-3 px-4">P.W</th><th class="py-3 px-4">Dự đoán</th><th class="py-3 px-4">Tin cậy</th></tr></thead>
          <tbody id="history-tbody"></tbody></table></div>
        <p id="history-empty" class="hidden text-center muted text-sm py-8">Chưa có dự đoán nào. Bấm "Phân tích ngay" để bắt đầu.</p>
      </div>
    </div>

    <script>
      const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
      let scatterChart, lastPayload = null;
      const KNAME = { linear:'Linear', rbf:'RBF', poly:'Polynomial', sigmoid:'Sigmoid' };

      function initCharts() {
        const c = [cssv('--c-setosa'), cssv('--c-versicolor'), cssv('--c-virginica')];
        const mk = (label, data, i) => ({ label, data, backgroundColor:c[i] + '99' });
        scatterChart = new Chart(document.getElementById('scatterChart').getContext('2d'), { type:'scatter',
          data:{ datasets:[
            mk('Setosa', [{x:1.4,y:0.2},{x:1.5,y:0.2},{x:1.3,y:0.2},{x:1.6,y:0.2}], 0),
            mk('Versicolor', [{x:4.7,y:1.4},{x:4.5,y:1.5},{x:4.9,y:1.5}], 1),
            mk('Virginica', [{x:6.0,y:2.5},{x:5.1,y:1.9},{x:5.9,y:2.1}], 2),
            { label:'Mẫu của bạn', data:[], backgroundColor:cssv('--text'), pointRadius:8, pointBorderColor:cssv('--bg'), pointBorderWidth:2 } ] },
          options:{ responsive:true, maintainAspectRatio:false, plugins:{ legend:{ position:'bottom' } } } });
      }
      function syncVal(s, t) { document.getElementById(t).textContent = document.getElementById(s).value; }
      function randomizeInputs() {
        const r = (a, b) => (Math.random() * (b - a) + a).toFixed(1);
        const range = { sl:[4,8], sw:[2,4.5], pl:[1,7], pw:[0.1,2.6] };
        Object.keys(range).forEach(id => { document.getElementById(id).value = r(...range[id]); syncVal(id, 'txt-' + id); });
      }
      function renderHistory(data) {
        localStorage.setItem('iris_history', JSON.stringify(data));
        document.getElementById('history-empty').classList.toggle('hidden', data.length > 0);
        document.getElementById('history-tbody').innerHTML = data.map(r => `<tr class="border-b" style="border-color:var(--border)"><td class="py-2 px-4 text-xs font-mono muted">${r.timestamp}</td><td class="py-2 px-4">${r.pl}</td><td class="py-2 px-4">${r.pw}</td><td class="py-2 px-4 font-bold accent">${r.species}</td><td class="py-2 px-4 font-mono">${r.confidence}%</td></tr>`).join('');
      }
      async function fetchHistory() { renderHistory(await (await fetch('/api/history')).json()); }

      async function clearHistory() {
        const ok = await confirmModal('Xóa lịch sử?', 'Toàn bộ lịch sử dự đoán sẽ bị xóa và không thể khôi phục.');
        if (!ok) return;
        await fetch('/api/history', { method:'DELETE' });
        localStorage.removeItem('iris_history');
        renderHistory([]);
        toast('Đã xóa lịch sử');
      }

      async function executeInference(silent) {
        const view = document.getElementById('result-view'), skel = document.getElementById('skeleton-view');
        const btn = document.getElementById('btn-predict');
        btn.disabled = true; btn.textContent = 'Đang xử lý...';
        view.classList.add('opacity-0'); skel.classList.remove('hidden');
        const pl = parseFloat(document.getElementById('pl').value), pw = parseFloat(document.getElementById('pw').value);
        const payload = { sepal_length:parseFloat(document.getElementById('sl').value), sepal_width:parseFloat(document.getElementById('sw').value), petal_length:pl, petal_width:pw };
        try {
          const [res] = await Promise.all([ fetch('/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) }), new Promise(r => setTimeout(r, 900)) ]);
          if (!res.ok) throw new Error();
          const data = await res.json();
          document.getElementById('specimen-title').textContent = data.prediction;
          document.getElementById('specimen-title').style.color = data.color;
          document.getElementById('main-conf').textContent = Math.max(...data.confidences) + '%';
          scatterChart.data.datasets[3].data = [{ x:pl, y:pw }]; scatterChart.update();
          lastPayload = payload;
          const cb = document.getElementById('btn-compare');
          cb.classList.remove('hidden'); cb.classList.add('pop'); cb.disabled = false;   // chỉ hiện SAU khi có kết quả
          document.getElementById('compare-panel').classList.add('hidden');
          fetchHistory();
        } catch (e) { toast('Dự đoán thất bại. Kiểm tra file svm_model.pkl.'); }
        skel.classList.add('hidden'); view.classList.remove('opacity-0');
        btn.disabled = false; btn.textContent = currentLang==='en'?'Analyze now':'Phân tích ngay';
      }

      async function compareKernels() {
        if (!lastPayload) return;
        const cb = document.getElementById('btn-compare'); cb.disabled = true; cb.textContent = 'Đang so sánh...';
        const panel = document.getElementById('compare-panel'), grid = document.getElementById('compare-grid');
        panel.classList.remove('hidden');
        grid.innerHTML = [1,2,3,4].map(() => '<div class="skel h-28"></div>').join('');
        try {
          const res = await fetch('/api/compare_kernels', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(lastPayload) });
          if (!res.ok) throw new Error();
          const rows = await res.json();
          grid.innerHTML = rows.map((r, i) => `<div class="soft p-4 pop" style="animation-delay:${i*70}ms"><div class="text-xs muted mb-1">Kernel ${KNAME[r.kernel]}</div><div class="font-bold" style="color:${r.color}">${r.prediction}</div><div class="font-mono text-2xl font-black mt-1">${r.confidence}%</div><div class="text-xs muted mt-1">Chính xác chung: ${r.accuracy}%</div></div>`).join('');
        } catch (e) { panel.classList.add('hidden'); toast('Không so sánh được. Hãy cài scikit-learn.'); }
        cb.disabled = false; cb.textContent = currentLang==='en'?'Compare 4 Kernels':'So sánh 4 nhân (Kernels) của mô hình';
      }
      function onThemeChange() { if (scatterChart) { scatterChart.data.datasets[3].backgroundColor = cssv('--text'); scatterChart.update(); } }
      window.addEventListener('DOMContentLoaded', () => {
        initCharts();
        try { renderHistory(JSON.parse(localStorage.getItem('iris_history') || '[]')); } catch (e) {}
        fetchHistory();
      });
    </script>"""
    return get_base_html("Phân tích", content)


@app.get("/vector3d", response_class=HTMLResponse)
def vector3d_page():
    content = """
    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-8 w-full">
      <h2 class="text-2xl font-bold">Không gian 3D</h2>
      <p class="text-sm muted mt-1 mb-6">Mỗi chấm là một bông hoa. Hoa càng giống nhau thì chấm càng nằm gần nhau. Kéo chuột để xoay, cuộn để thu phóng, rê chuột lên chấm để xem giải thích.</p>
      <div class="grid lg:grid-cols-4 gap-6">
        <div class="lg:col-span-3 card p-2"><div id="plot3d" class="w-full h-[560px]"></div></div>
        <div class="space-y-4">
          <div class="soft p-4"><b>🔵 Một chấm là gì?</b><p class="text-sm muted mt-1">Một bông hoa. 4 số đo của nó được nén xuống 3 trục để bạn nhìn được trên màn hình.</p></div>
          <div class="soft p-4"><b>🎨 Màu sắc</b><p class="text-sm muted mt-1">Mỗi màu là một loài. Nhóm màu cam nằm riêng, tím và xanh ngọc nằm gần nhau và hơi lẫn.</p></div>
          <div class="soft p-4"><b>📐 Ba trục</b><p class="text-sm muted mt-1">Các trục là "tổng hợp" của cả 4 số đo (kỹ thuật PCA). <span id="var-note"></span></p></div>
          <div class="soft p-4"><b>🧠 Mô hình làm gì?</b><p class="text-sm muted mt-1">SVM tìm một "ranh giới" chia các nhóm. Hoa mới rơi vào vùng nào thì được xếp vào loài đó.</p></div>
        </div>
      </div>
    </div>
    <script>
      const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
      const NAMES = ['Setosa','Versicolor','Virginica'];
      const TIPS = ['Cánh hoa nhỏ nhất. Rất dễ nhận ra, thường không bị nhầm.', 'Kích thước trung bình. Đôi khi bị nhầm với Virginica.', 'Cánh hoa lớn nhất. Vùng giáp Versicolor là chỗ mô hình dễ nhầm.'];
      let PTS;
      function draw() {
        const c = [cssv('--c-setosa'), cssv('--c-versicolor'), cssv('--c-virginica')], txt = cssv('--muted');
        const traces = [0,1,2].map(i => { const p = PTS.points.filter(q => q.cls === i);
          return { type:'scatter3d', mode:'markers', name:NAMES[i], x:p.map(q=>q.x), y:p.map(q=>q.y), z:p.map(q=>q.z),
            marker:{ size:5, color:c[i], opacity:.9 }, hovertemplate:'<b>Iris ' + NAMES[i] + '</b><br>' + TIPS[i] + '<extra></extra>' }; });
        const ax = t => ({ title:{ text:t, font:{ color:txt } }, showgrid:true, gridcolor:cssv('--border'), zeroline:false, showticklabels:false, backgroundcolor:'rgba(0,0,0,0)' });
        Plotly.react('plot3d', traces, { margin:{l:0,r:0,b:0,t:0}, paper_bgcolor:'rgba(0,0,0,0)',
          scene:{ xaxis:ax('Trục 1: kích cỡ chung'), yaxis:ax('Trục 2'), zaxis:ax('Trục 3') },
          legend:{ x:0.02, y:0.98, font:{ color:cssv('--text') } } }, { responsive:true, displaylogo:false });
      }
      function onThemeChange() { if (PTS) draw(); }
      window.addEventListener('DOMContentLoaded', async () => {
        try { PTS = await (await fetch('/api/pca')).json();
          document.getElementById('var-note').textContent = 'Ba trục giữ lại ' + PTS.variance.reduce((a,b)=>a+b,0).toFixed(1) + '% thông tin gốc.';
          draw(); } catch (e) { toast('Không tải được dữ liệu 3D. Hãy cài scikit-learn.'); }
      });
    </script>"""
    return get_base_html("Không gian 3D", content)


@app.get("/kernels", response_class=HTMLResponse)
def kernels_page():
    content = """
    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-8 w-full">
      <h2 class="text-2xl font-bold">So sánh Kernels</h2>
      <p class="text-sm muted mt-1 mb-6">Kernel là cách mô hình "nhìn" dữ liệu để vẽ ranh giới giữa các loài. Chọn một kernel để xem giải thích.</p>
      <div class="grid lg:grid-cols-2 gap-6">
        <div class="card p-6"><h3 class="text-sm font-bold mb-4">Độ chính xác (kiểm định chéo 5 lần, %)</h3><div class="h-64"><canvas id="accChart"></canvas></div></div>
        <div class="card p-6">
          <div id="tabs" class="flex flex-wrap gap-2 mb-4"></div>
          <div id="k-info" class="pop"></div>
        </div>
      </div>
    </div>
    <script>
      const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
      const INFO = {
        linear:{ n:'Linear', icon:'📏', how:'Vẽ một đường thẳng (hoặc mặt phẳng) để chia các nhóm.', ex:'Giống như kẻ một đường trên bản đồ để chia hai khu vực.', good:'Nhanh, dễ hiểu. Hợp khi các nhóm tách nhau khá rõ, như Iris.', bad:'Không xử lý được ranh giới cong.' },
        rbf:{ n:'RBF', icon:'🎯', how:'Đo khoảng cách từ điểm mới đến từng điểm cũ, rồi vẽ ranh giới cong bao quanh các nhóm.', ex:'Giống như khoanh vùng "ai sống gần ai" trên bản đồ.', good:'Linh hoạt, dùng được cho nhiều loại dữ liệu.', bad:'Dễ học vẽ quá khít dữ liệu cũ nếu chỉnh tham số không cẩn thận.' },
        poly:{ n:'Polynomial', icon:'🌊', how:'Kết hợp các số đo với nhau (nhân, bình phương...) để tạo ra ranh giới uốn lượn.', ex:'Giống như dùng đường cong thay cho đường thẳng khi vẽ ranh giới.', good:'Bắt được mối quan hệ giữa các số đo.', bad:'Bậc càng cao càng chậm và dễ quá khít.' },
        sigmoid:{ n:'Sigmoid', icon:'🧬', how:'Hoạt động giống một nơ-ron của mạng nơ-ron: bật hoặc tắt theo ngưỡng.', ex:'Giống công tắc: vượt ngưỡng thì bật, chưa vượt thì tắt.', good:'Có nền tảng từ mạng nơ-ron.', bad:'Với Iris cho kết quả thấp nhất, vì không hợp với dữ liệu này.' }
      };
      let scores = {}, current = 'linear', chart;
      const order = ['linear','rbf','poly','sigmoid'];
      function info(k) { const i = INFO[k];
        document.getElementById('k-info').innerHTML = `<div class="flex items-center gap-3 mb-3"><span class="text-3xl">${i.icon}</span><h4 class="text-xl font-bold">Kernel ${i.n}</h4><span class="ml-auto font-mono font-bold accent">${scores[k] ?? '–'}%</span></div>
          <p class="text-sm mb-2"><b>Cách hoạt động:</b> ${i.how}</p><p class="text-sm muted mb-2">💡 ${i.ex}</p>
          <p class="text-sm mb-1"><span class="vi-c font-bold">Ưu điểm:</span> ${i.good}</p><p class="text-sm"><span class="sw-c font-bold">Hạn chế:</span> ${i.bad}</p>`; }
      function tabs() { document.getElementById('tabs').innerHTML = order.map(k => `<button class="btn ${k===current?'btn-primary':'btn-ghost'} !py-2" onclick="pick('${k}')">${INFO[k].n}</button>`).join(''); }
      function pick(k) { current = k; tabs(); info(k); paintChart(); }
      function paintChart() {
        if (chart) chart.destroy();
        const cc = [cssv('--accent'), cssv('--c-versicolor'), cssv('--c-virginica'), cssv('--muted')];
        chart = new Chart(document.getElementById('accChart'), { type:'bar',
          data:{ labels:order.map(k=>INFO[k].n), datasets:[{ data:order.map(k=>scores[k]), backgroundColor:order.map((k,i)=>k===current?cc[i]:cc[i]+'55'), borderRadius:8 }] },
          options:{ maintainAspectRatio:false, plugins:{ legend:{ display:false } }, scales:{ y:{ min:50, max:100 } },
            onClick:(e, els) => { if (els.length) pick(order[els[0].index]); } } });
      }
      function onThemeChange() { if (chart) paintChart(); }
      window.addEventListener('DOMContentLoaded', async () => {
        try { scores = await (await fetch('/api/kernel_scores')).json(); } catch (e) { toast('Không tải được độ chính xác.'); }
        tabs(); info(current); paintChart();
      });
    </script>"""
    return get_base_html("Kernels", content)


@app.get("/game", response_class=HTMLResponse)
def game_page():
    content = """
    <div class="max-w-4xl mx-auto px-4 sm:px-6 py-8 w-full">
      <div class="card p-8 text-center flex flex-col items-center">
        <h2 class="text-2xl font-bold mb-2 accent">Flappy Flower 🌸</h2>
        <p class="text-xs muted mb-6">Nhấn chuột hoặc phím Space để bay lên</p>
        <canvas id="gameCanvas" width="320" height="480" class="rounded-xl cursor-pointer" style="border:4px solid var(--border);background:var(--surface-2)"></canvas>
        <button onclick="startGame()" class="btn btn-dark mt-6">Chơi lại</button>
      </div>
    </div>
    <script>
      const cvs = document.getElementById("gameCanvas"), ctx = cvs.getContext("2d");
      let frames = 0, score = 0, gameState = 0; // 0: chờ, 1: chơi, 2: thua
      const bird = { x:50, y:150, w:30, h:30, gravity:0.25, jump:4.6, speed:0,
        draw() { ctx.font = "30px Arial"; ctx.fillText("🌸", this.x, this.y + this.h); },
        flap() { this.speed = -this.jump; },
        update() { if (gameState === 1) { this.speed += this.gravity; this.y += this.speed; if (this.y + this.h >= cvs.height || this.y <= 0) gameState = 2; } } };
      const pipes = { position:[], w:40, gap:120, dx:2,
        draw() { ctx.fillStyle = "#84cc16"; for (const p of this.position) { ctx.fillRect(p.x, 0, this.w, p.y); ctx.fillRect(p.x, p.y + this.gap, this.w, cvs.height - p.y - this.gap); } },
        update() {
          if (gameState !== 1) return;
          if (frames % 100 === 0) this.position.push({ x:cvs.width, y:Math.max(50, Math.random() * (cvs.height - this.gap - 50)) });
          for (let i = 0; i < this.position.length; i++) {
            const p = this.position[i]; p.x -= this.dx;
            if (bird.x + bird.w > p.x && bird.x < p.x + this.w && (bird.y < p.y || bird.y + bird.h > p.y + this.gap)) gameState = 2;
            if (p.x + this.w === bird.x) score++;
            if (p.x + this.w < 0) { this.position.shift(); i--; }
          } } };
      function draw() {
        ctx.clearRect(0, 0, cvs.width, cvs.height); pipes.draw(); bird.draw();
        ctx.fillStyle = "#fff"; ctx.strokeStyle = "#000"; ctx.lineWidth = 2; ctx.font = "bold 35px monospace";
        ctx.fillText(score, cvs.width/2 - 10, 50); ctx.strokeText(score, cvs.width/2 - 10, 50);
        ctx.font = "20px Arial";
        if (gameState === 0) ctx.fillText("Click để chơi", 100, cvs.height/2);
        if (gameState === 2) { ctx.fillStyle = "red"; ctx.fillText("GAME OVER", 100, cvs.height/2); }
      }
      function loop() { bird.update(); pipes.update(); draw(); frames++; requestAnimationFrame(loop); }
      function startGame() { bird.y = 150; bird.speed = 0; pipes.position = []; score = 0; frames = 0; gameState = 1; }
      const act = () => { if (gameState === 0 || gameState === 2) startGame(); else bird.flap(); };
      window.addEventListener("keydown", e => { if (e.code === "Space") { e.preventDefault(); act(); } });
      cvs.addEventListener("mousedown", act);
      loop();
    </script>"""
    return get_base_html("Flappy Flower", content)
