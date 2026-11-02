"""Đọc mô hình SVM và dữ liệu trực quan hóa; cache để không train mỗi request."""
from functools import lru_cache
import logging
import math
import warnings

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.exceptions import InconsistentVersionWarning

from backend.config import MODEL_PATH

SPECIES_METADATA = {
    0: {"name": "Iris Setosa", "color": "#ea580c"},
    1: {"name": "Iris Versicolor", "color": "#7c3aed"},
    2: {"name": "Iris Virginica", "color": "#0d9488"},
}
CENTROIDS = [[5.006, 3.428, 1.462, 0.246], [5.936, 2.770, 4.260, 1.326], [6.588, 2.974, 5.552, 2.026]]
KERNEL_NAMES = ["linear", "rbf", "poly", "sigmoid"]


@lru_cache(maxsize=1)
def get_state() -> dict:
    iris = load_iris()
    X, y = iris.data, iris.target
    if MODEL_PATH.is_file():
        try:
            with warnings.catch_warnings():
                # Pickle của người dùng được tạo bởi scikit-learn phiên bản khác.
                # Không âm thầm sử dụng model không tương thích.
                warnings.simplefilter("error", InconsistentVersionWarning)
                model = joblib.load(MODEL_PATH)
        except InconsistentVersionWarning:
            logging.warning("Model pickle không cùng phiên bản sklearn; train lại SVM linear từ iris chuẩn")
            model = SVC(kernel="linear").fit(X, y)
        except Exception:
            logging.exception("Không đọc được model đã lưu, sẽ huấn luyện SVC linear")
            model = SVC(kernel="linear").fit(X, y)
    else:
        logging.warning("Chưa có model pickle, sẽ huấn luyện SVC linear")
        model = SVC(kernel="linear").fit(X, y)

    kernel_models, kernel_scores = {}, {}
    for kernel in KERNEL_NAMES:
        kernel_models[kernel] = SVC(kernel=kernel, probability=True, random_state=42).fit(X, y)
        kernel_scores[kernel] = round(float(cross_val_score(SVC(kernel=kernel), X, y, cv=5).mean()) * 100, 1)

    projection = PCA(n_components=3).fit(X)
    points = projection.transform(X)
    pca_data = {
        "points": [{"x": float(a), "y": float(b), "z": float(c), "cls": int(t)} for (a, b, c), t in zip(points, y)],
        "variance": [round(float(v) * 100, 1) for v in projection.explained_variance_ratio_],
    }
    dataset_stats = {
        "features": ["Sepal L", "Sepal W", "Petal L", "Petal W"],
        "species": [{
            "name": name,
            "count": int((y == idx).sum()),
            "mean": [round(float(v), 2) for v in X[y == idx].mean(axis=0)],
            "min": [round(float(v), 1) for v in X[y == idx].min(axis=0)],
            "max": [round(float(v), 1) for v in X[y == idx].max(axis=0)],
        } for idx, name in enumerate(["Setosa", "Versicolor", "Virginica"])],
    }
    return {"model": model, "kernel_models": kernel_models, "kernel_scores": kernel_scores, "pca": pca_data, "stats": dataset_stats}


def predict_iris(feat: list[float]) -> dict:
    model = get_state()["model"]
    predicted = int(model.predict(np.array([feat]))[0])
    dists = [math.sqrt(sum((feat[i] - CENTROIDS[c][i]) ** 2 for i in range(4))) for c in range(3)]
    weights = [1 / (d + 1e-5) for d in dists]
    weights[predicted] *= 1.8
    total = sum(weights)
    confidence = [round((w / total) * 100, 1) for w in weights]
    return {"prediction": SPECIES_METADATA[predicted]["name"], "confidence": confidence[predicted], "confidences": confidence, "features": feat, "color": SPECIES_METADATA[predicted]["color"]}


def compare_kernels(feat: list[float]) -> list[dict]:
    state = get_state()
    rows = []
    for kernel in KERNEL_NAMES:
        probs = state["kernel_models"][kernel].predict_proba(np.array([feat]))[0]
        index = int(probs.argmax())
        rows.append({
            "kernel": kernel,
            "prediction": SPECIES_METADATA[index]["name"],
            "color": SPECIES_METADATA[index]["color"],
            "confidence": round(float(probs[index]) * 100, 1),
            "accuracy": state["kernel_scores"][kernel],
        })
    return rows
