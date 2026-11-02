"""Train SVM Linear và lưu vào model/iris_svm.pkl, dùng: python train.py"""
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.svm import SVC

out = Path(__file__).resolve().parent / "model" / "svm_model.pkl"
out.parent.mkdir(parents=True, exist_ok=True)
iris = load_iris()
model = SVC(kernel="linear").fit(iris.data, iris.target)
joblib.dump(model, out)
print("Đã lưu model tại:", out)
