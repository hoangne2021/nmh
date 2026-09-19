from sklearn import datasets
from sklearn.svm import SVC
import os
import joblib

iris = datasets.load_iris()
X = iris.data
y = iris.target

model = SVC(kernel="linear")
model.fit(X, y)

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "svm_model.pkl")

joblib.dump(model, model_path)

print("Đã lưu model tại:", model_path)