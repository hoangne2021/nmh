from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "svm_model.pkl")

model = joblib.load(model_path)   # 👈 THÊM DÒNG NÀY

# Tạo FastAPI
app = FastAPI(
    title="Iris Classification API",
    description="SVM model for the Iris dataset",
    version="1.0.0",
)

# Input data
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Tên các loài hoa
species = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}


# Trang chủ
@app.get("/")
def home():
    return {"message": "Iris SVM API is running"}


# Health check
@app.get("/health")
def health():
    return {"status": "healthy"}


# Dự đoán
@app.post("/predict")
def predict(data: IrisInput):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]

    prediction = int(model.predict(features)[0])

    return {
        "class_id": prediction,
        "prediction": species[prediction],
    }