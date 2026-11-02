"""Các API phân loại hoa, dữ liệu, biểu đồ và lịch sử người dùng."""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import History, User
from backend.schemas import IrisInput
from backend.security import get_current_user
from backend.services.iris_ml import compare_kernels, get_state, predict_iris

router = APIRouter(tags=["Iris Classifier"])


@router.post("/predict")
def predict(payload: IrisInput, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    feat = [payload.sepal_length, payload.sepal_width, payload.petal_length, payload.petal_width]
    result = predict_iris(feat)
    db.add(History(
        username=user.username,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        sl=feat[0], sw=feat[1], pl=feat[2], pw=feat[3],
        species=result["prediction"],
        confidence=result["confidence"],
    ))
    db.commit()
    return result


@router.get("/api/history")
def history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(History).where(History.username == user.username).order_by(History.id.desc()).limit(50)).all()
    return [{
        "timestamp": r.timestamp,
        "sl": r.sl, "sw": r.sw, "pl": r.pl, "pw": r.pw,
        "species": r.species, "confidence": r.confidence,
    } for r in rows]


@router.delete("/api/history")
def clear_history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.execute(delete(History).where(History.username == user.username))
    db.commit()
    return {"status": "success"}


@router.post("/api/compare_kernels")
def compare(payload: IrisInput, _user: User = Depends(get_current_user)):
    return compare_kernels([payload.sepal_length, payload.sepal_width, payload.petal_length, payload.petal_width])


@router.get("/api/dataset")
def dataset_stats(_user: User = Depends(get_current_user)):
    return get_state()["stats"]


@router.get("/api/pca")
def pca_points(_user: User = Depends(get_current_user)):
    return get_state()["pca"]


@router.get("/api/kernel_scores")
def kernel_scores(_user: User = Depends(get_current_user)):
    return get_state()["kernel_scores"]
