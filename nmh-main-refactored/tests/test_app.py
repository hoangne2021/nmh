"""Smoke/integration test end-to-end với SQLite tạm."""
import os
import tempfile
from pathlib import Path

# Đặt biến môi trường trước khi import backend.*
TEST_DIR = Path(tempfile.mkdtemp(prefix="iris_svm_test_"))
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DIR / 'test.db'}"
os.environ["SECRET_KEY"] = "integration-test-secret-with-sufficient-entropy-only"

from fastapi.testclient import TestClient  # noqa: E402
from backend.main import app  # noqa: E402


def test_all_routes_and_auth():
    with TestClient(app) as client:
        for page in ["/", "/login", "/dataset", "/lab", "/vector3d", "/kernels"]:
            response = client.get(page)
            assert response.status_code == 200
            assert "<html" in response.text.lower()
            assert "/static/css/" in response.text
            assert "<style>" not in response.text
        assert client.get("/static/css/base.css").status_code == 200
        assert client.get("/static/js/login.js").status_code == 200
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/no-such-page").status_code == 404
        assert client.get("/api/history").status_code == 401
        assert client.post("/predict", json={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}).status_code == 401

        p = {"username": "user1", "password": "secret-password-987"}
        reg = client.post("/auth/register", json=p)
        assert reg.status_code == 201
        recovery_code = reg.json()["recovery_code"]
        assert len(recovery_code) >= 20
        assert client.post("/auth/register", json=p).status_code == 409
        assert client.post("/auth/login", json={"username": "user1", "password": "wrong"}).status_code == 401
        login = client.post("/auth/login", json=p)
        assert login.status_code == 200
        headers = {"Authorization": "Bearer " + login.json()["token"]}
        assert client.get("/auth/me", headers=headers).json()["username"] == "user1"
        pred_input = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
        result = client.post("/predict", json=pred_input, headers=headers)
        assert result.status_code == 200
        assert result.json()["prediction"] == "Iris Setosa"
        assert len(result.json()["confidences"]) == 3
        history = client.get("/api/history", headers=headers)
        assert history.status_code == 200 and len(history.json()) == 1
        # User 2 không nhìn thấy lịch sử user1, kể cả API history
        other = {"username": "user2", "password": "other-secret-123"}
        assert client.post("/auth/register", json=other).status_code == 201
        other_token = client.post("/auth/login", json=other).json()["token"]
        other_headers = {"Authorization": "Bearer " + other_token}
        assert client.get("/api/history", headers=other_headers).json() == []
        assert client.delete("/api/history", headers=other_headers).status_code == 200
        assert len(client.get("/api/history", headers=headers).json()) == 1
        assert client.get("/api/dataset", headers=headers).json()["features"][0] == "Sepal L"
        assert len(client.get("/api/pca", headers=headers).json()["points"]) == 150
        assert set(client.get("/api/kernel_scores", headers=headers).json()) == {"linear", "rbf", "poly", "sigmoid"}
        assert len(client.post("/api/compare_kernels", json=pred_input, headers=headers).json()) == 4
        assert client.delete("/api/history", headers=headers).status_code == 200
        assert client.get("/api/history", headers=headers).json() == []
        assert client.post("/predict", json={**pred_input, "sepal_length": -1}, headers=headers).status_code == 422

        # Mã không đúng không thể đổi password, kể cả biết phép cộng.
        bad = {"username": "user1", "recovery_code": "not-the-correct-code-123", "new_password": "new-password-123"}
        assert client.post("/auth/forgot-password", json=bad).status_code == 401
        ok = client.post("/auth/forgot-password", json={**bad, "recovery_code": recovery_code})
        assert ok.status_code == 200
        assert ok.json()["recovery_code"] != recovery_code
        assert client.get("/auth/me", headers=headers).status_code == 401  # JWT bị thu hồi
        assert client.post("/auth/login", json=p).status_code == 401
        assert client.post("/auth/login", json={**p, "password": "new-password-123"}).status_code == 200
        assert client.post("/auth/forgot-password", json={**bad, "recovery_code": recovery_code}).status_code == 401
