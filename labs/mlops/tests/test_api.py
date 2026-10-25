from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/healthz").status_code == 200


def test_ready():
    assert client.get("/readyz").status_code == 200


def test_predict():
    r = client.post("/predict", json={"age": 70, "systolic_bp": 140, "creatinine": 1.0, "prior_admissions": 1})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["risk"] <= 1
    assert body["decision"] in {"routine", "review"}


def test_invalid_input():
    r = client.post("/predict", json={"age": 500, "systolic_bp": 140, "creatinine": 1.0, "prior_admissions": 1})
    assert r.status_code == 422


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "biomed_predictions_total" in r.text
