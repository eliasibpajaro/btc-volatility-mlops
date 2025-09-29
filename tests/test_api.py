# tests/test_api.py
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert isinstance(data["available"], dict)
    assert len(data["available"]) >= 1  # al menos un lag

def test_predict_ok_with_first_available_lag():
    avail = client.get("/health").json()["available"]
    # toma el primer lag disponible (p. ej., 7)
    lag = int(sorted(map(int, avail.keys()))[0])
    horizon = int(avail[str(lag)]["horizon"])
    # ventana dummy del tamaño correcto
    window = [0.01] * lag

    r = client.post("/predict", json={"lag": lag, "window": window})
    assert r.status_code == 200
    out = r.json()
    assert out["used_lag"] == lag
    assert isinstance(out["yhat"], list)
    assert len(out["yhat"]) == horizon

def test_predict_wrong_window_length():
    avail = client.get("/health").json()["available"]
    lag = int(sorted(map(int, avail.keys()))[0])
    window = [0.01] * max(1, lag - 1)  # tamaño incorrecto

    r = client.post("/predict", json={"lag": lag, "window": window})
    assert r.status_code == 400
    assert "Se esperaban" in r.json()["detail"]

def test_predict_lag_not_available():
    avail = client.get("/health").json()["available"]
    bad_lag = 999
    assert str(bad_lag) not in avail
    r = client.post("/predict", json={"lag": bad_lag, "window": [0.01]*7})
    assert r.status_code == 400
    assert "no disponible" in r.json()["detail"]

def test_predict_422_invalid_body():
    # falta "window"
    r = client.post("/predict", json={"lag": 7})
    assert r.status_code == 422
