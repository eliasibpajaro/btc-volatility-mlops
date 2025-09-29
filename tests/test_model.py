# tests/test_model.py
from pathlib import Path
from joblib import load
import numpy as np
import math

APP_DIR = Path(__file__).resolve().parents[1] / "app"

def available_artifacts():
    return sorted(APP_DIR.glob("model_lag*.joblib"))

def test_artifacts_exist():
    files = available_artifacts()
    assert len(files) > 0, "No se encontró ningún artifact app/model_lag*.joblib"

def test_artifact_structure_and_prediction_shape():
    for f in available_artifacts():
        art = load(f)
        for k in ("model","scaler_x","scaler_y","config"):
            assert k in art, f"Falta '{k}' en {f.name}"
        L = int(art["config"]["lags"])
        H = int(art["config"].get("horizon", 7))

        x = np.random.rand(1, L)
        x_scaled = art["scaler_x"].transform(x)
        y_scaled = art["model"].predict(x_scaled)
        y = art["scaler_y"].inverse_transform(y_scaled)

        assert y.shape == (1, H), f"Esperado (1,{H}), fue {y.shape} en {f.name}"
        assert all(math.isfinite(float(v)) for v in y.ravel()), f"NaN/Inf en {f.name}"
