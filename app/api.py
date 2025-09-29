# app/api.py (resumen esencial)
from fastapi import FastAPI, HTTPException
from joblib import load
from pathlib import Path
import numpy as np
from app.schemas import PredictMultiLagRequest, PredictMultiLagResponse, HealthMultiLagResponse

app = FastAPI(title="BTC Volatility API (multi-lag)")
AVAILABLE_LAGS = (7,14,21,28)
MODELS = {}

for lag in AVAILABLE_LAGS:
    p = Path(f"app/model_lag{lag}.joblib")
    if p.exists():
        art = load(p)
        if isinstance(art, dict) and all(k in art for k in ("model","scaler_x","scaler_y","config")):
            MODELS[lag] = art

if not MODELS:
    raise RuntimeError("No hay artifacts model_lag*.joblib en app/")

@app.get("/")
def root():
    return {"msg":"BTC Volatility API (multi-lag)","endpoints":["/health (GET)","/predict (POST)"],"docs":"/docs"}

@app.get("/health", response_model=HealthMultiLagResponse)
def health():
    info = {lag: {"horizon": MODELS[lag]["config"].get("horizon", 7)} for lag in MODELS}
    return HealthMultiLagResponse(status="ok", available=info)

@app.post("/predict", response_model=PredictMultiLagResponse)
def predict(req: PredictMultiLagRequest):
    if req.lag not in MODELS:
        raise HTTPException(status_code=400, detail=f"Lag {req.lag} no disponible. {sorted(MODELS.keys())}")
    art = MODELS[req.lag]
    model, sx, sy, cfg = art["model"], art["scaler_x"], art["scaler_y"], art["config"]
    L = int(cfg.get("lags", req.lag)); H = int(cfg.get("horizon", 7))
    if len(req.window) != L:
        raise HTTPException(status_code=400, detail=f"Se esperaban {L} valores para window, recibidos {len(req.window)}")
    x = np.asarray(req.window, dtype=float).reshape(1, -1)
    yhat = sy.inverse_transform(model.predict(sx.transform(x)))
    return PredictMultiLagResponse(used_lag=L, horizon=H, yhat=yhat.ravel().astype(float).tolist())
