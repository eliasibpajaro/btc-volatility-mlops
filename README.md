<<<<<<< HEAD
=======
# BTC Volatility MLOps

Pipeline completo para estimar **volatilidad de BTC** con:
- EDA y features (retornos log y volatilidad histórica)
- Validación cruzada *time-aware* (GroupKFold temporal con `tsxv`)
- MLP multisalida (horizonte H=7)
- Diagnóstico de residuos (BDS test)
- API de inferencia con FastAPI (multi-lag)
- Docker y tests (pytest)

## Estructura

btc-volatility-mlops/
├── data/
│ └── btc_1d_data_2018_to_2025.csv
├── notebooks/
│ ├── 1_eda_volatility.ipynb
│ ├── 2_model_training.ipynb
│ └── 3_residual_analysis.ipynb
├── app/
│ ├── init.py
│ ├── api.py
│ ├── schemas.py
│ ├── model.joblib # (opcional: single-lag)
│ ├── model_lag7.joblib # artifacts multi-lag
│ ├── model_lag14.joblib
│ ├── model_lag21.joblib
│ └── model_lag28.joblib
├── tests/
│ ├── test_api.py
│ └── test_model.py
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .github/workflows/
│ └── ci.yml
└── README.md

422 Unprocessable Entity: body no coincide con schema. Usa claves lag y window y tipos numéricos. La longitud de window debe ser lag.

400 Bad Request: longitud de window ≠ lag.

/health no muestra lags: faltan model_lag*.joblib dentro de app/.

Docker no arranca en Windows: abrir Docker Desktop (WSL2 enabled). Probar docker run hello-world
>>>>>>> ed749fe (bootstrap: estructura del proyecto + CI)

