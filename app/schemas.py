# app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict

class PredictMultiLagRequest(BaseModel):
    lag: int = Field(..., description="7, 14, 21 o 28")
    window: List[float] = Field(..., description="Últimos 'lag' valores del objetivo")

class PredictMultiLagResponse(BaseModel):
    used_lag: int
    horizon: int
    yhat: List[float]

class HealthMultiLagResponse(BaseModel):
    status: str
    available: Dict[int, Dict[str, int]]
