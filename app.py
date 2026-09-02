"""
app.py
FastAPI server for the SmartBuild AI Engine.

Endpoint names match the original Phase 4 spec exactly:
  POST /api/v1/predict-cost
  POST /api/v1/score-supplier
  GET  /api/v1/forecast-trend/{material}
  POST /api/v1/smart-recommendation
  GET  /health
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import List, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Load models & preprocessors ONCE at startup, not per-request
# ---------------------------------------------------------------------------
try:
    logger.info("Loading ML models and preprocessors...")
    model_cost = joblib.load(config.MODELS_DIR / "model_cost.joblib")
    cost_preprocessor = joblib.load(config.MODELS_DIR / "preprocessor_cost.joblib")

    model_supplier = joblib.load(config.MODELS_DIR / "model_supplier.joblib")
    supplier_preprocessor = joblib.load(config.MODELS_DIR / "preprocessor_supplier.joblib")

    model_trend_dict = joblib.load(config.MODELS_DIR / "model_trend.joblib")
    logger.info("All models loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load models. Did you run pipeline_preprocessing.py and train_models.py? Error: {e}")
    raise

# ---------------------------------------------------------------------------
# 2. FastAPI setup
# ---------------------------------------------------------------------------
app = FastAPI(title="SmartBuild AI Recommendation Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Basic request logging middleware: method, path, status, duration."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %d (%.1fms)", request.method, request.url.path, response.status_code, duration_ms)
    return response


TruckType = Literal["mini", "6-wheeler", "10-wheeler"]
Urgency = Literal["standard", "express"]

# ---------------------------------------------------------------------------
# 3. Request schemas (with validation -- quantity/price/distance must be > 0)
# ---------------------------------------------------------------------------
class CostRequest(BaseModel):
    supplier_id: str
    material: str
    quantity: float = Field(..., gt=0)
    base_material_price: float = Field(..., gt=0)
    destination_city: str
    distance_km: float = Field(..., gt=0)
    truck_type: TruckType
    urgency: Urgency


class TrustRequest(BaseModel):
    supplier_id: str
    rating: float = Field(..., ge=1, le=5)
    dispute_count: int = Field(..., ge=0)


class Candidate(BaseModel):
    supplier_id: str
    company_name: str
    material: str
    quantity: float = Field(..., gt=0)
    base_material_price: float = Field(..., gt=0)
    destination_city: str
    distance_km: float = Field(..., gt=0)
    truck_type: TruckType
    urgency: Urgency
    rating: float = Field(..., ge=1, le=5)
    dispute_count: int = Field(..., ge=0)


class RecommendRequest(BaseModel):
    candidates: List[Candidate] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# 4. Response schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str


class CostResponse(BaseModel):
    supplier_id: str
    material_subtotal: float
    predicted_landed_cost: float


class TrustResponse(BaseModel):
    supplier_id: str
    trust_probability: float
    is_reliable: bool


class TrendResponse(BaseModel):
    material: str
    predicted_next_week_price: float


class RecommendationItem(BaseModel):
    supplier_id: str
    company_name: str
    landed_cost: float
    trust_score: float
    trend_score: float
    cost_score: float
    composite_score: float
    smart_rank: int


class RecommendResponse(BaseModel):
    recommendations: List[RecommendationItem]


# ---------------------------------------------------------------------------
# 5. Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness check. If this responds, all three models loaded
    successfully at startup (a load failure raises and prevents the app
    from starting at all, per section 1 above)."""
    return HealthResponse(status="ok")


@app.get("/api/v1/forecast-trend/{material}", response_model=TrendResponse)
def forecast_trend(material: str) -> TrendResponse:
    """Predict next week's baseline market price for a material."""
    if material not in model_trend_dict:
        raise HTTPException(status_code=404, detail=f"No trend model for material '{material}'")

    model_type, model_obj = model_trend_dict[material]

    try:
        if model_type == "prophet":
            future = model_obj.make_future_dataframe(periods=1, freq="W-MON")
            forecast = model_obj.predict(future)
            next_price = float(forecast["yhat"].iloc[-1])
        else:
            next_date = datetime.now() + timedelta(days=7)
            is_monsoon = 1 if next_date.month in config.MONSOON_MONTHS else 0
            forecast = model_obj.forecast(steps=1, exog=[[is_monsoon]])
            next_price = float(forecast.iloc[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return TrendResponse(material=material, predicted_next_week_price=round(next_price, 2))


@app.post("/api/v1/score-supplier", response_model=TrustResponse)
def score_supplier(req: TrustRequest) -> TrustResponse:
    """Predict probability of a supplier delivering reliably."""
    df = pd.DataFrame([req.model_dump()])
    X = supplier_preprocessor.transform(df)

    prob = float(model_supplier.predict_proba(X)[0][1])
    is_reliable = bool(model_supplier.predict(X)[0])

    return TrustResponse(supplier_id=req.supplier_id, trust_probability=round(prob, 4), is_reliable=is_reliable)


@app.post("/api/v1/predict-cost", response_model=CostResponse)
def predict_cost(req: CostRequest) -> CostResponse:
    """Predict total landed cost directly (material + freight + urgency
    markup all learned jointly by the model, not assembled by hand here --
    see pipeline_preprocessing.py's COST_TARGET note for why)."""
    df = pd.DataFrame([req.model_dump()])
    X = cost_preprocessor.transform(df)
    predicted_landed_cost = float(model_cost.predict(X)[0])

    material_subtotal = req.base_material_price * req.quantity

    return CostResponse(
        supplier_id=req.supplier_id,
        material_subtotal=round(material_subtotal, 2),
        predicted_landed_cost=round(predicted_landed_cost, 2),
    )


@app.post("/api/v1/smart-recommendation", response_model=RecommendResponse)
def smart_recommendation(req: RecommendRequest) -> RecommendResponse:
    """Rank competing suppliers using a composite cost/trust/trend score."""
    material = req.candidates[0].material
    trend_res = forecast_trend(material)
    market_benchmark = trend_res.predicted_next_week_price

    results = []
    for cand in req.candidates:
        trust_df = pd.DataFrame([{"supplier_id": cand.supplier_id, "rating": cand.rating, "dispute_count": cand.dispute_count}])
        X_trust = supplier_preprocessor.transform(trust_df)
        trust_score = float(model_supplier.predict_proba(X_trust)[0][1])

        cost_df = pd.DataFrame([cand.model_dump()])
        X_cost = cost_preprocessor.transform(cost_df)
        landed_cost = float(model_cost.predict(X_cost)[0])

        expected_market_cost = (market_benchmark * cand.quantity) + (landed_cost - cand.base_material_price * cand.quantity)
        cost_score = min(expected_market_cost / landed_cost, 1.0) if landed_cost > 0 else 0.0

        price_ratio = cand.base_material_price / market_benchmark
        if price_ratio > 1.0:
            trend_score = 1.0 / price_ratio
        elif price_ratio < 0.80:
            trend_score = price_ratio
        else:
            trend_score = 1.0

        results.append({
            "supplier_id": cand.supplier_id,
            "company_name": cand.company_name,
            "landed_cost": round(landed_cost, 2),
            "trust_score": round(trust_score, 4),
            "trend_score": round(trend_score, 4),
            "cost_score": round(cost_score, 4),
        })

    w_cost = config.RECOMMENDATION_WEIGHTS["w1_cost"]
    w_trust = config.RECOMMENDATION_WEIGHTS["w2_trust"]
    w_trend = config.RECOMMENDATION_WEIGHTS["w3_trend"]

    for r in results:
        r["composite_score"] = round(
            (w_cost * r["cost_score"]) + (w_trust * r["trust_score"]) + (w_trend * r["trend_score"]), 4
        )

    results.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, r in enumerate(results):
        r["smart_rank"] = i + 1

    return RecommendResponse(recommendations=[RecommendationItem(**r) for r in results])