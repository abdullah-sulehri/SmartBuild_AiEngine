"""
app.py
FastAPI server for the SmartBuild AI Engine.

Endpoint names match the original Phase 4 spec exactly:
  POST /api/v1/predict-cost
  POST /api/v1/score-supplier
  GET  /api/v1/forecast-trend/{material}
  POST /api/v1/smart-recommendation
  GET  /health

Plus "-raw" and "-cached" variants (added for wiring into the live
SmartBuild app):
  - "-raw" accepts Prisma-shaped fields directly and maps them via
    adapter.py before calling the same underlying logic -- see
    adapter.py's module docstring.
  - "-cached" accepts a precomputed trust_score instead of rating/
    dispute_count, and reads market_benchmark from an in-process cache
    that the nightly refresh job warms -- see smart_recommendation_cached()
    docstring for why.
  - /api/v1/materials and /api/v1/cities expose the static vocabularies
    (config.MATERIALS, config.CITY_DISTANCES_FROM_LAHORE_KM) so Next.js
    never has to hardcode or duplicate them.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from typing import List, Literal
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import adapter
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)
import subprocess
from datetime import datetime

RETRAIN_SECRET = os.environ.get("RETRAIN_SECRET", "")

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
    rating: float = Field(..., ge=0, le=5)
    dispute_count: int = Field(..., ge=0)


class SupplierBatchItem(BaseModel):
    supplier_id: str
    rating: float = Field(..., ge=0, le=5)
    dispute_count: int = Field(..., ge=0)


class SupplierBatchRequest(BaseModel):
    suppliers: List[SupplierBatchItem] = Field(..., min_length=1)


class Candidate(BaseModel):
    ref: str  # opaque caller-supplied id (e.g. the ad's id), echoed back
              # unchanged in the response so the caller can match results
              # back to its own records even when one supplier has several
              # candidates (e.g. multiple products in the same category) --
              # matching by supplier_id alone collapses those into one.
    supplier_id: str
    company_name: str
    material: str
    quantity: float = Field(..., gt=0)
    base_material_price: float = Field(..., gt=0)
    destination_city: str
    distance_km: float = Field(..., gt=0)
    truck_type: TruckType
    urgency: Urgency
    rating: float = Field(..., ge=0, le=5)
    dispute_count: int = Field(..., ge=0)


class RecommendRequest(BaseModel):
    candidates: List[Candidate] = Field(..., min_length=1)


# --- "-raw" schemas: Prisma-shaped fields, pre-adapter ---
class RawCandidate(BaseModel):
    """One supplier quote as it exists in Prisma, before any AI-Engine mapping.

    Field names match what Next.js can pull directly off Product/
    SupplierProfile/RiderProfile/Order without any lookup logic --
    adapter.py does the material/vehicle/city translation server-side.
    """
    ref: str                    # caller's own id for this offer (e.g. ad.id)
    supplier_id: str
    company_name: str
    category_name: str          # Category.name -- fuzzy-matched to a material
    vehicle_type: str           # RiderProfile.vehicleType -- keyword-matched to a truck_type
    quantity: float = Field(..., gt=0)
    base_material_price: float = Field(..., gt=0)  # Product.price
    delivery_city: str          # Order.deliveryCity -- looked up for distance_km
    urgency: Urgency
    rating: float = Field(..., ge=0, le=5)          # SupplierProfile.rating
    dispute_count: int = Field(..., ge=0)           # count(Dispute) for this supplier


class RawRecommendRequest(BaseModel):
    candidates: List[RawCandidate] = Field(..., min_length=1)


# --- "-cached" schemas: trust_score precomputed nightly, not scored live ---
class CachedCandidate(BaseModel):
    """Same shape as RawCandidate, but rating/dispute_count are replaced by
    trust_score -- precomputed nightly by the RF classifier and stored on
    SupplierProfile, instead of being scored live on every request."""
    ref: str
    supplier_id: str
    company_name: str
    category_name: str
    vehicle_type: str
    quantity: float = Field(..., gt=0)
    base_material_price: float = Field(..., gt=0)
    delivery_city: str
    urgency: Urgency
    trust_score: float = Field(..., ge=0, le=1)


class CachedRecommendRequest(BaseModel):
    candidates: List[CachedCandidate] = Field(..., min_length=1)


def _adapt_candidate(raw: RawCandidate) -> Candidate:
    return Candidate(
        ref=raw.ref,
        supplier_id=raw.supplier_id,
        company_name=raw.company_name,
        material=adapter.map_category_to_material(raw.category_name),
        quantity=raw.quantity,
        base_material_price=raw.base_material_price,
        destination_city=raw.delivery_city,
        distance_km=adapter.get_distance_km(raw.delivery_city),
        truck_type=adapter.map_vehicle_to_truck_type(raw.vehicle_type),
        urgency=raw.urgency,
        rating=raw.rating,
        dispute_count=raw.dispute_count,
    )


def _adapt_cached_candidate(raw: CachedCandidate) -> dict:
    return {
        "ref": raw.ref,
        "supplier_id": raw.supplier_id,
        "company_name": raw.company_name,
        "material": adapter.map_category_to_material(raw.category_name),
        "quantity": raw.quantity,
        "base_material_price": raw.base_material_price,
        "destination_city": raw.delivery_city,
        "distance_km": adapter.get_distance_km(raw.delivery_city),
        "truck_type": adapter.map_vehicle_to_truck_type(raw.vehicle_type),
        "urgency": raw.urgency,
        "trust_score": raw.trust_score,
    }


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


class SupplierBatchResult(BaseModel):
    supplier_id: str
    trust_score: float


class SupplierBatchResponse(BaseModel):
    results: List[SupplierBatchResult]


class TrendResponse(BaseModel):
    material: str
    predicted_next_week_price: float


class RecommendationItem(BaseModel):
    ref: str  # echoes Candidate.ref -- use this to match back to your own records
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
# 5. Health + static vocabulary endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")

@app.post("/api/v1/retrain")
def retrain(request: Request):
    auth = request.headers.get("authorization")
    if auth != f"Bearer {RETRAIN_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    logger.info("Retrain triggered — running preprocessing + training pipeline...")

    result1 = subprocess.run(["python", "pipeline_preprocessing.py"], capture_output=True, text=True)
    if result1.returncode != 0:
        logger.error(f"Preprocessing failed: {result1.stderr}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {result1.stderr}")

    result2 = subprocess.run(["python", "train_models.py"], capture_output=True, text=True)
    if result2.returncode != 0:
        logger.error(f"Training failed: {result2.stderr}")
        raise HTTPException(status_code=500, detail=f"Training failed: {result2.stderr}")

    global model_cost, cost_preprocessor, model_supplier, supplier_preprocessor, model_trend_dict
    model_cost = joblib.load(config.MODELS_DIR / "model_cost.joblib")
    cost_preprocessor = joblib.load(config.MODELS_DIR / "preprocessor_cost.joblib")
    model_supplier = joblib.load(config.MODELS_DIR / "model_supplier.joblib")
    supplier_preprocessor = joblib.load(config.MODELS_DIR / "preprocessor_supplier.joblib")
    model_trend_dict = joblib.load(config.MODELS_DIR / "model_trend.joblib")

    get_cached_market_benchmark.cache_clear()  # old trend predictions are now stale, tied to the old model

    logger.info("Retrain complete — new models loaded into memory.")
    return {"status": "retrained", "timestamp": datetime.now().isoformat()}


@app.get("/api/v1/materials")
def get_materials() -> dict:
    return {"materials": config.MATERIALS}


@app.get("/api/v1/cities")
def get_cities() -> dict:
    return {"cities": list(config.CITY_DISTANCES_FROM_LAHORE_KM.keys())}


# ---------------------------------------------------------------------------
# 6. Cost prediction
# ---------------------------------------------------------------------------
@app.post("/api/v1/predict-cost", response_model=CostResponse)
def predict_cost(req: CostRequest) -> CostResponse:
    df = pd.DataFrame([req.model_dump()])
    X = cost_preprocessor.transform(df)
    predicted_landed_cost = float(model_cost.predict(X)[0])

    material_subtotal = req.base_material_price * req.quantity

    return CostResponse(
        supplier_id=req.supplier_id,
        material_subtotal=round(material_subtotal, 2),
        predicted_landed_cost=round(predicted_landed_cost, 2),
    )


# ---------------------------------------------------------------------------
# 7. Supplier trust scoring (single + batch)
# ---------------------------------------------------------------------------
@app.post("/api/v1/score-supplier", response_model=TrustResponse)
def score_supplier(req: TrustRequest) -> TrustResponse:
    df = pd.DataFrame([req.model_dump()])
    X = supplier_preprocessor.transform(df)

    prob = float(model_supplier.predict_proba(X)[0][1])
    is_reliable = bool(model_supplier.predict(X)[0])

    return TrustResponse(supplier_id=req.supplier_id, trust_probability=round(prob, 4), is_reliable=is_reliable)


@app.post("/api/v1/score-suppliers-batch", response_model=SupplierBatchResponse)
def score_suppliers_batch(req: SupplierBatchRequest) -> SupplierBatchResponse:
    """Vectorized scoring for the nightly cache-refresh job.

    Scoring N suppliers one at a time means N HTTP round trips plus N
    separate preprocessor.transform() calls. This batches them into a
    single DataFrame -> transform -> predict_proba, the same pattern
    _rank_candidates() already uses.
    """
    df = pd.DataFrame([s.model_dump() for s in req.suppliers])
    X = supplier_preprocessor.transform(df)
    probs = model_supplier.predict_proba(X)[:, 1]

    results = [
        SupplierBatchResult(supplier_id=s.supplier_id, trust_score=round(float(p), 4))
        for s, p in zip(req.suppliers, probs)
    ]
    return SupplierBatchResponse(results=results)


# ---------------------------------------------------------------------------
# 8. Trend forecasting + in-process cache
# ---------------------------------------------------------------------------
@app.get("/api/v1/forecast-trend/{material}", response_model=TrendResponse)
def forecast_trend(material: str) -> TrendResponse:
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


@lru_cache(maxsize=32)
def get_cached_market_benchmark(material: str) -> float:
    """Caches the heavy Prophet/SARIMAX prediction to avoid per-request lag.

    In normal operation this cache is warmed once a day by the nightly
    refresh job hitting /api/v1/forecast-trend for every material -- so a
    live request almost never triggers an actual Prophet/SARIMAX call, only
    a dict lookup. Only the first call per material after a process restart
    pays the real cost.
    """
    try:
        res = forecast_trend(material)
        return res.predicted_next_week_price
    except Exception as e:
        logger.error(f"Trend forecast failed for {material}: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# 9. Smart recommendation (live, raw, and cached variants)
# ---------------------------------------------------------------------------
def _rank_candidates(candidates: List[Candidate]) -> RecommendResponse:
    if not candidates:
        return RecommendResponse(recommendations=[])

    material = candidates[0].material
    market_benchmark = get_cached_market_benchmark(material)

    # 1. BATCH PROCESSING: Single DataFrame conversion
    cand_dicts = [cand.model_dump() for cand in candidates]

    trust_df = pd.DataFrame([
        {"supplier_id": c["supplier_id"], "rating": c["rating"], "dispute_count": c["dispute_count"]}
        for c in cand_dicts
    ])
    cost_df = pd.DataFrame(cand_dicts)

    # 2. VECTORIZED INFERENCE: Predict all rows at once
    X_trust = supplier_preprocessor.transform(trust_df)
    trust_probs = model_supplier.predict_proba(X_trust)[:, 1]

    X_cost = cost_preprocessor.transform(cost_df)
    landed_costs = model_cost.predict(X_cost)

    results = []
    w_cost = config.RECOMMENDATION_WEIGHTS["w1_cost"]
    w_trust = config.RECOMMENDATION_WEIGHTS["w2_trust"]
    w_trend = config.RECOMMENDATION_WEIGHTS["w3_trend"]

    # 3. Calculate Scores
    for i, cand in enumerate(candidates):
        landed_cost = float(landed_costs[i])
        trust_score = float(trust_probs[i])

        benchmark = market_benchmark if market_benchmark > 0 else cand.base_material_price

        expected_market_cost = (benchmark * cand.quantity) + (landed_cost - cand.base_material_price * cand.quantity)
        cost_score = min(expected_market_cost / landed_cost, 1.0) if landed_cost > 0 else 0.0

        price_ratio = cand.base_material_price / benchmark
        if price_ratio > 1.0:
            trend_score = 1.0 / price_ratio
        elif price_ratio < 0.80:
            trend_score = price_ratio
        else:
            trend_score = 1.0

        results.append({
            "ref": cand.ref,
            "supplier_id": cand.supplier_id,
            "company_name": cand.company_name,
            "landed_cost": round(landed_cost, 2),
            "trust_score": round(trust_score, 4),
            "trend_score": round(trend_score, 4),
            "cost_score": round(cost_score, 4),
        })

    for r in results:
        r["composite_score"] = round(
            (w_cost * r["cost_score"]) + (w_trust * r["trust_score"]) + (w_trend * r["trend_score"]), 4
        )

    results.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, r in enumerate(results):
        r["smart_rank"] = i + 1

    return RecommendResponse(recommendations=[RecommendationItem(**r) for r in results])


@app.post("/api/v1/smart-recommendation", response_model=RecommendResponse)
def smart_recommendation(req: RecommendRequest) -> RecommendResponse:
    """Rank competing suppliers using a composite cost/trust/trend score."""
    return _rank_candidates(req.candidates)


@app.post("/api/v1/smart-recommendation-raw", response_model=RecommendResponse)
def smart_recommendation_raw(req: RawRecommendRequest) -> RecommendResponse:
    """Same as /smart-recommendation, but accepts Prisma-shaped fields.

    Call this directly from Next.js with raw category_name/vehicle_type/
    delivery_city values -- adapter.py maps them to the model's training
    vocabulary here, so Next.js never needs to know config.MATERIALS,
    TRUCK_TYPES, or the city-distance table.
    """
    try:
        candidates = [_adapt_candidate(c) for c in req.candidates]
    except ValueError as e:
        # Bad/unmappable input data (unknown category or city) -> 422, not
        # a 500 -- this is a client data problem, not a server bug.
        raise HTTPException(status_code=422, detail=str(e))
    return _rank_candidates(candidates)


@app.post("/api/v1/smart-recommendation-cached", response_model=RecommendResponse)
def smart_recommendation_cached(req: CachedRecommendRequest) -> RecommendResponse:
    """Same composite ranking as /smart-recommendation, but skips the two
    lookups that don't need to happen per-request anymore:

      - trust_score comes straight from the caller (SupplierProfile.trustScore,
        refreshed nightly by /api/v1/score-suppliers-batch) instead of a live
        model_supplier.predict_proba() call here.
      - market_benchmark still goes through get_cached_market_benchmark(), but
        the nightly job already warms this lru_cache by hitting
        /api/v1/forecast-trend for every material once a day -- so in normal
        operation this is a dict lookup, not a live Prophet call. Only the
        very first request after a process restart pays the Prophet cost.

    Only the XGBoost landed-cost prediction runs live, vectorized in one
    batch across all candidates -- everything else is O(1) lookups.
    """
    try:
        candidates = [_adapt_cached_candidate(c) for c in req.candidates]
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    material = candidates[0]["material"]
    market_benchmark = get_cached_market_benchmark(material)
    if market_benchmark <= 0:
        market_benchmark = candidates[0]["base_material_price"]

    cost_df = pd.DataFrame(candidates)
    X_cost = cost_preprocessor.transform(cost_df)
    landed_costs = model_cost.predict(X_cost)

    w_cost = config.RECOMMENDATION_WEIGHTS["w1_cost"]
    w_trust = config.RECOMMENDATION_WEIGHTS["w2_trust"]
    w_trend = config.RECOMMENDATION_WEIGHTS["w3_trend"]

    results = []
    for i, c in enumerate(candidates):
        landed_cost = float(landed_costs[i])
        trust_score = float(c["trust_score"])

        expected_market_cost = (
            (market_benchmark * c["quantity"])
            + (landed_cost - c["base_material_price"] * c["quantity"])
        )
        cost_score = min(expected_market_cost / landed_cost, 1.0) if landed_cost > 0 else 0.0

        price_ratio = c["base_material_price"] / market_benchmark
        if price_ratio > 1.0:
            trend_score = 1.0 / price_ratio
        elif price_ratio < 0.80:
            trend_score = price_ratio
        else:
            trend_score = 1.0

        results.append({
            "ref": c["ref"],
            "supplier_id": c["supplier_id"],
            "company_name": c["company_name"],
            "landed_cost": round(landed_cost, 2),
            "trust_score": round(trust_score, 4),
            "trend_score": round(trend_score, 4),
            "cost_score": round(cost_score, 4),
        })

    for r in results:
        r["composite_score"] = round(
            (w_cost * r["cost_score"]) + (w_trust * r["trust_score"]) + (w_trend * r["trend_score"]), 4
        )

    results.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, r in enumerate(results):
        r["smart_rank"] = i + 1

    return RecommendResponse(recommendations=[RecommendationItem(**r) for r in results])