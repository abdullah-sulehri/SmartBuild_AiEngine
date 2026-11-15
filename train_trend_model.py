"""
train_trend_model.py

Standalone training module for Model 3: Multivariate Prophet Material Price Forecaster.
Uses per-category ProphetProfiles, Pakistan construction holiday windows, and
exogenous macroeconomic regressors (diesel_price_pkr, usd_pkr) calibrated to
PBS October 2026 & SSRN 4432245.
"""

from __future__ import annotations

import json
import logging
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def _fit_configured_prophet(
    df_in: pd.DataFrame, material: str
) -> "Prophet":
    from prophet import Prophet

    profile = config.get_prophet_profile(material)
    holidays_df = pd.DataFrame(config.PAKISTAN_CONSTRUCTION_EVENTS)

    dfp = df_in.rename(columns={"date": "ds", "price": "y"}).copy()
    dfp["ds"] = pd.to_datetime(dfp["ds"])

    m = Prophet(
        changepoint_prior_scale=profile.changepoint_prior_scale,
        seasonality_prior_scale=profile.seasonality_prior_scale,
        seasonality_mode=profile.seasonality_mode,
        yearly_seasonality=profile.fourier_order_yearly,
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=holidays_df,
    )

    if profile.use_diesel_regressor and "diesel_price_pkr" in dfp.columns:
        m.add_regressor("diesel_price_pkr", prior_scale=0.5, mode="multiplicative")
    if profile.use_fx_regressor and "usd_pkr" in dfp.columns:
        m.add_regressor("usd_pkr", prior_scale=0.5, mode="multiplicative")

    m.fit(dfp)
    return m


def _try_prophet_eval_and_refit(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    material: str,
) -> tuple[np.ndarray, "Prophet", float]:
    """1. Fit on train_df & predict test_df for honest out-of-sample MAPE.
    2. Refit on full_df & compute 1-step-ahead next_week_price with macro regressors.
    """
    profile = config.get_prophet_profile(material)

    # Step 1: Out-of-sample evaluation on held-out test_df
    eval_model = _fit_configured_prophet(train_df, material)
    eval_future = test_df.rename(columns={"date": "ds"}).copy()
    eval_future["ds"] = pd.to_datetime(eval_future["ds"])
    eval_forecast = eval_model.predict(eval_future)
    test_preds = eval_forecast["yhat"].to_numpy()

    # Step 2: Refit on full history (up to Oct 2026) for live production serving
    prod_model = _fit_configured_prophet(full_df, material)
    prod_future = prod_model.make_future_dataframe(periods=1, freq="W-MON")

    # Populate macro regressors on future dataframe (carry forward latest macro values)
    if profile.use_diesel_regressor and "diesel_price_pkr" in full_df.columns:
        last_diesel = float(full_df["diesel_price_pkr"].iloc[-1])
        prod_future["diesel_price_pkr"] = last_diesel
        prod_future.loc[: len(full_df) - 1, "diesel_price_pkr"] = full_df["diesel_price_pkr"].to_numpy()

    if profile.use_fx_regressor and "usd_pkr" in full_df.columns:
        last_fx = float(full_df["usd_pkr"].iloc[-1])
        prod_future["usd_pkr"] = last_fx
        prod_future.loc[: len(full_df) - 1, "usd_pkr"] = full_df["usd_pkr"].to_numpy()

    prod_forecast = prod_model.predict(prod_future)
    next_week_price = round(float(prod_forecast["yhat"].iloc[-1]), 2)

    # Attach latest regressor defaults onto the model object so app.py can also run live .predict()
    prod_model._last_diesel_pkr = float(full_df["diesel_price_pkr"].iloc[-1]) if "diesel_price_pkr" in full_df.columns else 396.20
    prod_model._last_usd_pkr = float(full_df["usd_pkr"].iloc[-1]) if "usd_pkr" in full_df.columns else 278.50
    prod_model._precomputed_next_week_price = next_week_price

    return test_preds, prod_model, next_week_price


def _sarimax_eval_and_refit(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> tuple[np.ndarray, "SARIMAXResultsWrapper", float]:
    def _build_exog(df_sub: pd.DataFrame) -> np.ndarray:
        monsoon = df_sub["date"].dt.month.isin(config.MONSOON_MONTHS).astype(float)
        if "diesel_price_pkr" in df_sub.columns and "usd_pkr" in df_sub.columns:
            return np.column_stack([
                monsoon.to_numpy(),
                df_sub["diesel_price_pkr"].to_numpy() / 300.0,
                df_sub["usd_pkr"].to_numpy() / 250.0,
            ])
        return monsoon.to_numpy().reshape(-1, 1)

    exog_train = _build_exog(train_df)
    exog_test = _build_exog(test_df)

    eval_model = SARIMAX(
        train_df["price"].to_numpy(),
        exog=exog_train,
        order=(1, 1, 1),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    eval_fit = eval_model.fit(disp=False)
    test_preds = np.asarray(eval_fit.forecast(steps=len(test_df), exog=exog_test))

    # Refit on full history for production serving
    exog_full = _build_exog(full_df)
    prod_model = SARIMAX(
        full_df["price"].to_numpy(),
        exog=exog_full,
        order=(1, 1, 1),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    prod_fit = prod_model.fit(disp=False)
    next_exog = exog_full[-1:]
    next_week_price = round(float(np.asarray(prod_fit.forecast(steps=1, exog=next_exog))[0]), 2)
    prod_fit._precomputed_next_week_price = next_week_price

    return test_preds, prod_fit, next_week_price


def train_trend_model() -> None:
    logger.info("--- Training Price Trend Forecaster (Model 3: Multivariate Prophet per Material) ---")

    features_df = pd.read_csv(config.DATA_DIR / "price_trends_features.csv")
    features_df["date"] = pd.to_datetime(features_df["date"])

    prophet_available = True
    try:
        import prophet  # noqa: F401
    except ImportError:
        prophet_available = False
        logger.info("Prophet not installed -- using SARIMAX with macro exog for all materials.")

    fitted_models: dict[str, object] = {}
    next_week_cache: dict[str, float] = {}
    results: list[dict] = []

    for material in config.MATERIALS:
        mat_df = features_df[features_df["material"] == material].sort_values("date").reset_index(drop=True)
        train_df = mat_df[mat_df["split"].isin(["train", "drop"])].reset_index(drop=True)
        test_df = mat_df[mat_df["split"] == "test"].reset_index(drop=True)

        if prophet_available:
            try:
                preds, prophet_model, next_wk = _try_prophet_eval_and_refit(
                    train_df, test_df, mat_df, material
                )
                fitted_models[material] = ("prophet", prophet_model)
                next_week_cache[material] = next_wk
            except Exception as exc:
                logger.warning(
                    "Prophet failed on %s (%s) -- falling back to SARIMAX.",
                    material, exc,
                )
                prophet_available = False
                preds, fit_result, next_wk = _sarimax_eval_and_refit(train_df, test_df, mat_df)
                fitted_models[material] = ("sarimax", fit_result)
                next_week_cache[material] = next_wk
        else:
            preds, fit_result, next_wk = _sarimax_eval_and_refit(train_df, test_df, mat_df)
            fitted_models[material] = ("sarimax", fit_result)
            next_week_cache[material] = next_wk

        actual = test_df["price"].to_numpy()
        mape = mean_absolute_percentage_error(actual, preds) * 100
        r2 = r2_score(actual, preds)
        results.append(
            {
                "material": material,
                "category": config.MATERIAL_CATEGORY.get(material, "default"),
                "oct2026_pbs_anchor_pkr": float(mat_df["price"].iloc[-1]),
                "predicted_next_week_pkr": next_wk,
                "mape_pct": round(float(mape), 3),
                "r2_score": round(float(r2), 4),
                "n_test_weeks": len(test_df),
            }
        )

    results_df = pd.DataFrame(results).sort_values("mape_pct", ascending=False)
    results_df.to_csv(config.ARTIFACTS_DIR / "trend_mape_by_material.csv", index=False)
    logger.info("Trend Model Evaluation (MAPE %% per material):\n%s", results_df.to_string(index=False))
    logger.info(
        "Trend Model Evaluation -> Mean Out-of-Sample MAPE: %.2f%% | Mean R^2: %.4f",
        results_df["mape_pct"].mean(),
        results_df["r2_score"].mean(),
    )

    joblib.dump(fitted_models, config.MODELS_DIR / "model_trend.joblib")
    logger.info("Saved %s", config.MODELS_DIR / "model_trend.joblib")

    cache_path = config.MODELS_DIR / "trend_next_week_cache.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(next_week_cache, f, indent=2)
    logger.info("Saved precomputed next-week forecast cache to %s", cache_path)


if __name__ == "__main__":
    train_trend_model()