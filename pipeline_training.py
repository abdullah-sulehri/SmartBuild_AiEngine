"""
train_models.py

Phase 3 of the SmartBuild AI Engine.

1. Cost Predictor         -- XGBoost Regressor on landed_cost (see
                              pipeline_preprocessing.py's COST_TARGET note
                              for why landed_cost, not freight_cost).
                              Random split. MAE, RMSE, R^2, feature
                              importance table, SHAP importance table.
2. Reliability Classifier -- Random Forest, class_weight='balanced'.
                              Precision/recall/F1, confusion matrix table,
                              SHAP importance table.
3. Trend Forecaster       -- one model PER MATERIAL, chronological split,
                              MAPE per material. Prophet first, SARIMAX
                              fallback.

Artifacts are CSV tables, not charts: matplotlib bar charts / SHAP summary
plots hit a RecursionError inside matplotlib.path.Path.__deepcopy__ on a
target environment (Windows, Python 3.14, a recent matplotlib) -- a
matplotlib compatibility issue, not a modeling bug. Every plotting call is
replaced with the equivalent numbers as a CSV instead.
"""

from __future__ import annotations

import logging
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from statsmodels.tsa.statespace.sarimax import SARIMAX

import config
import pipeline_preprocessing as pp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

XGB_PARAMS_COST: dict = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "random_state": config.RANDOM_SEED,
    "n_jobs": -1,
}

RF_PARAMS_SUPPLIER: dict = {
    "n_estimators": 150,
    "max_depth": 5,
    "class_weight": "balanced",
    "random_state": config.RANDOM_SEED,
    "n_jobs": -1,
}

SHAP_SAMPLE_SIZE: int = 500


def train_cost_model() -> None:
    logger.info("--- Training Landed Cost Predictor (Model 1) ---")

    orders_df = pd.read_csv(config.LANDED_COST_CSV)
    splits_df = pd.read_csv(config.DATA_DIR / "landed_cost_split.csv")
    orders_df = orders_df.merge(splits_df, on="order_id")

    preprocessor = joblib.load(config.MODELS_DIR / "preprocessor_cost.joblib")
    features = pp.COST_CATEGORICAL_LOW_CARD + pp.COST_CATEGORICAL_HIGH_CARD + pp.COST_NUMERIC
    target = pp.COST_TARGET  # "landed_cost"

    train_df = orders_df[orders_df["split"] == "train"]
    test_df = orders_df[orders_df["split"] == "test"]

    X_train = np.load(config.DATA_DIR / "X_train_cost.npy")
    y_train = train_df[target].to_numpy()

    if X_train.shape[0] != len(y_train):
        raise ValueError(
            f"X_train_cost.npy has {X_train.shape[0]} rows but the train split "
            f"has {len(y_train)} rows -- pipeline_preprocessing.py and this "
            f"script have drifted out of sync. Re-run pipeline_preprocessing.py."
        )

    X_test = preprocessor.transform(test_df[features])
    y_test = test_df[target].to_numpy()

    model = xgb.XGBRegressor(**XGB_PARAMS_COST)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)
    logger.info(
        "Landed Cost Model Evaluation -> MAE: PKR %.2f | RMSE: PKR %.2f | R^2: %.4f",
        mae, rmse, r2,
    )

    feature_names = preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(config.ARTIFACTS_DIR / "cost_feature_importance.csv", index=False)
    logger.info("Wrote %s", config.ARTIFACTS_DIR / "cost_feature_importance.csv")
    logger.info("Landed Cost Predictor top 5 features:\n%s", importance_df.head(5).to_string(index=False))

    X_shap = X_test[:SHAP_SAMPLE_SIZE]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    shap_importance_df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap_value": np.abs(shap_values).mean(axis=0)}
    ).sort_values("mean_abs_shap_value", ascending=False)
    shap_importance_df.to_csv(config.ARTIFACTS_DIR / "cost_shap_importance.csv", index=False)
    logger.info("Wrote %s", config.ARTIFACTS_DIR / "cost_shap_importance.csv")

    joblib.dump(model, config.MODELS_DIR / "model_cost.joblib")
    logger.info("Saved %s", config.MODELS_DIR / "model_cost.joblib")


def train_supplier_model() -> None:
    logger.info("--- Training Supplier Trust Classifier (Model 2) ---")

    suppliers_raw = pd.read_csv(config.SUPPLIER_TRUST_CSV)
    splits_df = pd.read_csv(config.DATA_DIR / "supplier_trust_split.csv")
    suppliers_df = suppliers_raw.merge(splits_df, on="supplier_id")

    preprocessor = joblib.load(config.MODELS_DIR / "preprocessor_supplier.joblib")
    features = pp.SUPPLIER_NUMERIC_FEATURES
    target = "reliable"

    train_df = suppliers_df[suppliers_df["split"] == "train"]
    test_df = suppliers_df[suppliers_df["split"] == "test"]

    X_train = preprocessor.transform(train_df[features])
    y_train = train_df[target].to_numpy()
    X_test = preprocessor.transform(test_df[features])
    y_test = test_df[target].to_numpy()

    model = RandomForestClassifier(**RF_PARAMS_SUPPLIER)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    report = classification_report(y_test, preds, target_names=["Unreliable", "Reliable"])
    logger.info("Supplier Model Evaluation:\n%s", report)

    cm = confusion_matrix(y_test, preds)
    cm_df = pd.DataFrame(
        cm, index=["actual_unreliable", "actual_reliable"], columns=["predicted_unreliable", "predicted_reliable"]
    )
    cm_df.to_csv(config.ARTIFACTS_DIR / "supplier_confusion_matrix.csv")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_for_reliable = shap_values[1]
    elif shap_values.ndim == 3:
        shap_for_reliable = shap_values[:, :, 1]
    else:
        shap_for_reliable = shap_values
    shap_importance_df = pd.DataFrame(
        {"feature": features, "mean_abs_shap_value": np.abs(shap_for_reliable).mean(axis=0)}
    ).sort_values("mean_abs_shap_value", ascending=False)
    shap_importance_df.to_csv(config.ARTIFACTS_DIR / "supplier_shap_importance.csv", index=False)
    logger.info("Wrote %s", config.ARTIFACTS_DIR / "supplier_shap_importance.csv")

    joblib.dump(model, config.MODELS_DIR / "model_supplier.joblib")
    logger.info("Saved %s", config.MODELS_DIR / "model_supplier.joblib")


def _try_prophet_forecast(train_df: pd.DataFrame, test_len: int) -> tuple[np.ndarray, "Prophet"]:
    from prophet import Prophet

    dfp = train_df.rename(columns={"date": "ds", "price": "y"})[["ds", "y"]]
    m = Prophet()
    m.fit(dfp)
    future = m.make_future_dataframe(periods=test_len, freq="W-MON")
    forecast = m.predict(future)
    return forecast["yhat"].tail(test_len).to_numpy(), m


def _sarimax_forecast(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[np.ndarray, "SARIMAXResultsWrapper"]:
    exog_train = train_df["date"].dt.month.isin(config.MONSOON_MONTHS).astype(int).to_numpy().reshape(-1, 1)
    exog_test = test_df["date"].dt.month.isin(config.MONSOON_MONTHS).astype(int).to_numpy().reshape(-1, 1)

    model = SARIMAX(
        train_df["price"].to_numpy(),
        exog=exog_train,
        order=(1, 1, 1),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit_result = model.fit(disp=False)
    forecast = fit_result.forecast(steps=len(test_df), exog=exog_test)
    return np.asarray(forecast), fit_result


def train_trend_model() -> None:
    logger.info("--- Training Price Trend Forecaster (Model 3, per material) ---")

    features_df = pd.read_csv(config.DATA_DIR / "price_trends_features.csv")
    features_df["date"] = pd.to_datetime(features_df["date"])

    prophet_available = True
    try:
        import prophet  # noqa: F401
    except ImportError:
        prophet_available = False
        logger.info("Prophet not installed -- using SARIMAX for all materials.")

    fitted_models: dict[str, object] = {}
    results: list[dict] = []

    for material in config.MATERIALS:
        mat_df = features_df[features_df["material"] == material].sort_values("date")
        train_df = mat_df[mat_df["split"].isin(["train", "drop"])]
        test_df = mat_df[mat_df["split"] == "test"]

        if prophet_available:
            try:
                preds, prophet_model = _try_prophet_forecast(train_df, len(test_df))
                fitted_models[material] = ("prophet", prophet_model)
            except Exception as exc:
                logger.warning(
                    "Prophet failed on %s (%s) -- falling back to SARIMAX for all remaining materials.",
                    material, exc,
                )
                prophet_available = False
                preds, fit_result = _sarimax_forecast(train_df, test_df)
                fitted_models[material] = ("sarimax", fit_result)
        else:
            preds, fit_result = _sarimax_forecast(train_df, test_df)
            fitted_models[material] = ("sarimax", fit_result)

        actual = test_df["price"].to_numpy()
        mape = mean_absolute_percentage_error(actual, preds) * 100
        results.append({"material": material, "mape_pct": mape, "n_test_weeks": len(test_df)})

    results_df = pd.DataFrame(results).sort_values("mape_pct", ascending=False)
    results_df.to_csv(config.ARTIFACTS_DIR / "trend_mape_by_material.csv", index=False)
    logger.info("Trend Model Evaluation (MAPE %% per material):\n%s", results_df.to_string(index=False))
    logger.info("Trend Model Evaluation -> mean MAPE across materials: %.2f%%", results_df["mape_pct"].mean())

    joblib.dump(fitted_models, config.MODELS_DIR / "model_trend.joblib")
    logger.info("Saved %s", config.MODELS_DIR / "model_trend.joblib")


def main() -> None:
    logger.info("Starting Phase 3: Model Training")
    train_cost_model()
    train_supplier_model()
    train_trend_model()
    logger.info("Phase 3 complete. Models in /models, charts/tables in /artifacts.")


if __name__ == "__main__":
    main()