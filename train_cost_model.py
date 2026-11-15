"""
train_cost_model.py

Standalone training module for Model 1: XGBoost Landed Cost Predictor.
Can be run independently (`python train_cost_model.py`) or via pipeline_training.py.
"""

from __future__ import annotations

import logging
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

SHAP_SAMPLE_SIZE: int = 500


def train_cost_model() -> None:
    logger.info("--- Training Landed Cost Predictor (Model 1: XGBoost) ---")

    orders_df = pd.read_csv(config.LANDED_COST_CSV)
    splits_df = pd.read_csv(config.DATA_DIR / "landed_cost_split.csv")
    orders_df = orders_df.merge(splits_df, on="order_id")

    preprocessor = joblib.load(config.MODELS_DIR / "preprocessor_cost.joblib")
    features = pp.COST_CATEGORICAL_LOW_CARD + pp.COST_CATEGORICAL_HIGH_CARD + pp.COST_NUMERIC
    target = pp.COST_TARGET

    train_df = orders_df[orders_df["split"] == "train"]
    test_df = orders_df[orders_df["split"] == "test"]

    X_train = np.load(config.DATA_DIR / "X_train_cost.npy")
    y_train = train_df[target].to_numpy()

    if X_train.shape[0] != len(y_train):
        raise ValueError(
            f"X_train_cost.npy has {X_train.shape[0]} rows but the train split "
            f"has {len(y_train)} rows. Re-run pipeline_preprocessing.py."
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


if __name__ == "__main__":
    train_cost_model()