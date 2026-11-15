"""
train_supplier_model.py

Standalone training module for Model 2: Random Forest Supplier Reliability Classifier.
Can be run independently (`python train_supplier_model.py`) or via pipeline_training.py.
"""

from __future__ import annotations

import logging
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import config
import pipeline_preprocessing as pp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

RF_PARAMS_SUPPLIER: dict = {
    "n_estimators": 150,
    "max_depth": 5,
    "class_weight": "balanced",
    "random_state": config.RANDOM_SEED,
    "n_jobs": -1,
}


def train_supplier_model() -> None:
    logger.info("--- Training Supplier Trust Classifier (Model 2: Random Forest) ---")

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
        cm,
        index=["actual_unreliable", "actual_reliable"],
        columns=["predicted_unreliable", "predicted_reliable"],
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


if __name__ == "__main__":
    train_supplier_model()