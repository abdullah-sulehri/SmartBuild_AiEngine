

# """
# pipeline_preprocessing.py

# Phase 2 of the SmartBuild AI Engine.

# COST_TARGET is "landed_cost" (NOT "freight_cost" -- see the note in
# build_cost_preprocessor() below for why freight_cost was a dead end).
# """

# from __future__ import annotations

# import logging

# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

# import config

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
# )
# logger = logging.getLogger(__name__)

# # ---------------------------------------------------------------------------
# # Trend feature engineering
# # ---------------------------------------------------------------------------
# LAG_WEEKS: tuple[int, ...] = (1, 4, 12)
# ROLLING_WINDOWS: tuple[int, ...] = (4, 12)
# TREND_TEST_FRACTION: float = 0.2

# # ---------------------------------------------------------------------------
# # Cost predictor feature set
# # ---------------------------------------------------------------------------
# COST_CATEGORICAL_LOW_CARD: list[str] = ["material", "truck_type", "urgency", "destination_city"]
# COST_CATEGORICAL_HIGH_CARD: list[str] = ["supplier_id"]
# COST_NUMERIC: list[str] = ["quantity", "distance_km", "base_material_price"]

# # COST_TARGET = "landed_cost", not "freight_cost". freight_cost is an EXACT
# # deterministic function of just truck_type and distance_km in this
# # synthetic data (freight_cost = TRUCK_BASE_FEE[truck_type] +
# # TRUCK_PER_KM_RATE[truck_type] * distance_km, verified against
# # generate_datasets.py -- max residual across 2000 test rows was PKR 6,
# # which is exactly round(x, 2) noise, not real error). A model trained on
# # freight_cost would just be re-deriving two numbers already hardcoded in
# # config.py, giving a suspiciously perfect R^2 with no real predictive
# # work being done, and 6 of the 8 input features would carry zero signal.
# # landed_cost is a genuinely harder target: it depends on
# # quantity * base_material_price (a multiplicative interaction trees only
# # approximate, never reproduce exactly) plus the urgency markup plus
# # freight -- real composite structure, and predicting it directly means
# # the urgency markup is learned as part of the model instead of needing a
# # separate, easy-to-forget addition at serving time.
# COST_TARGET: str = "landed_cost"
# COST_TEST_FRACTION: float = 0.2

# # ---------------------------------------------------------------------------
# # Supplier reliability feature set
# # ---------------------------------------------------------------------------
# SUPPLIER_NUMERIC_FEATURES: list[str] = ["dispute_count", "rating"]
# SUPPLIER_TEST_FRACTION: float = 0.2


# def engineer_trend_features(price_df: pd.DataFrame) -> pd.DataFrame:
#     df = price_df.copy()
#     df["date"] = pd.to_datetime(df["date"])
#     df = df.sort_values(["material", "date"]).reset_index(drop=True)

#     for k in LAG_WEEKS:
#         df[f"lag_{k}w"] = df.groupby("material")["price"].shift(k)

#     for w in ROLLING_WINDOWS:
#         df[f"rolling_mean_{w}w"] = df.groupby("material")["price"].transform(
#             lambda s, w=w: s.shift(1).rolling(window=w, min_periods=w).mean()
#         )

#     max_window = max(ROLLING_WINDOWS)
#     n_full_history = df[f"rolling_mean_{max_window}w"].notna().sum()
#     logger.info(
#         "Engineered trend features: %d rows total, %d with full %d-week history",
#         len(df), n_full_history, max_window,
#     )
#     return df


# def split_trend_data(trend_features_df: pd.DataFrame) -> pd.DataFrame:
#     df = trend_features_df.copy()
#     max_window = max(ROLLING_WINDOWS)
#     has_full_history = df[f"rolling_mean_{max_window}w"].notna()

#     unique_dates = np.sort(df.loc[has_full_history, "date"].unique())
#     cutoff_idx = int(len(unique_dates) * (1 - TREND_TEST_FRACTION))
#     cutoff_date = unique_dates[cutoff_idx]

#     df["split"] = "drop"
#     df.loc[has_full_history & (df["date"] < cutoff_date), "split"] = "train"
#     df.loc[has_full_history & (df["date"] >= cutoff_date), "split"] = "test"

#     counts = df["split"].value_counts().to_dict()
#     logger.info(
#         "Trend chronological split at %s: %s (test = most recent %.0f%% of weeks)",
#         pd.Timestamp(cutoff_date).date(), counts, TREND_TEST_FRACTION * 100,
#     )
#     return df


# def build_trend_preprocessor() -> StandardScaler:
#     return StandardScaler()


# def split_cost_data(orders_df: pd.DataFrame) -> pd.DataFrame:
#     df = orders_df.copy()
#     train_idx, test_idx = train_test_split(
#         df.index, test_size=COST_TEST_FRACTION, random_state=config.RANDOM_SEED,
#     )
#     df["split"] = "train"
#     df.loc[test_idx, "split"] = "test"
#     logger.info(
#         "Cost data random split: %d train / %d test",
#         (df["split"] == "train").sum(), (df["split"] == "test").sum(),
#     )
#     return df


# def build_cost_preprocessor() -> ColumnTransformer:
#     """Build (unfit) the ColumnTransformer for the landed-cost predictor's inputs.

#     material_cost, freight_cost, and urgency_markup themselves are NOT
#     included as features even though they're present in landed_cost_sim.csv
#     -- they are deterministic arithmetic components of landed_cost
#     (landed_cost = material_cost + freight_cost + urgency_markup), so
#     including any of them would make "prediction" a trivial re-summation.
#     The model instead learns landed_cost from the underlying drivers:
#     material, quantity, base_material_price, truck_type, distance_km,
#     urgency, destination_city, supplier_id.

#     TargetEncoder cross-fitting note: TargetEncoder only performs its
#     internal cross-fitting inside fit_transform(). A later, separate
#     transform() call on the SAME rows the encoder was fit on does NOT
#     cross-fit and would leak each row's own target back into its encoding.
#     main() below calls fit_transform() exactly once on the train split and
#     persists the result (data/X_train_cost.npy); training code loads that
#     array directly rather than calling transform() on the train split
#     again. transform() remains correct for the held-out test split.

#     OneHotEncoder uses sparse_output=False so np.save()/np.load() round-trip
#     the combined matrix as a plain dense ndarray instead of silently
#     pickling a scipy sparse matrix into a 0-d object array (which then
#     needs allow_pickle=True to load).
#     """
#     return ColumnTransformer(
#         transformers=[
#             ("cat_low", OneHotEncoder(handle_unknown="ignore", sparse_output=False), COST_CATEGORICAL_LOW_CARD),
#             ("cat_high", TargetEncoder(random_state=config.RANDOM_SEED), COST_CATEGORICAL_HIGH_CARD),
#             ("num", StandardScaler(), COST_NUMERIC),
#         ],
#     )


# def derive_supplier_label(suppliers_df: pd.DataFrame) -> pd.DataFrame:
#     df = suppliers_df.copy()
#     df["reliable"] = (
#         df["on_time_delivery_rate"] >= config.RELIABLE_ON_TIME_THRESHOLD
#     ).astype(int)
#     rate = df["reliable"].mean()
#     logger.info(
#         "Derived supplier `reliable` label at threshold %.2f: %.1f%% positive class",
#         config.RELIABLE_ON_TIME_THRESHOLD, rate * 100,
#     )
#     return df


# def split_supplier_data(suppliers_df: pd.DataFrame) -> pd.DataFrame:
#     df = suppliers_df.copy()
#     train_idx, test_idx = train_test_split(
#         df.index,
#         test_size=SUPPLIER_TEST_FRACTION,
#         random_state=config.RANDOM_SEED,
#         stratify=df["reliable"],
#     )
#     df["split"] = "train"
#     df.loc[test_idx, "split"] = "test"
#     logger.info(
#         "Supplier data stratified split: %d train / %d test",
#         (df["split"] == "train").sum(), (df["split"] == "test").sum(),
#     )
#     return df


# def build_supplier_preprocessor() -> ColumnTransformer:
#     return ColumnTransformer(
#         transformers=[
#             ("num", StandardScaler(), SUPPLIER_NUMERIC_FEATURES),
#         ],
#     )


# def main() -> None:
#     price_df = pd.read_csv(config.PRICE_TRENDS_CSV)
#     orders_df = pd.read_csv(config.LANDED_COST_CSV)
#     suppliers_df = pd.read_csv(config.SUPPLIER_TRUST_CSV)

#     trend_features_df = engineer_trend_features(price_df)
#     trend_features_df = split_trend_data(trend_features_df)
#     trend_features_df.to_csv(config.DATA_DIR / "price_trends_features.csv", index=False)
#     logger.info("Wrote %s", config.DATA_DIR / "price_trends_features.csv")

#     trend_train = trend_features_df[trend_features_df["split"] == "train"]
#     trend_lag_roll_cols = [f"lag_{k}w" for k in LAG_WEEKS] + [
#         f"rolling_mean_{w}w" for w in ROLLING_WINDOWS
#     ]
#     trend_scaler = build_trend_preprocessor()
#     trend_scaler.fit(trend_train[trend_lag_roll_cols])
#     joblib.dump(trend_scaler, config.MODELS_DIR / "preprocessor_trend.joblib")
#     logger.info("Saved %s", config.MODELS_DIR / "preprocessor_trend.joblib")

#     orders_df = split_cost_data(orders_df)
#     orders_df[["order_id", "split"]].to_csv(
#         config.DATA_DIR / "landed_cost_split.csv", index=False
#     )
#     logger.info("Wrote %s", config.DATA_DIR / "landed_cost_split.csv")

#     cost_train = orders_df[orders_df["split"] == "train"]
#     cost_preprocessor = build_cost_preprocessor()

#     X_train_cost = cost_preprocessor.fit_transform(
#         cost_train[COST_CATEGORICAL_LOW_CARD + COST_CATEGORICAL_HIGH_CARD + COST_NUMERIC],
#         cost_train[COST_TARGET],
#     )
#     np.save(config.DATA_DIR / "X_train_cost.npy", X_train_cost)
#     logger.info("Wrote %s (leakage-safe cost train features)", config.DATA_DIR / "X_train_cost.npy")

#     joblib.dump(cost_preprocessor, config.MODELS_DIR / "preprocessor_cost.joblib")
#     logger.info("Saved %s", config.MODELS_DIR / "preprocessor_cost.joblib")

#     suppliers_df = derive_supplier_label(suppliers_df)
#     suppliers_df = split_supplier_data(suppliers_df)
#     suppliers_df[["supplier_id", "reliable", "split"]].to_csv(
#         config.DATA_DIR / "supplier_trust_split.csv", index=False
#     )
#     logger.info("Wrote %s", config.DATA_DIR / "supplier_trust_split.csv")

#     supplier_train = suppliers_df[suppliers_df["split"] == "train"]
#     supplier_preprocessor = build_supplier_preprocessor()
#     supplier_preprocessor.fit(supplier_train[SUPPLIER_NUMERIC_FEATURES])
#     joblib.dump(supplier_preprocessor, config.MODELS_DIR / "preprocessor_supplier.joblib")
#     logger.info("Saved %s", config.MODELS_DIR / "preprocessor_supplier.joblib")


# if __name__ == "__main__":
#     main()

"""
pipeline_preprocessing.py

Phase 2 of the SmartBuild AI Engine.
Preserves macroeconomic regressor columns (diesel_price_pkr, usd_pkr) in
price_trends_features.csv so train_trend_model.py can feed them to Prophet.
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trend feature engineering
# ---------------------------------------------------------------------------
LAG_WEEKS: tuple[int, ...] = (1, 4, 12)
ROLLING_WINDOWS: tuple[int, ...] = (4, 12)
TREND_TEST_FRACTION: float = 0.2

# ---------------------------------------------------------------------------
# Cost predictor feature set
# ---------------------------------------------------------------------------
COST_CATEGORICAL_LOW_CARD: list[str] = ["material", "truck_type", "urgency", "destination_city"]
COST_CATEGORICAL_HIGH_CARD: list[str] = ["supplier_id"]
COST_NUMERIC: list[str] = ["quantity", "distance_km", "base_material_price"]
COST_TARGET: str = "landed_cost"
COST_TEST_FRACTION: float = 0.2

# ---------------------------------------------------------------------------
# Supplier reliability feature set
# ---------------------------------------------------------------------------
SUPPLIER_NUMERIC_FEATURES: list[str] = ["dispute_count", "rating"]
SUPPLIER_TEST_FRACTION: float = 0.2


def engineer_trend_features(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["material", "date"]).reset_index(drop=True)

    # Ensure macro regressor columns exist even if an older CSV is loaded
    if "diesel_price_pkr" not in df.columns:
        df["diesel_price_pkr"] = 396.20
    if "usd_pkr" not in df.columns:
        df["usd_pkr"] = 278.50

    for k in LAG_WEEKS:
        df[f"lag_{k}w"] = df.groupby("material")["price"].shift(k)

    for w in ROLLING_WINDOWS:
        df[f"rolling_mean_{w}w"] = df.groupby("material")["price"].transform(
            lambda s, w=w: s.shift(1).rolling(window=w, min_periods=w).mean()
        )

    max_window = max(ROLLING_WINDOWS)
    n_full_history = df[f"rolling_mean_{max_window}w"].notna().sum()
    logger.info(
        "Engineered trend features: %d rows total, %d with full %d-week history",
        len(df), n_full_history, max_window,
    )
    return df


def split_trend_data(trend_features_df: pd.DataFrame) -> pd.DataFrame:
    df = trend_features_df.copy()
    max_window = max(ROLLING_WINDOWS)
    has_full_history = df[f"rolling_mean_{max_window}w"].notna()

    unique_dates = np.sort(df.loc[has_full_history, "date"].unique())
    cutoff_idx = int(len(unique_dates) * (1 - TREND_TEST_FRACTION))
    cutoff_date = unique_dates[cutoff_idx]

    df["split"] = "drop"
    df.loc[has_full_history & (df["date"] < cutoff_date), "split"] = "train"
    df.loc[has_full_history & (df["date"] >= cutoff_date), "split"] = "test"

    counts = df["split"].value_counts().to_dict()
    logger.info(
        "Trend chronological split at %s: %s (test = most recent %.0f%% of weeks)",
        pd.Timestamp(cutoff_date).date(), counts, TREND_TEST_FRACTION * 100,
    )
    return df


def build_trend_preprocessor() -> StandardScaler:
    return StandardScaler()


def split_cost_data(orders_df: pd.DataFrame) -> pd.DataFrame:
    df = orders_df.copy()
    train_idx, test_idx = train_test_split(
        df.index, test_size=COST_TEST_FRACTION, random_state=config.RANDOM_SEED,
    )
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    logger.info(
        "Cost data random split: %d train / %d test",
        (df["split"] == "train").sum(), (df["split"] == "test").sum(),
    )
    return df


def build_cost_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat_low", OneHotEncoder(handle_unknown="ignore", sparse_output=False), COST_CATEGORICAL_LOW_CARD),
            ("cat_high", TargetEncoder(random_state=config.RANDOM_SEED), COST_CATEGORICAL_HIGH_CARD),
            ("num", StandardScaler(), COST_NUMERIC),
        ],
    )


def derive_supplier_label(suppliers_df: pd.DataFrame) -> pd.DataFrame:
    df = suppliers_df.copy()
    df["reliable"] = (
        df["on_time_delivery_rate"] >= config.RELIABLE_ON_TIME_THRESHOLD
    ).astype(int)
    rate = df["reliable"].mean()
    logger.info(
        "Derived supplier `reliable` label at threshold %.2f: %.1f%% positive class",
        config.RELIABLE_ON_TIME_THRESHOLD, rate * 100,
    )
    return df


def split_supplier_data(suppliers_df: pd.DataFrame) -> pd.DataFrame:
    df = suppliers_df.copy()
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=SUPPLIER_TEST_FRACTION,
        random_state=config.RANDOM_SEED,
        stratify=df["reliable"],
    )
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    logger.info(
        "Supplier data stratified split: %d train / %d test",
        (df["split"] == "train").sum(), (df["split"] == "test").sum(),
    )
    return df


def build_supplier_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), SUPPLIER_NUMERIC_FEATURES),
        ],
    )


def main() -> None:
    price_df = pd.read_csv(config.PRICE_TRENDS_CSV)
    orders_df = pd.read_csv(config.LANDED_COST_CSV)
    suppliers_df = pd.read_csv(config.SUPPLIER_TRUST_CSV)

    trend_features_df = engineer_trend_features(price_df)
    trend_features_df = split_trend_data(trend_features_df)
    trend_features_df.to_csv(config.DATA_DIR / "price_trends_features.csv", index=False)
    logger.info("Wrote %s", config.DATA_DIR / "price_trends_features.csv")

    trend_train = trend_features_df[trend_features_df["split"] == "train"]
    trend_lag_roll_cols = [f"lag_{k}w" for k in LAG_WEEKS] + [
        f"rolling_mean_{w}w" for w in ROLLING_WINDOWS
    ]
    trend_scaler = build_trend_preprocessor()
    trend_scaler.fit(trend_train[trend_lag_roll_cols])
    joblib.dump(trend_scaler, config.MODELS_DIR / "preprocessor_trend.joblib")
    logger.info("Saved %s", config.MODELS_DIR / "preprocessor_trend.joblib")

    orders_df = split_cost_data(orders_df)
    orders_df[["order_id", "split"]].to_csv(
        config.DATA_DIR / "landed_cost_split.csv", index=False
    )
    logger.info("Wrote %s", config.DATA_DIR / "landed_cost_split.csv")

    cost_train = orders_df[orders_df["split"] == "train"]
    cost_preprocessor = build_cost_preprocessor()

    X_train_cost = cost_preprocessor.fit_transform(
        cost_train[COST_CATEGORICAL_LOW_CARD + COST_CATEGORICAL_HIGH_CARD + COST_NUMERIC],
        cost_train[COST_TARGET],
    )
    np.save(config.DATA_DIR / "X_train_cost.npy", X_train_cost)
    logger.info("Wrote %s (leakage-safe cost train features)", config.DATA_DIR / "X_train_cost.npy")

    joblib.dump(cost_preprocessor, config.MODELS_DIR / "preprocessor_cost.joblib")
    logger.info("Saved %s", config.MODELS_DIR / "preprocessor_cost.joblib")

    suppliers_df = derive_supplier_label(suppliers_df)
    suppliers_df = split_supplier_data(suppliers_df)
    suppliers_df[["supplier_id", "reliable", "split"]].to_csv(
        config.DATA_DIR / "supplier_trust_split.csv", index=False
    )
    logger.info("Wrote %s", config.DATA_DIR / "supplier_trust_split.csv")

    supplier_train = suppliers_df[suppliers_df["split"] == "train"]
    supplier_preprocessor = build_supplier_preprocessor()
    supplier_preprocessor.fit(supplier_train[SUPPLIER_NUMERIC_FEATURES])
    joblib.dump(supplier_preprocessor, config.MODELS_DIR / "preprocessor_supplier.joblib")
    logger.info("Saved %s", config.MODELS_DIR / "preprocessor_supplier.joblib")


if __name__ == "__main__":
    main()