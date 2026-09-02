# """
# pipeline_preprocessing.py

# Phase 2 of the SmartBuild AI Engine: turns the Phase 1 synthetic CSVs into
# model-ready feature sets, and persists every fitted transformer with
# joblib so training (Phase 3) and serving (Phase 4) load the IDENTICAL
# fitted object rather than each re-fitting "the same" transformer from
# scratch. This matters because a transformer re-fit at serving time on
# whatever data happens to be available then is a classic train/serve skew
# bug: category sets, feature ordering, or scaler statistics can all drift
# silently even when the code looks unchanged. Loading one serialized object
# everywhere removes that failure mode by construction.

# Two more leakage-discipline decisions live here, deliberately, not by
# accident:

# 1. Every transformer is fit on a TRAIN split only, and that split is
#    persisted alongside the transformer (as a CSV of IDs + a "split"
#    column). Phase 3 loads the split file rather than re-splitting itself,
#    so the rows the fitted transformer was validated against are always
#    the exact rows it will be evaluated against — a different random or
#    chronological split in Phase 3 would silently invalidate the fitted
#    encoder/scaler.
# 2. The supplier reliability label (`reliable`) is derived FROM
#    on_time_delivery_rate, so on_time_delivery_rate is deliberately
#    EXCLUDED from the classifier's input features. Including it would make
#    the "classifier" a trivial threshold check on its own label rather
#    than a model learning from independent signals (dispute_count, rating)
#    — a leakage bug that's easy to miss precisely because the column
#    looks like an ordinary numeric feature.

# Produces:
# - data/price_trends_features.csv      (lag + rolling window features, split col)
# - data/landed_cost_split.csv          (order_id -> train/test)
# - data/supplier_trust_split.csv       (supplier_id -> train/test, + label)
# - models/preprocessor_cost.joblib     (ColumnTransformer, cost predictor)
# - models/preprocessor_supplier.joblib (ColumnTransformer, reliability clf)
# - models/preprocessor_trend.joblib    (StandardScaler, trend forecaster features)
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
# TREND_TEST_FRACTION: float = 0.2  # most recent 20% of weeks held out

# # ---------------------------------------------------------------------------
# # Cost predictor feature set
# # ---------------------------------------------------------------------------
# COST_CATEGORICAL_LOW_CARD: list[str] = ["material", "truck_type", "urgency", "destination_city"]
# COST_CATEGORICAL_HIGH_CARD: list[str] = ["supplier_id"]
# COST_NUMERIC: list[str] = ["quantity", "distance_km", "base_material_price"]
# COST_TARGET: str = "landed_cost"
# COST_TEST_FRACTION: float = 0.2  # orders aren't sequential -> random split is fine here

# # ---------------------------------------------------------------------------
# # Supplier reliability feature set
# # ---------------------------------------------------------------------------
# # on_time_delivery_rate is intentionally NOT a feature -- see module
# # docstring point (2). Only dispute_count and rating are used as inputs.
# SUPPLIER_NUMERIC_FEATURES: list[str] = ["dispute_count", "rating"]
# SUPPLIER_TEST_FRACTION: float = 0.2


# def engineer_trend_features(price_df: pd.DataFrame) -> pd.DataFrame:
#     """Add lag and rolling-average features to the weekly price trend data.

#     Every lag/rolling value at row t is computed strictly from rows < t
#     within the same material's series (rolling windows are built on a
#     shift(1)'d series, never a centered or trailing-inclusive one), so no
#     feature at time t can see t's own price or any later price. This is
#     the leakage discipline that has to hold before any train/test split is
#     even applied -- a leaky feature is leaky regardless of which rows end
#     up on which side of the split.

#     Args:
#         price_df: Output of generate_price_trends() -- columns date,
#             material, price, unit, data_source.

#     Returns:
#         Copy of price_df, sorted by (material, date), with added columns
#         lag_{k}w for k in LAG_WEEKS and rolling_mean_{w}w for w in
#         ROLLING_WINDOWS. Early rows in each material's series lack enough
#         history and contain NaN -- callers must drop or impute these
#         explicitly rather than have them silently filled here.
#     """
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
#     """Assign a chronological train/test split to the trend feature set.

#     This is the single highest-risk leakage point in the whole pipeline
#     (per Phase 3 ground rules) -- a RANDOM split here would let the model
#     train on weeks that come chronologically after some of its own test
#     weeks, which is invisible in an aggregate error metric but is exactly
#     the first thing a technically literate examiner checks. The cutoff is
#     a single global date (all materials share the same weekly calendar),
#     picked so the most recent TREND_TEST_FRACTION of the date range is
#     held out for every material identically.

#     Args:
#         trend_features_df: Output of engineer_trend_features().

#     Returns:
#         Copy of trend_features_df with an added "split" column ("train" /
#         "test"). Rows lacking full lag/rolling history are marked "drop" --
#         Phase 3 should exclude these rather than impute them, since an
#         imputed lag feature is a fabricated signal, not a missing one.
#     """
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
#     """Build (unfit) the scaler for trend lag/rolling numeric features."""
#     return StandardScaler()


# def split_cost_data(orders_df: pd.DataFrame) -> pd.DataFrame:
#     """Assign a random train/test split to the landed-cost orders.

#     Orders are independent B2B transactions, not a single ordered
#     sequence the way weekly prices are -- per the Phase 3 ground rules, a
#     standard random split is the appropriate (non-leaky) choice here,
#     unlike the trend data above.

#     Returns:
#         Copy of orders_df with an added "split" column ("train" / "test").
#     """
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
#     """Build (unfit) the ColumnTransformer for the cost predictor's inputs.

#     Low-cardinality categoricals (material, truck_type, urgency,
#     destination_city) get OneHotEncoder. supplier_id is high-cardinality
#     (hundreds of distinct suppliers) so a one-hot would blow up the
#     feature space with mostly-empty columns -- TargetEncoder is used
#     instead.

#     IMPORTANT (leakage point, fixed at call-site in main()): TargetEncoder
#     only performs its internal cross-fitting -- the mechanism that keeps
#     a row's encoding from leaking that row's own target back into itself
#     -- inside fit_transform(). A later, separate transform() call on the
#     SAME rows the encoder was fit on does NOT cross-fit; it just applies
#     the whole-train category means, so the encoding for a training row
#     would include that row's own landed_cost. main() below calls this
#     preprocessor's fit_transform() exactly once, on the cost train split,
#     and persists the resulting matrix (data/X_train_cost.npy) so
#     downstream training code never needs to call transform() on the train
#     split a second time; transform() remains correct (no leak) when
#     applied to the held-out test split, since the encoder was never fit
#     on those rows.

#     OneHotEncoder is built with sparse_output=False so the
#     ColumnTransformer's combined output is a plain dense ndarray -- this
#     also means np.save()/np.load() round-trip it as an ordinary array
#     rather than pickling a scipy sparse matrix into a 0-d object array
#     (which then needs allow_pickle=True to load back). The feature set
#     here is small enough that dense output has no meaningful memory cost.

#     Numeric features are scaled with StandardScaler so XGBoost's
#     regularization terms (and any linear baseline compared against it)
#     treat PKR-scale and km-scale inputs comparably.

#     Note what's deliberately excluded: material_cost, freight_cost, and
#     urgency_markup are not included as features even though they're
#     present in landed_cost_sim.csv, because they are deterministic
#     arithmetic components of the landed_cost target itself
#     (landed_cost = material_cost + freight_cost + urgency_markup) --
#     including them would make "prediction" a trivial re-summation rather
#     than a genuine regression problem.

#     Returns:
#         Unfit ColumnTransformer. Caller fits it on the TRAIN split only.
#     """
#     return ColumnTransformer(
#         transformers=[
#             ("cat_low", OneHotEncoder(handle_unknown="ignore", sparse_output=False), COST_CATEGORICAL_LOW_CARD),
#             ("cat_high", TargetEncoder(random_state=config.RANDOM_SEED), COST_CATEGORICAL_HIGH_CARD),
#             ("num", StandardScaler(), COST_NUMERIC),
#         ],
#     )


# def derive_supplier_label(suppliers_df: pd.DataFrame) -> pd.DataFrame:
#     """Derive the binary `reliable` label from on_time_delivery_rate.

#     Returns:
#         Copy of suppliers_df with an added `reliable` column (1 if
#         on_time_delivery_rate >= config.RELIABLE_ON_TIME_THRESHOLD, else 0).
#     """
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
#     """Assign a stratified random train/test split to supplier profiles.

#     Stratified on the `reliable` label so both splits keep roughly the
#     same (likely imbalanced -- most suppliers reliable, few bad) class
#     proportion; Phase 3 still needs to check and handle that imbalance
#     explicitly (e.g. class_weight='balanced'), this just prevents the
#     split itself from making the imbalance worse by chance.

#     Returns:
#         Copy of suppliers_df (with `reliable` label already present) plus
#         an added "split" column ("train" / "test").
#     """
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
#     """Build (unfit) the ColumnTransformer for the reliability classifier's inputs.

#     Only dispute_count and rating are included -- see module docstring
#     point (2) for why on_time_delivery_rate must not appear here.

#     Returns:
#         Unfit ColumnTransformer. Caller fits it on the TRAIN split only.
#     """
#     return ColumnTransformer(
#         transformers=[
#             ("num", StandardScaler(), SUPPLIER_NUMERIC_FEATURES),
#         ],
#     )


# def main() -> None:
#     """Run Phase 2 end-to-end: engineer features, split, fit, and persist."""
#     price_df = pd.read_csv(config.PRICE_TRENDS_CSV)
#     orders_df = pd.read_csv(config.LANDED_COST_CSV)
#     suppliers_df = pd.read_csv(config.SUPPLIER_TRUST_CSV)

#     # --- Trend forecaster features ---
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

#     # --- Cost predictor features ---
#     orders_df = split_cost_data(orders_df)
#     orders_df[["order_id", "split"]].to_csv(
#         config.DATA_DIR / "landed_cost_split.csv", index=False
#     )
#     logger.info("Wrote %s", config.DATA_DIR / "landed_cost_split.csv")

#     cost_train = orders_df[orders_df["split"] == "train"]
#     cost_preprocessor = build_cost_preprocessor()

#     # fit_transform() (NOT fit() followed later by a separate transform())
#     # so TargetEncoder's internal cross-fitting actually runs -- see
#     # build_cost_preprocessor() docstring above. The resulting matrix is
#     # the leakage-safe encoding of the train split and is persisted
#     # directly; downstream training code loads it rather than
#     # re-deriving it via transform().
#     X_train_cost = cost_preprocessor.fit_transform(
#         cost_train[COST_CATEGORICAL_LOW_CARD + COST_CATEGORICAL_HIGH_CARD + COST_NUMERIC],
#         cost_train[COST_TARGET],
#     )
#     np.save(config.DATA_DIR / "X_train_cost.npy", X_train_cost)
#     logger.info("Wrote %s (leakage-safe cost train features)", config.DATA_DIR / "X_train_cost.npy")

#     joblib.dump(cost_preprocessor, config.MODELS_DIR / "preprocessor_cost.joblib")
#     logger.info("Saved %s", config.MODELS_DIR / "preprocessor_cost.joblib")

#     # --- Supplier reliability features ---
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

# """
# pipeline_preprocessing.py
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

# # FIXED: Target is now freight_cost so ML only predicts the logistics variables
# COST_TARGET: str = "freight_cost"
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

#     # --- Trend forecaster features ---
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

#     # --- Cost predictor features ---
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

#     # --- Supplier reliability features ---
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

COST_TARGET is "landed_cost" (NOT "freight_cost" -- see the note in
build_cost_preprocessor() below for why freight_cost was a dead end).
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

# COST_TARGET = "landed_cost", not "freight_cost". freight_cost is an EXACT
# deterministic function of just truck_type and distance_km in this
# synthetic data (freight_cost = TRUCK_BASE_FEE[truck_type] +
# TRUCK_PER_KM_RATE[truck_type] * distance_km, verified against
# generate_datasets.py -- max residual across 2000 test rows was PKR 6,
# which is exactly round(x, 2) noise, not real error). A model trained on
# freight_cost would just be re-deriving two numbers already hardcoded in
# config.py, giving a suspiciously perfect R^2 with no real predictive
# work being done, and 6 of the 8 input features would carry zero signal.
# landed_cost is a genuinely harder target: it depends on
# quantity * base_material_price (a multiplicative interaction trees only
# approximate, never reproduce exactly) plus the urgency markup plus
# freight -- real composite structure, and predicting it directly means
# the urgency markup is learned as part of the model instead of needing a
# separate, easy-to-forget addition at serving time.
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
    """Build (unfit) the ColumnTransformer for the landed-cost predictor's inputs.

    material_cost, freight_cost, and urgency_markup themselves are NOT
    included as features even though they're present in landed_cost_sim.csv
    -- they are deterministic arithmetic components of landed_cost
    (landed_cost = material_cost + freight_cost + urgency_markup), so
    including any of them would make "prediction" a trivial re-summation.
    The model instead learns landed_cost from the underlying drivers:
    material, quantity, base_material_price, truck_type, distance_km,
    urgency, destination_city, supplier_id.

    TargetEncoder cross-fitting note: TargetEncoder only performs its
    internal cross-fitting inside fit_transform(). A later, separate
    transform() call on the SAME rows the encoder was fit on does NOT
    cross-fit and would leak each row's own target back into its encoding.
    main() below calls fit_transform() exactly once on the train split and
    persists the result (data/X_train_cost.npy); training code loads that
    array directly rather than calling transform() on the train split
    again. transform() remains correct for the held-out test split.

    OneHotEncoder uses sparse_output=False so np.save()/np.load() round-trip
    the combined matrix as a plain dense ndarray instead of silently
    pickling a scipy sparse matrix into a 0-d object array (which then
    needs allow_pickle=True to load).
    """
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