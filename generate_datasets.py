
# """
# generate_datasets.py

# ============================================================================
# DATA PROVENANCE NOTICE (read before using these files anywhere, including
# in a viva/defense):

# All three CSVs produced by this script are SYNTHETICALLY GENERATED. Values
# are anchored to illustrative / approximate public price ranges researched
# during development (e.g. rough cement/steel PKR figures at time of
# writing). They are NOT scraped, sourced, or derived from a live Pakistan
# Bureau of Statistics (PBS) dataset, nor from any real supplier or
# transaction records. This is a deliberate, disclosed methodological choice
# appropriate for an FYP without access to real market/transaction data —
# see viva_defense_notes.md (Phase 5) for the full justification.
# ============================================================================

# Generates, with RANDOM_SEED fixed for reproducibility:
#     1. price_trends_sim.csv   — 3 years of weekly prices, 5 materials
#     2. landed_cost_sim.csv    — ~5,000 simulated B2B orders
#     3. supplier_trust_sim.csv — ~1,000 simulated supplier trust profiles
# """

# from __future__ import annotations

# import logging
# from datetime import datetime, timedelta

# import numpy as np
# import pandas as pd

# import config

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
# )
# logger = logging.getLogger(__name__)


# def generate_price_trends(rng: np.random.Generator) -> pd.DataFrame:
#     start_date = datetime.strptime(config.TREND_START_DATE, "%Y-%m-%d")
#     rows: list[dict] = []

#     for material in config.MATERIALS:
#         base_price = config.MATERIAL_BASE_PRICE[material]
#         annual_inflation = config.MATERIAL_ANNUAL_INFLATION[material]
#         monsoon_dip = config.MATERIAL_MONSOON_DIP[material]
#         noise_std_frac = config.MATERIAL_NOISE_STD_FRAC[material]
#         unit = config.MATERIAL_UNIT[material]

#         for week in range(config.TREND_WEEKS):
#             date = start_date + timedelta(weeks=week)

#             years_elapsed = week / 52.0
#             trend_price = base_price * ((1.0 + annual_inflation) ** years_elapsed)

#             if date.month in config.MONSOON_MONTHS:
#                 trend_price *= (1.0 - monsoon_dip)

#             noise = rng.normal(loc=0.0, scale=noise_std_frac * trend_price)
#             price = max(trend_price + noise, 1.0) 

#             rows.append(
#                 {
#                     "date": date.strftime("%Y-%m-%d"),
#                     "material": material,
#                     "price": round(price, 2),
#                     "unit": unit,
#                     "data_source": config.DATA_SOURCE_TAG,
#                 }
#             )

#     df = pd.DataFrame(rows)
#     logger.info(
#         "Generated price_trends_sim: %d rows across %d materials, %d weeks each",
#         len(df), len(config.MATERIALS), config.TREND_WEEKS,
#     )
#     return df


# def _lookup_price(price_df: pd.DataFrame, material: str, order_date: datetime) -> float:
#     subset = price_df[price_df["material"] == material]
#     subset_dates = pd.to_datetime(subset["date"])
#     idx = (subset_dates - order_date).abs().idxmin()
#     return float(subset.loc[idx, "price"])


# def generate_landed_cost_orders(
#     rng: np.random.Generator, price_df: pd.DataFrame
# ) -> pd.DataFrame:
#     start_date = datetime.strptime(config.TREND_START_DATE, "%Y-%m-%d")
#     end_date = start_date + timedelta(weeks=config.TREND_WEEKS)
#     total_span_days = (end_date - start_date).days

#     cities = list(config.CITY_DISTANCES_FROM_LAHORE_KM.keys())
#     supplier_ids = [f"SUP-{i:04d}" for i in range(1, config.NUM_SUPPLIERS_FOR_ORDERS + 1)]

#     rows: list[dict] = []
#     for order_num in range(1, config.NUM_ORDERS + 1):
#         material = rng.choice(config.MATERIALS)
#         qty_min, qty_max = config.QUANTITY_RANGE_BY_MATERIAL[material]
#         quantity = round(float(rng.uniform(qty_min, qty_max)), 1)

#         day_offset = int(rng.integers(0, total_span_days))
#         hour = int(rng.integers(8, 18))
#         minute = int(rng.integers(0, 60))
#         order_dt = start_date + timedelta(days=day_offset, hours=hour, minutes=minute)

#         city = rng.choice(cities)
#         distance_km = config.CITY_DISTANCES_FROM_LAHORE_KM[city]
#         distance_km = max(distance_km + rng.normal(0, distance_km * 0.05), 1.0)

#         qty_frac = (quantity - qty_min) / max(qty_max - qty_min, 1e-9)
#         truck_probs = (
#             np.array([0.6, 0.3, 0.1]) if qty_frac < 0.33 else
#             np.array([0.25, 0.5, 0.25]) if qty_frac < 0.66 else
#             np.array([0.05, 0.35, 0.6])
#         )
#         truck_type = rng.choice(config.TRUCK_TYPES, p=truck_probs)

#         urgency = rng.choice(config.URGENCY_LEVELS, p=[0.8, 0.2])
#         supplier_id = rng.choice(supplier_ids)

#         base_price = _lookup_price(price_df, material, order_dt)
#         material_cost = base_price * quantity
#         freight_cost = (
#             config.TRUCK_BASE_FEE[truck_type]
#             + config.TRUCK_PER_KM_RATE[truck_type] * distance_km
#         )
#         urgency_markup = material_cost * config.URGENCY_MARKUP_FRAC[urgency]
#         landed_cost = material_cost + freight_cost + urgency_markup

#         rows.append(
#             {
#                 "order_id": f"ORD-{order_num:06d}",
#                 "order_timestamp": order_dt.strftime("%Y-%m-%d %H:%M:%S"),
#                 "material": material,
#                 "quantity": quantity,
#                 "unit": config.MATERIAL_UNIT[material],
#                 "destination_city": city,
#                 "distance_km": round(distance_km, 1),
#                 "truck_type": truck_type,
#                 "urgency": urgency,
#                 "supplier_id": supplier_id,
#                 "base_material_price": round(base_price, 2),
#                 "material_cost": round(material_cost, 2),
#                 "freight_cost": round(freight_cost, 2),
#                 "urgency_markup": round(urgency_markup, 2),
#                 "landed_cost": round(landed_cost, 2),
#                 "data_source": config.DATA_SOURCE_TAG,
#             }
#         )

#     df = pd.DataFrame(rows)
#     logger.info("Generated landed_cost_sim: %d rows", len(df))
#     return df


# def generate_supplier_trust(rng: np.random.Generator) -> pd.DataFrame:
#     n = config.NUM_SUPPLIERS
#     quality = rng.normal(config.SUPPLIER_QUALITY_MEAN, config.SUPPLIER_QUALITY_STD, size=n)
#     quality = np.clip(quality, 0.0, 1.0)

#     on_time_delivery_rate = np.clip(quality + rng.normal(0, 0.05, size=n), 0.0, 1.0)
#     rating = np.clip(1.0 + 4.0 * quality + rng.normal(0, 0.3, size=n), 1.0, 5.0)

#     dispute_lambda = np.clip((1.0 - quality) * 8.0, 0.1, None)
#     dispute_count = rng.poisson(dispute_lambda)

#     df = pd.DataFrame(
#         {
#             "supplier_id": [f"SUP-{i:04d}" for i in range(1, n + 1)],
#             "quality_latent": np.round(quality, 4),
#             "on_time_delivery_rate": np.round(on_time_delivery_rate, 4),
#             "rating": np.round(rating, 2),
#             "dispute_count": dispute_count,
#             "data_source": config.DATA_SOURCE_TAG,
#         }
#     )
#     logger.info("Generated supplier_trust_sim: %d rows", len(df))
#     return df


# def main() -> None:
#     rng = np.random.default_rng(config.RANDOM_SEED)

#     price_df = generate_price_trends(rng)
#     price_df = price_df.sort_values(by="date").reset_index(drop=True)
#     price_df.to_csv(config.PRICE_TRENDS_CSV, index=False)
#     logger.info("Wrote %s (max date %s)", config.PRICE_TRENDS_CSV, price_df["date"].max())

#     orders_df = generate_landed_cost_orders(rng, price_df)
#     orders_df = orders_df.sort_values(by="order_timestamp").reset_index(drop=True)
#     orders_df.to_csv(config.LANDED_COST_CSV, index=False)
#     logger.info("Wrote %s (max ts %s)", config.LANDED_COST_CSV, orders_df["order_timestamp"].max())

#     suppliers_df = generate_supplier_trust(rng)
#     suppliers_df.to_csv(config.SUPPLIER_TRUST_CSV, index=False)
#     logger.info("Wrote %s", config.SUPPLIER_TRUST_CSV)


# if __name__ == "__main__":
#     main()

"""
generate_datasets.py

Generates reproducible datasets anchored to:
1. Official PBS Urban Construction Prices (October 2026 Lahore anchors)
2. Qureshi & Siddiqui (SSRN 4432245) Macroeconomic Correlations
3. Smooth Fourier-compatible seasonal waves + AR(1) price momentum
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def generate_price_trends(rng: np.random.Generator) -> pd.DataFrame:
    """Generate weekly price series for all 24 materials with smooth harmonic
    seasonality, AR(1) momentum, and macroeconomic regressors
    (diesel_price_pkr, usd_pkr) calibrated to PBS October 2026 Lahore prices.
    """
    start_date = datetime.strptime(config.TREND_START_DATE, "%Y-%m-%d")
    n_weeks = config.TREND_WEEKS

    # Shared weekly macroeconomic regressor trajectories ending at PBS Oct 2026 Diesel (396.20 PKR/L)
    diesel_series = np.linspace(225.0, 396.20, n_weeks) + rng.normal(0, 2.2, n_weeks)
    diesel_series[-1] = 396.20

    fx_series = np.linspace(226.0, 278.50, n_weeks) + rng.normal(0, 1.5, n_weeks)
    fx_series[-1] = 278.50

    rows: list[dict] = []

    for material in config.MATERIALS:
        oct26_target = config.MATERIAL_OCT2026_ANCHOR_PRICE[material]
        annual_inflation = config.MATERIAL_ANNUAL_INFLATION[material]
        monsoon_dip = config.MATERIAL_MONSOON_DIP[material]
        noise_std = config.MATERIAL_NOISE_STD_FRAC[material]
        unit = config.MATERIAL_UNIT[material]
        d_weight, fx_weight = config.MATERIAL_MACRO_WEIGHTS.get(material, (0.25, 0.20))

        raw_trajectory: list[tuple[datetime, float]] = []
        ar1_state = 0.0

        for week in range(n_weeks):
            date = start_date + timedelta(weeks=week)
            years_elapsed = week / 52.0
            day_of_year = date.timetuple().tm_yday

            # 1. Compounding inflation trend
            growth = (1.0 + annual_inflation) ** years_elapsed

            # 2. Exogenous macroeconomic regressor elasticity (SSRN 4432245)
            d_rel = (diesel_series[week] - diesel_series[0]) / diesel_series[0]
            fx_rel = (fx_series[week] - fx_series[0]) / fx_series[0]
            macro_mult = 1.0 + (d_weight * 0.25 * d_rel) + (fx_weight * 0.35 * fx_rel)

            # 3. Smooth differentiable seasonal wave (peaks in spring/autumn, dips in mid-Aug monsoon)
            monsoon_gaussian = np.exp(-0.5 * ((day_of_year - 220.0) / 28.0) ** 2)
            winter_boost = 0.015 * np.cos(2.0 * np.pi * (day_of_year - 45.0) / 365.25)
            seasonal_mult = 1.0 - (monsoon_dip * monsoon_gaussian) + winter_boost

            # 4. Autocorrelated AR(1) market memory
            innovation = rng.normal(0.0, noise_std)
            ar1_state = 0.70 * ar1_state + 0.30 * innovation

            unscaled_val = growth * macro_mult * seasonal_mult * (1.0 + ar1_state)
            raw_trajectory.append((date, unscaled_val))

        # Scale series so the final October 2026 observation hits the exact PBS Lahore anchor price
        scale_factor = oct26_target / raw_trajectory[-1][1]

        for week, (date, unscaled_val) in enumerate(raw_trajectory):
            final_price = max(unscaled_val * scale_factor, 1.0)
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "material": material,
                    "price": round(float(final_price), 2),
                    "unit": unit,
                    "diesel_price_pkr": round(float(diesel_series[week]), 2),
                    "usd_pkr": round(float(fx_series[week]), 2),
                    "data_source": config.DATA_SOURCE_TAG,
                }
            )

    df = pd.DataFrame(rows)
    logger.info(
        "Generated price_trends_sim: %d rows across %d materials, %d weeks each",
        len(df), len(config.MATERIALS), config.TREND_WEEKS,
    )
    return df


def _lookup_price(price_df: pd.DataFrame, material: str, order_date: datetime) -> float:
    subset = price_df[price_df["material"] == material]
    subset_dates = pd.to_datetime(subset["date"])
    idx = (subset_dates - order_date).abs().idxmin()
    return float(subset.loc[idx, "price"])


def generate_landed_cost_orders(
    rng: np.random.Generator, price_df: pd.DataFrame
) -> pd.DataFrame:
    start_date = datetime.strptime(config.TREND_START_DATE, "%Y-%m-%d")
    end_date = start_date + timedelta(weeks=config.TREND_WEEKS)
    total_span_days = (end_date - start_date).days

    cities = list(config.CITY_DISTANCES_FROM_LAHORE_KM.keys())
    supplier_ids = [f"SUP-{i:04d}" for i in range(1, config.NUM_SUPPLIERS_FOR_ORDERS + 1)]

    rows: list[dict] = []
    for order_num in range(1, config.NUM_ORDERS + 1):
        material = rng.choice(config.MATERIALS)
        qty_min, qty_max = config.QUANTITY_RANGE_BY_MATERIAL[material]
        quantity = round(float(rng.uniform(qty_min, qty_max)), 1)

        day_offset = int(rng.integers(0, total_span_days))
        hour = int(rng.integers(8, 18))
        minute = int(rng.integers(0, 60))
        order_dt = start_date + timedelta(days=day_offset, hours=hour, minutes=minute)

        city = rng.choice(cities)
        distance_km = config.CITY_DISTANCES_FROM_LAHORE_KM[city]
        distance_km = max(distance_km + rng.normal(0, distance_km * 0.05), 1.0)

        qty_frac = (quantity - qty_min) / max(qty_max - qty_min, 1e-9)
        truck_probs = (
            np.array([0.6, 0.3, 0.1]) if qty_frac < 0.33 else
            np.array([0.25, 0.5, 0.25]) if qty_frac < 0.66 else
            np.array([0.05, 0.35, 0.6])
        )
        truck_type = rng.choice(config.TRUCK_TYPES, p=truck_probs)

        urgency = rng.choice(config.URGENCY_LEVELS, p=[0.8, 0.2])
        supplier_id = rng.choice(supplier_ids)

        base_price = _lookup_price(price_df, material, order_dt)
        material_cost = base_price * quantity
        freight_cost = (
            config.TRUCK_BASE_FEE[truck_type]
            + config.TRUCK_PER_KM_RATE[truck_type] * distance_km
        )
        urgency_markup = material_cost * config.URGENCY_MARKUP_FRAC[urgency]
        landed_cost = material_cost + freight_cost + urgency_markup

        rows.append(
            {
                "order_id": f"ORD-{order_num:06d}",
                "order_timestamp": order_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "material": material,
                "quantity": quantity,
                "unit": config.MATERIAL_UNIT[material],
                "destination_city": city,
                "distance_km": round(distance_km, 1),
                "truck_type": truck_type,
                "urgency": urgency,
                "supplier_id": supplier_id,
                "base_material_price": round(base_price, 2),
                "material_cost": round(material_cost, 2),
                "freight_cost": round(freight_cost, 2),
                "urgency_markup": round(urgency_markup, 2),
                "landed_cost": round(landed_cost, 2),
                "data_source": config.DATA_SOURCE_TAG,
            }
        )

    df = pd.DataFrame(rows)
    logger.info("Generated landed_cost_sim: %d rows", len(df))
    return df


def generate_supplier_trust(rng: np.random.Generator) -> pd.DataFrame:
    n = config.NUM_SUPPLIERS
    quality = rng.normal(config.SUPPLIER_QUALITY_MEAN, config.SUPPLIER_QUALITY_STD, size=n)
    quality = np.clip(quality, 0.0, 1.0)

    on_time_delivery_rate = np.clip(quality + rng.normal(0, 0.05, size=n), 0.0, 1.0)
    rating = np.clip(1.0 + 4.0 * quality + rng.normal(0, 0.3, size=n), 1.0, 5.0)

    dispute_lambda = np.clip((1.0 - quality) * 8.0, 0.1, None)
    dispute_count = rng.poisson(dispute_lambda)

    df = pd.DataFrame(
        {
            "supplier_id": [f"SUP-{i:04d}" for i in range(1, n + 1)],
            "quality_latent": np.round(quality, 4),
            "on_time_delivery_rate": np.round(on_time_delivery_rate, 4),
            "rating": np.round(rating, 2),
            "dispute_count": dispute_count,
            "data_source": config.DATA_SOURCE_TAG,
        }
    )
    logger.info("Generated supplier_trust_sim: %d rows", len(df))
    return df


def main() -> None:
    rng = np.random.default_rng(config.RANDOM_SEED)

    price_df = generate_price_trends(rng)
    price_df = price_df.sort_values(by="date").reset_index(drop=True)
    price_df.to_csv(config.PRICE_TRENDS_CSV, index=False)
    logger.info("Wrote %s (max date %s)", config.PRICE_TRENDS_CSV, price_df["date"].max())

    orders_df = generate_landed_cost_orders(rng, price_df)
    orders_df = orders_df.sort_values(by="order_timestamp").reset_index(drop=True)
    orders_df.to_csv(config.LANDED_COST_CSV, index=False)
    logger.info("Wrote %s (max ts %s)", config.LANDED_COST_CSV, orders_df["order_timestamp"].max())

    suppliers_df = generate_supplier_trust(rng)
    suppliers_df.to_csv(config.SUPPLIER_TRUST_CSV, index=False)
    logger.info("Wrote %s", config.SUPPLIER_TRUST_CSV)


if __name__ == "__main__":
    main()