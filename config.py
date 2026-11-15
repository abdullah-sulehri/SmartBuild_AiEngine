

# """
# config.py

# Single source of truth for reproducibility settings, file paths, and domain
# parameters used across the SmartBuild AI Engine pipeline (data generation,
# preprocessing, training, and serving).

# Centralizing these values (rather than scattering magic numbers through
# each script) is a deliberate choice for FYP defensibility: a panel member
# can be pointed at one file to see every assumption the pipeline makes.
# """

# from __future__ import annotations

# from logging import config
# import os
# from pathlib import Path

# # ---------------------------------------------------------------------------
# # Reproducibility
# # ---------------------------------------------------------------------------
# # Fixed globally and imported everywhere random state is touched (numpy,
# # python's random module, scikit-learn's random_state=, XGBoost's seed=,
# # etc.) so every run — and the viva demo — reproduces identical numbers.
# RANDOM_SEED: int = 42

# # ---------------------------------------------------------------------------
# # Paths
# # ---------------------------------------------------------------------------
# BASE_DIR: Path = Path(__file__).resolve().parent
# DATA_DIR: Path = BASE_DIR / "data"
# MODELS_DIR: Path = BASE_DIR / "models"
# ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"  # charts, SHAP plots, confusion matrices

# for _dir in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR):
#     os.makedirs(_dir, exist_ok=True)


# PRICE_TRENDS_CSV: Path = DATA_DIR / "price_trends_sim.csv"
# LANDED_COST_CSV: Path = DATA_DIR / "landed_cost_sim.csv"
# SUPPLIER_TRUST_CSV: Path = DATA_DIR / "supplier_trust_sim.csv"

# # ---------------------------------------------------------------------------
# # Data provenance flag
# # ---------------------------------------------------------------------------
# # Every synthetic row carries this literal string in a `data_source` /
# # `data_provenance` column. This is NOT real PBS (Pakistan Bureau of
# # Statistics) or any live-scraped data — it is illustrative, hand-anchored
# # to approximate public price ranges researched during development, used
# # only because this is an FYP without access to real market/transaction
# # data. See Phase 5 viva notes for the full justification.
# DATA_SOURCE_TAG: str = "synthetic-illustrative"

# # ---------------------------------------------------------------------------
# # Phase 1 — price_trends_sim.csv parameters
# # ---------------------------------------------------------------------------
# TREND_WEEKS: int = 191  # ~3.67 years, reaching into Sept 2026
# TREND_START_DATE: str = "2023-01-02"  # a Monday, for tidy weekly stamps

# # Comprehensive catalog: grey structure, masonry, finishing, electrical,
# # plumbing/sanitary, roofing, and doors/windows hardware.
# MATERIALS: list[str] = [
#     "Cement", "Steel", "Bricks", "Sand", "Aggregate", "Gravel",
#     "Concrete Blocks", "Lime",
#     "Wood", "Plywood", "Glass", "Marble", "Granite",
#     "Paint", "Tiles", "Gypsum Board", "Wall Putty",
#     "Pipes", "Sanitary Fittings",
#     "Wires", "Electrical Switches", "Circuit Breakers",
#     "Roofing Sheets", "Aluminum Profiles",
# ]

# # Illustrative starting prices (PKR), unit noted per material. These are
# # approximate public-range anchors researched during development, NOT
# # live/current market quotes — flagged as such in defense.
# MATERIAL_BASE_PRICE: dict[str, float] = {
#     "Cement": 1_300.0,           # PKR per 50kg bag
#     "Steel": 250_000.0,          # PKR per ton
#     "Bricks": 14_000.0,          # PKR per 1,000 bricks
#     "Sand": 40_000.0,            # PKR per 100 cft (truck-load unit)
#     "Aggregate": 45_000.0,       # PKR per 100 cft
#     "Gravel": 42_000.0,          # PKR per 100 cft
#     "Concrete Blocks": 16_000.0, # PKR per 1,000 blocks
#     "Lime": 900.0,               # PKR per 50kg bag
#     "Wood": 4_000.0,             # PKR per cft
#     "Plywood": 3_500.0,          # PKR per 4x8ft sheet
#     "Glass": 250.0,              # PKR per sq ft
#     "Marble": 180.0,             # PKR per sq ft
#     "Granite": 220.0,            # PKR per sq ft
#     "Paint": 15_000.0,           # PKR per 16L drum
#     "Tiles": 2_500.0,            # PKR per sq meter
#     "Gypsum Board": 2_200.0,     # PKR per sheet
#     "Wall Putty": 1_800.0,       # PKR per 40kg bag
#     "Pipes": 2_000.0,            # PKR per length
#     "Sanitary Fittings": 8_000.0,  # PKR per fixture (basin/faucet set)
#     "Wires": 8_000.0,            # PKR per coil
#     "Electrical Switches": 350.0,  # PKR per piece (switch/socket)
#     "Circuit Breakers": 1_200.0,   # PKR per piece (MCB)
#     "Roofing Sheets": 3_200.0,   # PKR per corrugated GI sheet
#     "Aluminum Profiles": 950.0,  # PKR per kg
# }

# MATERIAL_UNIT: dict[str, str] = {
#     "Cement": "PKR/50kg bag",
#     "Steel": "PKR/ton",
#     "Bricks": "PKR/1000 bricks",
#     "Sand": "PKR/100 cft",
#     "Aggregate": "PKR/100 cft",
#     "Gravel": "PKR/100 cft",
#     "Concrete Blocks": "PKR/1000 blocks",
#     "Lime": "PKR/50kg bag",
#     "Wood": "PKR/cft",
#     "Plywood": "PKR/sheet",
#     "Glass": "PKR/sq ft",
#     "Marble": "PKR/sq ft",
#     "Granite": "PKR/sq ft",
#     "Paint": "PKR/16L drum",
#     "Tiles": "PKR/sq meter",
#     "Gypsum Board": "PKR/sheet",
#     "Wall Putty": "PKR/40kg bag",
#     "Pipes": "PKR/length",
#     "Sanitary Fittings": "PKR/fixture",
#     "Wires": "PKR/coil",
#     "Electrical Switches": "PKR/piece",
#     "Circuit Breakers": "PKR/piece",
#     "Roofing Sheets": "PKR/sheet",
#     "Aluminum Profiles": "PKR/kg",
# }

# # Gradual annualized inflation rate applied per material (illustrative,
# # construction-material inflation in Pakistan has historically run high
# # and volatile — these are approximate, not official CPI sub-indices).
# # Cement held near current real-world max observed during development;
# # metal-linked materials (Steel, Wires, Aluminum Profiles) carry the
# # highest inflation of the set.
# MATERIAL_ANNUAL_INFLATION: dict[str, float] = {
#     "Cement": 0.07,
#     "Steel": 0.18,
#     "Bricks": 0.10,
#     "Sand": 0.08,
#     "Aggregate": 0.08,
#     "Gravel": 0.08,
#     "Concrete Blocks": 0.09,
#     "Lime": 0.08,
#     "Wood": 0.10,
#     "Plywood": 0.10,
#     "Glass": 0.09,
#     "Marble": 0.09,
#     "Granite": 0.09,
#     "Paint": 0.12,
#     "Tiles": 0.08,
#     "Gypsum Board": 0.11,
#     "Wall Putty": 0.10,
#     "Pipes": 0.11,
#     "Sanitary Fittings": 0.10,
#     "Wires": 0.15,
#     "Electrical Switches": 0.10,
#     "Circuit Breakers": 0.10,
#     "Roofing Sheets": 0.13,
#     "Aluminum Profiles": 0.14,
# }

# # Monsoon season (Jul-Sep): construction activity slows, which we model as
# # a demand-driven dip in price. Magnitude differs per material — Sand/
# # Aggregate/Gravel (riverbed-sourced) and Paint/Wall Putty (can't apply
# # well in rain/humidity) are hit hardest; quarried stone (Marble/Granite)
# # and indoor plumbing/electrical/hardware items are largely unaffected.
# MATERIAL_MONSOON_DIP: dict[str, float] = {
#     "Cement": 0.02,
#     "Steel": 0.015,
#     "Bricks": 0.03,
#     "Sand": 0.06,
#     "Aggregate": 0.05,
#     "Gravel": 0.05,
#     "Concrete Blocks": 0.03,
#     "Lime": 0.02,
#     "Wood": 0.04,
#     "Plywood": 0.03,
#     "Glass": 0.01,
#     "Marble": 0.01,
#     "Granite": 0.01,
#     "Paint": 0.08,
#     "Tiles": 0.02,
#     "Gypsum Board": 0.0,
#     "Wall Putty": 0.06,
#     "Pipes": 0.0,
#     "Sanitary Fittings": 0.0,
#     "Wires": 0.0,
#     "Electrical Switches": 0.0,
#     "Circuit Breakers": 0.0,
#     "Roofing Sheets": 0.0,
#     "Aluminum Profiles": 0.0,
# }
# MONSOON_MONTHS: tuple[int, ...] = (7, 8, 9)

# # Weekly noise as a fraction of the current (trend-adjusted) price, i.e.
# # noise scale = MATERIAL_NOISE_STD_FRAC * current_price.
# MATERIAL_NOISE_STD_FRAC: dict[str, float] = {
#     "Cement": 0.01,
#     "Steel": 0.015,
#     "Bricks": 0.012,
#     "Sand": 0.02,
#     "Aggregate": 0.02,
#     "Gravel": 0.02,
#     "Concrete Blocks": 0.012,
#     "Lime": 0.01,
#     "Wood": 0.03,
#     "Plywood": 0.02,
#     "Glass": 0.01,
#     "Marble": 0.015,
#     "Granite": 0.015,
#     "Paint": 0.01,
#     "Tiles": 0.01,
#     "Gypsum Board": 0.015,
#     "Wall Putty": 0.015,
#     "Pipes": 0.01,
#     "Sanitary Fittings": 0.02,
#     "Wires": 0.025,
#     "Electrical Switches": 0.015,
#     "Circuit Breakers": 0.015,
#     "Roofing Sheets": 0.02,
#     "Aluminum Profiles": 0.02,
# }


# # ---------------------------------------------------------------------------
# # Phase 1 — landed_cost_sim.csv parameters
# # ---------------------------------------------------------------------------
# NUM_ORDERS: int = 20_000  # bumped for 24-material catalog so each material keeps ~800+ rows
# NUM_SUPPLIERS_FOR_ORDERS: int = 400  # supplier_id pool orders are drawn from

# TRUCK_TYPES: list[str] = ["mini", "6-wheeler", "10-wheeler"]

# # Illustrative freight structure: a fixed base fee plus a per-km rate that
# # varies by truck capacity (bigger trucks cost more per trip but the
# # per-km rate itself is roughly flat to slightly higher due to fuel draw).
# TRUCK_BASE_FEE: dict[str, float] = {
#     "mini": 3_000.0,
#     "6-wheeler": 6_000.0,
#     "10-wheeler": 9_500.0,
# }
# TRUCK_PER_KM_RATE: dict[str, float] = {
#     "mini": 55.0,
#     "6-wheeler": 85.0,
#     "10-wheeler": 120.0,
# }

# URGENCY_LEVELS: list[str] = ["standard", "express"]
# URGENCY_MARKUP_FRAC: dict[str, float] = {
#     "standard": 0.0,
#     "express": 0.12,  # +12% on (base material price) for express handling
# }

# # Real Lahore/Punjab city pairs with approximate road distances (km) from
# # Lahore, used to simulate realistic delivery legs. Approximate, illustrative.
# CITY_DISTANCES_FROM_LAHORE_KM: dict[str, float] = {
#     "Lahore": 15.0,          # intra-city delivery, short leg
#     "Sheikhupura": 45.0,
#     "Kasur": 55.0,
#     "Gujranwala": 80.0,
#     "Sialkot": 125.0,
#     "Faisalabad": 130.0,
#     "Sargodha": 190.0,
#     "Gujrat": 155.0,
#     "Multan": 340.0,
#     "Rawalpindi": 275.0,
#     # Added for national coverage (originally Punjab/Lahore-only scope) --
#     # approximate road distances, illustrative like the rest of Phase 1.
#     "Islamabad": 375.0,
#     "Peshawar": 510.0,
#     "Karachi": 1_210.0,
#     "Hyderabad": 1_150.0,
#     "Quetta": 940.0,
#     "Sukkur": 820.0,
#     "Bahawalpur": 430.0,
#     "Sahiwal": 180.0,
#     "Okara": 120.0,
#     "Dera Ghazi Khan": 520.0,
# }


# QUANTITY_RANGE_BY_MATERIAL: dict[str, tuple[float, float]] = {
#     # (min, max) quantity per order, in the unit given in MATERIAL_UNIT
#     "Cement": (50, 2_000),            # bags
#     "Steel": (1, 40),                 # tons
#     "Bricks": (5, 200),               # thousands of bricks
#     "Sand": (2, 60),                  # units of 100 cft
#     "Aggregate": (2, 60),             # units of 100 cft
#     "Gravel": (2, 60),                # units of 100 cft
#     "Concrete Blocks": (5, 200),      # thousands of blocks
#     "Lime": (50, 1_500),              # bags
#     "Wood": (50, 1_000),              # cft
#     "Plywood": (20, 500),             # sheets
#     "Glass": (100, 5_000),            # sq ft
#     "Marble": (100, 5_000),           # sq ft
#     "Granite": (100, 5_000),          # sq ft
#     "Paint": (10, 200),               # 16L drums
#     "Tiles": (100, 5_000),            # sq meters
#     "Gypsum Board": (20, 500),        # sheets
#     "Wall Putty": (10, 300),          # 40kg bags
#     "Pipes": (50, 1_000),             # lengths
#     "Sanitary Fittings": (5, 200),    # fixtures
#     "Wires": (50, 1_000),             # coils
#     "Electrical Switches": (50, 2_000),  # pieces
#     "Circuit Breakers": (10, 500),    # pieces
#     "Roofing Sheets": (10, 500),      # sheets
#     "Aluminum Profiles": (50, 2_000), # kg
# }

# # ---------------------------------------------------------------------------
# # Phase 1 — supplier_trust_sim.csv parameters
# # ---------------------------------------------------------------------------
# NUM_SUPPLIERS: int = 1_000

# # Latent "quality" factor suppliers are drawn from (clipped to [0, 1]),
# # used to correlate on-time rate, rating, and dispute count so the
# # classifier in Phase 3 has genuine structure to learn rather than noise.
# SUPPLIER_QUALITY_MEAN: float = 0.72
# SUPPLIER_QUALITY_STD: float = 0.16

# # Reliability label threshold: suppliers at/above this on-time rate are
# # labeled "reliable" for the Phase 3 classifier target. Chosen (0.60,
# # against a supplier-pool mean on-time rate of ~0.72) to produce a
# # realistic "most suppliers reliable, meaningful minority are not" split
# # rather than an inverted one -- see Phase 3 imbalance handling.
# RELIABLE_ON_TIME_THRESHOLD: float = 0.60

# # ---------------------------------------------------------------------------
# # Phase 4 — smart-recommendation composite weights
# # ---------------------------------------------------------------------------
# # Composite score = w1*cost_component + w2*trust_component + w3*trend_component
# # All three components are normalized to [0, 1] before combination (see
# # pipeline_preprocessing.py / app.py) so raw PKR values never dominate a
# # 0-1 trust score. Weights are configurable here, not hardcoded in app.py.
# RECOMMENDATION_WEIGHTS: dict[str, float] = {
#     "w1_cost": 0.5,
#     "w2_trust": 0.3,
#     "w3_trend": 0.2,
# }


"""
config.py

Single source of truth for reproducibility settings, file paths, domain
parameters (anchored to PBS October 2026 Lahore prices & SSRN 4432245
macroeconomic correlations), and per-material Prophet hyperparameter profiles.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"

for _dir in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR):
    os.makedirs(_dir, exist_ok=True)

PRICE_TRENDS_CSV: Path = DATA_DIR / "price_trends_sim.csv"
LANDED_COST_CSV: Path = DATA_DIR / "landed_cost_sim.csv"
SUPPLIER_TRUST_CSV: Path = DATA_DIR / "supplier_trust_sim.csv"

DATA_SOURCE_TAG: str = "pbs-oct2026-ssrn4432245-calibrated"

# ---------------------------------------------------------------------------
# Phase 1 — price_trends_sim.csv parameters
# ---------------------------------------------------------------------------
TREND_WEEKS: int = 199  # Jan 2, 2023 -> Oct 19, 2026 (matches Oct 2026 PBS release)
TREND_START_DATE: str = "2023-01-02"

MATERIALS: list[str] = [
    "Cement", "Steel", "Bricks", "Sand", "Aggregate", "Gravel",
    "Concrete Blocks", "Lime",
    "Wood", "Plywood", "Glass", "Marble", "Granite",
    "Paint", "Tiles", "Gypsum Board", "Wall Putty",
    "Pipes", "Sanitary Fittings",
    "Wires", "Electrical Switches", "Circuit Breakers",
    "Roofing Sheets", "Aluminum Profiles",
]

# Category groupings for Prophet tuning & macroeconomic sensitivity
MATERIAL_CATEGORY: dict[str, str] = {
    "Cement": "binder",
    "Lime": "binder",
    "Wall Putty": "binder",
    "Gypsum Board": "binder",
    "Steel": "metal",
    "Wires": "metal",
    "Aluminum Profiles": "metal",
    "Roofing Sheets": "metal",
    "Bricks": "masonry_quarry",
    "Concrete Blocks": "masonry_quarry",
    "Sand": "riverbed_aggregate",
    "Aggregate": "riverbed_aggregate",
    "Gravel": "riverbed_aggregate",
    "Marble": "masonry_quarry",
    "Granite": "masonry_quarry",
    "Wood": "timber",
    "Plywood": "timber",
    "Glass": "finishing",
    "Paint": "weather_sensitive_finish",
    "Tiles": "finishing",
    "Pipes": "petrochemical_mep",
    "Sanitary Fittings": "petrochemical_mep",
    "Electrical Switches": "petrochemical_mep",
    "Circuit Breakers": "petrochemical_mep",
}

# Exact October 2026 target prices anchored to PBS Urban Construction Prices
# (Lahore row in Urbn-Construction-12.xlsx) and Lahore market quotes.
MATERIAL_OCT2026_ANCHOR_PRICE: dict[str, float] = {
    "Cement": 1_619.98,           # PBS Lahore Oct 2026 (50kg bag)
    "Steel": 235_380.45,          # PBS Lahore Oct 2026 (per tonne)
    "Bricks": 25_830.87,          # PBS Lahore Oct 2026 (per 1,000 bricks)
    "Sand": 7_691.11,             # PBS Lahore Oct 2026 (per 100 cft)
    "Aggregate": 17_772.19,       # PBS Lahore Oct 2026 Bajri/Crush (per 100 cft)
    "Gravel": 16_500.00,          # PKR per 100 cft
    "Concrete Blocks": 73_995.50, # PBS Oct 2026 scaled to 1,000 blocks
    "Lime": 980.00,               # PKR per 50kg bag
    "Wood": 4_600.00,             # PKR per cft
    "Plywood": 3_950.00,          # PKR per 4x8ft sheet
    "Glass": 310.00,              # PKR per sq ft
    "Marble": 215.00,             # PKR per sq ft
    "Granite": 265.00,            # PKR per sq ft
    "Paint": 25_813.13,           # PBS Lahore Oct 2026 scaled to 16L drum
    "Tiles": 2_850.00,            # PKR per sq meter
    "Gypsum Board": 2_480.00,     # PKR per sheet
    "Wall Putty": 2_050.00,       # PKR per 40kg bag
    "Pipes": 2_350.00,            # PKR per length
    "Sanitary Fittings": 9_200.00,# PKR per fixture
    "Wires": 9_800.00,            # PKR per coil
    "Electrical Switches": 410.00,# PKR per piece
    "Circuit Breakers": 1_420.00, # PKR per piece
    "Roofing Sheets": 3_850.00,   # PKR per sheet
    "Aluminum Profiles": 1_180.00,# PKR per kg
}

# Retained for backward compatibility with any external script importing MATERIAL_BASE_PRICE
MATERIAL_BASE_PRICE: dict[str, float] = MATERIAL_OCT2026_ANCHOR_PRICE.copy()

MATERIAL_UNIT: dict[str, str] = {
    "Cement": "PKR/50kg bag",
    "Steel": "PKR/ton",
    "Bricks": "PKR/1000 bricks",
    "Sand": "PKR/100 cft",
    "Aggregate": "PKR/100 cft",
    "Gravel": "PKR/100 cft",
    "Concrete Blocks": "PKR/1000 blocks",
    "Lime": "PKR/50kg bag",
    "Wood": "PKR/cft",
    "Plywood": "PKR/sheet",
    "Glass": "PKR/sq ft",
    "Marble": "PKR/sq ft",
    "Granite": "PKR/sq ft",
    "Paint": "PKR/16L drum",
    "Tiles": "PKR/sq meter",
    "Gypsum Board": "PKR/sheet",
    "Wall Putty": "PKR/40kg bag",
    "Pipes": "PKR/length",
    "Sanitary Fittings": "PKR/fixture",
    "Wires": "PKR/coil",
    "Electrical Switches": "PKR/piece",
    "Circuit Breakers": "PKR/piece",
    "Roofing Sheets": "PKR/sheet",
    "Aluminum Profiles": "PKR/kg",
}

MATERIAL_ANNUAL_INFLATION: dict[str, float] = {
    "Cement": 0.085,
    "Steel": 0.140,
    "Bricks": 0.110,
    "Sand": 0.080,
    "Aggregate": 0.085,
    "Gravel": 0.080,
    "Concrete Blocks": 0.090,
    "Lime": 0.075,
    "Wood": 0.095,
    "Plywood": 0.095,
    "Glass": 0.105,
    "Marble": 0.080,
    "Granite": 0.085,
    "Paint": 0.115,
    "Tiles": 0.085,
    "Gypsum Board": 0.095,
    "Wall Putty": 0.095,
    "Pipes": 0.105,
    "Sanitary Fittings": 0.095,
    "Wires": 0.135,
    "Electrical Switches": 0.090,
    "Circuit Breakers": 0.095,
    "Roofing Sheets": 0.120,
    "Aluminum Profiles": 0.130,
}

# Smooth harmonic monsoon dip amplitude (peaks mid-August)
MATERIAL_MONSOON_DIP: dict[str, float] = {
    "Cement": 0.035,
    "Steel": 0.020,
    "Bricks": 0.055,
    "Sand": 0.075,
    "Aggregate": 0.060,
    "Gravel": 0.055,
    "Concrete Blocks": 0.035,
    "Lime": 0.025,
    "Wood": 0.040,
    "Plywood": 0.030,
    "Glass": 0.015,
    "Marble": 0.015,
    "Granite": 0.015,
    "Paint": 0.080,
    "Tiles": 0.020,
    "Gypsum Board": 0.015,
    "Wall Putty": 0.065,
    "Pipes": 0.015,
    "Sanitary Fittings": 0.010,
    "Wires": 0.010,
    "Electrical Switches": 0.005,
    "Circuit Breakers": 0.005,
    "Roofing Sheets": 0.025,
    "Aluminum Profiles": 0.010,
}
MONSOON_MONTHS: tuple[int, ...] = (7, 8, 9)

# Calibrated innovation noise std fraction (paired with AR(1) momentum)
MATERIAL_NOISE_STD_FRAC: dict[str, float] = {
    "Cement": 0.005,
    "Steel": 0.008,
    "Bricks": 0.006,
    "Sand": 0.008,
    "Aggregate": 0.008,
    "Gravel": 0.008,
    "Concrete Blocks": 0.006,
    "Lime": 0.005,
    "Wood": 0.009,
    "Plywood": 0.007,
    "Glass": 0.005,
    "Marble": 0.006,
    "Granite": 0.006,
    "Paint": 0.006,
    "Tiles": 0.005,
    "Gypsum Board": 0.006,
    "Wall Putty": 0.007,
    "Pipes": 0.006,
    "Sanitary Fittings": 0.007,
    "Wires": 0.008,
    "Electrical Switches": 0.005,
    "Circuit Breakers": 0.005,
    "Roofing Sheets": 0.007,
    "Aluminum Profiles": 0.007,
}

# Empirical macroeconomic sensitivities derived from Qureshi & Siddiqui (SSRN 4432245):
# (diesel_transport_weight, fx_import_weight)
MATERIAL_MACRO_WEIGHTS: dict[str, tuple[float, float]] = {
    "Cement": (0.40, 0.25),
    "Steel": (0.15, 0.45),
    "Bricks": (0.20, 0.10),
    "Sand": (0.35, 0.05),
    "Aggregate": (0.35, 0.05),
    "Gravel": (0.30, 0.05),
    "Concrete Blocks": (0.30, 0.15),
    "Lime": (0.25, 0.10),
    "Wood": (0.20, 0.20),
    "Plywood": (0.20, 0.25),
    "Glass": (0.35, 0.30),
    "Marble": (0.25, 0.10),
    "Granite": (0.25, 0.15),
    "Paint": (0.20, 0.35),
    "Tiles": (0.30, 0.20),
    "Gypsum Board": (0.25, 0.20),
    "Wall Putty": (0.25, 0.20),
    "Pipes": (0.20, 0.35),
    "Sanitary Fittings": (0.15, 0.30),
    "Wires": (0.15, 0.45),
    "Electrical Switches": (0.15, 0.25),
    "Circuit Breakers": (0.15, 0.30),
    "Roofing Sheets": (0.20, 0.40),
    "Aluminum Profiles": (0.20, 0.45),
}


# ---------------------------------------------------------------------------
# Prophet Per-Category Hyperparameter Profiles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProphetProfile:
    changepoint_prior_scale: float
    seasonality_prior_scale: float
    seasonality_mode: str
    fourier_order_yearly: int
    use_diesel_regressor: bool
    use_fx_regressor: bool


PROPHET_PROFILES_BY_CATEGORY: dict[str, ProphetProfile] = {
    "metal": ProphetProfile(
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=5.0,
        seasonality_mode="multiplicative",
        fourier_order_yearly=6,
        use_diesel_regressor=True,
        use_fx_regressor=True,
    ),
    "riverbed_aggregate": ProphetProfile(
        changepoint_prior_scale=0.06,
        seasonality_prior_scale=12.0,
        seasonality_mode="multiplicative",
        fourier_order_yearly=8,
        use_diesel_regressor=True,
        use_fx_regressor=False,
    ),
    "weather_sensitive_finish": ProphetProfile(
        changepoint_prior_scale=0.07,
        seasonality_prior_scale=12.0,
        seasonality_mode="multiplicative",
        fourier_order_yearly=8,
        use_diesel_regressor=False,
        use_fx_regressor=True,
    ),
    "binder": ProphetProfile(
        changepoint_prior_scale=0.08,
        seasonality_prior_scale=8.0,
        seasonality_mode="multiplicative",
        fourier_order_yearly=8,
        use_diesel_regressor=True,
        use_fx_regressor=True,
    ),
    "default": ProphetProfile(
        changepoint_prior_scale=0.08,
        seasonality_prior_scale=8.0,
        seasonality_mode="multiplicative",
        fourier_order_yearly=8,
        use_diesel_regressor=True,
        use_fx_regressor=True,
    ),
}


def get_prophet_profile(material: str) -> ProphetProfile:
    cat = MATERIAL_CATEGORY.get(material, "default")
    return PROPHET_PROFILES_BY_CATEGORY.get(cat, PROPHET_PROFILES_BY_CATEGORY["default"])


PAKISTAN_CONSTRUCTION_EVENTS: list[dict[str, str | int]] = [
    {"holiday": "ramadan_eid_slowdown", "ds": "2023-04-10", "lower_window": -14, "upper_window": 7},
    {"holiday": "ramadan_eid_slowdown", "ds": "2024-04-01", "lower_window": -14, "upper_window": 7},
    {"holiday": "ramadan_eid_slowdown", "ds": "2025-03-20", "lower_window": -14, "upper_window": 7},
    {"holiday": "ramadan_eid_slowdown", "ds": "2026-03-10", "lower_window": -14, "upper_window": 7},
    {"holiday": "punjab_smog_kiln_ban", "ds": "2023-11-15", "lower_window": -7, "upper_window": 21},
    {"holiday": "punjab_smog_kiln_ban", "ds": "2024-11-15", "lower_window": -7, "upper_window": 21},
    {"holiday": "punjab_smog_kiln_ban", "ds": "2025-11-15", "lower_window": -7, "upper_window": 21},
]

# ---------------------------------------------------------------------------
# Phase 1 — landed_cost_sim.csv parameters
# ---------------------------------------------------------------------------
NUM_ORDERS: int = 20_000
NUM_SUPPLIERS_FOR_ORDERS: int = 400
TRUCK_TYPES: list[str] = ["mini", "6-wheeler", "10-wheeler"]

TRUCK_BASE_FEE: dict[str, float] = {
    "mini": 3_000.0,
    "6-wheeler": 6_000.0,
    "10-wheeler": 9_500.0,
}
TRUCK_PER_KM_RATE: dict[str, float] = {
    "mini": 55.0,
    "6-wheeler": 85.0,
    "10-wheeler": 120.0,
}

URGENCY_LEVELS: list[str] = ["standard", "express"]
URGENCY_MARKUP_FRAC: dict[str, float] = {
    "standard": 0.0,
    "express": 0.12,
}

CITY_DISTANCES_FROM_LAHORE_KM: dict[str, float] = {
    "Lahore": 15.0,
    "Sheikhupura": 45.0,
    "Kasur": 55.0,
    "Gujranwala": 80.0,
    "Sialkot": 125.0,
    "Faisalabad": 130.0,
    "Sargodha": 190.0,
    "Gujrat": 155.0,
    "Multan": 340.0,
    "Rawalpindi": 275.0,
    "Islamabad": 375.0,
    "Peshawar": 510.0,
    "Karachi": 1_210.0,
    "Hyderabad": 1_150.0,
    "Quetta": 940.0,
    "Sukkur": 820.0,
    "Bahawalpur": 430.0,
    "Sahiwal": 180.0,
    "Okara": 120.0,
    "Dera Ghazi Khan": 520.0,
}

QUANTITY_RANGE_BY_MATERIAL: dict[str, tuple[float, float]] = {
    "Cement": (50, 2_000),
    "Steel": (1, 40),
    "Bricks": (5, 200),
    "Sand": (2, 60),
    "Aggregate": (2, 60),
    "Gravel": (2, 60),
    "Concrete Blocks": (5, 200),
    "Lime": (50, 1_500),
    "Wood": (50, 1_000),
    "Plywood": (20, 500),
    "Glass": (100, 5_000),
    "Marble": (100, 5_000),
    "Granite": (100, 5_000),
    "Paint": (10, 200),
    "Tiles": (100, 5_000),
    "Gypsum Board": (20, 500),
    "Wall Putty": (10, 300),
    "Pipes": (50, 1_000),
    "Sanitary Fittings": (5, 200),
    "Wires": (50, 1_000),
    "Electrical Switches": (50, 2_000),
    "Circuit Breakers": (10, 500),
    "Roofing Sheets": (10, 500),
    "Aluminum Profiles": (50, 2_000),
}

# ---------------------------------------------------------------------------
# Phase 1 — supplier_trust_sim.csv parameters
# ---------------------------------------------------------------------------
NUM_SUPPLIERS: int = 1_000
SUPPLIER_QUALITY_MEAN: float = 0.72
SUPPLIER_QUALITY_STD: float = 0.16
RELIABLE_ON_TIME_THRESHOLD: float = 0.60

# ---------------------------------------------------------------------------
# Phase 4 — smart-recommendation composite weights
# ---------------------------------------------------------------------------
RECOMMENDATION_WEIGHTS: dict[str, float] = {
    "w1_cost": 0.5,
    "w2_trust": 0.3,
    "w3_trend": 0.2,
}