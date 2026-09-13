"""
adapter.py

Translation layer between SmartBuild's Prisma schema (real app data) and
the AI Engine's trained feature space (config.MATERIALS, TRUCK_TYPES,
CITY_DISTANCES_FROM_LAHORE_KM). Nothing here touches the database --
Next.js queries Prisma and sends raw field values over HTTP; this module
only maps those raw strings/numbers onto the vocabulary the models were
trained on.

Three mappings are needed because the schema and the training vocabulary
don't share a taxonomy:
1. Category.name (free text)      -> one of config.MATERIALS (fuzzy match)
2. RiderProfile.vehicleType (free text) -> one of config.TRUCK_TYPES (keyword match)
3. Order.deliveryCity / SupplierLocation.city -> distance_km (static lookup)

Supplier reliability (dispute_count, rating, on-time rate) is NOT mapped
here -- compute that in Next.js via Prisma (Dispute count per supplier,
Dispatch.deliveredAt vs promisedDeliveryDate) and send the resulting
numbers directly; the AI Engine has no way to know that without DB access.
"""

from __future__ import annotations

import difflib
from typing import Literal

import config

# ---------------------------------------------------------------------------
# 1. Category name -> material
# ---------------------------------------------------------------------------
def map_category_to_material(category_name: str) -> str:
    """Match a free-text Category.name to a config.MATERIALS entry.

    Three tiers, most reliable first:
    1. Exact match (case-insensitive).
    2. Substring containment (case-insensitive) -- catches real-world
       names like "Portland Cement" or "GI Roofing Sheets" that a naive
       whole-string fuzzy ratio scores too low on due to length mismatch.
    3. difflib fuzzy ratio, for near-miss spelling/wording only.

    Raises ValueError (not a silent fallback) if nothing matches well
    enough -- a wrong material guess silently corrupts a cost prediction,
    so the caller (Next.js) should surface this as "please map this
    category to a material" rather than have it fail invisibly.
    """
    lowered = category_name.strip().lower()

    for material in config.MATERIALS:
        if material.lower() == lowered:
            return material

    for material in config.MATERIALS:
        m = material.lower()
        if m in lowered or lowered in m:
            return material

    matches = difflib.get_close_matches(category_name, config.MATERIALS, n=1, cutoff=0.6)
    if matches:
        return matches[0]

    raise ValueError(
        f"No material match for category '{category_name}'. "
        f"Known materials: {config.MATERIALS}"
    )


# ---------------------------------------------------------------------------
# 2. Vehicle type -> truck_type bucket
# ---------------------------------------------------------------------------
# Ordered by specificity -- checked top to bottom, first match wins.
_VEHICLE_KEYWORDS: list[tuple[str, Literal["mini", "6-wheeler", "10-wheeler"]]] = [
    ("10-wheel", "10-wheeler"),
    ("10 wheel", "10-wheeler"),
    ("trailer", "10-wheeler"),
    ("flatbed", "10-wheeler"),
    ("6-wheel", "6-wheeler"),
    ("6 wheel", "6-wheeler"),
    ("truck", "6-wheeler"),
    ("mini", "mini"),
    ("pickup", "mini"),
    ("suzuki", "mini"),
    ("van", "mini"),
]
_DEFAULT_TRUCK_TYPE: Literal["mini", "6-wheeler", "10-wheeler"] = "6-wheeler"


def map_vehicle_to_truck_type(vehicle_type: str) -> str:
    """Keyword-match RiderProfile.vehicleType (free text) to a truck_type bucket.

    Unlike map_category_to_material(), this defaults rather than raises --
    a mis-bucketed vehicle still gives a reasonable freight estimate (the
    default is the middle tier), whereas a wrong material would corrupt
    the whole prediction. Log the fallback on the caller's side if you
    want visibility into how often this happens.
    """
    lowered = vehicle_type.lower()
    for keyword, truck_type in _VEHICLE_KEYWORDS:
        if keyword in lowered:
            return truck_type
    return _DEFAULT_TRUCK_TYPE


# ---------------------------------------------------------------------------
# 3. City -> distance_km
# ---------------------------------------------------------------------------
def get_distance_km(city_name: str) -> float:
    """Look up distance_km for a delivery city.

    Case-insensitive exact match against config.CITY_DISTANCES_FROM_LAHORE_KM
    first; falls back to a fuzzy match (handles minor spelling variants
    like "Sheikhupura" vs "shekhupura"); raises if nothing is close enough,
    since a wrong distance silently skews freight_cost.
    """
    known_cities = list(config.CITY_DISTANCES_FROM_LAHORE_KM.keys())
    for city in known_cities:
        if city.lower() == city_name.strip().lower():
            return config.CITY_DISTANCES_FROM_LAHORE_KM[city]

    matches = difflib.get_close_matches(city_name, known_cities, n=1, cutoff=0.7)
    if matches:
        return config.CITY_DISTANCES_FROM_LAHORE_KM[matches[0]]

    raise ValueError(
        f"No distance mapping for city '{city_name}'. "
        f"Known cities: {known_cities}. Add it to "
        f"config.CITY_DISTANCES_FROM_LAHORE_KM."
    )