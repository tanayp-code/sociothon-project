"""
Valuation and Assessment Engine for Circl-E
Calculates inherent metal values based on materials.json and prices.json.
Determines action (dropoff vs swap), generates data wipe checklists, and provides nearest dropoff info.
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_json(filename: str) -> Any:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_metal_value(materials: Dict[str, Any], prices: Dict[str, Any]) -> Tuple[float, float]:
    """
    metal_value_inr = sum(grams * price) for min and max ranges.
    Returns (min_inr, max_inr) rounded to 2 decimals.
    """
    rates = prices.get("rates", {})
    gold_price_per_g = rates.get("gold_inr_per_g", 7850.0)
    silver_price_per_g = rates.get("silver_inr_per_g", 95.0)
    pd_price_per_g = rates.get("palladium_inr_per_g", 3200.0)
    copper_price_per_kg = rates.get("copper_inr_per_kg", 750.0)
    copper_price_per_g = copper_price_per_kg / 1000.0

    def parse_range(val: Any) -> Tuple[float, float]:
        if val is None:
            return (0.0, 0.0)
        if isinstance(val, (int, float)):
            return (float(val), float(val))
        if isinstance(val, list) and len(val) >= 2:
            return (float(val[0]), float(val[1]))
        return (0.0, 0.0)

    # Gold (mg -> g)
    gold_min_mg, gold_max_mg = parse_range(materials.get("gold_mg"))
    gold_min_val = (gold_min_mg / 1000.0) * gold_price_per_g
    gold_max_val = (gold_max_mg / 1000.0) * gold_price_per_g

    # Silver (mg -> g)
    silver_min_mg, silver_max_mg = parse_range(materials.get("silver_mg"))
    silver_min_val = (silver_min_mg / 1000.0) * silver_price_per_g
    silver_max_val = (silver_max_mg / 1000.0) * silver_price_per_g

    # Palladium (mg -> g)
    pd_min_mg, pd_max_mg = parse_range(materials.get("palladium_mg"))
    pd_min_val = (pd_min_mg / 1000.0) * pd_price_per_g
    pd_max_val = (pd_max_mg / 1000.0) * pd_price_per_g

    # Copper (g)
    cu_min_g, cu_max_g = parse_range(materials.get("copper_g"))
    cu_min_val = cu_min_g * copper_price_per_g
    cu_max_val = cu_max_g * copper_price_per_g

    total_min = round(gold_min_val + silver_min_val + pd_min_val + cu_min_val, 2)
    total_max = round(gold_max_val + silver_max_val + pd_max_val + cu_max_val, 2)

    return (total_min, total_max)


def assess_item(item_class: str, powers_on: bool, data_wiped: bool) -> Dict[str, Any]:
    """
    Implements POST /api/assess contract from TEAM.md:
    returns item_class, action, materials, metal_value_inr, recovery_note,
    prices_as_of, wipe_checklist, nearest_dropoff, sources.
    """
    materials_catalog = load_json("materials.json")
    prices_data = load_json("prices.json")
    dropoffs = load_json("dropoffs.json")

    item_info = materials_catalog.get(item_class, materials_catalog.get("unknown", {}))
    stores_data = item_info.get("stores_data", False)

    # Action triage: swap if functional, dropoff if dead/scrap
    action = "swap" if powers_on else "dropoff"

    # Materials payload
    materials_payload = {
        "gold_mg": item_info.get("gold_mg"),
        "silver_mg": item_info.get("silver_mg"),
        "palladium_mg": item_info.get("palladium_mg"),
        "copper_g": item_info.get("copper_g"),
        "lithium_g": item_info.get("lithium_g"),
    }

    # Metal value calculation
    min_val, max_val = calculate_metal_value(item_info, prices_data)

    # Wipe checklist
    wipe_checklist: List[str] = []
    if stores_data and not data_wiped:
        wipe_checklist = [
            "Sign out of Google / Apple / Microsoft accounts to remove Factory Reset Protection (FRP).",
            "Back up personal media, documents, and credentials to cloud or external storage.",
            "Trigger a full Factory Data Reset from Settings (encrypt & erase storage).",
            "Physically eject all SIM cards and MicroSD storage cards before dropping off.",
            "If device is unbootable/dead: Request certified physical flash destruction at the collection hub.",
        ]

    # Nearest dropoff (first campus non-bulk location)
    nearest_dropoff = None
    if isinstance(dropoffs, list):
        for d in dropoffs:
            if not d.get("bulk_only"):
                nearest_dropoff = d
                break

    # Sources list
    sources = []
    if item_info.get("source"):
        sources.append(f"Materials: {item_info.get('source')}")
    if prices_data.get("sources"):
        for metal, url in prices_data["sources"].items():
            sources.append(f"{metal.capitalize()} price: {url}")

    return {
        "item_class": item_class,
        "action": action,
        "materials": materials_payload,
        "metal_value_inr": [min_val, max_val],
        "recovery_note": "recyclers recover only part of this value",
        "prices_as_of": prices_data.get("as_of", "2026-10-24"),
        "is_indicative": prices_data.get("is_indicative", True),
        "wipe_checklist": wipe_checklist,
        "nearest_dropoff": nearest_dropoff,
        "sources": sources,
    }
