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


# Total copper cross-section (mm²) by cable type — engineering estimates, shown to the user as assumptions.
CABLE_COPPER_MM2 = {
    "usb_data": (0.2, 0.5),
    "power_cord": (1.5, 2.5),
    "hdmi_video": (0.4, 0.8),
    "audio": (0.1, 0.3),
}
COPPER_DENSITY_G_PER_CM3 = 8.96
LITHIUM_G_PER_AH = 0.3  # rule of thumb for Li-ion cells (see claude_research.md)


def _num(details: Dict[str, Any], key: str) -> Optional[float]:
    try:
        v = float(details.get(key))
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def resolve_details(item_class: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Only accepts values from catalog.json; anything else counts as 'Other / local brand'."""
    catalog = load_json("catalog.json")
    section = catalog.get(item_class, {})
    brands = [b for b in section if not b.startswith("_")] or section.get("_brands", [])
    out: Dict[str, Any] = {}
    brand = raw.get("brand")
    if brand in brands:
        out["brand"] = brand
        model = raw.get("model")
        specs = section.get(brand, {}).get(model) if isinstance(section.get(brand), dict) else None
        if specs:
            out["model"] = model
            out.update(specs)  # weight_g / battery_mah come from the catalog, never from the client
    else:
        out["local"] = True
    if item_class == "cable":
        if raw.get("cable_type") in CABLE_COPPER_MM2:
            out["cable_type"] = raw["cable_type"]
        try:
            length = float(raw.get("cable_length_m"))
            if length in catalog.get("cable_lengths_m", []):
                out["cable_length_m"] = length
        except (TypeError, ValueError):
            pass
    if item_class == "battery_powerbank":
        try:
            mah = int(float(raw.get("battery_mah")))
            if mah in section.get("_capacities_mah", []):
                out["battery_mah"] = mah
        except (TypeError, ValueError):
            pass
    return out


def refine_materials(item_class: str, item_info: Dict[str, Any], details: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Adjusts the generic per-item figures using what we know about this exact item."""
    info = dict(item_info)
    basis: List[str] = []
    if item_class == "cable":
        length = _num(details, "cable_length_m") or 1.0
        ctype = details.get("cable_type") if details.get("cable_type") in CABLE_COPPER_MM2 else "usb_data"
        lo, hi = CABLE_COPPER_MM2[ctype]
        if details.get("local"):
            # Unbranded cables often use thinner wire or copper-clad aluminium (assumption: about half the copper).
            lo, hi = round(lo * 0.5, 2), round(hi * 0.5, 2)
        # mm² × m = cm³
        info["copper_g"] = [round(length * lo * COPPER_DENSITY_G_PER_CM3, 1), round(length * hi * COPPER_DENSITY_G_PER_CM3, 1)]
        basis.append(f"Copper = {length:g} m × {lo}–{hi} mm² ({ctype.replace('_', ' ')}) × 8.96 g/cm³"
                     + ("" if _num(details, "cable_length_m") else " (length assumed 1 m)")
                     + (" — local brand: assumed thinner / copper-clad wire" if details.get("local") else ""))
    mah = _num(details, "battery_mah")
    if mah:
        info["lithium_g"] = round(mah / 1000.0 * LITHIUM_G_PER_AH, 2)
        basis.append(f"Lithium ≈ {mah:g} mAh × 0.3 g per Ah")
    weight = _num(details, "weight_g")
    typical = item_info.get("typical_weight_g")
    if weight and typical and item_class != "cable":
        ratio = max(0.3, min(3.0, weight / typical))
        for k in ("gold_mg", "silver_mg", "palladium_mg", "copper_g"):
            v = info.get(k)
            if isinstance(v, list):
                info[k] = [round(x * ratio, 1) for x in v]
            elif isinstance(v, (int, float)):
                info[k] = round(v * ratio, 1)
        basis.append(f"Metals scaled by weight: {weight:g} g vs typical {typical:g} g (×{ratio:.2f})")
    if details.get("local") and item_class != "cable":
        basis.append("Other / local brand: generic figures for a typical " + item_class.replace("_", " "))
    return info, basis


def assess_item(item_class: str, powers_on: bool, data_wiped: bool, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
    details = resolve_details(item_class, details or {})
    item_info, estimate_basis = refine_materials(item_class, item_info, details)
    if not estimate_basis or (len(estimate_basis) == 1 and estimate_basis[0].startswith("Other / local")):
        estimate_basis = estimate_basis or ["Generic figures for a typical " + item_class.replace("_", " ") + ". Add details for a sharper estimate."]

    # Action triage: swap if functional, dropoff if dead/scrap
    action = "swap" if powers_on else "dropoff"
    # A working device that still holds personal data must be wiped before it changes hands.
    wipe_before_swap = action == "swap" and stores_data and not data_wiped

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
        "wipe_before_swap": wipe_before_swap,
        "estimate_precision": "refined" if any(details.get(k) for k in ("weight_g", "battery_mah", "cable_length_m", "cable_type")) else "generic",
        "local_brand": bool(details.get("local")),
        "estimate_basis": estimate_basis,
        "brand": details.get("brand"),
        "model": details.get("model"),
        "nearest_dropoff": nearest_dropoff,
        "sources": sources,
    }
