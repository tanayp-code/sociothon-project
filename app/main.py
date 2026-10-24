"""
Circl-E FastAPI Application
Backend serving prediction, valuation/assessment, swap board, dropoffs, stats, and static frontend.
"""

import os
import json
import time
import uuid
import hashlib
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ml.predict import predict, predict_pile
from app.valuation import assess_item, load_json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATS_FILE = os.path.join(DATA_DIR, "stats.json")
SWAP_FILE = os.path.join(DATA_DIR, "swap.json")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="Circl-E API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_stats() -> Dict[str, Any]:
    if not os.path.exists(STATS_FILE):
        initial = {
            "scans_count": 0,
            "dropoffs_count": 0,
            "swaps_count": 0,
            "kg_diverted": 0.0,
            "kg_diverted_is_estimate": True,
        }
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(initial, f, indent=2)
        return initial
    try:
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "scans_count": 0,
            "dropoffs_count": 0,
            "swaps_count": 0,
            "kg_diverted": 0.0,
            "kg_diverted_is_estimate": True,
        }


def update_stats(scan_inc: int = 0, dropoff_inc: int = 0, swap_inc: int = 0, kg_inc: float = 0.0) -> None:
    stats = get_stats()
    stats["scans_count"] = stats.get("scans_count", 0) + scan_inc
    stats["dropoffs_count"] = stats.get("dropoffs_count", 0) + dropoff_inc
    stats["swaps_count"] = stats.get("swaps_count", 0) + swap_inc
    current_kg = stats.get("kg_diverted", 0.0) + kg_inc
    stats["kg_diverted"] = round(current_kg, 2)
    stats["kg_diverted_is_estimate"] = True
    try:
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"Error persisting stats: {e}")


class AssessRequest(BaseModel):
    item_class: str
    powers_on: bool = False
    data_wiped: bool = False
    preview: bool = False  # My Drawer valuations: don't count as real drop-offs/swaps


class SwapListingCreate(BaseModel):
    title: str = Field(..., min_length=2, max_length=120)
    category: str
    condition: str
    powers_on: bool = True
    description: str = Field(..., max_length=500)
    price_inr: int = Field(0, ge=0)
    contact: str = Field(..., min_length=2, max_length=100)


# Same photo -> same answer, instantly and without spending API quota.
SCAN_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "scan_cache.json")
try:
    with open(SCAN_CACHE_PATH) as f:
        _scan_cache: Dict[str, Any] = json.load(f)
except Exception:
    _scan_cache = {}


def cached_pile(contents: bytes) -> Dict[str, Any]:
    key = hashlib.sha256(contents).hexdigest()
    if key in _scan_cache:
        return {**_scan_cache[key], "cached": True}
    result = predict_pile(contents)
    if result.get("items"):  # never cache failures
        _scan_cache[key] = result
        try:
            with open(SCAN_CACHE_PATH, "w") as f:
                json.dump(_scan_cache, f)
        except Exception:
            pass
    return result


@app.post("/api/scan_pile")
async def scan_pile(file: UploadFile = File(...)) -> Dict[str, Any]:
    """One photo of many items -> list of detected items with counts."""
    result = cached_pile(await file.read())
    update_stats(scan_inc=sum(i["count"] for i in result["items"]))
    return result


@app.post("/api/scan")
async def scan_item(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Receives image upload, executes ml.predict, and logs scan stats."""
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        # One AI call per photo: the pile scan also gives us the main item.
        pile = cached_pile(contents)
        items = pile.get("items", [])
        if items:
            main = max(items, key=lambda i: (i["item_class"] != "unknown", i["item_class"] in ("laptop", "smartphone")))
            result = {"item_class": main["item_class"], "confidence": 0.9, "top3": [[main["item_class"], 0.9]],
                      "visible_damage": main.get("visible_damage"), "description": main.get("description", ""),
                      "provider": pile.get("provider"), "items": items}
            if any(i.get("safety_warning") for i in items):
                result["safety_warning"] = next(i["safety_warning"] for i in items if i.get("safety_warning"))
        else:
            result = {"item_class": "unknown", "confidence": 0.0, "top3": [["unknown", 1.0]], "visible_damage": None,
                      "description": "Could not identify the item. Pick it from the list.",
                      "provider": pile.get("provider"), "items": [], "error": pile.get("error")}
        update_stats(scan_inc=1)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image scan failed: {str(e)}")


@app.post("/api/assess")
def assess_endpoint(req: AssessRequest) -> Dict[str, Any]:
    """Assesses item value, wipe checklist, and dropoff/swap action."""
    assessment = assess_item(req.item_class, req.powers_on, req.data_wiped)

    # Estimate diverted kg based on typical weights
    materials = load_json("materials.json")
    item_meta = materials.get(req.item_class, {})
    typical_weight_g = item_meta.get("typical_weight_g") or 100.0
    kg_added = round(typical_weight_g / 1000.0, 3)

    if req.preview:
        return assessment
    if assessment["action"] == "swap":
        update_stats(swap_inc=1, kg_inc=kg_added)
    else:
        update_stats(dropoff_inc=1, kg_inc=kg_added)

    return assessment


@app.get("/api/dropoffs")
def get_dropoffs() -> List[Dict[str, Any]]:
    """Returns authorized campus e-waste bins and partner refinery."""
    return load_json("dropoffs.json")


@app.get("/api/swap")
def get_swap_listings() -> List[Dict[str, Any]]:
    """Returns active campus hardware swap listings."""
    return load_json("swap.json")


@app.post("/api/swap")
def create_swap_listing(listing: SwapListingCreate) -> Dict[str, Any]:
    """Posts a new item onto the campus swap board."""
    current_listings = load_json("swap.json")
    if not isinstance(current_listings, list):
        current_listings = []

    new_entry = {
        "id": f"swap-{str(uuid.uuid4())[:8]}",
        "title": listing.title,
        "category": listing.category,
        "condition": listing.condition,
        "powers_on": listing.powers_on,
        "description": listing.description,
        "price_inr": listing.price_inr,
        "contact": listing.contact,
        "posted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    current_listings.insert(0, new_entry)
    with open(SWAP_FILE, "w", encoding="utf-8") as f:
        json.dump(current_listings, f, indent=2)

    return new_entry


@app.get("/api/stats")
def get_stats_endpoint() -> Dict[str, Any]:
    """Returns cumulative scan, dropoff, swap, and diverted e-waste stats."""
    return get_stats()


# Mount static files at root
if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
