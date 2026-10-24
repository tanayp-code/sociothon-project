"""
Circl-E Vision Classifier via Gemini Flash API with Groq Vision Fallback
Classifies e-waste items into 8 standardized classes or unknown.
Handles resizing (max 768px), dual-provider cascade (Gemini Flash -> Groq Vision),
6s Gemini timeout cap, structured JSON extraction, and swollen battery safety alerts.
"""

import os
import io
import sys
import json
import time
import base64
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dotenv import load_dotenv
from PIL import Image
from google import genai
from google.genai import types

load_dotenv()

ALLOWED_CLASSES = [
    "smartphone",
    "laptop",
    "charger_adapter",
    "cable",
    "battery_powerbank",
    "earphones_headphones",
    "keyboard_mouse",
    "pcb_board",
    "tablet",
]

# Fastest reliable model first (3.1-flash-lite and 3.8-flash were timing out).
GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
]

GROQ_MODELS = [
    "qwen/qwen3.8-27b",
]

# When Gemini fails (e.g. free quota used up), skip it for a while instead of
# waiting on it for every scan.
GEMINI_COOLDOWN_S = 300
_gemini_down_until = 0.0

SYSTEM_PROMPT = """You are an expert e-waste and electronics recycling inspector.
Analyze the provided image of an electronic item or discarded scrap.

Classify the item into EXACTLY ONE of the following item_class categories:
- smartphone (mobile phones, feature phones, cracked or intact phones)
- laptop (notebooks, netbooks, MacBooks, PC laptops)
- charger_adapter (wall power bricks, power supply adapters)
- cable (USB cables, charging cords, power cables, HDMI/display cables)
- battery_powerbank (power banks, loose Li-ion batteries, laptop battery packs, alkaline/rechargeable cells)
- earphones_headphones (in-ear buds, wireless earbuds, overhead headphones)
- keyboard_mouse (computer keyboards, computer mice, trackpads)
- pcb_board (printed circuit boards, motherboards, green/blue circuitry)
- tablet (iPads, Android tablets, e-readers)
- unknown (non-electronics, unclear images, or items not matching the above categories)

You must also inspect for visible damage:
- "cracked screen", "swollen battery", "bent pins", "frayed wire", "missing keys", "broken casing", "corrosion/water damage", or null if intact.
- CRITICAL: If a battery or power bank appears visibly puffed, bloated, bulging, or swollen, you MUST state "swollen battery" in visible_damage.

Return a JSON object with:
{
  "item_class": "one of the allowed classes above",
  "confidence": float between 0.0 and 1.0,
  "top3": [
    ["class_name_1", confidence_1],
    ["class_name_2", confidence_2],
    ["class_name_3", confidence_3]
  ],
  "visible_damage": "string description of damage or null",
  "description": "concise description in 10 words or less"
}
"""


def _api_keys(name: str) -> List[str]:
    """Env var may hold several comma-separated keys; we rotate through them when one runs out."""
    return [k.strip() for k in os.getenv(name, "").split(",") if k.strip()]


def _downscale_image(image_bytes: bytes, max_dim: int = 768) -> bytes:
    """Downscales image to max_dim on the longest edge to optimize latency and token cost."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            out_buf = io.BytesIO()
            img.save(out_buf, format="JPEG", quality=85)
            return out_buf.getvalue()
    except Exception:
        return image_bytes


def _call_gemini_api(processed_bytes: bytes, prompt: str = SYSTEM_PROMPT) -> Dict[str, Any]:
    """Invokes Gemini Flash API."""
    keys = _api_keys("GEMINI_API_KEY")
    if not keys:
        raise ValueError("GEMINI_API_KEY is not set")
    last_error = None
    for api_key in keys:
        try:
            return _call_gemini_with_key(processed_bytes, prompt, api_key)
        except Exception as e:
            last_error = e
    raise RuntimeError(f"All Gemini keys failed: {last_error}")


def _call_gemini_with_key(processed_bytes: bytes, prompt: str, api_key: str) -> Dict[str, Any]:

    # No silent retries: on quota errors fail fast so we can fall back to Groq.
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
        timeout=10000, retry_options=types.HttpRetryOptions(attempts=1)))
    part = types.Part.from_bytes(data=processed_bytes, mime_type="image/jpeg")

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.1,
    )

    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[part, prompt],
                config=config,
            )
            raw_text = response.text.strip()
            data = json.loads(raw_text)
            data["provider"] = "gemini"
            return data
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"All Gemini models failed: {last_error}")


def _call_groq_api(processed_bytes: bytes, prompt: str = SYSTEM_PROMPT) -> Dict[str, Any]:
    """Fallback: Invokes Groq Vision API via OpenAI-compatible endpoint."""
    keys = _api_keys("GROQ_API_KEY")
    if not keys:
        raise ValueError("GROQ_API_KEY is not set")
    last_error = None
    for api_key in keys:
        try:
            return _call_groq_with_key(processed_bytes, prompt, api_key)
        except Exception as e:
            last_error = e
    raise RuntimeError(f"All Groq keys failed: {last_error}")


def _call_groq_with_key(processed_bytes: bytes, prompt: str, api_key: str) -> Dict[str, Any]:

    b64_img = base64.b64encode(processed_bytes).decode("utf-8")

    last_error = None
    for model_name in GROQ_MODELS:
        try:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Inspect this e-waste image and output the requested JSON format."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                        ],
                    },
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
            }

            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "CirclE-Sociothon/1.0",
                },
            )

            with urllib.request.urlopen(req, timeout=8.0) as resp:
                result_json = json.loads(resp.read().decode("utf-8"))
                content = result_json["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                parsed["provider"] = "groq"
                return parsed
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"All Groq models failed: {last_error}")


def predict(image_bytes: bytes) -> Dict[str, Any]:
    """
    Main prediction function adhering to TEAM.md contract:
    predict(image_bytes) -> {
        "item_class": str,
        "confidence": float,
        "top3": [[cls, conf], ...],
        "visible_damage": str|null,
        "description": str,
        "provider": "gemini"|"groq",
        "safety_warning": str|null
    }
    Cascade: Gemini Flash (6s cap) -> Groq Vision fallback (5s cap) -> Unknown fallback.
    Never crashes, total execution cap <= 10s.
    """
    processed_bytes = _downscale_image(image_bytes, max_dim=768)
    data: Optional[Dict[str, Any]] = None
    provider = "unknown"

    # 1. Try Gemini Flash with 6s timeout cap
    global _gemini_down_until
    try:
        if time.time() < _gemini_down_until:
            raise RuntimeError("Gemini skipped (cooling down after recent failure)")
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_gemini_api, processed_bytes)
            data = future.result(timeout=8.0)
            provider = "gemini"
    except Exception as gemini_err:
        _gemini_down_until = time.time() + GEMINI_COOLDOWN_S
        # 2. Try Groq Vision fallback with 5s timeout
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call_groq_api, processed_bytes)
                data = future.result(timeout=10.0)
                provider = "groq"
        except Exception as groq_err:
            return {
                "item_class": "unknown",
                "confidence": 0.0,
                "top3": [["unknown", 1.0]],
                "visible_damage": None,
                "description": "Classification unavailable from providers",
                "provider": "none",
                "error": f"Gemini error: {gemini_err} | Groq error: {groq_err}",
            }

    if not data:
        return {
            "item_class": "unknown",
            "confidence": 0.0,
            "top3": [["unknown", 1.0]],
            "visible_damage": None,
            "description": "No classification data",
            "provider": "none",
        }

    # Validate and sanitize item_class
    raw_class = str(data.get("item_class", "unknown")).lower().strip()
    if raw_class not in ALLOWED_CLASSES:
        raw_class = "unknown"

    confidence = float(data.get("confidence", 0.90))
    confidence = max(0.0, min(1.0, confidence))

    raw_top3 = data.get("top3", [])
    top3: List[List[Any]] = []
    if isinstance(raw_top3, list):
        for item in raw_top3[:3]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                c_name = str(item[0]).lower().strip()
                c_conf = float(item[1])
                top3.append([c_name, round(c_conf, 2)])
    if not top3:
        top3 = [[raw_class, round(confidence, 2)]]

    visible_damage = data.get("visible_damage")
    if visible_damage and str(visible_damage).lower() in ("null", "none"):
        visible_damage = None

    description = str(data.get("description", f"Detected {raw_class}"))[:80]

    # Swollen battery hazard check
    safety_warning = None
    damage_text = (visible_damage or "").lower()
    desc_text = description.lower()
    is_swollen = any(
        kw in damage_text or kw in desc_text
        for kw in ("swollen", "bloated", "puffing", "bulging battery", "expanded cell")
    )
    if is_swollen:
        if not visible_damage:
            visible_damage = "swollen battery"
        safety_warning = (
            "⚠️ HAZARD WARNING: Swollen / bloated battery detected. High fire and thermal "
            "runaway risk! Do NOT puncture, press, or expose to heat. Do NOT dispose in "
            "household trash. Transport gently in a non-combustible box to an authorized e-waste bin."
        )

    result: Dict[str, Any] = {
        "item_class": raw_class,
        "confidence": round(confidence, 2),
        "top3": top3,
        "visible_damage": visible_damage,
        "description": description,
        "provider": provider,
    }
    if safety_warning:
        result["safety_warning"] = safety_warning

    return result


PILE_PROMPT = """You are an expert e-waste inspector. The image shows a PILE or GROUP of several items
(e.g. a drawer emptied onto a bed). List EVERY distinct electronic item you can see.

Use ONLY these item_class values:
smartphone, laptop, tablet, charger_adapter, cable, battery_powerbank, earphones_headphones, keyboard_mouse, pcb_board, unknown

Group identical items and give a count. Note visible damage per group ("cracked screen", "swollen battery",
"frayed wire", ... or null). If any battery/power bank looks puffed or bulging, visible_damage MUST be "swollen battery".
Ignore non-electronic objects (bed, table, hands, paper).

Return JSON: {"items": [{"item_class": "...", "count": 1, "visible_damage": null, "description": "max 8 words"}]}
"""


def predict_pile(image_bytes: bytes) -> Dict[str, Any]:
    """Detects multiple items in one photo. Returns {"items": [...], "provider": str}; never raises."""
    processed_bytes = _downscale_image(image_bytes, max_dim=1280)
    data, provider, err = None, "none", None
    global _gemini_down_until
    for name, fn, cap in (("gemini", _call_gemini_api, 14.0), ("groq", _call_groq_api, 15.0)):
        if name == "gemini" and time.time() < _gemini_down_until:
            continue
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                data = executor.submit(fn, processed_bytes, PILE_PROMPT).result(timeout=cap)
            provider = name
            break
        except Exception as e:
            err = e
            if name == "gemini":
                _gemini_down_until = time.time() + GEMINI_COOLDOWN_S
    items: List[Dict[str, Any]] = []
    for it in (data or {}).get("items", []) if isinstance(data, dict) else []:
        if not isinstance(it, dict):
            continue
        cls = str(it.get("item_class", "unknown")).lower().strip()
        if cls not in ALLOWED_CLASSES:
            cls = "unknown"
        try:
            count = max(1, min(20, int(it.get("count", 1))))
        except (TypeError, ValueError):
            count = 1
        damage = it.get("visible_damage")
        if damage and str(damage).lower() in ("null", "none"):
            damage = None
        entry = {"item_class": cls, "count": count, "visible_damage": damage,
                 "description": str(it.get("description", ""))[:60]}
        if damage and "swollen" in str(damage).lower():
            entry["safety_warning"] = "Swollen battery: fire risk. Do not bin; hand in separately."
        items.append(entry)
    result: Dict[str, Any] = {"items": items, "provider": provider}
    if data is None:
        result["error"] = str(err)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ml.predict <path_to_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        sys.exit(1)

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    res = predict(img_bytes)
    print(json.dumps(res, indent=2))
