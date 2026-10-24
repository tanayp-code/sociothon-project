# 🌱 Circl-E: AI E-Waste Diagnostic & Campus Urban-Mining Locker
**Event:** NSS IIT Roorkee presents Sociothon 2026  
**Theme:** Smart and Sustainable Waste Management  
**Track:** AI Identification + Waste-to-Resource + Circular Economy  

---

## 📌 Executive Summary
**Circl-E** is an AI-powered diagnostic and reverse-logistics platform that transforms discarded consumer electronics ("zombie tech") into circular resources for IIT Roorkee.

Instead of building cost-prohibitive custom hardware, Circl-E combines:
1. **Gemini Vision AI Diagnostic Engine:** Classifies gadgets, detects visible physical damage, and immediately flags fire hazards (swollen/bloated Li-ion batteries).
2. **Urban-Mining Mineral Valuation:** Calculates the recoverable precious metals (Gold, Silver, Palladium, Copper, Lithium) inside dead gadgets using verified teardown data.
3. **Data Security Protocol:** Eliminates the #1 reason students hoard dead phones by guiding them through mandatory data-wipe verification (FRP logout, factory reset, SD ejection).
4. **Campus Peer Swap Board:** Automatically triages functional items (working chargers, cables, tested 18650 cells) for peer exchange among robotics teams and hostel students, prioritizing reuse over destruction.
5. **Local Industrial Grounding:** Routes non-reusable e-waste to campus collection lockers which aggregate for bulk delivery to **Attero Recycling**—India's largest CPCB-authorised e-waste & battery refinery located right in **Bhagwanpur, Roorkee (20 km from campus)**.

---

## 🗃️ My Drawer (bulk inventory)
Scan many gadgets at once (or add manually), compare their metal value in a sorted table, toggle "works?" to route each item to swap or e-waste, and **export / import CSV** — for a student's room, a hostel wing, or a department store room.

## 🏗️ Architecture

```
                    [ Student Device Camera / File Upload ]
                                      │
                                      ▼
                        POST /api/scan (Image Bytes)
                                      │
                                      ▼
                  ┌───────────────────────────────────────┐
                  │       Gemini Flash Vision AI          │
                  │ • Classifies into 8 standard classes  │
                  │ • Checks visible damage & swelling    │
                  │ • Generates instant hazard warning    │
                  └───────────────────┬───────────────────┘
                                      │
                                      ▼
                      [ Triage: Powers On? Data Wiped? ]
                                      │
                        POST /api/assess (Class + State)
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
           [ Action: SWAP ]                   [ Action: DROPOFF ]
      • Item is functional               • Inherent metal value (₹)
      • Post to Campus Swap Board        • Au, Ag, Pd, Cu, Li breakdown
      • Free or peer price               • Mandatory data-wipe checklist
                                         • Route to campus dropoff bin
                                         • Issue Green Voucher QR
```

---

## 🚀 Quickstart & Local Installation

### Prerequisites
* Python 3.10+
* Google Gemini API Key (`GEMINI_API_KEY`)

### Setup Instructions
```bash
# 1. Clone repository & enter workspace
git clone <repo-url>
cd sociothon

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
echo "GEMINI_API_KEY=your_api_key_here" > .env

# 5. Run development server
uvicorn app.main:app --reload --port 8000
```
Open **`http://localhost:8000`** in any web or mobile browser.

---

## 🧪 CLI Image Classification
You can test the computer vision engine directly from the command line:
```bash
python -m ml.predict ml/test_images/phone.jpg
python -m ml.predict ml/test_images/swollen_battery.jpg
```

---

## 📊 API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/scan` | `POST` | Uploads image (multipart), returns predicted class, confidence, top3, damage, and hazard alerts. |
| `/api/assess` | `POST` | Receives class and triage answers; returns action (`dropoff` or `swap`), metal values, wipe checklist, and nearest bin. |
| `/api/dropoffs`| `GET` | Returns list of campus collection bins and Attero bulk refinery details. |
| `/api/swap` | `GET` | Returns active listings on the campus hardware swap board. |
| `/api/swap` | `POST` | Posts a new working gadget or salvaged component to the swap board. |
| `/api/stats` | `GET` | Returns cumulative scan counts, formal dropoffs, swaps, and diverted kilograms. |

---

## 📑 Verified Data Sources & Facts

* **National E-Waste Volume:** India generates **1.25–1.75 million tonnes** of e-waste annually; only **43–71%** is formally processed (Sources: *CPCB EPR Portal, Parliament Data, Down To Earth*).
* **Local Processing Partner:** **Attero Recycling Pvt. Ltd.** operates an authorized refinery processing ~144,000 t/yr e-waste + 15,000 t/yr Li-ion batteries in **Raipur Industrial Area, Bhagwanpur, Roorkee** (*Argus, Jul 2024*).
* **Smartphone Mineral Yields:**
  * Gold ($\text{Au}$): **24–36 mg** (valued at ₹188–₹282)
  * Silver ($\text{Ag}$): **90–250 mg** (valued at ₹8.5–₹23.7)
  * Palladium ($\text{Pd}$): **~9 mg** (valued at ~₹28.8)
  * Copper ($\text{Cu}$): **9–15 g** (valued at ₹6.7–₹11.2)
  * Lithium ($\text{Li}$): **~1.1 g**
  *(Teardown benchmarks: Plymouth University 2019, Remedia/PoliMi, Falconbridge via US EPA/USGS).*
* **Commodity Spot Rates (Oct 2026):**
  * Gold 24K: ₹7,850/g (*Paytm/Bullion spot*)
  * Silver 999: ₹95/g (*Groww spot*)
  * Palladium: ₹3,200/g (*Goldpriceindia spot*)
  * Copper Scrap: ₹750/kg (*Scrapyard/Mandi scrap rate*)
