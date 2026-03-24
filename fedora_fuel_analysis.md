# Fedora Fuel ML — Full System Audit Report
> Conducted: 2026-03-24 | Scope: All source files, UI, ML Model, Architecture

---

## 1. System Overview

Fedora Fuel ML is a **FastAPI + SARIMAX** fuel price forecasting web application targeting Kenya (KES).  
It has a **two-page Jinja2 frontend** (main engine + admin console), a REST API backend, and a pre-trained SARIMAX model persisted via `joblib`.

```
src/fuel_pricing/
├── api/
│   ├── main.py        ← FastAPI app, all routes, upload, train, predict logic
│   ├── auth.py        ← JWT + bcrypt authentication
│   ├── static/css/style.css
│   └── templates/index.html + admin.html
├── core/config.py     ← Path constants
├── data/loader.py     ← CSV loader (minimal)
├── ml/sarimax_model.py ← Core SARIMAX model class
├── optimization/pricing.py ← Stub: apply_cap()
└── pipelines/predict_pipeline.py ← Run prediction wrapper
```

---

## 2. 🐛 Bugs & Mistakes Found

### 2.1 [loader.py](file:///home/kiriinya/Antigravity/Projects/Fedora_Fuel_ML/src/fuel_pricing/data/loader.py) — Typo in Docstring
**File:** [src/fuel_pricing/data/loader.py](file:///home/kiriinya/Antigravity/Projects/Fedora_Fuel_ML/src/fuel_pricing/data/loader.py), Line 6
```diff
- Loads CSV and prepares it for SARI0MAX.
+ Loads CSV and prepares it for SARIMAX.
```
**Severity:** Low — cosmetic, but signals careless documentation.

---

### 2.2 [sarimax_model.py](file:///home/kiriinya/Antigravity/Projects/Fedora_Fuel_ML/src/fuel_pricing/ml/sarimax_model.py) — Relative `MODEL_PATH` (Critical Bug)
**File:** [src/fuel_pricing/ml/sarimax_model.py](file:///home/kiriinya/Antigravity/Projects/Fedora_Fuel_ML/src/fuel_pricing/ml/sarimax_model.py), Lines 30–31
```python
MODEL_PATH = Path("data/processed/sarimax_model.pkl")
METRICS_PATH = Path("data/processed/metrics.pkl")
```
These are **relative paths**, so they resolve relative to the **current working directory** at runtime — which is unpredictable. If the server is launched from a different directory than the project root (e.g., inside Docker, via `gunicorn`, or pytest), `MODEL_PATH.exists()` silently returns `False` and training/prediction fails with a misleading error.

**Fix:** Use the same absolute-path pattern as [config.py](file:///home/kiriinya/Antigravity/Projects/Fedora_Fuel_ML/src/fuel_pricing/core/config.py):
```python
from pathlib import Path
_HERE = Path(__file__).resolve().parents[3]
MODEL_PATH = _HERE / "data/processed/sarimax_model.pkl"
METRICS_PATH = _HERE / "data/processed/metrics.pkl"
```

---

### 2.3 [sarimax_model.py](file:///home/kiriinya/Antigravity/Projects/Fedora_Fuel_ML/src/fuel_pricing/ml/sarimax_model.py) — Commented-out Metrics Save
**File:** Lines 201–202
```python
# joblib.dump(metrics, METRICS_PATH)
```
This dead code left the original save path commented out while the actual save (line 204) uses a different expression. This is confusing and invites regression if the live line is accidentally deleted.

---

### 2.4 [sarimax_model.py](file:///home/kiriinya/Antigravity/Projects/Fedora_Fuel_ML/src/fuel_pricing/ml/sarimax_model.py) — [metrics](file:///home/kiriinya/Antigravity/Projects/Fedora_Fuel_ML/src/fuel_pricing/ml/sarimax_model.py#115-133) dict spacing
**File:** Line 42
```python
self.metrics ={} # ← missing space before {
```
Minor PEP 8 violation but signals inconsistent formatting.

---

### 2.5 [main.py](file:///home/kiriinya/Antigravity/Projects/Fedora_Fuel_ML/src/fuel_pricing/api/main.py) — Import block broken across file
**File:** [main.py](file:///home/kiriinya/Antigravity/Projects/Fedora_Fuel_ML/src/fuel_pricing/api/main.py), Line 197–198
```python
from fuel_pricing.core.config import PROCESSED_DIR
import os
```
These imports appear **mid-file** after the upload route, instead of at the top with all other imports. This violates PEP 8 and makes it harder to audit what the module depends on.

---

### 2.6 [main.py](file:///home/kiriinya/Antigravity/Projects/Fedora_Fuel_ML/src/fuel_pricing/api/main.py) — Hardcoded Admin Credentials
**File:** Lines 331–332
```python
correct_username = secrets.compare_digest(credentials.username, "admin")
correct_password = secrets.compare_digest(credentials.password, "admin123")
```
Credentials are **hardcoded strings** instead of reading from [.env](file:///home/kiriinya/Antigravity/Projects/Fedora_Fuel_ML/.env). The [.env.example](file:///home/kiriinya/Antigravity/Projects/Fedora_Fuel_ML/.env.example) defines `ADMIN_USERNAME` and `ADMIN_PASSWORD`, but [main.py](file:///home/kiriinya/Antigravity/Projects/Fedora_Fuel_ML/src/fuel_pricing/api/main.py) never uses `os.getenv()` for these. This is a **security vulnerability** — any deployment using these defaults is exposed.

**Fix:**
```python
import os
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
```

---

### 2.7 [auth.py](file:///home/kiriinya/Antigravity/Projects/Fedora_Fuel_ML/src/fuel_pricing/api/auth.py) — `datetime.utcnow()` Deprecation
**File:** Lines 57, 59
```python
expire = datetime.utcnow() + expires_delta
expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
```
`datetime.utcnow()` is **deprecated in Python 3.12+**. Should use `datetime.now(timezone.utc)`.

**Fix:**
```python
from datetime import datetime, timedelta, timezone
expire = datetime.now(timezone.utc) + expires_delta
```

---

### 2.8 [auth.py](file:///home/kiriinya/Antigravity/Projects/Fedora_Fuel_ML/src/fuel_pricing/api/auth.py) — Fake In-Memory User "Database"
**File:** Lines 132–147
The system has a single hardcoded `"admin"` user with `"admin123"` as the default password, stored entirely in-memory. There is no persistence, no user management, and no way to change passwords without redeploying. `sqlalchemy` and `psycopg2-binary` are in [requirements.txt](file:///home/kiriinya/Antigravity/Projects/Fedora_Fuel_ML/requirements.txt) but **never used**.

---

### 2.9 [cpi_monthly_clean.csv](file:///home/kiriinya/Antigravity/Projects/Fedora_Fuel_ML/data/processed/market_indicators/cpi_monthly_clean.csv) — Empty Data File
**File:** [data/processed/market_indicators/cpi_monthly_clean.csv](file:///home/kiriinya/Antigravity/Projects/Fedora_Fuel_ML/data/processed/market_indicators/cpi_monthly_clean.csv)
```
date,month
(empty)
```
This file has a header only with **zero data rows**. The [get_training_data()](file:///home/kiriinya/Antigravity/Projects/Fedora_Fuel_ML/src/fuel_pricing/api/main.py#200-263) merger will process it silently (no data merged, no error), but it's misleading — the CPI feature is wired but absent, degrading model accuracy.

---

### 2.10 `data/external/` — Completely Empty
The `data/external/` directory is empty. `test_model.py` (line 22) imports from `data/external/sample_fuel_data.csv` which doesn't exist — **all tests will crash immediately** with a `FileNotFoundError`.

---

### 2.11 `predict_pipeline.py` — Entirely Dead Code at Top
**File:** Lines 1–18
The first 18 lines are a fully commented-out older implementation of the pipeline. This should be removed — it pollutes the file and could confuse anyone maintaining it.

---

### 2.12 `optimization/pricing.py` — Stub Never Called
```python
def apply_cap(predicted_price: float, cap: float):
    """Enforce regulatory price cap."""
    return min(predicted_price, cap)
```
This stub exists but is **never called anywhere** in the active codebase. The predict endpoint returns raw SARIMAX output with no regulatory cap applied. The feature is wired in the commented-out pipeline (lines 1–18 predict_pipeline.py) but disconnected.

---

### 2.13 `main.py` — `get_training_data()` Uses First Upload Only
**File:** Lines 205–211
```python
csv_files = list(UPLOAD_DIR.glob("*.csv"))
if csv_files:
    df = pd.read_csv(csv_files[0])  # ← always takes first file
```
If multiple CSVs are uploaded, only the first (in filesystem order, not upload order) is used — silently. There's no feedback to the user about this. Upload order is non-deterministic.

---

### 2.14 `index.html` — Duplicate `display` Attribute
**File:** Line 97
```html
<div id="results" class="results" style="display: none; flex: 1; display: flex; ...">
```
`display` is defined **twice** inline — `display: none` then `display: flex`. The second declaration wins, meaning the `results` div is **never actually hidden** on initial render. The placeholder logic is broken — both divs are visible simultaneously until JavaScript overrides them.

---

### 2.15 `index.html` — `autoGenerateDefault()` Silently Trains on Page Load
**File:** Lines 376–410
Every time a user visits the homepage without a cached prediction, the app **automatically fires `/train/` then `/predict/`** in the background. This means:
- Cold-start training (~30s) blocks every fresh visitor 
- No user consent for the action
- If training fails, the user sees an error with no way to retry
- The status bar says `"Engine Trained"` even if the user did nothing

---

### 2.16 `admin.html` — `clearData()` is a No-Op
**File:** Lines 205–207
```javascript
function clearData() {
    alert('⚠️ Database purging must be performed securely via Docker Volume...');
}
```
The "💥 Purge Data" button does absolutely **nothing** — it shows an alert telling the user to do it manually. This is a dead UI action that erodes trust.

---

### 2.17 `admin.html` — File Status Always Shows "Verified"
**File:** Line 66
```html
<td><span class="badge badge-success">Verified</span></td>
```
Every file is shown as "Verified" regardless of whether it passed any actual validation. Files that fail column checks are deleted before reaching the admin, so the admin table is technically correct — but shows zero useful metadata (size, upload date, row count).

---

### 2.18 `style.css` — `transition` on Every Element
**File:** Line 42
```css
* {
    transition: background-color 0.4s ease, border-color 0.4s ease;
}
```
Applying `transition` to the universal selector `*` is a **performance anti-pattern**. It forces the browser to compute transitions on every single DOM element on every style change, including during layout and scroll.

---

### 2.19 `requirements.txt` — Unpinned Dependencies
```
fastapi
pandas
numpy
...
```
Zero version pins. Any `pip install` will grab the latest versions, making builds **non-reproducible**. `pulp` and `apscheduler` are listed but **never imported anywhere in the codebase**.

---

### 2.20 `pyproject.toml` — Incomplete
Missing: `requires-python`, `dependencies`, `description`, `authors`, `license`. Only 9 lines with bare-minimum setuptools config. Using both `requirements.txt` and `pyproject.toml` without synchronizing them is ambiguous.

---

### 2.21 `.gitignore` Conflicts with `data/processed/*.pkl`
```gitignore
data/processed/*.pkl   # line 44
*.pkl                  # line 65
```
Line 44 ignores `.pkl` in `data/processed/`, and line 65 ignores **all** `.pkl` files globally — making line 44 redundant. More importantly, the trained model (`sarimax_model.pkl`) is **never committed to git**, meaning any fresh clone of the repo has no model and must immediately retrain.

---

## 3. 🎨 UI — Strengths & Improvements

### ✅ What Works Well
- Glassmorphism cards with backdrop-filter look premium
- Animated gradient background (`gradientBG`) is visually striking
- Dark/light theme toggle with `localStorage` persistence
- Spinner + loading overlay on async actions
- Status badge with pulsing indicator is intuitive

### ❌ UI Improvements Needed

| # | Issue | Recommendation |
|---|-------|---------------|
| 1 | **No favicon** — browser shows blank icon tab | Add a fuel-themed SVG favicon |
| 2 | **No meta description** — poor SEO | Add `<meta name="description" ...>` |
| 3 | **No Google Fonts fallback** — offline usage breaks | Add `font-family: 'Outfit', Inter, sans-serif` fallback |
| 4 | **Model metrics not shown** — after training, MAE/RMSE/MAPE are saved to `.pkl` but never surfaced in the UI | Expose a `/metrics/` endpoint and display in admin panel |
| 5 | **Forecast data shown as raw `<pre>` text** — not a table | Replace `<pre>` output with a styled table: Month, Projected Price (KES), Δ Change |
| 6 | **No confidence intervals on chart** — SARIMAX naturally produces confidence bands | Add upper/lower bound shaded area to the Chart.js line |
| 7 | **"Purge Data" button is broken** | Implement a real `DELETE /upload/{filename}` endpoint or remove the button |
| 8 | **Admin dashboard shows no file metadata** | Show file size, upload date, row count — all obtainable from `Path.stat()` and `pd.read_csv()` |
| 9 | **No toast notification system** — messages appear inline, easy to miss | Replace with corner toast pop-ups using CSS animations |
| 10 | **`overflow: hidden` on `body`** — the page cannot scroll on desktop | Risky; any content overflow is silently clipped. Use `overflow: auto` for resilience |
| 11 | **No loading skeleton for chart area** — blank white box during auto-train | Add pulse skeleton placeholder in the chart div |
| 12 | **Mobile `<900px`: cards set to `min-height: 300px`** — not enough for chart canvas | Increase to `min-height: 450px` for chart card on mobile |
| 13 | **Theme toggle text hidden on small screens** — `.text` class has no responsive hide | Add `@media (max-width: 600px) { .theme-toggle .text { display: none; } }` |
| 14 | **Chart X-axis labels say "Month 1–6"** — no real dates | Compute actual future month labels from the current date |
| 15 | **No historical price line on chart** — can't see how forecast compares to past | Add a second dataset on the chart with the last N actual prices |

---

## 4. 🤖 ML Model — Strengths & Improvements

### ✅ Current Architecture
- **SARIMAX(1,1,1)(1,1,1,12)** — well-suited for monthly, seasonal fuel price data
- Exogenous variable support: crude prices, exchange rate, inflation, pipeline events
- Non-linear feature engineering: interaction terms, log-transform of transport cost
- 80/20 train-test split with MAE/RMSE/MAPE metrics
- Model persisted via `joblib` for fast prediction

### ❌ Model Improvements Needed

| # | Issue | Recommendation |
|---|-------|---------------|
| 1 | **Fixed SARIMAX order `(1,1,1)(1,1,1,12)`** — no hyperparameter tuning | Run `auto_arima` (from `pmdarima`) to find optimal `p,d,q,P,D,Q` per dataset |
| 2 | **Future exogenous values are copies of past data** | `future_exog = df.drop(columns=["price"]).iloc[-steps:].copy()` — using historical values as future proxies is incorrect for genuine forecasting; exog values should be future projections or at minimum flagged as assumed |
| 3 | **No prediction confidence intervals surfaced** | `get_forecast()` returns `conf_int()` — expose these as `lower_bound` / `upper_bound` in the API response |
| 4 | **CPI data is empty** — feature silently absent | Fill `cpi_monthly_clean.csv` with real data or remove it from the pipeline |
| 5 | **No stationarity test** | Run ADF/KPSS test before fitting; warn if time series is not stationary enough for SARIMAX |
| 6 | **No cross-validation** | Time series cross-validation (walk-forward) gives more reliable metric estimates than a single 80/20 split |
| 7 | **No model versioning** — retraining overwrites `sarimax_model.pkl` with no history | Save models with timestamps: `sarimax_model_20260324.pkl` and keep the last N versions |
| 8 | **Training with multi-worker gunicorn is unsafe** | Multiple workers may try to write to `sarimax_model.pkl` simultaneously — use a file lock or train in a separate background worker |
| 9 | **`historical_exchange_rate_clean.csv` is 2.8 MB** — ~170K rows | This is far too granular for monthly SARIMAX; it should be aggregated to monthly mean before merging (already partially done by `usd_kes_monthly_mean.csv`) |
| 10 | **`non_linear_market_shocks.csv` is 172 KB** — unknown format | File exists but never explicitly referenced; the merge loop picks it up generically — validate its column names and date alignment |
| 11 | **No feature importance analysis** | SARIMAX provides coefficient p-values per exogenous variable via `results.summary()` — expose insignificant features as warnings |
| 12 | **`apply_cap()` is never called** | The regulatory price cap function exists but is disconnected — wire it into the predict endpoint with a configurable cap value |
| 13 | **`test_model.py` depends on non-existent file** | `data/external/sample_fuel_data.csv` is missing — all tests fail immediately |
| 14 | **`tests/` directory is empty** | No pytest unit tests exist; edge case testing is in a standalone script, not a proper test suite |

---

## 5. 🏗️ Architecture Improvements

| # | Issue | Recommendation |
|---|-------|---------------|
| 1 | **`routes/` and `services/` are empty stubs** | Route logic lives entirely in `main.py` (373 lines) — refactor routes into `routes/` modules (upload, train, predict, admin) |
| 2 | **`sqlalchemy` + `psycopg2` in requirements, never used** | Either implement a proper DB layer for uploads/predictions/users, or remove the dependencies |
| 3 | **`pulp` and `apscheduler` in requirements, never used** | Remove or implement: `pulp` for price optimization LP, `apscheduler` for scheduled retraining |
| 4 | **Redis defined in docker-compose but never used** | Integrate Redis for caching predictions — avoid retraining on every page load |
| 5 | **No background task for training** | Training takes 20–60s; doing it synchronously blocks the API thread. Use FastAPI `BackgroundTasks` or Celery |
| 6 | **No API versioning** | All endpoints are at root `/` — prefix with `/api/v1/` for future compatibility |
| 7 | **`SECRET_KEY` defaults to `"supersecretkey"`** | If `.env` is not loaded, the default is insecure — at minimum, raise an error on startup if using default |
| 8 | **No rate limiting** | `/train/` endpoint can be hammered — add `slowapi` rate limiting |
| 9 | **No structured logging** | `print()` used inside `sarimax_model.py` (lines 206–209) — replace with `logger.info()` |
| 10 | **Dockerfile copies `data/processed/` but `.gitignore` excludes `.pkl`** | Fresh Docker build won't have a trained model — add a startup check or pre-baked model |

---

## 6. Priority Action Plan

### 🔴 Critical (Fix Immediately)
1. Fix `MODEL_PATH` to use absolute path (`sarimax_model.py`)
2. Read `ADMIN_USERNAME` / `ADMIN_PASSWORD` from env vars (`main.py`)
3. Fix the duplicate `display` inline style breaking results visibility (`index.html` L97)
4. Add sample data file or skip test gracefully (`data/external/`)
5. Fix `datetime.utcnow()` deprecation (`auth.py`)

### 🟡 High Priority
6. Pin all versions in `requirements.txt`
7. Surface model metrics (MAE/RMSE/MAPE) in the admin UI
8. Implement the "Purge Data" button with a real DELETE endpoint
9. Replace `<pre>` forecast output with a styled results table
10. Add confidence intervals to the forecast chart

### 🟢 Medium Priority
11. Wire `apply_cap()` into `/predict/` with a configurable parameter
12. Replace auto-train-on-load with an explicit "Load Default Model" button
13. Add real pytest unit tests to `tests/`
14. Move mid-file imports to the top of `main.py`
15. Add model versioning / timestamped saves
16. Refactor `main.py` routes into `routes/` modules

### 🔵 Low / Enhancement
17. Add `auto_arima` hyperparameter search
18. Add real future exogenous projections (not historic copies)
19. Add confidence interval bands to API response
20. Implement Redis caching for predictions
21. Add background task for async training
22. Add real date labels to chart X-axis
23. Add historical line to forecast chart
