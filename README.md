# Fedora_Fuel_ML — SARIMAX Dynamic Fuel Pricing System

This system predicts fuel prices (KES) using:
- SARIMAX (Seasonal ARIMA)
- Optional external factors (crude, inflation, exchange rate)
- Regulatory price cap enforcement
- FastAPI Web UI
- Admin Dashboard

----------------------------------------
HOW TO RUN

1. Install dependencies:
   pip install -r requirements.txt

2. Run server:
   uvicorn fuel_pricing.api.main:app --reload

3. Open:
   http://127.0.0.1:8000

----------------------------------------
REQUIRED CSV FORMAT

Columns required:
- date
- price

Optional:
- crude_price
- exchange_rate
- inflation
