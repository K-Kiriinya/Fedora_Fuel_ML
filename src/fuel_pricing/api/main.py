"""
FastAPI Application for Fuel Price Forecasting
----------------------------------------------

Features:
- CSV Upload
- Model Training
- Forecast Prediction
- Admin Dashboard
"""

from certifi import contents
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import pandas as pd
import shutil

from fuel_pricing.ml.sarimax_model import FuelSARIMAXModel
from fuel_pricing.data.loader import load_and_prepare
from fuel_pricing.pipelines.predict_pipeline import run_prediction
from fuel_pricing.core.config import UPLOAD_DIR

# -------------------------------------------------------
# PROJECT CONFIGURATION
# -------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
# UPLOAD_DIR = BASE_DIR / "uploads"

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI
app = FastAPI()

# Templates directory
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# -------------------------------------------------------
# HOME ROUTE
# -------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Render homepage.
    """
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# -------------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------------

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload CSV dataset.
    """

    if not file.filename.endswith(".csv"):
        return {"error": "Only CSV files are allowed."}

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": f"{file.filename} uploaded successfully."}


# -------------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------------

@app.post("/train/")
def train_model():
    """
    Train SARIMAX model using latest uploaded CSV.
    """

    csv_files = list(UPLOAD_DIR.glob("*.csv"))

    if not csv_files:
        return {"error": "No CSV files uploaded."}

    # Load first CSV
    df = pd.read_csv(csv_files[0])

    # Ensure 'date' column exists
    if "date" not in df.columns:
        return {"error": "CSV must contain a 'date' column."}

    # Convert date
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Initialize model
    model = FuelSARIMAXModel()

    # Train model
    model.train_sarimax(df)

    return {"message": "Model trained successfully."}


# -------------------------------------------------------
# PREDICT FUTURE PRICES
# -------------------------------------------------------

@app.post("/predict/")
def predict_price(steps: int = Form(...)):
    """
    Predict future fuel prices.

    Parameters:
    steps: Number of months to forecast.
    """

    csv_files = list(UPLOAD_DIR.glob("*.csv"))

    if not csv_files:
        return {"error": "No CSV files uploaded."}

    df = pd.read_csv(csv_files[0])

    if "date" not in df.columns:
        return {"error": "CSV must contain a 'date' column."}

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Use last known exogenous values for forecasting
    future_exog = df.drop(columns=["price"]).iloc[-steps:].copy()

    model = FuelSARIMAXModel()

    try:
        forecast = model.predict(steps=steps, future_exog=future_exog)
    except Exception as e:
        return {"error": str(e)}

    return {
        "forecast_steps": steps,
        "predicted_prices": forecast.tolist()
    }


# -------------------------------------------------------
# ADMIN DASHBOARD
# -------------------------------------------------------

@app.get("/admin/", response_class=HTMLResponse)
def admin_dashboard(request: Request):
    """
    View uploaded CSV files.
    """

    files = list(UPLOAD_DIR.glob("*.csv"))

    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "files": files
        }
    )