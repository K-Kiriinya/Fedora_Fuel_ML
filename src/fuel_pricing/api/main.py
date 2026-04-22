"""
FastAPI Application for Fuel Price Forecasting
----------------------------------------------

Features:
- CSV Upload
- Model Training
- Forecast Prediction
- Admin Dashboard
"""

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Request,
    Form,
    Depends,
    HTTPException,
    status,
)

from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import timedelta, datetime

import pandas as pd
import numpy as np
import re
import shutil
import secrets
import logging
import joblib
import os

from fuel_pricing.ml.sarimax_model import (
    FuelSARIMAXModel,
    get_model_path,
    get_metrics_path,
)
from fuel_pricing.data.loader import load_and_prepare
from fuel_pricing.pipelines.predict_pipeline import run_prediction
from fuel_pricing.optimization.pricing import apply_cap
from fuel_pricing.core.config import UPLOAD_DIR, PROCESSED_DIR
from fuel_pricing.api.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------
# PROJECT CONFIGURATION
# ----------------------
BASE_DIR = Path(__file__).resolve().parent

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Metrics storage path (absolute via config)
METRICS_PATH = PROCESSED_DIR / "metrics.pkl"

# Initialize FastAPI
app = FastAPI(
    title="Fedora Fuel ML API",
    description="SARIMAX-based fuel price forecasting system",
    version="1.0.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Templates directory
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# HTTP Basic Auth for simple admin access
security_basic = HTTPBasic()

# File upload constraints
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".csv"}


# ----------
# HOME ROUTE
# ----------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Render homepage.
    """
    return templates.TemplateResponse(
        request=request, name="index.html", context={"request": request}
    )


# ---------------
# AUTHENTICATION
# ---------------
@app.post("/api/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """
    Authenticate user and return JWT token.
    """
    user = authenticate_user(username, password)
    if not user:
        logger.warning(f"Failed login attempt for username: {username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )

    logger.info(f"User {username} logged in successfully")
    return {"access_token": access_token, "token_type": "bearer"}


# ------------
# FILE UPLOAD
# ------------
def validate_csv_file(file: UploadFile) -> None:
    """
    Validate uploaded CSV file.
    """
    # Check file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Only {', '.join(ALLOWED_EXTENSIONS)} files are allowed.",
        )

    if ".." in file.filename or "/" in file.filename or "\\" in file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload CSV dataset with security validation.
    """
    try:
        # Validate file
        validate_csv_file(file)

        # Secure filename
        filename = Path(file.filename).name
        file_path = UPLOAD_DIR / filename

        # Check file size
        file_size = 0
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(8192):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    buffer.close()
                    file_path.unlink()  # Delete incomplete file
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024)}MB",
                    )
                buffer.write(chunk)

        # Validate CSV structure
        try:
            df = pd.read_csv(file_path)
            # Support either the old 'date/price' or the new regulatory columns
            required_cols = ["From", "To", "Super (PMS)"]  # Basic requirements
            has_new = all(col in df.columns for col in required_cols)
            has_old = "date" in df.columns and "price" in df.columns

            if not (has_new or has_old):
                file_path.unlink()  # Delete invalid file
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid CSV structure. Missing mandatory regulatory columns (From, To, Super (PMS))"
                )

            # Basic data validation
            if len(df) == 0:
                file_path.unlink()
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="CSV file is empty"
                )

        except HTTPException:
            raise
        except Exception as e:
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid CSV format: {str(e)}"
            )

        logger.info(f"File uploaded successfully: {filename} ({file_size} bytes)")
        return {"message": f"{filename} uploaded successfully."}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


def flexible_date_parse(date_val):
    if pd.isna(date_val):
        return pd.NaT

    date_str = str(date_val).strip()

    # 1. Explicit DD-MM-YYYY and DD/MM/YYYY
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d/%m/%y"):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except (ValueError, TypeError):
            pass

    # 2. Handle 6-digit numeric (ddmmyy or yymmdd)
    if re.match(r"^\d{6}$", date_str):
        try:
            return pd.to_datetime(date_str, format="%d%m%y")
        except:
            try:
                return pd.to_datetime(date_str, format="%y%m%d")
            except:
                pass

    # 3. Handle ordinal month names: 15th April 26, 1st May 26
    clean_str = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_str, flags=re.IGNORECASE)

    try:
        return pd.to_datetime(clean_str, dayfirst=True)
    except:
        return pd.NaT


def get_training_data(town: str = "Nairobi", fuel_type: str = "pms") -> pd.DataFrame:
    # 1. PRIORITY: Check for manual uploads (user-uploaded custom data)
    csv_files = list(UPLOAD_DIR.glob("*.csv"))
    if csv_files:
        csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        df = pd.read_csv(csv_files[0])
    else:
        # 2. FALLBACK: Main regulatory dataset
        prices_file = PROCESSED_DIR / "local_prices" / "EPRA_Pump_Prices.csv"
        if not prices_file.exists():
            raise Exception("System Error: No historical or uploaded datasets found.")

        df = pd.read_csv(prices_file)

    # Filter for selected town
    if "Town" in df.columns:
        df = df[df["Town"].str.strip().str.lower() == town.lower()]

    # Mapping: Resolve requested fuel type to actual column names
    price_aliases = {
        "pms": ["Super (PMS)", "Super", "PMS", "Petrol", "Super Petrol", "Super (PMS)"],
        "ago": ["Diesel (AGO)", "Diesel", "AGO", "Automotive Gas Oil", "Diesel (AGO)"],
        "ik": [
            "Kerosene (IK)",
            "Kerosene",
            "IK",
            "Illuminating Kerosene",
            "Kerosene (IK)",
        ],
    }

    target_fuel = fuel_type.lower()
    search_list = price_aliases.get(target_fuel, price_aliases["pms"])

    found_col = None
    for alias in search_list:
        if alias in df.columns:
            found_col = alias
            break

    if found_col:
        df["price"] = df[found_col]
    elif "price" in df.columns:
        pass
    elif "Local Price in KSH" in df.columns:
        df["price"] = df["Local Price in KSH"]
    else:
        for col in ["Super (PMS)", "Diesel (AGO)", "Kerosene (IK)", "PMS", "AGO", "IK"]:
            if col in df.columns:
                df["price"] = df[col]
                break

    if "To" in df.columns:
        # Use 'To' column to align with the end of the fuel cycle
        df["month"] = df["To"].apply(flexible_date_parse)
        df["month"] = df["month"].dt.to_period("M").dt.to_timestamp()
    elif "From" in df.columns:
        df["month"] = df["From"].apply(flexible_date_parse)
        df["month"] = df["month"].dt.to_period("M").dt.to_timestamp()
    elif "month" in df.columns:
        df["month"] = df["month"].apply(flexible_date_parse)
        df["month"] = df["month"].dt.to_period("M").dt.to_timestamp()
    else:
        df["month"] = df["date"].apply(flexible_date_parse)
        df["month"] = df["month"].dt.to_period("M").dt.to_timestamp()

    df.sort_values("month", inplace=True)

    # 3. Dynamically merge ALL market_indicators
    indicators_dir = PROCESSED_DIR / "market_indicators"
    if indicators_dir.exists() and indicators_dir.is_dir():
        for csv_path in indicators_dir.glob("*.csv"):
            try:
                df_ind = pd.read_csv(csv_path)
                
                # Identify date column
                target_date_col = None
                if "month" in df_ind.columns:
                    target_date_col = "month"
                elif "date" in df_ind.columns:
                    target_date_col = "date"
                
                if target_date_col:
                    # Standardize to monthly start timestamp
                    df_ind["month"] = (
                        pd.to_datetime(df_ind[target_date_col])
                        .dt.to_period("M")
                        .dt.to_timestamp()
                    )
                    
                    # ⚠️ CRITICAL: Aggregate by month to prevent row explosion on merge
                    # Multiple indicator entries in a single month would otherwise create a Cartesian product
                    df_ind_numeric = df_ind.select_dtypes(include=[np.number]).copy()
                    df_ind_numeric["month"] = df_ind["month"]
                    df_ind_grouped = df_ind_numeric.groupby("month").mean().reset_index()
                    
                    # Filter columns to avoid duplicates (except month)
                    cols_to_use = df_ind_grouped.columns.difference(df.columns).tolist() + ["month"]
                    df = pd.merge(df, df_ind_grouped[cols_to_use], on="month", how="left")
            except Exception as e:
                logger.warning(f"Skipped indicator {os.path.basename(csv_path)}: {e}")

    # 4. Standardize format for ML Pipeline
    df.set_index("month", inplace=True)
    df.index.name = "date"

    # 5. Extract fully numeric feature matrix
    numeric_df = df.select_dtypes(include=["number"]).copy()

    # ⚠️ CRITICAL: Remove all potential target columns from features to prevent leakage/cheating
    # If the target price is also in exog, the model will just learn price = price
    price_cols_to_drop = [
        "Super (PMS)", "Diesel (AGO)", "Kerosene (IK)",
        "PMS", "AGO", "IK", "Super", "Diesel", "Kerosene",
        "Petrol", "Super Petrol", "Automotive Gas Oil", "Illuminating Kerosene",
        "Local Price in KSH"
    ]
    for col in price_cols_to_drop:
        if col in numeric_df.columns and col != "price":
            numeric_df.drop(columns=[col], inplace=True)

    # Safely interpolate missing values across mismatched indicator date ranges
    numeric_df = numeric_df.ffill().bfill()
    numeric_df = numeric_df.dropna(axis=1, how="all")

    if "price" not in numeric_df.columns:
        raise Exception("Failed to identify 'price' column in merged dataset.")

    return numeric_df


# -----------
# TRAIN MODEL
# -----------
@app.post("/train/")
async def train_endpoint(town: str = Form(...)):
    try:
        fuels = ["pms", "ago", "ik"]
        metrics_summary = {}

        # Suppress warnings for clean output
        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning

        for fuel in fuels:
            df = get_training_data(town, fuel)
            model = FuelSARIMAXModel()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.train_sarimax(df, fuel_type=fuel)

            metrics_summary[fuel] = model.metrics

        return {
            "message": f"Successfully trained engine for {town}.",
            "metrics": metrics_summary,
            "pms_metrics": metrics_summary.get("pms", {}),
        }

    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )


# -------------
# MODEL METRICS
# -------------
@app.get("/metrics/")
async def get_metrics(fuel_type: str = "pms"):
    try:
        metrics_path = get_metrics_path(fuel_type)
        if not metrics_path.exists():
            generic_metrics_path = PROCESSED_DIR / "metrics.pkl"
            if not generic_metrics_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No metrics available. Train the engine first."
                )
            metrics_path = generic_metrics_path

        metrics = joblib.load(metrics_path)
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrics fetch error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ----------------------
# PREDICT FUTURE PRICES
# ----------------------
@app.post("/predict/")
async def predict_price(
    steps: int = Form(6), town: str = Form("Nairobi"), cap: float = Form(None)
):

    try:
        fuels = ["pms", "ago", "ik"]
        combined_results = {}
        date_labels = None

        for fuel in fuels:
            df = get_training_data(town, fuel)

            # Handle cases where df might be empty or lack sufficient data
            if df.empty or len(df) < 2:  # Need at least 2 points for SARIMAX
                combined_results[fuel] = {
                    "prices": [0.0] * steps,
                    "lower": [0.0] * steps,
                    "upper": [0.0] * steps,
                }
                if date_labels is None:  # Generate date labels even if no data
                    prediction_start = pd.Timestamp.now().to_period(
                        "M"
                    ).to_timestamp() + pd.DateOffset(months=1)
                    future_dates = pd.date_range(
                        start=prediction_start, periods=steps, freq="MS"
                    )
                    date_labels = [d.strftime("%b %Y") for d in future_dates]
                continue

            last_date = df.index[-1]
            prediction_start = last_date + pd.DateOffset(months=1)

            # Prepare future exogenous variables
            historic_exog = df.drop(columns=["price"])
            if len(historic_exog) == 0:
                combined_results[fuel] = {
                    "prices": [0.0] * steps,
                    "lower": [0.0] * steps,
                    "upper": [0.0] * steps,
                }
                continue

            if steps <= len(historic_exog):
                active_exog = historic_exog.iloc[-steps:].copy()
            else:
                # 🛠 IMPROVED: Use last-value persistence for future exogenous forecasting
                # instead of repeating the entire historical block (tiling).
                # This assumes a 'random walk' for external shocks which is a standard baseline.
                last_known_vals = historic_exog.iloc[-1:]
                # Create a dataframe of the same structure filled with the last known row
                persistence_exog = pd.DataFrame(
                    np.repeat(last_known_vals.values, steps, axis=0),
                    columns=historic_exog.columns,
                )
                active_exog = persistence_exog.copy()

            model = FuelSARIMAXModel()
            forecast_result = model.predict(
                steps=steps, future_exog=active_exog, fuel_type=fuel
            )

            predicted = forecast_result["predicted_mean"]
            lower_ci = forecast_result["lower_ci"]
            upper_ci = forecast_result["upper_ci"]

            # Apply regulatory cap if provided
            if cap is not None:
                predicted = [apply_cap(p, cap) for p in predicted]
                lower_ci = [apply_cap(p, cap) for p in lower_ci]
                upper_ci = [apply_cap(p, cap) for p in upper_ci]
            else:
                predicted = predicted.tolist()
                lower_ci = lower_ci.tolist()
                upper_ci = upper_ci.tolist()

            # Ensure all values are float and serializable (strip numpy types and NaNs)
            def clean_list(vals):
                # Enforce non-negativity and clean serializable floats
                return [max(0.0, float(v)) if np.isfinite(v) else 0.0 for v in vals]

            combined_results[fuel] = {
                "prices": clean_list(predicted),
                "lower": clean_list(lower_ci),
                "upper": clean_list(upper_ci),
                "current_price": float(df["price"].iloc[-1]) if not df.empty else 0.0,
            }

            if date_labels is None:
                # Generate real future date labels
                future_dates = pd.date_range(
                    start=prediction_start, periods=steps, freq="MS"
                )
                date_labels = [d.strftime("%b %Y") for d in future_dates]

        # Ensure response dictionary contains all fields even if some fuels failed
        return {
            "pms": combined_results.get(
                "pms", {"prices": [], "lower": [], "upper": []}
            ),
            "ago": combined_results.get(
                "ago", {"prices": [], "lower": [], "upper": []}
            ),
            "ik": combined_results.get("ik", {"prices": [], "lower": [], "upper": []}),
            "date_labels": date_labels or [],
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ---------------
# ADMIN DASHBOARD
# ----------------
def verify_admin(credentials: HTTPBasicCredentials = Depends(security_basic)):
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@app.get("/admin/", response_class=HTMLResponse)
def admin_dashboard(request: Request, username: str = Depends(verify_admin)):
    """
    View uploaded CSV files and system information.
    Requires HTTP Basic Authentication.
    """
    files = list(UPLOAD_DIR.glob("*.csv"))

    logger.info(f"Admin dashboard accessed by: {username}")
    return templates.TemplateResponse(
        request=request, name="admin.html", context={"request": request, "files": files}
    )


# -------------------
# PURGE UPLOADED FILE
# -------------------
@app.post("/admin/purge/{filename}")
def purge_file(
    filename: str,
    request: Request,
    username: str = Depends(verify_admin),
):

    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    file_path.unlink()
    logger.info(f"File '{filename}' deleted by admin: {username}")
    return RedirectResponse(url="/admin/", status_code=303)


# ------------
# HEALTH CHECK
# ------------
@app.get("/health")
def health_check():
    """System health check endpoint."""
    return {"status": "healthy", "service": "Fedora Fuel ML", "version": "1.0.0"}


# ------------------------
# PURGE ALL UPLOADED FILES
# ------------------------
@app.post("/admin/purge_all")
def purge_all_files(
    request: Request,
    username: str = Depends(verify_admin),
):
    """
    Delete ALL uploaded CSV files.
    """
    files = list(UPLOAD_DIR.glob("*.csv"))
    for f in files:
        f.unlink()

    # Also reset model trained status in a real scenario, but local_prices data still exists
    logger.info(f"Total {len(files)} uploaded files purged by admin: {username}")
    return RedirectResponse(url="/admin/", status_code=303)


# ----------------
# HISTORICAL DATA
# ----------------
@app.get("/current-prices/")
def get_current_prices(town: str = "Nairobi"):
    """
    Returns the most recent pump prices
    """
    try:
        prices_file = PROCESSED_DIR / "local_prices" / "EPRA_Pump_Prices.csv"
        if not prices_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No price data available."
            )
        df = pd.read_csv(prices_file)

        # Filter by town if column exists
        if "Town" in df.columns:
            town_df = df[df["Town"].astype(str).str.strip().str.lower() == town.lower()]
            if not town_df.empty:
                df = town_df

        # Parse dates
        from_col = next(
            (c for c in ["From", "date", "timestamp"] if c in df.columns), None
        )
        to_col = next((c for c in ["To", "Period"] if c in df.columns), None)

        if from_col:
            df["_from_date"] = df[from_col].apply(flexible_date_parse)
            df = df[df["_from_date"].notna()]
            max_from = df["_from_date"].max()
            latest_rows = df[df["_from_date"] == max_from]
            latest = latest_rows.iloc[0]
            period_from = max_from.strftime("%d %b %Y")
            period_to = (
                latest[to_col].strip() if to_col and pd.notna(latest[to_col]) else None
            )
            # Try to parse and reformat period_to neatly
            if period_to:
                parsed_to = flexible_date_parse(period_to)
                period_to = (
                    parsed_to.strftime("%d %b %Y") if pd.notna(parsed_to) else period_to
                )
        else:
            latest = df.iloc[0]  # fallback
            period_from = None
            period_to = None

        # Match column names for each fuel type
        def find_price(row, patterns):
            for p in patterns:
                for col in row.index:
                    if p.upper() in col.upper():
                        try:
                            val = float(row[col])
                            if val > 0:
                                return round(val, 2)
                        except (ValueError, TypeError):
                            pass
            return None

        pms = find_price(latest, ["PMS", "SUPER", "PETROL"])
        ago = find_price(latest, ["AGO", "DIESEL"])
        kero = find_price(latest, ["IK", "KERO"])

        return {
            "pms": pms,
            "ago": ago,
            "kero": kero,
            "period_from": period_from,
            "period_to": period_to,
            "town": town,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Current prices error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/towns/")
def get_towns():
    """Returns a list of all available towns from the official regulatory."""
    try:
        prices_file = PROCESSED_DIR / "local_prices" / "EPRA_Pump_Prices.csv"
        if not prices_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No historical data available."
            )

        df = pd.read_csv(prices_file)
        if "Town" not in df.columns:
            return {"towns": []}

        # Get unique sorted towns, handling NaN and empty
        towns = sorted(
            list(set([str(t).strip() for t in df["Town"].dropna() if str(t).strip()]))
        )
        return {"towns": towns}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Towns fetch error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/history/")
def get_history_data(town: str = "Nairobi"):
    """
    Returns historical price data for visualization using the official regulatory data.
    """
    try:
        prices_file = PROCESSED_DIR / "local_prices" / "EPRA_Pump_Prices.csv"
        if not prices_file.exists():
            prices_file = (
                PROCESSED_DIR / "local_prices" / "kenyan_oil_prices_monthly_clean.csv"
            )

        if not prices_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Historical repository missing."
            )

        df = pd.read_csv(prices_file)

        # Filter for the chosen town
        if "Town" in df.columns:
            # Handle case-insensitive town
            df["_town_clean"] = df["Town"].astype(str).str.strip().str.lower()
            df = df[df["_town_clean"] == town.lower()]

        # Identify date column
        date_col = None
        for col in ["To", "From", "date", "timestamp", "Period"]:
            if col in df.columns:
                date_col = col
                break

        if not date_col:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Time column not found in dataset."
            )

        df["month"] = df[date_col].apply(flexible_date_parse)
        df = df[df["month"].notna()]  # Drop NaT to avoid crash
        df.sort_values("month", inplace=True)

        if df.empty:
            return {"labels": [], "pms": [], "ago": [], "kero": []}

        # Prepare labels and values
        labels = [d.strftime("%b %Y") for d in df["month"]]

        # --- Robust Fuel Column Mapping ---
        pms_col = next(
            (
                c
                for c in df.columns
                if any(p in c.upper() for p in ["PMS", "PETROL", "SUPER"])
            ),
            None,
        )
        ago_col = next(
            (c for c in df.columns if any(a in c.upper() for a in ["AGO", "DIESEL"])),
            None,
        )
        kero_col = next(
            (c for c in df.columns if any(k in c.upper() for k in ["IK", "KERO"])), None
        )

        def clean_series(col_name):
            if col_name and col_name in df.columns:
                return df[col_name].ffill().bfill().round(2).fillna(0).tolist()
            return []

        return {
            "labels": labels,
            "pms": clean_series(pms_col),
            "ago": clean_series(ago_col),
            "kero": clean_series(kero_col),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"History data error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
