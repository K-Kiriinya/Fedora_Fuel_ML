"""
FastAPI Application for Fuel Price Forecasting
----------------------------------------------

Features:
- CSV Upload
- Model Training
- Forecast Prediction
- Admin Dashboard
- JWT Authentication
"""

from fastapi import FastAPI, UploadFile, File, Request, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import timedelta

import pandas as pd
import shutil
import secrets
import logging
import joblib
import os

from fuel_pricing.ml.sarimax_model import FuelSARIMAXModel
from fuel_pricing.data.loader import load_and_prepare
from fuel_pricing.pipelines.predict_pipeline import run_prediction
from fuel_pricing.core.config import UPLOAD_DIR, PROCESSED_DIR
from fuel_pricing.api.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------
# PROJECT CONFIGURATION
# -------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Metrics storage path (absolute via config)
METRICS_PATH = PROCESSED_DIR / "metrics.pkl"

# Initialize FastAPI
app = FastAPI(
    title="Fedora Fuel ML API",
    description="SARIMAX-based fuel price forecasting system",
    version="1.0.0"
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


# -------------------------------------------------------
# HOME ROUTE
# -------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Render homepage.
    """
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )


# -------------------------------------------------------
# AUTHENTICATION
# -------------------------------------------------------

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


# -------------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------------

def validate_csv_file(file: UploadFile) -> None:
    """
    Validate uploaded CSV file.

    Raises
    ------
    HTTPException
        If validation fails
    """
    # Check file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Only {', '.join(ALLOWED_EXTENSIONS)} files are allowed."
        )

    # Sanitize filename to prevent path traversal
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
            while chunk := await file.read(8192):  # Read in 8KB chunks
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    buffer.close()
                    file_path.unlink()  # Delete incomplete file
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024)}MB"
                    )
                buffer.write(chunk)

        # Validate CSV structure
        try:
            df = pd.read_csv(file_path)
            if "date" not in df.columns or "price" not in df.columns:
                file_path.unlink()  # Delete invalid file
                return {"error": "CSV must contain 'date' and 'price' columns"}

            # Basic data validation
            if len(df) == 0:
                file_path.unlink()
                return {"error": "CSV file is empty"}

        except Exception as e:
            if file_path.exists():
                file_path.unlink()
            return {"error": f"Invalid CSV format: {str(e)}"}

        logger.info(f"File uploaded successfully: {filename} ({file_size} bytes)")
        return {"message": f"{filename} uploaded successfully."}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return {"error": f"Upload failed: {str(e)}"}


def get_training_data() -> pd.DataFrame:
    """
    Dynamically loads and merges all available processed datasets.
    """
    # 1. Check for manual uploads
    csv_files = list(UPLOAD_DIR.glob("*.csv"))
    if csv_files:
        df = pd.read_csv(csv_files[0])
        if "date" in df.columns and "price" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            return df
            
    # 2. Main dataset (local_prices)
    prices_file = PROCESSED_DIR / "local_prices" / "kenyan_oil_prices_monthly_clean.csv"
    if not prices_file.exists():
        raise Exception("System Error: Default Processed local prices dataset missing.")
        
    df = pd.read_csv(prices_file)
    
    if "Local Price in KSH" in df.columns:
        df["price"] = df["Local Price in KSH"]
    elif "PMS" in df.columns:
        df["price"] = df["PMS"]
        
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"])
    else:
        df["month"] = pd.to_datetime(df["date"])
        
    # 3. Dynamically merge ALL market_indicators
    indicators_dir = PROCESSED_DIR / "market_indicators"
    if indicators_dir.exists() and indicators_dir.is_dir():
        for csv_path in indicators_dir.glob("*.csv"):
            try:
                df_ind = pd.read_csv(csv_path)
                if 'month' in df_ind.columns:
                    df_ind['month'] = pd.to_datetime(df_ind['month'])
                    cols_to_use = df_ind.columns.difference(df.columns).tolist() + ['month']
                    df = pd.merge(df, df_ind[cols_to_use], on='month', how='left')
                elif 'date' in df_ind.columns:
                    df_ind['month'] = pd.to_datetime(df_ind['date'])
                    cols_to_use = df_ind.columns.difference(df.columns).tolist() + ['month']
                    df = pd.merge(df, df_ind[cols_to_use], on='month', how='left')
            except Exception as e:
                logger.warning(f"Skipped indicator {os.path.basename(csv_path)}: {e}")
                
    # 4. Standardize format for ML Pipeline
    df.set_index("month", inplace=True)
    df.index.name = "date"
    
    # 5. Extract fully numeric feature matrix
    numeric_df = df.select_dtypes(include=['number'])
    
    # Safely interpolate missing values across mismatched indicator date ranges 
    # to protect SARIMAX strict constraint against NaN
    numeric_df = numeric_df.ffill().bfill()
    numeric_df = numeric_df.dropna(axis=1, how='all')

    if "price" not in numeric_df.columns:
        raise Exception("Failed to identify 'price' column in merged dataset.")

    return numeric_df

# -------------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------------

@app.post("/train/")
def train_model():
    """
    Train SARIMAX model dynamically. Uses merged pre-processed datasets
    by default, or the user's uploaded dataset if overridden.
    """
    try:
        df = get_training_data()

        # Initialize and Train model
        model = FuelSARIMAXModel()
        logger.info(f"Training model initiated with shape {df.shape}")
        
        # Suppress possible statsmodels warnings temporarily for clean logs
        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            model.train_sarimax(df)

        logger.info("Model training completed successfully")
        return {
            "message": "Model trained successfully.",
            "metrics": model.metrics,
        }

    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return {"error": f"Training failed: {str(e)}"}


# -------------------------------------------------------
# MODEL METRICS
# -------------------------------------------------------

@app.get("/metrics/")
def get_metrics():
    """Return the latest model training evaluation metrics."""
    try:
        if not METRICS_PATH.exists():
            return {"error": "No metrics found. Train the model first."}
        metrics = joblib.load(METRICS_PATH)
        return metrics
    except Exception as e:
        logger.error(f"Metrics fetch error: {str(e)}")
        return {"error": str(e)}


# -------------------------------------------------------
# PREDICT FUTURE PRICES
# -------------------------------------------------------

@app.post("/predict/")
def predict_price(steps: int = Form(...)):
    """
    Predict future fuel prices with confidence intervals and real date labels.
    """
    try:
        df = get_training_data()

        # Use last known exogenous values for forecasting forward
        future_exog = df.drop(columns=["price"]).iloc[-steps:].copy()

        model = FuelSARIMAXModel()
        forecast_result = model.predict(steps=steps, future_exog=future_exog)

        predicted  = forecast_result["predicted_mean"]
        lower_ci   = forecast_result["lower_ci"]
        upper_ci   = forecast_result["upper_ci"]

        # Generate real future date labels from the training data's last date
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date, periods=steps + 1, freq="MS")[1:]
        date_labels = [d.strftime("%b %Y") for d in future_dates]

        return {
            "forecast_steps": steps,
            "predicted_prices": predicted.tolist(),
            "lower_ci":         lower_ci.tolist(),
            "upper_ci":         upper_ci.tolist(),
            "date_labels":      date_labels.tolist(),
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": str(e)}


# -------------------------------------------------------
# ADMIN DASHBOARD (Protected with HTTP Basic Auth)
# -------------------------------------------------------

def verify_admin(credentials: HTTPBasicCredentials = Depends(security_basic)):
    """Simple HTTP Basic Auth for admin access."""
    correct_username = secrets.compare_digest(credentials.username, "admin")
    correct_password = secrets.compare_digest(credentials.password, "admin123")

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
        request=request,
        name="admin.html",
        context={
            "request": request,
            "files": files
        }
    )


# -------------------------------------------------------
# PURGE UPLOADED FILE (Admin-protected)
# -------------------------------------------------------

@app.post("/admin/purge/{filename}")
def purge_file(
    filename: str,
    request: Request,
    username: str = Depends(verify_admin),
):
    """
    Delete a specific uploaded CSV file.
    Protected by HTTP Basic Auth (same as admin dashboard).
    """
    # Sanitize filename — block path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    file_path.unlink()
    logger.info(f"File '{filename}' deleted by admin: {username}")
    # Redirect back to admin dashboard after deletion
    return RedirectResponse(url="/admin/", status_code=303)


# -------------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------------

@app.get("/health")
def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "service": "Fedora Fuel ML",
        "version": "1.0.0"
    }