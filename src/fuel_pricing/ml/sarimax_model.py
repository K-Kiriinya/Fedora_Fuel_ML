"""
SARIMAX Model for Fuel Price Forecasting
----------------------------------------

This module defines a SARIMAX model for modeling fuel prices with exogenous shock variables.

Features:
- Data preparation & merging
- Non-linear feature engineering
- Model training with evaluation
- Model persistence (joblib)
- Future forecasting

Author: Kiriinya
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from statsmodels.tsa.statespace.sarimax import SARIMAX

from fuel_pricing.core.config import PROCESSED_DIR


def get_model_path(fuel_type: str = "pms") -> Path:
    return PROCESSED_DIR / f"sarimax_model_{fuel_type.lower()}.pkl"


def get_metrics_path(fuel_type: str = "pms") -> Path:
    return PROCESSED_DIR / f"metrics_{fuel_type.lower()}.pkl"


class FuelSARIMAXModel:
    """
    SARIMAX model wrapper for fuel price prediction.
    """

    def __init__(self):
        self.model = None
        self.results = None
        self.metrics = {}

        # Ensure directory exists
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------
    # STEP 1: LOAD & MERGE DATA
    # --------------------------
    def prepare_dataset(
        self, price_df: pd.DataFrame, shock_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge price data with exogenous shock variables.

        Returns:
            Cleaned and merged dataset indexed by date.
        """

        # Ensure datetime format
        price_df["date"] = pd.to_datetime(price_df["date"])
        shock_df["date"] = pd.to_datetime(shock_df["date"])

        # Merge datasets
        df = pd.merge(price_df, shock_df, on="date", how="left")

        # Replace missing shocks with 0 (no event assumption)
        df.fillna(0, inplace=True)

        # Sort chronologically
        df = df.sort_values("date")
        df.set_index("date", inplace=True)

        return df

    # ----------------------------
    # STEP 2: FEATURE ENGINEERING
    # ----------------------------
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction and transformation features to capture non-linear relationships.
        """

        df = df.copy()

        # FX shock interaction
        if "pipeline_burst" in df.columns and "exchange_rate_usd_ksh" in df.columns:
            df["burst_fx_interaction"] = (
                df["pipeline_burst"] * df["exchange_rate_usd_ksh"]
            )

        # Inflation shortage interaction
        if "fuel_shortage" in df.columns and "inflation_rate" in df.columns:
            df["shortage_inflation_interaction"] = (
                df["fuel_shortage"] * df["inflation_rate"]
            )

        # Log transformation for transport costs
        if "transport_cost_index" in df.columns:
            df["log_transport_cost"] = np.log1p(df["transport_cost_index"])

        return df

    # --------------------
    # METRICS CALCULATION
    # --------------------
    def calculate_metrics(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # Safe MAPE (avoid division by zero)
        y_true_safe = np.where(y_true == 0, 1e-10, y_true)
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

        self.metrics = {
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "MAPE": round(float(mape), 4),
        }

        return self.metrics

    # --------------------
    # STEP 3: TRAIN MODEL
    # --------------------
    def train_sarimax(
        self, df: pd.DataFrame, fuel_type: str = "pms", test_size: float = 0.2
    ):
        """
        Train SARIMAX model and evaluate performance.
        """

        if "price" not in df.columns:
            raise ValueError("Dataset must contain 'price' column.")

        # Create features
        df = self.create_features(df)

        # Ensure only clean numeric rows are passed
        df = df.select_dtypes(include=[np.number])
        df = df.ffill().bfill().dropna()

        # Resample to ensure monthly frequency ONLY if needed (and fill gaps)
        if isinstance(df.index, pd.DatetimeIndex):
            # Check if resampling is actually decreasing points or just filling gaps
            df = df.resample("MS").mean().ffill().bfill().dropna()

        print(f"DEBUG: Training dataset size: {len(df)}")
        if len(df) < 5:
            raise ValueError(
                f"Insufficient data for training: {len(df)} points found. Minimum 5 required."
            )

        # Train/test split for evaluation
        test_size = 0.2 if len(df) >= 10 else 0.1
        split_index = max(1, int(len(df) * (1 - test_size)))
        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]

        # Fixed dimensionality: Ensure 1D numpy arrays for endog
        y_train = train_df["price"].to_numpy().flatten()
        y_test = test_df["price"].to_numpy().flatten()

        # Exogenous variables
        exog_train = train_df.drop(columns=["price"]).to_numpy()
        exog_test = test_df.drop(columns=["price"]).to_numpy()

        # Determine safe order for small datasets
        # Simple ARIMA(1,1,0) if N < 15 to avoid complex startup failure
        p, d, q = (1, 1, 1) if len(y_train) >= 15 else (1, 1, 0)

        # Evaluation model (on training split)
        seasonal_train = (1, 1, 1, 12) if len(y_train) >= 24 else (0, 0, 0, 0)

        eval_model = SARIMAX(
            y_train,
            exog=exog_train,
            order=(p, d, q),
            seasonal_order=seasonal_train,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        eval_results = eval_model.fit(disp=False, maxiter=200)

        # Forecast test period
        forecast = eval_results.get_forecast(steps=len(y_test), exog=exog_test)
        predictions = forecast.predicted_mean

        # Use .values to avoid any residual index-alignment issues
        metrics = self.calculate_metrics(y_test, predictions)

        # --- FINAL FULL-DATA REFIT FOR PRODUCTION FORECASTS ---
        y_full = df["price"].to_numpy().flatten()
        exog_full = df.drop(columns=["price"]).to_numpy()

        seasonal_full = (1, 1, 1, 12) if len(y_full) >= 24 else (0, 0, 0, 0)

        # Re-calc order for full set
        p_f, d_f, q_f = (1, 1, 1) if len(y_full) >= 15 else (1, 1, 0)

        self.model = SARIMAX(
            y_full,
            exog=exog_full,
            order=(p_f, d_f, q_f),
            seasonal_order=seasonal_full,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        self.results = self.model.fit(disp=False, maxiter=200)

        # Save trained final model
        model_path = get_model_path(fuel_type)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.results, model_path)

        # Save metrics
        joblib.dump(self.metrics, get_metrics_path(fuel_type))

        print("\nModel Evaluation")
        print("----------------")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        return predictions, y_test

    # ------------------------------------------------------------------------------
    # STEP 4: PREDICT FUTURE: Forecast future fuel prices with confidence intervals
    # ------------------------------------------------------------------------------
    def predict(
        self, steps: int, future_exog: pd.DataFrame, fuel_type: str = "pms"
    ) -> dict:
        model_path = get_model_path(fuel_type)
        if not model_path.exists():
            fallback_path = PROCESSED_DIR / "sarimax_model.pkl"
            if not fallback_path.exists():
                raise FileNotFoundError(
                    f"Model file for {fuel_type} (or fallback) not found. Train the engine first."
                )
            model_path = fallback_path

        # Load saved model
        self.results = joblib.load(model_path)

        # Create features for future exogenous data
        future_exog = self.create_features(future_exog)

        # Force exog to numpy if we trained with numpy
        exog_np = future_exog.to_numpy()

        forecast = self.results.get_forecast(steps=steps, exog=exog_np)

        predicted = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Handle both pandas objects and numpy arrays from statsmodels results
        if hasattr(predicted, "to_numpy"):
            predicted = predicted.to_numpy()

        if hasattr(conf_int, "iloc"):
            lower = conf_int.iloc[:, 0].to_numpy()
            upper = conf_int.iloc[:, 1].to_numpy()
        else:
            # Statsmodels returns ndarray if no index was provided during training
            lower = conf_int[:, 0]
            upper = conf_int[:, 1]

        # Sanity check: Cap forecasts at reasonable fuel price extremes
        # This prevents divergent linear trends from showing absurd values
        def sanity_cap(arr):
            return np.clip(arr, 0.0, 500.0)

        return {
            "predicted_mean": sanity_cap(predicted.flatten()),
            "lower_ci": sanity_cap(lower.flatten()),
            "upper_ci": sanity_cap(upper.flatten()),
        }

    # --------------
    # MODEL SUMMARY
    # --------------
    def summary(self):

        if self.results is None:
            raise Exception("Model not trained or loaded.")

        return self.results.summary()
