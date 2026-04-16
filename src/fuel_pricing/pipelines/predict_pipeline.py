"""
Prediction Pipeline:  Run prediction using trained SARIMAX model.
"""

import pandas as pd
from fuel_pricing.ml.sarimax_model import FuelSARIMAXModel


def run_prediction(df: pd.DataFrame, steps: int = 1):

    if "price" not in df.columns:
        raise ValueError("Dataset must contain 'price' column.")

    # Ensure datetime index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    # Extract exogenous variables
    future_exog = df.drop(columns=["price"]).iloc[-steps:].copy()

    model = FuelSARIMAXModel()

    # model.predict() now returns a dict; extract predicted_mean
    forecast_result = model.predict(steps=steps, future_exog=future_exog)

    return forecast_result["predicted_mean"].tolist()
