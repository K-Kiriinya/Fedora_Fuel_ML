# from fuel_pricing.ml.sarimax_model import predict_next
# from fuel_pricing.optimization.pricing import apply_cap
#
# def run_prediction(df, cap):
#     """
#     1. Predict next price using SARIMAX model.
#     2. Apply regulatory cap.
#     3. Return final KES price.
#     """
#
#     forecast = predict_next(df, steps=1)
#
#     predicted_price = float(forecast.iloc[0])
#
#     final_price = apply_cap(predicted_price, cap)
#
#     return final_price
#     # return apply_cap(predict_next(df), 1000)

"""
Prediction Pipeline
-------------------

Handles model loading and forecasting logic.
"""

import pandas as pd
from fuel_pricing.ml.sarimax_model import FuelSARIMAXModel


def run_prediction(df: pd.DataFrame, steps: int = 1):
    """
    Run prediction using trained SARIMAX model.

    Parameters
    ----------
    df : DataFrame
        Historical dataset.
    steps : int
        Number of periods to forecast.

    Returns
    -------
    float or list
        Predicted price(s)
    """

    if "price" not in df.columns:
        raise ValueError("Dataset must contain 'price' column.")

    # Ensure datetime index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    # Extract exogenous variables
    future_exog = df.drop(columns=["price"]).iloc[-steps:].copy()

    model = FuelSARIMAXModel()

    forecast = model.predict(
        steps=steps,
        future_exog=future_exog
    )

    return forecast.tolist()
