import pytest
import pandas as pd
import numpy as np
from fuel_pricing.ml.sarimax_model import FuelSARIMAXModel

def test_metrics_calculation():
    """Test metrics logic for accuracy."""
    model = FuelSARIMAXModel()
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.1, 1.9, 3.2]
    
    metrics = model.calculate_metrics(y_true, y_pred)
    
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAPE" in metrics
    assert metrics["MAE"] > 0
    assert metrics["MAPE"] > 0

def test_feature_engineering():
    """Test feature generation function."""
    model = FuelSARIMAXModel()
    df = pd.DataFrame({
        "pipeline_burst": [0, 1],
        "exchange_rate_usd_ksh": [110, 120],
        "fuel_shortage": [0, 1],
        "inflation_rate": [5.0, 6.0],
        "transport_cost_index": [100, 105]
    })
    
    df_feat = model.create_features(df)
    
    assert "burst_fx_interaction" in df_feat.columns
    assert "shortage_inflation_interaction" in df_feat.columns
    assert "log_transport_cost" in df_feat.columns
    assert df_feat["log_transport_cost"].iloc[0] == np.log1p(100)
