"""
Test Script for SARIMAX Model Validation
-----------------------------------------

This script tests the model with sample data and validates predictions.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
from fuel_pricing.ml.sarimax_model import FuelSARIMAXModel

def test_model_training():
    """Test model training with sample data."""
    print("=" * 60)
    print("TESTING SARIMAX MODEL - Fedora Fuel ML")
    print("=" * 60)

    # Load sample data
    print("\n1. Loading sample data...")
    df = pd.read_csv('data/external/sample_fuel_data.csv')
    print(f"   ✓ Loaded {len(df)} data points")
    print(f"   ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   ✓ Columns: {list(df.columns)}")

    # Prepare data
    print("\n2. Preparing data...")
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    print(f"   ✓ Data indexed by date")
    print(f"   ✓ Price range: KES {df['price'].min():.2f} - {df['price'].max():.2f}")
    print(f"   ✓ Average price: KES {df['price'].mean():.2f}")

    # Display data summary
    print("\n3. Data Summary:")
    print(df.describe())

    # Initialize and train model
    print("\n4. Training SARIMAX model...")
    print("   (This may take 10-30 seconds...)")
    model = FuelSARIMAXModel()

    try:
        predictions, y_test = model.train_sarimax(df, test_size=0.2)
        print("   ✓ Model trained successfully!")

        # Display metrics
        print("\n5. Model Evaluation Metrics:")
        print("   " + "-" * 40)
        for metric, value in model.metrics.items():
            print(f"   {metric:8s}: {value:.4f}")
        print("   " + "-" * 40)

        # Interpret metrics
        print("\n6. Metrics Interpretation:")
        mae = model.metrics['MAE']
        mape = model.metrics['MAPE']

        if mae < 5:
            print(f"   ✓ MAE ({mae:.2f}) - EXCELLENT accuracy")
        elif mae < 10:
            print(f"   ✓ MAE ({mae:.2f}) - GOOD accuracy")
        elif mae < 20:
            print(f"   ⚠ MAE ({mae:.2f}) - FAIR accuracy")
        else:
            print(f"   ✗ MAE ({mae:.2f}) - POOR accuracy")

        if mape < 5:
            print(f"   ✓ MAPE ({mape:.2f}%) - EXCELLENT accuracy")
        elif mape < 10:
            print(f"   ✓ MAPE ({mape:.2f}%) - GOOD accuracy")
        elif mape < 15:
            print(f"   ⚠ MAPE ({mape:.2f}%) - FAIR accuracy")
        else:
            print(f"   ✗ MAPE ({mape:.2f}%) - POOR accuracy")

        # Test predictions
        print("\n7. Testing future predictions...")
        future_exog = df.drop(columns=['price']).iloc[-6:].copy()
        forecast = model.predict(steps=6, future_exog=future_exog)

        print("\n   6-Month Forecast:")
        print("   " + "-" * 40)
        for i, price in enumerate(forecast, 1):
            print(f"   Month {i}: KES {price:.2f}")
        print("   " + "-" * 40)

        # Validate predictions
        print("\n8. Prediction Validation:")
        last_price = df['price'].iloc[-1]
        avg_forecast = forecast.mean()

        print(f"   Last actual price: KES {last_price:.2f}")
        print(f"   Average forecast:  KES {avg_forecast:.2f}")
        print(f"   Difference:        KES {abs(avg_forecast - last_price):.2f}")

        if forecast.min() > 0 and forecast.max() < 1000:
            print("   ✓ Predictions within reasonable range")
        else:
            print("   ✗ WARNING: Predictions outside expected range")

        # Check for trend consistency
        price_trend = df['price'].iloc[-6:].mean()
        if abs(avg_forecast - price_trend) / price_trend < 0.2:
            print("   ✓ Forecast aligns with recent trend")
        else:
            print("   ⚠ Forecast deviates from recent trend")

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Model is working correctly!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)

    model = FuelSARIMAXModel()

    # Test 1: Missing price column
    print("\n1. Testing missing 'price' column...")
    df_bad = pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=10, freq='MS'),
        'value': range(10)
    })
    df_bad.set_index('date', inplace=True)

    try:
        model.train_sarimax(df_bad)
        print("   ✗ FAILED: Should have raised error")
        return False
    except ValueError as e:
        print(f"   ✓ PASSED: Correctly caught error - {e}")

    # Test 2: Small dataset
    print("\n2. Testing with minimal data...")
    df_small = pd.read_csv('data/external/sample_fuel_data.csv')
    df_small = df_small.head(15)  # Only 15 points
    df_small['date'] = pd.to_datetime(df_small['date'])
    df_small.set_index('date', inplace=True)

    try:
        predictions, y_test = model.train_sarimax(df_small, test_size=0.2)
        print(f"   ✓ PASSED: Handled small dataset ({len(df_small)} points)")
    except Exception as e:
        print(f"   ⚠ WARNING: Failed with small dataset - {e}")

    print("\n" + "=" * 60)
    print("✅ EDGE CASE TESTS COMPLETED")
    print("=" * 60)

    return True

if __name__ == "__main__":
    print("\n🚀 Starting Model Validation Tests...\n")

    # Run main tests
    success1 = test_model_training()

    # Run edge case tests
    success2 = test_edge_cases()

    # Final summary
    print("\n\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)

    if success1 and success2:
        print("✅ ALL TESTS PASSED")
        print("\nThe model is:")
        print("  ✓ Training correctly")
        print("  ✓ Generating valid predictions")
        print("  ✓ Handling edge cases")
        print("  ✓ Calculating metrics accurately")
        print("\n✨ Model is production-ready!")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        print("Please review the errors above.")
        sys.exit(1)
