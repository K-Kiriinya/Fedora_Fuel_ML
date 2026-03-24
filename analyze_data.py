import pandas as pd
from pathlib import Path

prices_file = Path("data/processed/local_prices/kenyan_oil_prices_monthly_clean.csv")
df_main = pd.read_csv(prices_file)
df_main['month'] = pd.to_datetime(df_main['month'])
print(f"Main Target (Oil Prices): {df_main['month'].min().date()} to {df_main['month'].max().date()} ({len(df_main)} rows)")

indicators_dir = Path("data/processed/market_indicators")
for f in indicators_dir.glob("*.csv"):
    try:
        df = pd.read_csv(f)
        if 'month' in df.columns:
            df['month'] = pd.to_datetime(df['month'])
            print(f"- {f.name}: {df['month'].min().date()} to {df['month'].max().date()} ({len(df)} rows)")
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            print(f"- {f.name}: {df['date'].min().date()} to {df['date'].max().date()} ({len(df)} rows) [uses 'date']")
        else:
            print(f"- {f.name}: No month/date column found.")
    except Exception as e:
        print(f"- {f.name}: Error reading - {e}")
