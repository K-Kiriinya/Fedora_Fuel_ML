import pandas as pd


def load_and_prepare(path: str):
    """
    Loads CSV and prepares it for SARI0MAX.

    Steps:
    1. Reads CSV
    2. Converts 'date' to datetime
    3. Sorts chronologically
    4. Sets date as index (required for time series)
    """

    df = pd.read_csv(path)

    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Sort by date (chronological order)
    df = df.sort_values("date")

    # Set date as index
    df = df.set_index("date")

    return df
