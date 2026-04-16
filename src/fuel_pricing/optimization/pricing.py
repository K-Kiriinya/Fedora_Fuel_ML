def apply_cap(predicted_price: float, cap: float):
    """Enforce regulatory price cap and non-negativity floor."""
    return max(0.0, min(predicted_price, cap))