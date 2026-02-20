def apply_cap(predicted_price: float, cap: float):
    """Enforce regulatory price cap."""
    return min(predicted_price, cap)