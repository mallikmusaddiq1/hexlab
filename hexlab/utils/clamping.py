# File: clamping.py
def _clamp01(v: float) -> float:
    """Clamp value to range [0.0, 1.0]."""
    # Handle NaN
    if v != v:
        return 0.0
    return max(0.0, min(1.0, v))


def _clamp255(v: float) -> float:
    """Clamp value to range [0.0, 255.0]."""
    # Handle NaN
    if v != v:
        return 0.0
    return max(0.0, min(255.0, v))
