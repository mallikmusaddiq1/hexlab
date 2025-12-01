def _clamp01(v: float) -> float:
    if v != v:
        return 0.0
    return max(0.0, min(1.0, v))


def _clamp255(v: float) -> float:
    if v != v:
        return 0.0
    return max(0.0, min(255.0, v))
