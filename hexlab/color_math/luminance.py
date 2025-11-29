from .conversions import _srgb_to_linear

def get_luminance(r: int, g: int, b: int) -> float:
    return 0.2126 * _srgb_to_linear(r) + 0.7152 * _srgb_to_linear(g) + 0.0722 * _srgb_to_linear(b)
