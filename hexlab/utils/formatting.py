# File: formatting.py
from typing import Tuple


def format_colorspace(fmt: str, *args) -> str:
    """Format color values into standard string representations."""
    if fmt == 'rgb':
        return f"rgb({args[0]}, {args[1]}, {args[2]})"
    elif fmt == 'hsl':
        h, s, l = args
        return f"hsl({h:.1f}°, {s * 100:.1f}%, {l * 100:.1f}%)"
    elif fmt == 'hsv':
        h, s, v = args
        return f"hsv({h:.1f}°, {s * 100:.1f}%, {v * 100:.1f}%)"
    elif fmt == 'hwb':
        h, w, b = args
        return f"hwb({h:.1f}°, {w * 100:.1f}%, {b * 100:.1f}%)"
    elif fmt == 'cmyk':
        c, m, y, k = args
        return f"cmyk({c * 100:.1f}%, {m * 100:.1f}%, {y * 100:.1f}%, {k * 100:.1f}%)"
    elif fmt == 'xyz':
        return f"xyz({args[0]:.4f}, {args[1]:.4f}, {args[2]:.4f})"
    elif fmt == 'lab':
        return f"lab({args[0]:.4f}, {args[1]:.4f}, {args[2]:.4f})"
    elif fmt == 'lch':
        return f"lch({args[0]:.4f}, {args[1]:.4f}, {args[2]:.4f}°)"
    elif fmt == 'luv':
        return f"luv({args[0]:.4f}, {args[1]:.4f}, {args[2]:.4f})"
    elif fmt == 'oklab':
        return f"oklab({args[0]:.4f}, {args[1]:.4f}, {args[2]:.4f})"
    elif fmt == 'oklch':
        return f"oklch({args[0]:.4f}, {args[1]:.4f}, {args[2]:.4f}°)"

    return ""
