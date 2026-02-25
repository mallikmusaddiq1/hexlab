# File: formatting.py
from typing import Tuple


def format_colorspace(fmt: str, *args) -> str:
    """
    Format color values into standard CSS-like string representations.
    
    Args:
        fmt (str): The target colorspace format (e.g., 'rgb', 'hsl', 'oklab').
        *args: The numeric values corresponding to the colorspace channels.
        
    Returns:
        str: A formatted string ready for CLI output or display.
    """
    if fmt == 'rgb':
        return f"rgb({args[0]}, {args[1]}, {args[2]})"
    elif fmt == 'hsl':
        h, s, l = args
        return f"hsl({h:.2f}deg, {s * 100:.2f}%, {l * 100:.2f}%)"
    elif fmt == 'hsv':
        h, s, v = args
        return f"hsv({h:.2f}deg, {s * 100:.2f}%, {v * 100:.2f}%)"
    elif fmt == 'hwb':
        h, w, b = args
        return f"hwb({h:.2f}deg {w * 100:.2f}% {b * 100:.2f}%)"
    elif fmt == 'cmyk':
        c, m, y, k = args
        return f"cmyk({c * 100:.2f}%, {m * 100:.2f}%, {y * 100:.2f}%, {k * 100:.2f}%)"
    elif fmt == 'xyz':
        return f"xyz({args[0]:.4f}, {args[1]:.4f}, {args[2]:.4f})"
    elif fmt == 'lab':
        return f"lab({args[0]:.4f} {args[1]:.4f} {args[2]:.4f})"
    elif fmt == 'lch':
        return f"lch({args[0]:.4f} {args[1]:.4f} {args[2]:.4f}deg)"
    elif fmt == 'luv':
        return f"luv({args[0]:.4f} {args[1]:.4f} {args[2]:.4f})"
    elif fmt == 'oklab':
        return f"oklab({args[0]:.4f} {args[1]:.4f} {args[2]:.4f})"
    elif fmt == 'oklch':
        return f"oklch({args[0]:.4f} {args[1]:.4f} {args[2]:.4f}deg)"

    return ""