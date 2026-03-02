#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/logic/convert/renderer.py

from hexlab.core import config as c
from hexlab.core import conversions as conv
from hexlab.shared.naming import get_title_for_hex
from hexlab.shared.formatting import format_colorspace

def render_convert_info(r: int, g: int, b: int, fmt: str) -> str:
    """Composes RGB into a formatted output string."""
    def bold(t): return f"{c.BOLD_WHITE}{t}{c.RESET}"
    hx = conv.rgb_to_hex(r, g, b)

    if fmt == "hex": return bold(f"#{hx.upper()}")
    if fmt == "index": return bold(int(hx, 16))
    if fmt == "name":
        n = get_title_for_hex(hx, fallback="unknown")
        return bold(n) if n != "unknown" else f"{c.MSG_BOLD_COLORS['error']}unknown{c.RESET}"

    maps = {
        "rgb": lambda: (int(round(r)), int(round(g)), int(round(b))),
        "hsl": lambda: conv.rgb_to_hsl(r, g, b),
        "hsv": lambda: conv.rgb_to_hsv(r, g, b),
        "hwb": lambda: conv.rgb_to_hwb(r, g, b),
        "cmyk": lambda: conv.rgb_to_cmyk(r, g, b),
        "xyz": lambda: conv.rgb_to_xyz(r, g, b),
        "lab": lambda: conv.rgb_to_lab(r, g, b),
        "lch": lambda: conv.rgb_to_lch(r, g, b),
        "oklab": lambda: conv.rgb_to_oklab(r, g, b),
        "oklch": lambda: conv.rgb_to_oklch(r, g, b),
        "luv": lambda: conv.rgb_to_luv(r, g, b)
    }
    
    return bold(format_colorspace(fmt, *maps[fmt]())) if fmt in maps else ""