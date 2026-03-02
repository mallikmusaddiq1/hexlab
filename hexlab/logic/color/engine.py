#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/logic/color/engine.py

import argparse
import random
from typing import Dict, Any

from hexlab.core import config as c
from hexlab.core import conversions as conv
from hexlab.core.luminance import get_luminance
from hexlab.core.contrast import get_wcag_contrast
from .resolver import resolve_color_input
from .renderer import render_color_info

def run(args: argparse.Namespace, parser: argparse.ArgumentParser = None) -> None:
    """Main execution engine for the color command"""

    # If --all-tech-infos is used, activate every key in TECH_INFO_KEYS
    if getattr(args, "all_tech_infos", False):
        for key in c.TECH_INFO_KEYS:
            setattr(args, key, True)

    # Enable neighbor colors flags if --mods is used
    if getattr(args, "mods", False):
        args.next = True
        args.previous = True
        args.negative = True

    base_hex, title = resolve_color_input(args)
    base_dec = int(base_hex, 16)

    neighbors = {}
    if getattr(args, "next", False):
        neighbors["next"] = f"{(base_dec + 1) % (c.MAX_DEC + 1):06X}"
    if getattr(args, "previous", False):
        neighbors["previous"] = f"{(base_dec - 1) % (c.MAX_DEC + 1):06X}"
    if getattr(args, "negative", False):
        neighbors["negative"] = f"{c.MAX_DEC - base_dec:06X}"

    r, g, b = conv.hex_to_rgb(base_hex)
    l_rel = get_luminance(r, g, b)

    tech_data = get_color_data(r, g, b, args)

    wcag_data = None
    if getattr(args, "contrast", False):
        wcag_data = get_wcag_contrast(l_rel)

    render_color_info(
        hex_code=base_hex,
        title=title,
        args=args,
        rgb=(r, g, b),
        luminance=l_rel,
        neighbors=neighbors if neighbors else None,
        tech_data=tech_data,
        wcag_data=wcag_data
    )

def get_color_data(r: int, g: int, b: int, args: argparse.Namespace) -> Dict[str, Any]:
    """Helper to extract technical conversion data with flat logic."""
    data = {}
    
    if getattr(args, "hsl", False):
        data["hsl"] = conv.rgb_to_hsl(r, g, b)        
    if getattr(args, "hsv", False):
        data["hsv"] = conv.rgb_to_hsv(r, g, b)        
    if getattr(args, "hwb", False):
        data["hwb"] = conv.rgb_to_hwb(r, g, b)        
    if getattr(args, "cmyk", False):
        data["cmyk"] = conv.rgb_to_cmyk(r, g, b)        
    if getattr(args, "xyz", False):
        data["xyz"] = conv.rgb_to_xyz(r, g, b)        
    if getattr(args, "lab", False):
        data["lab"] = conv.rgb_to_lab(r, g, b)       
    if getattr(args, "lch", False):
        data["lch"] = conv.rgb_to_lch(r, g, b)        
    if getattr(args, "cieluv", False):
        data["luv"] = conv.rgb_to_luv(r, g, b)        
    if getattr(args, "oklab", False):
        data["oklab"] = conv.rgb_to_oklab(r, g, b)        
    if getattr(args, "oklch", False):
        data["oklch"] = conv.oklab_to_oklch(r, g, b)

    return data