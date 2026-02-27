#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/logic/convert/resolve.py

import re
import sys
from typing import Tuple

from hexlab.core import conversions as conv
from hexlab.core import config as c
from hexlab.shared.naming import resolve_color_name_or_exit
from hexlab.shared.parser import STRING_PARSERS
from hexlab.shared.logger import log

def to_rgb(val: str, fmt: str) -> Tuple[int, int, int]:
    """Resolves any input format into an RGB tuple."""
    if fmt == "hex":
        return conv.hex_to_rgb(val)
        
    elif fmt == "index":
        try:
            dec = int(re.findall(r"[-+]?\d+", str(val))[0])
            dec = max(0, min(c.MAX_DEC, dec))
            return conv.hex_to_rgb(f"{dec:06X}")
        except:
            log("error", f"invalid index '{val}'")
            sys.exit(2)
            
    elif fmt == "name":
        return conv.hex_to_rgb(resolve_color_name_or_exit(val))

    if fmt in STRING_PARSERS:
        v = STRING_PARSERS[fmt](val)
        maps = {
            "rgb": lambda: v,
            "hsl": lambda: conv.hsl_to_rgb(*v),
            "hsv": lambda: conv.hsv_to_rgb(*v),
            "hwb": lambda: conv.hwb_to_rgb(*v),
            "cmyk": lambda: conv.cmyk_to_rgb(*v),
            "xyz": lambda: conv.xyz_to_rgb(*v),
            "lab": lambda: conv.lab_to_rgb(*v),
            "lch": lambda: conv.lch_to_rgb(*v),
            "oklab": lambda: conv.oklab_to_rgb(*v),
            "oklch": lambda: conv.oklch_to_rgb(*v),
            "luv": lambda: conv.luv_to_rgb(*v)
        }
        return maps[fmt]() if fmt in maps else (0, 0, 0)

    return (0, 0, 0)