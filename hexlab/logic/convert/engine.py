#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/logic/convert/engine.py

import random
from hexlab.core import config as c
from hexlab.core.conversions import hex_to_rgb
from .resolve import to_rgb
from .composer import to_string

def run(val, f_fmt, t_fmt, is_rand=False, seed=None, verbose=False) -> str:
    """Main execution engine for color conversion."""
    if seed is not None: random.seed(seed)
    
    r, g, b = hex_to_rgb(f"{random.randint(0, c.MAX_DEC):06X}") if is_rand else to_rgb(val, f_fmt)
    out = to_string(r, g, b, t_fmt)

    if verbose:
        src = to_string(r, g, b, f_fmt)
        return f"{src} {c.MSG_BOLD_COLORS['info']}->{c.RESET} {out}"
    
    return out