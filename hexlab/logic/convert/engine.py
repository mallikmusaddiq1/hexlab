#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/logic/convert/engine.py

import argparse
import random

from hexlab.core import config as c
from hexlab.core import conversions as conv
from .resolver import resolve_convert_input
from .renderer import render_convert_info

def run(args: argparse.Namespace, parser: argparse.ArgumentParser = None) -> None:
    """Main execution engine for color conversion"""
    if args.seed is not None:
        random.seed(args.seed)

    if args.random:
        r, g, b = conv.hex_to_rgb(f"{random.randint(0, c.MAX_DEC):06X}")
    else:
        r, g, b = resolve_convert_input(args.value, args.from_format)

    out = render_convert_info(r, g, b, args.to_format)

    if args.verbose:
        src = render_convert_info(r, g, b, args.from_format)
        print(f"{src} {c.MSG_BOLD_COLORS['info']}->{c.RESET} {out}")
    else:
        print(out)