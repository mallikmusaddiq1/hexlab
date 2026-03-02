#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/logic/gradient/resolver.py

import argparse
import random
import sys
from typing import List

from hexlab.core import conversions as conv
from hexlab.core import config as c
from hexlab.shared.naming import resolve_color_name_or_exit
from hexlab.shared.logger import log
from .engine import get_interpolated_color, convert_rgb_to_space
from .renderer import render_gradient

def resolve_gradient_input(args: argparse.Namespace) -> None:
    """Orchestrate input resolution and gradient generation."""
    if args.seed is not None:
        random.seed(args.seed)

    colors_hex: List[str] = []
    if args.random:
        count = max(2, min(c.MAX_COUNT, args.count if args.count > 0 else random.randint(2, 3)))
        colors_hex = [f"{random.randint(0, c.MAX_DEC):06X}" for _ in range(count)]
    else:
        input_list = []
        if args.hex: input_list.extend(args.hex)
        if args.color_name:
            for nm in args.color_name: input_list.append(resolve_color_name_or_exit(nm))
        if getattr(args, "decimal_index", None):
            for di in args.decimal_index: input_list.append(di)

        if len(input_list) < 2:
            log(
                "error",
                "at least two hex codes, color names, decimal indexes are required for a gradient",
            )
            log("info", "use -H HEX, -cn NAME, -di INDEX multiple times or -r")
            sys.exit(2)

        colors_hex = input_list

    num_steps = max(1, min(c.MAX_STEPS, args.steps))
    if num_steps == 1:
        render_gradient([colors_hex[0]])
        return

    colors_in_space = [convert_rgb_to_space(*conv.hex_to_rgb(h), args.colorspace) for h in colors_hex]
    num_segments = len(colors_in_space) - 1
    total_intervals = num_steps - 1
    gradient_colors: List[str] = []

    for i in range(total_intervals + 1):
        t_global = (i / total_intervals) if total_intervals > 0 else 0
        t_scaled = t_global * num_segments
        idx = min(int(t_scaled), num_segments - 1)
        r_f, g_f, b_f = get_interpolated_color(colors_in_space[idx], colors_in_space[idx+1], t_scaled - idx, args.colorspace)
        gradient_colors.append(conv.rgb_to_hex(r_f, g_f, b_f))

    render_gradient(gradient_colors)