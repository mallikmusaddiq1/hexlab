#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/logic/gradient/renderer.py

from typing import List

from hexlab.core import config as c
from hexlab.shared.preview import print_color_block

def render_gradient(gradient_colors: List[str]) -> None:
    """Print the generated gradient steps to the terminal."""
    print()
    for i, hex_code in enumerate(gradient_colors):
        label = f"{c.MSG_BOLD_COLORS['info']}step{f'{i + 1}':>11}{c.RESET}"
        print_color_block(hex_code, label)
    print()