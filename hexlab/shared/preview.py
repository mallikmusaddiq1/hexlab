#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/shared/preview.py

import re

from hexlab.core.conversions import hex_to_rgb
from hexlab.core import config as c


def get_visible_len(s: str) -> int:
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return len(ansi_escape.sub('', s))


def print_color_block(hex_code: str, title: str = "color", end: str = "\n") -> None:
    r, g, b = hex_to_rgb(hex_code)
    vis_len = get_visible_len(title)
    padding = " " * max(0, 18 - vis_len)

    print(f"{title}{padding}{c.BOLD_WHITE}:{c.RESET}   \033[48;2;{r};{g};{b}m                {c.RESET}  {c.BOLD_WHITE}#{hex_code}{c.RESET}", end=end)