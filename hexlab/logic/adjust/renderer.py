#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/logic/adjust/renderer.py

import argparse
from typing import List, Tuple

from hexlab.core import config as c
from hexlab.shared.logger import log
from hexlab.shared.preview import print_color_block

def render_adjust_info(
    base_hex: str, 
    res_hex: str, 
    title: str, 
    mods: List[Tuple[str, str]], 
    args: argparse.Namespace
) -> None:
    
    base_hex_upper = base_hex.upper()
    is_hex_title = (
        isinstance(title, str)
        and title.startswith("#")
        and title[1:].upper() == base_hex_upper
    )

    print()
    label = "original" if is_hex_title else title
    print_color_block(base_hex, f"{c.BOLD_WHITE}{label}{c.RESET}")
    
    if mods:
        print()
        print_color_block(res_hex, f"{c.MSG_BOLD_COLORS['info']}adjusted{c.RESET}")

    if getattr(args, "verbose", False):
        print()
        if not mods:
            log("info", "steps: no adjustments applied yet")
        else:
            log("info", "steps:")
            for i, (label, val) in enumerate(mods, 1):
                if getattr(args, "steps_compact", False):
                    print(f"{c.MSG_COLORS['info']}    {i}. {label}")
                else:
                    detail = f" {val}" if val else ""
                    print(f"{c.MSG_COLORS['info']}    {i}. {label}{detail}")
    print()