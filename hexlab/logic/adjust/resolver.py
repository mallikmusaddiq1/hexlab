#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/logic/adjust/resolver.py

import argparse
import random
import sys
from typing import Tuple

from hexlab.core import config as c
from hexlab.shared.naming import (
    get_title_for_hex,
    resolve_color_name_or_exit,
)
from hexlab.shared.logger import log

def resolve_adjust_input(args: argparse.Namespace) -> Tuple[str, str]:
    """Resolve color input for the adjust command into hex and title.
    
    Strictly handles input logic: Random, Color Names, Hex, or Decimal Index.
    """
    if args.seed is not None:
        random.seed(args.seed)

    base_hex, title = None, "original"

    # 1. Random Color
    if args.random:
        base_hex, title = f"{random.randint(0, c.MAX_DEC):06X}", "random"
    
    # 2. Color Name
    elif args.color_name:
        base_hex = resolve_color_name_or_exit(args.color_name)
        title = get_title_for_hex(base_hex)
        if title.lower() == "unknown":
            title = args.color_name
            
    # 3. Direct Hex
    elif args.hex:
        base_hex = args.hex
        title = get_title_for_hex(args.hex)
        
    # 4. Decimal Index
    elif getattr(args, "decimal_index", None):
        base_hex = args.decimal_index
        title = f"index {int(args.decimal_index, 16)}"

    # Error handling if no input provided
    if not base_hex:
        log("error", "one of the arguments -H/--hex -r/--random -cn/--color-name -di/--decimal-index is required")
        sys.exit(2)

    return base_hex, title