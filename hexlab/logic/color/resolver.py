#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/logic/color/resolver.py

import argparse
import random
import sys
from typing import Tuple

from hexlab.core import config as c
from hexlab.shared.naming import get_title_for_hex, resolve_color_name_or_exit
from hexlab.shared.logger import log


def resolve_color_input(args: argparse.Namespace) -> Tuple[str, str]:
    """Resolve raw CLI input into a base hex"""

    if args.seed is not None:
        random.seed(args.seed)

    clean_hex = None
    title = "current"

    if args.random:
        clean_hex = f"{random.randint(0, c.MAX_DEC):06X}"
        title = "random"
    elif args.color_name:
        clean_hex = resolve_color_name_or_exit(args.color_name)
        title = get_title_for_hex(clean_hex)
    elif args.hex:
        clean_hex = args.hex
        title = get_title_for_hex(clean_hex)
    elif getattr(args, "decimal_index", None) is not None:
        clean_hex = args.decimal_index
        title = get_title_for_hex(clean_hex, f"index {int(clean_hex, 16)}")

    else:
        log(
            "error",
            "one of the arguments -H/--hex -r/--random -cn/--color-name -di/--decimal-index is required",
        )
        log("info", "use 'hexlab --help' for more information")
        sys.exit(2)

    return clean_hex, title