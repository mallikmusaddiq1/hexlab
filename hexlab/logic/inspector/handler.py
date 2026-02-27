#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/logic/engine/handler.py

import argparse
import random
import sys
from hexlab.core import config as c
from hexlab.logic.inspector.renderer import print_color_and_info
from hexlab.shared.naming import (
    get_title_for_hex,
    resolve_color_name_or_exit,
)
from hexlab.shared.logger import log


def handle_color_command(args: argparse.Namespace) -> None:
    # Enable all technical info flags if --all is used
    if args.all_tech_infos:
        for key in c.TECH_INFO_KEYS:
            setattr(args, key, True)

    # Enable neighbor colors flags if --mods is used
    if getattr(args, "mods", False):
        args.next = True
        args.previous = True
        args.negative = True

    clean_hex = None
    title = "current"

    if args.seed is not None:
        random.seed(args.seed)

    # 1. Resolve Input Color
    if args.random:
        current_dec = random.randint(0, c.MAX_DEC)
        clean_hex = f"{current_dec:06X}"
        title = "random"
    elif args.color_name:
        clean_hex = resolve_color_name_or_exit(args.color_name)
        title = get_title_for_hex(clean_hex)
    elif args.hex:
        clean_hex = args.hex
        title = get_title_for_hex(clean_hex)
    elif getattr(args, "decimal_index", None) is not None:
        clean_hex = args.decimal_index
        idx = int(clean_hex, 16)
        title = get_title_for_hex(clean_hex, f"index {idx}")
    else:
        log(
            "error",
            "one of the arguments -H/--hex -r/--random -cn/--color-name -di/--decimal-index is required",
        )
        log("info", "use 'hexlab --help' for more information")
        sys.exit(2)

    current_dec = int(clean_hex, 16)

    # 2. Compute Neighbors (Pure Logic)
    neighbors = {}
    if args.next:
        next_dec = (current_dec + 1) % (c.MAX_DEC + 1)
        neighbors["next"] = f"{next_dec:06X}"
    if args.previous:
        prev_dec = (current_dec - 1) % (c.MAX_DEC + 1)
        neighbors["previous"] = f"{prev_dec:06X}"
    if args.negative:
        neg_dec = c.MAX_DEC - current_dec
        neighbors["negative"] = f"{neg_dec:06X}"

    # 3. Delegate to View
    print_color_and_info(clean_hex, title, args, neighbors=neighbors if neighbors else None)
