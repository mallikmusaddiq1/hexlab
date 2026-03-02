#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/subcommands/gradient.py

import argparse
import sys
from hexlab.core import config as c
from hexlab.shared.logger import HexlabArgumentParser
from hexlab.shared.sanitizer import INPUT_HANDLERS
from hexlab.shared.truecolor import ensure_truecolor
from hexlab.logic.gradient.resolver import resolve_gradient_input

def get_gradient_parser() -> argparse.ArgumentParser:
    """Create argument parser for gradient command."""
    parser = HexlabArgumentParser(
        prog="hexlab gradient",
        description="hexlab gradient: generate color gradients between multiple hex codes",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-H",
        "--hex",
        action="append",
        type=INPUT_HANDLERS["hex"],
        help="use -H HEX multiple times for inputs"
    )
    parser.add_argument(
        "-r",
        "--random",
        action="store_true",
        help="generate gradient from random colors"
    )
    parser.add_argument(
        "-cn",
        "--color-name",
        action="append",
        type=INPUT_HANDLERS["color_name"],
        help="use -cn NAME multiple times for inputs by name"
    )
    parser.add_argument(
        "-di",
        "--decimal-index",
        action="append",
        type=INPUT_HANDLERS["decimal_index"],
        help="use -di INDEX multiple times for inputs by decimal index"
    )
    parser.add_argument(
        "-S",
        "--steps",
        type=INPUT_HANDLERS["steps"],
        default=10,
        help=f"total steps in gradient (default: 10, max: {c.MAX_STEPS})",
    )
    parser.add_argument(
        "-cs",
        "--colorspace",
        default="lab",
        type=INPUT_HANDLERS["colorspace"],
        choices=["srgb", "srgblinear", "lab", "lch", "oklab", "oklch", "luv"],
        help="colorspace interpolation (default: lab)"
    )
    parser.add_argument(
        "-c",
        "--count",
        type=INPUT_HANDLERS["count"],
        default=0,
        help=f"number of random colors for input (default: 2-3, max: {c.MAX_COUNT}"
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=INPUT_HANDLERS["seed"],
        default=None,
        help="seed for reproducibility of random"
    )

    return parser

def main() -> None:
    """Main entry point for gradient command."""
    parser = get_gradient_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    resolve_gradient_input(args)

if __name__ == "__main__":
    main()