#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/subcommands/convert.py

import argparse
import sys

from hexlab.core import config as c
from hexlab.shared.logger import HexlabArgumentParser, log
from hexlab.shared.sanitizer import INPUT_HANDLERS
from hexlab.shared.formatting import format_colorspace
from hexlab.shared.truecolor import ensure_truecolor
from hexlab.logic.convert import engine


def get_convert_parser() -> argparse.ArgumentParser:
    """Create argument parser for convert command with detailed examples."""
    parser = HexlabArgumentParser(
        prog="hexlab convert",
        description="hexlab convert: convert a color value from one format to another",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )
    
    formats_list = "hex rgb hsl hsv hwb cmyk xyz lab lch luv oklab oklch index name"
    
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="show this help message and exit",
    )
    
    parser.add_argument(
        "-f",
        "--from-format",
        required=True,
        type=INPUT_HANDLERS["from_format"],
        help="the format to convert from\n" f"all formats: {formats_list}",
    )
    
    parser.add_argument(
        "-t",
        "--to-format",
        required=True,
        type=INPUT_HANDLERS["to_format"],
        help="the format to convert to\n" f"all formats: {formats_list}",
    )

    # Help message examples (Formatting for consistency)
    ex_rgb = format_colorspace("rgb", 0, 0, 0)
    ex_hsl = format_colorspace("hsl", 0, 0, 0).replace("%", "%%")
    ex_hsv = format_colorspace("hsv", 0, 0, 0).replace("%", "%%")
    ex_hwb = format_colorspace("hwb", 0, 0, 1.0).replace("%", "%%")
    ex_cmyk = format_colorspace("cmyk", 0, 0, 0, 1.0).replace("%", "%%")
    ex_xyz = format_colorspace("xyz", 0, 0, 0)
    ex_lab = format_colorspace("lab", 0, 0, 0)
    ex_lch = format_colorspace("lch", 0, 0, 0)
    ex_luv = format_colorspace("luv", 0, 0, 0)
    ex_oklab = format_colorspace("oklab", 0.0001, 0, 0)
    ex_oklch = format_colorspace("oklch", 0.0001, 0, 90)

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-v",
        "--value",
        type=str,
        help=(
            "color value to convert must be in quotes\n"
            "examples:\n"
            '  -v "000000"\n'
            '  -v "0"\n'
            '  -v "black"\n'
            f'  -v "{ex_rgb}"\n'
            f'  -v "{ex_hsl}"\n'
            f'  -v "{ex_hsv}"\n'
            f'  -v "{ex_hwb}"\n'
            f'  -v "{ex_cmyk}"\n'
            f'  -v "{ex_xyz}"\n'
            f'  -v "{ex_lab}"\n'
            f'  -v "{ex_lch}"\n'
            f'  -v "{ex_luv}"\n'
            f'  -v "{ex_oklab}"\n'
            f'  -v "{ex_oklch}"'
        ),
    )
    
    input_group.add_argument(
        "-r",
        "--random",
        action="store_true",
        help="generate a random value for the --from-format",
    )
    
    parser.add_argument(
        "-s",
        "--seed",
        type=INPUT_HANDLERS["seed"],
        default=None,
        help="seed for reproducibility of random",
    )
    
    parser.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        help="print the conversion verbosely",
    )
    
    return parser

def handle_convert_command(args: argparse.Namespace) -> None:
    """Entry point for the convert subcommand.
    
    Resolves aliases and passes execution to the engine.
    """
    parser = get_convert_parser()

    try:
        args.from_format = c.FORMAT_ALIASES[args.from_format]
        args.to_format = c.FORMAT_ALIASES[args.to_format]
    except KeyError as e:
        log("error", f"invalid format: {e}")
        sys.exit(2)

    engine.run(args, parser)

def main() -> None:
    """Main entry point following adjust.py structure."""
    parser = get_convert_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    handle_convert_command(args)

if __name__ == "__main__":
    main()