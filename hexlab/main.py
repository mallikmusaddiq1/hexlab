#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/main.py

import argparse
import sys

from hexlab import __version__
from hexlab.core import config as c
from hexlab.logic.color import engine
from hexlab.subcommands.command_registry import SUBCOMMANDS
from hexlab.shared.naming import handle_list_color_names_action
from hexlab.shared.logger import log, HexlabArgumentParser
from hexlab.shared.sanitizer import INPUT_HANDLERS
from hexlab.shared.truecolor import ensure_truecolor


def get_color_parser() -> argparse.ArgumentParser:
    """Create argument parser for the main color (inspector) command."""
    parser = HexlabArgumentParser(
        prog="hexlab",
        description="hexlab: a feature-rich color exploration and manipulation tool",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )

    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="show this help message and exit",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"hexlab {__version__}",
        help="show program version and exit",
    )
    parser.add_argument(
        "-hf",
        "--help-full",
        action="store_true",
        help="show full help message including subcommands",
    )
    parser.add_argument(
        "--list-color-names",
        nargs="?",
        const="text",
        default=None,
        choices=["text", "json", "prettyjson"],
        type=INPUT_HANDLERS["color_name"],
        help="list available color names and exit",
    )

    # Color Input Group
    color_input_group = parser.add_mutually_exclusive_group()
    color_input_group.add_argument(
        "-H",
        "--hex",
        dest="hex",
        type=INPUT_HANDLERS["hex"],
        help="6-digit hex color code without # sign",
    )
    color_input_group.add_argument(
        "-r",
        "--random",
        action="store_true",
        help="generate a random hex color",
    )
    color_input_group.add_argument(
        "-cn",
        "--color-name",
        type=INPUT_HANDLERS["color_name"],
        help="color names from 'hexlab --list-color-names'",
    )
    color_input_group.add_argument(
        "-di",
        "--decimal-index",
        dest="decimal_index",
        type=INPUT_HANDLERS["decimal_index"],
        help=f"decimal index of the color (0 to {c.MAX_DEC})",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=INPUT_HANDLERS["seed"],
        default=None,
        help="seed for reproducibility of random",
    )

    # Color Modifications Group
    mod_group = parser.add_argument_group("color modifications")
    mod_group.add_argument(
        "-m",
        "--mods",
        action="store_true",
        help="show all color modifications",
    )
    mod_group.add_argument(
        "-n",
        "--next",
        action="store_true",
        help="show the next color",
    )
    mod_group.add_argument(
        "-p",
        "--previous",
        action="store_true",
        help="show the previous color",
    )
    mod_group.add_argument(
        "-N",
        "--negative",
        action="store_true",
        help="show the inverse color",
    )

    # Technical Information Flags
    info_group = parser.add_argument_group("technical information flags")
    info_group.add_argument(
        "-all",
        "--all-tech-infos",
        action="store_true",
        help="show all technical information",
    )
    info_group.add_argument(
        "-hb",
        "--hide-bars",
        action="store_true",
        help="hide visual color bars",
    )
    info_group.add_argument(
        "-i",
        "--index",
        action="store_true",
        help="show decimal index",
    )
    info_group.add_argument(
        "-rgb",
        "--red-green-blue",
        action="store_true",
        dest="rgb",
        help="show RGB values",
    )
    info_group.add_argument(
        "-l",
        "--luminance",
        action="store_true",
        help="show relative luminance",
    )
    info_group.add_argument(
        "-hsl",
        "--hue-saturation-lightness",
        action="store_true",
        dest="hsl",
        help="show HSL values",
    )
    info_group.add_argument(
        "-hsv",
        "--hue-saturation-value",
        action="store_true",
        dest="hsv",
        help="show HSV values",
    )
    info_group.add_argument(
        "-hwb",
        "--hue-whiteness-blackness",
        action="store_true",
        dest="hwb",
        help="show HWB values",
    )
    info_group.add_argument(
        "-cmyk",
        "--cyan-magenta-yellow-key",
        action="store_true",
        dest="cmyk",
        help="show CMYK values",
    )
    info_group.add_argument(
        "-xyz",
        "--ciexyz",
        dest="xyz",
        action="store_true",
        help="show CIE 1931 XYZ values",
    )
    info_group.add_argument(
        "-lab",
        "--cielab",
        dest="lab",
        action="store_true",
        help="show CIE 1976 LAB values",
    )
    info_group.add_argument(
        "-lch",
        "--lightness-chroma-hue",
        action="store_true",
        dest="lch",
        help="show CIE 1976 LCH values",
    )
    info_group.add_argument(
        "--cieluv",
        "-luv",
        action="store_true",
        dest="cieluv",
        help="show CIE 1976 LUV values",
    )
    info_group.add_argument(
        "--oklab",
        action="store_true",
        dest="oklab",
        help="show OKLAB values",
    )
    info_group.add_argument(
        "--oklch",
        action="store_true",
        dest="oklch",
        help="show OKLCH values",
    )
    info_group.add_argument(
        "-wcag",
        "--contrast",
        action="store_true",
        help="show WCAG contrast ratio",
    )
    info_group.add_argument(
        "--name",
        action="store_true",
        help="show color name if available",
    )

    parser.add_argument(
        "command",
        nargs="?",
        help=argparse.SUPPRESS,
    )
    return parser


def handle_color_command(args: argparse.Namespace) -> None:
    """Entry point for the core color command."""
    parser = get_color_parser()

    if args.list_color_names:
        handle_list_color_names_action(args.list_color_names)
        sys.exit(0)

    if args.help_full:
        parser.print_help()
        for name, module in SUBCOMMANDS.items():
            print("\n" * 2)
            try:
                getter = getattr(module, f"get_{name}_parser")
                getter().print_help()
            except AttributeError:
                log("info", f"help for '{name}' not available")
        sys.exit(0)

    # Routing Validation (if a command was passed in the wrong place)
    if args.command:
        if args.command.lower() in SUBCOMMANDS:
            log("error", f"the '{args.command}' command must be the first argument")
        else:
            log("error", f"unrecognized command or argument: '{args.command}'")
        sys.exit(2)

    # Execution
    engine.run(args, parser)


def main() -> None:
    """Main entry point for hexlab CLI"""
    # Subcommand Routing (Global behavior)
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd in SUBCOMMANDS:
            sys.argv.pop(1)
            ensure_truecolor()
            SUBCOMMANDS[cmd].main()
            sys.exit(0)

    parser = get_color_parser()
    args = parser.parse_args()
    ensure_truecolor()
    handle_color_command(args)


if __name__ == "__main__":
    main()