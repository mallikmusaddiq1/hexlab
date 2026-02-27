#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/subcommands/adjust.py

import argparse
import sys

from hexlab.logic.adjust import pipeline
from hexlab.shared.logger import HexlabArgumentParser
from hexlab.shared.sanitizer import INPUT_HANDLERS
from hexlab.shared.truecolor import ensure_truecolor


def get_adjust_parser() -> argparse.ArgumentParser:
    """Create argument parser for adjust command.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    p = HexlabArgumentParser(
        prog="hexlab adjust",
        description=(
            "hexlab adjust: advanced color manipulation\n\n"
            "by default, all operations are deterministic and applied in a fixed pipeline"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        usage=argparse.SUPPRESS,
    )
    p.add_argument(
        "usage_hack",
        nargs="?",
        help=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
    )
    original_print_help = p.print_help

    def custom_print_help(file=None):
        if file is None:
            file = sys.stdout
        print(
            "usage: hexlab adjust [-h] (-H HEX | -r | -cn NAME | -di INDEX) [OPTIONS...]",
            file=file,
        )
        print()
        original_print_help(file)

    p.print_help = custom_print_help
    
    # Input group setup
    input_group = p.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "-H",
        "--hex",
        type=INPUT_HANDLERS["hex"],
        help="base hex code",
    )
    input_group.add_argument(
        "-r",
        "--random",
        action="store_true",
        help="use a random base",
    )
    input_group.add_argument(
        "-cn",
        "--color-name",
        type=INPUT_HANDLERS["color_name"],
        help="base color name",
    )
    input_group.add_argument(
        "-di",
        "--decimal-index",
        type=INPUT_HANDLERS["decimal_index"],
        help=f"base decimal index",
    )
    
    # Global options
    p.add_argument(
        "-s",
        "--seed",
        type=INPUT_HANDLERS["seed"],
        help="seed for reproducibility of random",
    )
    p.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        help="log detailed pipeline steps",
    )
    p.add_argument(
        "--steps-compact",
        dest="steps_compact",
        action="store_true",
        help="show only operation names in verbose steps, hide numeric values",
    )
    p.add_argument(
        "-cp",
        "--custom-pipeline",
        action="store_true",
        help="disable fixed pipeline and apply adjustments in the order provided on CLI",
    )
    p.add_argument(
        "--list-fixed-pipeline",
        dest="list_fixed_pipeline",
        action="store_true",
        help="print the fixed pipeline order and exit",
    )

    # HSL and Hue group
    ga = p.add_argument_group("hsl and hue")
    ga.add_argument(
        "-l",
        "--lighten",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="increase lightness (0 to 100%%)",
    )
    ga.add_argument(
        "-d",
        "--darken",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="decrease lightness (0 to 100%%)",
    )
    ga.add_argument(
        "-sat",
        "--saturate",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="increase saturation (0 to 100%%)",
    )
    ga.add_argument(
        "-des",
        "--desaturate",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="decrease saturation (0 to 100%%)",
    )
    ga.add_argument(
        "-rot",
        "--rotate",
        type=INPUT_HANDLERS["float_signed_360"],
        metavar="N",
        help="rotate hue in HSL (-360 to 360°)",
    )
    ga.add_argument(
        "-rotl",
        "--rotate-oklch",
        dest="rotate_oklch",
        type=INPUT_HANDLERS["float_signed_360"],
        metavar="N",
        help="rotate hue in OKLCH (-360 to 360°)",
    )

    # Tone and Vividness group
    adv_group = p.add_argument_group("tone and vividness")
    bgroup = adv_group.add_mutually_exclusive_group()
    bgroup.add_argument(
        "-br",
        "--brightness",
        type=INPUT_HANDLERS["float_signed_100"],
        metavar="N",
        help="adjust linear brightness (-100 to 100%%)",
    )
    bgroup.add_argument(
        "-brs",
        "--brightness-srgb",
        dest="brightness_srgb",
        type=INPUT_HANDLERS["float_signed_100"],
        metavar="N",
        help="adjust sRGB brightness (-100 to 100%%)",
    )
    adv_group.add_argument(
        "-ct",
        "--contrast",
        type=INPUT_HANDLERS["float_signed_100"],
        metavar="N",
        help="adjust contrast (-100 to 100%%)",
    )
    adv_group.add_argument(
        "-cb",
        "--chroma-oklch",
        dest="chroma_oklch",
        type=INPUT_HANDLERS["float_signed_100"],
        metavar="N",
        help="scale chroma in OKLCH (-100 to 100%%)",
    )
    adv_group.add_argument(
        "-whiten",
        "--whiten-hwb",
        dest="whiten_hwb",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="adjust white in HWB (0 to 100%%)",
    )
    adv_group.add_argument(
        "-blacken",
        "--blacken-hwb",
        dest="blacken_hwb",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="adjust black in HWB (0 to 100%%)",
    )
    adv_group.add_argument(
        "-warm",
        "--warm-oklab",
        dest="warm_oklab",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="adjust warmth (0 to 100%%)",
    )
    adv_group.add_argument(
        "-cool",
        "--cool-oklab",
        dest="cool_oklab",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="adjust coolness (0 to 100%%)",
    )
    adv_group.add_argument(
        "-ll",
        "--lock-luminance",
        action="store_true",
        help="preserve base OKLAB lightness perceptual-L",
    )
    adv_group.add_argument(
        "-lY",
        "--lock-rel-luminance",
        dest="lock_rel_luminance",
        action="store_true",
        help="preserve base relative luminance",
    )
    adv_group.add_argument(
        "--target-rel-lum",
        dest="target_rel_lum",
        type=INPUT_HANDLERS["float"],
        metavar="Y",
        help="set absolute target relative luminance (0.0 to 1.0)",
    )
    adv_group.add_argument(
        "--min-contrast-with",
        dest="min_contrast_with",
        type=INPUT_HANDLERS["hex"],
        metavar="HEX",
        help="target hex color to ensure contrast against",
    )
    adv_group.add_argument(
        "--min-contrast",
        dest="min_contrast",
        type=INPUT_HANDLERS["float"],
        metavar="RATIO",
        help=(
            "minimum WCAG contrast ratio with --min-contrast-with, "
            "best effort within srgb gamut"
        ),
    )
    adv_group.add_argument(
        "--gamma",
        dest="gamma",
        type=INPUT_HANDLERS["float"],
        metavar="N",
        help="gamma correction in linear space (>0, typical 0.5 to 3.0)",
    )
    adv_group.add_argument(
        "--exposure",
        dest="exposure",
        type=INPUT_HANDLERS["float_signed_100"],
        metavar="N",
        help="exposure adjustment in stops (-100 to 100%%, 10%% = 1 stop)",
    )
    adv_group.add_argument(
        "-vb",
        "--vibrance-oklch",
        dest="vibrance_oklch",
        type=INPUT_HANDLERS["float_signed_100"],
        metavar="N",
        help="adjust vibrance in OKLCH, boosting low chroma (-100 to 100%%)",
    )

    # Filters and Channels group
    filter_group = p.add_argument_group("filters and channels")
    filter_group.add_argument(
        "-g",
        "--grayscale",
        action="store_true",
        help="convert to grayscale",
    )
    filter_group.add_argument(
        "-inv",
        "--invert",
        action="store_true",
        help="invert color",
    )
    filter_group.add_argument(
        "-sep",
        "--sepia",
        action="store_true",
        help="apply sepia filter",
    )
    filter_group.add_argument(
        "-red",
        "--red-channel",
        type=INPUT_HANDLERS["int_channel"],
        metavar="N",
        help="add or subtract red (-255 to 255)",
    )
    filter_group.add_argument(
        "-green",
        "--green-channel",
        type=INPUT_HANDLERS["int_channel"],
        metavar="N",
        help="add or subtract green (-255 to 255)",
    )
    filter_group.add_argument(
        "-blue",
        "--blue-channel",
        type=INPUT_HANDLERS["int_channel"],
        metavar="N",
        help="add or subtract blue (-255 to 255)",
    )
    filter_group.add_argument(
        "-op",
        "--opacity",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="opacity over black (0 to 100%%)",
    )
    filter_group.add_argument(
        "--posterize",
        dest="posterize",
        type=INPUT_HANDLERS["int_channel"],
        metavar="N",
        help="posterize RGB channels to N levels (2 to 256)",
    )
    filter_group.add_argument(
        "--threshold",
        dest="threshold",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="binarize by relative luminance threshold (0 to 100%%)",
    )
    filter_group.add_argument(
        "--threshold-low",
        dest="threshold_low",
        type=INPUT_HANDLERS["hex"],
        metavar="HEX",
        help="low output color for --threshold (default: 000000)",
    )
    filter_group.add_argument(
        "--threshold-high",
        dest="threshold_high",
        type=INPUT_HANDLERS["hex"],
        metavar="HEX",
        help="high output color for --threshold (default: FFFFFF)",
    )
    filter_group.add_argument(
        "--solarize",
        dest="solarize",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="solarize based on perceptual lightness OKLAB-L threshold (0 to 100%%)",
    )
    filter_group.add_argument(
        "--tint",
        dest="tint",
        type=INPUT_HANDLERS["hex"],
        metavar="HEX",
        help="tint result toward given hex color using OKLAB",
    )
    filter_group.add_argument(
        "--tint-strength",
        dest="tint_strength",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="tint strength (0 to 100%%, default: 20%%)",
    )
    return p


def handle_adjust_command(args: argparse.Namespace) -> None:
    """Entry point for the adjust subcommand."""
    parser = get_adjust_parser()
    pipeline.handle_adjust_command(args, parser)


def main() -> None:
    """Main entry point."""
    parser = get_adjust_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    handle_adjust_command(args)


if __name__ == "__main__":
    main()