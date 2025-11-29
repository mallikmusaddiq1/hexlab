#!/usr/bin/env python3

import argparse
import sys
import random

from ..utils.input_handler import INPUT_HANDLERS, HexlabArgumentParser
from ..utils.hexlab_logger import log
from ..utils.truecolor import ensure_truecolor
from ..utils.print_color_block import print_color_block
from ..utils.color_names_handler import resolve_color_name_or_exit, get_title_for_hex
from ..constants.constants import MAX_DEC, SCHEME_KEYS
from ..color_math.conversions import (
    hex_to_rgb, rgb_to_hex,
    rgb_to_hsl, hsl_to_rgb,
    rgb_to_xyz, xyz_to_lab,
    lab_to_lch, lch_to_rgb,
    rgb_to_oklch, oklch_to_rgb
)

def handle_scheme_command(args: argparse.Namespace) -> None:
    if args.all_schemes:
        for key in SCHEME_KEYS:
            if key == "custom_scheme":
                if getattr(args, "custom_scheme", None) is None:
                    setattr(args, key, [])
            else:
                setattr(args, key, True)
    if args.seed is not None:
        random.seed(args.seed)

    base_hex = None
    title = "base color"

    if args.random:
        base_hex = f"{random.randint(0, MAX_DEC):06X}"
        title = "random"
    elif args.color_name:
        base_hex = resolve_color_name_or_exit(args.color_name)
        title = get_title_for_hex(base_hex)
        if title.lower() == "unknown":
            title = args.color_name.title()
    elif args.hex:
        base_hex = args.hex
        title = get_title_for_hex(base_hex)
    elif getattr(args, "decimal_index", None) is not None:
        base_hex = args.decimal_index
        idx = int(base_hex, 16)
        title = get_title_for_hex(base_hex, f"index {idx}")

    print()
    print_color_block(base_hex, title)
    print()
    r, g, b = hex_to_rgb(base_hex)

    h, s, l, c = (0.0,) * 4

    model = (args.harmony_model or 'hsl').lower()

    if model not in ("hsl", "lch", "oklch"):
        log('error', f"unsupported harmony model '{args.harmony_model}'.")
        log('info', "valid options are: hsl, lch, oklch")
        sys.exit(2)

    if model == 'hsl':
        h, s, l = rgb_to_hsl(r, g, b)
    elif model == 'lch':
        x, y, z = rgb_to_xyz(r, g, b)
        l_lab, a_lab, b_lab = xyz_to_lab(x, y, z)
        l, c, h = lab_to_lch(l_lab, a_lab, b_lab)
    elif model == 'oklch':
        l, c, h = rgb_to_oklch(r, g, b)

    if not (isinstance(h, (int, float)) and h == h):
        log('error', "computed hue is invalid (NaN). aborting to avoid producing invalid output.")
        sys.exit(3)

    def get_scheme_hex(hue_shift: float) -> str:
        new_h = (h + hue_shift) % 360
        new_r, new_g, new_b = 0.0, 0.0, 0.0

        if model == 'hsl':
            new_r, new_g, new_b = hsl_to_rgb(new_h, s, l)
        elif model == 'lch':
            new_r, new_g, new_b = lch_to_rgb(l, c, new_h)
        elif model == 'oklch':
            new_r, new_g, new_b = oklch_to_rgb(l, c, new_h)

        return rgb_to_hex(new_r, new_g, new_b)

    def get_mono_hex(l_shift: float) -> str:
        new_r, new_g, new_b = 0.0, 0.0, 0.0

        if model == 'hsl':
            new_l = max(0.0, min(1.0, l + l_shift))
            new_r, new_g, new_b = hsl_to_rgb(h, s, new_l)
        elif model == 'lch':
            new_l = max(0.0, min(100.0, l + (l_shift * 100)))
            new_r, new_g, new_b = lch_to_rgb(new_l, c, h)
        elif model == 'oklch':
            new_l = max(0.0, min(1.0, l + l_shift))
            new_r, new_g, new_b = oklch_to_rgb(new_l, c, h)

        return rgb_to_hex(new_r, new_g, new_b)

    def _has_custom_scheme(val):
        return isinstance(val, (list, tuple)) and len(val) > 0

    any_specific_flag = (
        args.complementary or
        args.split_complementary or
        args.analogous or
        args.triadic or
        args.tetradic_square or
        args.tetradic_rectangular or
        args.monochromatic or
        _has_custom_scheme(args.custom_scheme)
    )

    if not any_specific_flag:
        print_color_block(get_scheme_hex(180), "comp        180°")
    else:
        if args.complementary:
            print_color_block(get_scheme_hex(180), "comp        180°")
        if args.split_complementary:
            print_color_block(get_scheme_hex(150), "split comp  150°")
            print_color_block(get_scheme_hex(210), "split comp  210°")
        if args.analogous:
            print_color_block(get_scheme_hex(-30), "analog      -30°")
            print_color_block(get_scheme_hex(30), "analog       30°")
        if args.triadic:
            print_color_block(get_scheme_hex(120), "tria        120°")
            print_color_block(get_scheme_hex(240), "tria        240°")
        if args.tetradic_square:
            print_color_block(get_scheme_hex(90), "tetra sq     90°")
            print_color_block(get_scheme_hex(180), "tetra sq    180°")
            print_color_block(get_scheme_hex(270), "tetra sq    270°")
        if args.tetradic_rectangular:
            print_color_block(get_scheme_hex(60), "tetra rec    60°")
            print_color_block(get_scheme_hex(180), "tetra rec   180°")
            print_color_block(get_scheme_hex(240), "tetra rec   240°")
        if args.monochromatic:
            print_color_block(get_mono_hex(-0.2), "mono       -20%L")
            print_color_block(get_mono_hex(0.2), "mono       +20%L")

        if _has_custom_scheme(args.custom_scheme):
            for angle in args.custom_scheme:
                print_color_block(get_scheme_hex(angle), f"         {angle}°")
    print()


def get_scheme_parser() -> argparse.ArgumentParser:
    parser = HexlabArgumentParser(
        prog="hexlab scheme",
        description="hexlab scheme: generate color harmonies",
        formatter_class=argparse.RawTextHelpFormatter
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-H", "--hex",
        type=INPUT_HANDLERS["hex"],
        help="base hex code for the scheme"
    )
    input_group.add_argument(
        "-r", "--random",
        action="store_true",
        help="generate a scheme from a random color"
    )
    input_group.add_argument(
        "-cn", "--color-name",
        type=INPUT_HANDLERS["color_name"],
        help="base color name from --list-color-names"
    )
    input_group.add_argument(
        "-di", "--decimal-index",
        type=INPUT_HANDLERS["decimal_index"],
        help="base color decimal index for the scheme"
    )
    parser.add_argument(
        "-s", "--seed",
        type=INPUT_HANDLERS["seed"],
        default=None,
        help="random seed for reproducibility"
    )
    parser.add_argument(
        "-hm", "--harmony-model",
        type=INPUT_HANDLERS["harmony_model"],
        choices=["hsl", "lch", "oklch"],
        default='hsl',
        help="harmony model: hsl lch oklch (default: hsl)"
    )

    scheme_group = parser.add_argument_group("scheme types")
    scheme_group.add_argument(
        '-all', '--all-schemes',
        action="store_true",
        help="show all color schemes"
    )
    scheme_group.add_argument(
        '-co', '--complementary',
        action="store_true",
        help="show complementary color 180°"
    )
    scheme_group.add_argument(
        '-sco', '--split-complementary',
        action="store_true",
        help="show split-complementary colors 150° 210°"
    )
    scheme_group.add_argument(
        '-an', '--analogous',
        action="store_true",
        help="show analogous colors -30° +30°"
    )
    scheme_group.add_argument(
        '-tr', '--triadic',
        action="store_true",
        help="show triadic colors 120° 240°"
    )
    scheme_group.add_argument(
        '-tsq', '--tetradic-square',
        action="store_true",
        help="show tetradic square colors 90° 180° 270°"
    )
    scheme_group.add_argument(
        '-trc', '--tetradic-rectangular',
        action="store_true",
        help="show tetradic rectangular colors 60° 180° 240°"
    )
    scheme_group.add_argument(
        '-mch', '--monochromatic',
        action="store_true",
        help="show monochromatic colors -20%%L +20%%L"
    )
    scheme_group.add_argument(
        '-cs', '--custom-scheme',
        action="append",
        type=INPUT_HANDLERS["float_signed_360"],
        help="custom hue shift in degrees (-360.0 to 360.0)"
    )
    return parser


def main() -> None:
    parser = get_scheme_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    handle_scheme_command(args)


if __name__ == "__main__":
    main()
