#!/usr/bin/env python3

import argparse
import sys
import random
import re
from typing import Tuple

from ..utils.input_handler import INPUT_HANDLERS, HexlabArgumentParser
from ..utils.hexlab_logger import log
from ..utils.string_parser import STRING_PARSERS
from ..utils.color_names_handler import resolve_color_name_or_exit, get_title_for_hex
from ..constants.constants import MAX_DEC, FORMAT_ALIASES
from ..utils.formatting import format_colorspace
from ..color_math.conversions import (
    hex_to_rgb, rgb_to_hex,
    rgb_to_hsl, hsl_to_rgb,
    rgb_to_hsv, hsv_to_rgb,
    rgb_to_cmyk, cmyk_to_rgb,
    rgb_to_xyz, xyz_to_rgb,
    rgb_to_lab, lab_to_rgb,
    rgb_to_lch, lch_to_rgb,
    rgb_to_oklab, oklab_to_rgb,
    rgb_to_oklch, oklch_to_rgb,
    rgb_to_hwb, hwb_to_rgb,
    rgb_to_luv, luv_to_rgb
)

def fmt_hex_for_output(hex_str: str) -> str:
    return f"#{hex_str.upper()}"

def _parse_value_to_rgb(clean_val: str, from_fmt: str) -> Tuple[int, int, int]:
    if from_fmt == 'hex':
        return hex_to_rgb(clean_val)
    elif from_fmt == 'index':
        try:
            dec_str = re.findall(r'[-+]?\d+', str(clean_val))[0]
            dec_val = int(dec_str)
            dec_val = max(0, min(MAX_DEC, dec_val))
            return hex_to_rgb(f"{dec_val:06X}")
        except Exception:
            log('error', f"invalid index value '{clean_val}'")
            sys.exit(2)
    elif from_fmt == 'name':
        hex_val = resolve_color_name_or_exit(clean_val)
        return hex_to_rgb(hex_val)

    if from_fmt in STRING_PARSERS:
        vals = STRING_PARSERS[from_fmt](clean_val)
        if from_fmt == 'rgb':
            return vals
        elif from_fmt == 'hsl':
            return hsl_to_rgb(vals[0], vals[1], vals[2])
        elif from_fmt == 'hsv':
            return hsv_to_rgb(vals[0], vals[1], vals[2])
        elif from_fmt == 'hwb':
            return hwb_to_rgb(vals[0], vals[1], vals[2])
        elif from_fmt == 'cmyk':
            return cmyk_to_rgb(vals[0], vals[1], vals[2], vals[3])
        elif from_fmt == 'xyz':
            return xyz_to_rgb(vals[0], vals[1], vals[2])
        elif from_fmt == 'lab':
            return lab_to_rgb(vals[0], vals[1], vals[2])
        elif from_fmt == 'lch':
            return lch_to_rgb(vals[0], vals[1], vals[2])
        elif from_fmt == 'oklab':
            return oklab_to_rgb(vals[0], vals[1], vals[2])
        elif from_fmt == 'oklch':
            return oklch_to_rgb(vals[0], vals[1], vals[2])
        elif from_fmt == 'luv':
            return luv_to_rgb(vals[0], vals[1], vals[2])
    
    return (0, 0, 0)


def _format_value_from_rgb(r: int, g: int, b: int, to_fmt: str) -> str:
    if to_fmt == 'hex':
        return fmt_hex_for_output(rgb_to_hex(r, g, b))
    elif to_fmt == 'index':
        hex_val = rgb_to_hex(r, g, b)
        return str(int(hex_val, 16))
    elif to_fmt == 'name':
        return get_title_for_hex(rgb_to_hex(r, g, b))

    elif to_fmt == 'rgb':
        return format_colorspace('rgb', int(round(r)), int(round(g)), int(round(b)))
    elif to_fmt == 'hsl':
        return format_colorspace('hsl', *rgb_to_hsl(r, g, b))
    elif to_fmt == 'hsv':
        return format_colorspace('hsv', *rgb_to_hsv(r, g, b))
    elif to_fmt == 'hwb':
        return format_colorspace('hwb', *rgb_to_hwb(r, g, b))
    elif to_fmt == 'cmyk':
        return format_colorspace('cmyk', *rgb_to_cmyk(r, g, b))
    elif to_fmt == 'xyz':
        return format_colorspace('xyz', *rgb_to_xyz(r, g, b))
    elif to_fmt == 'lab':
        return format_colorspace('lab', *rgb_to_lab(r, g, b))
    elif to_fmt == 'lch':
        return format_colorspace('lch', *rgb_to_lch(r, g, b))
    elif to_fmt == 'oklab':
        return format_colorspace('oklab', *rgb_to_oklab(r, g, b))
    elif to_fmt == 'oklch':
        return format_colorspace('oklch', *rgb_to_oklch(r, g, b))
    elif to_fmt == 'luv':
        return format_colorspace('luv', *rgb_to_luv(r, g, b))
    
    return ""


def handle_convert_command(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)
    try:
        from_fmt = FORMAT_ALIASES[args.from_format]
        to_fmt = FORMAT_ALIASES[args.to_format]
    except KeyError as e:
        log('error', f"invalid format specified: {e}")
        log('info', "use 'hexlab convert -h' to see all formats")
        sys.exit(2)

    r, g, b = (0, 0, 0)
    
    if args.random:
        dec_val = random.randint(0, MAX_DEC)
        r, g, b = hex_to_rgb(f"{dec_val:06X}")
    else:
        r, g, b = _parse_value_to_rgb(args.value, from_fmt)

    output_value_str = _format_value_from_rgb(r, g, b, to_fmt)
    
    if args.verbose:
        input_value_str = _format_value_from_rgb(r, g, b, from_fmt)
        print(f"{input_value_str} -> {output_value_str}")
    else:
        print(output_value_str)


def get_convert_parser() -> argparse.ArgumentParser:
    parser = HexlabArgumentParser(
        prog="hexlab convert",
        description="hexlab convert: convert a color value from one format to another",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False
    )
    formats_list = "hex rgb hsl hsv hwb cmyk xyz lab lch luv oklab oklch index name"
    parser.add_argument(
        '-h', '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='show this help message and exit'
    )
    parser.add_argument(
        "-f", "--from-format",
        required=True,
        type=INPUT_HANDLERS["from_format"],
        help="the format to convert from\n"
             f"all formats: {formats_list}\n"
             f"use quotes for better UX"
    )
    parser.add_argument(
        "-t", "--to-format",
        required=True,
        type=INPUT_HANDLERS["to_format"],
        help="the format to convert to\n"
             f"all formats: {formats_list}\n"
             f"use quotes for better UX"
    )

    ex_rgb = format_colorspace('rgb', 0, 0, 0)
    ex_hsl = format_colorspace('hsl', 0, 0, 0).replace('%', '%%')
    ex_hsv = format_colorspace('hsv', 0, 0, 0).replace('%', '%%')
    ex_hwb = format_colorspace('hwb', 0, 0, 1.0).replace('%', '%%')
    ex_cmyk = format_colorspace('cmyk', 0, 0, 0, 1.0).replace('%', '%%')
    ex_xyz = format_colorspace('xyz', 0, 0, 0)
    ex_lab = format_colorspace('lab', 0, 0, 0)
    ex_lch = format_colorspace('lch', 0, 0, 0)
    ex_luv = format_colorspace('luv', 0, 0, 0)
    ex_oklab = format_colorspace('oklab', 0.0001, 0, 0)
    ex_oklch = format_colorspace('oklch', 0.0001, 0, 90)

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-v", "--value",
        type=str, 
        help="write value to convert in quotes\n"
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
    )
    input_group.add_argument(
        "-r", "--random",
        action="store_true",
        help="generate a random value for the --from-format"
    )
    parser.add_argument(
        "-s", "--seed",
        type=INPUT_HANDLERS["seed"],
        default=None,
        help="seed for reproducibility of random"
    )
    parser.add_argument(
        "-V", "--verbose",
        action="store_true",
        help="print the conversion verbosely"
    )
    return parser


def main() -> None:
    parser = get_convert_parser()
    args = parser.parse_args(sys.argv[1:])
    handle_convert_command(args)


if __name__ == "__main__":
    main()
