#!/usr/bin/env python3

import argparse
import random
import re
import sys
from typing import Tuple

from ..color_math.conversions import (
    cmyk_to_rgb,
    hex_to_rgb,
    hsl_to_rgb,
    hsv_to_rgb,
    hwb_to_rgb,
    lab_to_rgb,
    lch_to_rgb,
    luv_to_rgb,
    oklab_to_rgb,
    oklch_to_rgb,
    rgb_to_cmyk,
    rgb_to_hex,
    rgb_to_hsl,
    rgb_to_hsv,
    rgb_to_hwb,
    rgb_to_lab,
    rgb_to_lch,
    rgb_to_luv,
    rgb_to_oklab,
    rgb_to_oklch,
    rgb_to_xyz,
    xyz_to_rgb,
)
from ..constants.constants import (
    FORMAT_ALIASES,
    MAX_DEC,

    # Standard ANSI Escape Codes for UI
    MSG_BOLD_COLORS,
    BOLD_WHITE,
    RESET,
)
from ..utils.color_names_handler import get_title_for_hex, resolve_color_name_or_exit
from ..utils.formatting import format_colorspace
from ..utils.hexlab_logger import log, HexlabArgumentParser
from ..utils.input_handler import INPUT_HANDLERS
from ..utils.string_parser import STRING_PARSERS


def fmt_hex_for_output(hex_str: str) -> str:
    """Format hex string for colored output.

    Args:
        hex_str (str): Hex color code.

    Returns:
        str: Formatted hex with ANSI colors.
    """
    return f"{BOLD_WHITE}#{hex_str.upper()}{RESET}"


def _parse_value_to_rgb(clean_val: str, from_fmt: str) -> Tuple[int, int, int]:
    """Parse input value to RGB based on format.

    Handles conversion from various formats to RGB tuple.

    Args:
        clean_val (str): Cleaned input value.
        from_fmt (str): Source format.

    Returns:
        Tuple[int, int, int]: RGB values.
    """
    if from_fmt == "hex":
        return hex_to_rgb(clean_val)
    elif from_fmt == "index":
        try:
            # Extract integer from string
            dec_str = re.findall(r"[-+]?\d+", str(clean_val))[0]
            dec_val = int(dec_str)
            # Clamp to valid range
            dec_val = max(0, min(MAX_DEC, dec_val))
            return hex_to_rgb(f"{dec_val:06X}")
        except Exception:
            log("error", f"invalid index value '{clean_val}'")
            sys.exit(2)
    elif from_fmt == "name":
        hex_val = resolve_color_name_or_exit(clean_val)
        return hex_to_rgb(hex_val)

    if from_fmt in STRING_PARSERS:
        vals = STRING_PARSERS[from_fmt](clean_val)
        if from_fmt == "rgb":
            return vals
        elif from_fmt == "hsl":
            return hsl_to_rgb(vals[0], vals[1], vals[2])
        elif from_fmt == "hsv":
            return hsv_to_rgb(vals[0], vals[1], vals[2])
        elif from_fmt == "hwb":
            return hwb_to_rgb(vals[0], vals[1], vals[2])
        elif from_fmt == "cmyk":
            return cmyk_to_rgb(vals[0], vals[1], vals[2], vals[3])
        elif from_fmt == "xyz":
            return xyz_to_rgb(vals[0], vals[1], vals[2])
        elif from_fmt == "lab":
            return lab_to_rgb(vals[0], vals[1], vals[2])
        elif from_fmt == "lch":
            return lch_to_rgb(vals[0], vals[1], vals[2])
        elif from_fmt == "oklab":
            return oklab_to_rgb(vals[0], vals[1], vals[2])
        elif from_fmt == "oklch":
            return oklch_to_rgb(vals[0], vals[1], vals[2])
        elif from_fmt == "luv":
            return luv_to_rgb(vals[0], vals[1], vals[2])

    # Default black if invalid
    return (0, 0, 0)


def _format_value_from_rgb(r: int, g: int, b: int, to_fmt: str) -> str:
    """Format RGB to target format string.

    Args:
        r (int): Red component.
        g (int): Green component.
        b (int): Blue component.
        to_fmt (str): Target format.

    Returns:
        str: Formatted color string.
    """
    if to_fmt == "hex":
        return fmt_hex_for_output(rgb_to_hex(r, g, b))
    elif to_fmt == "index":
        hex_val = rgb_to_hex(r, g, b)
        return f"{BOLD_WHITE}{int(hex_val, 16)}{RESET}"
    elif to_fmt == "name":
        name = get_title_for_hex(rgb_to_hex(r, g, b), fallback="unknown")
        if name == "unknown":
            return f"{MSG_BOLD_COLORS['error']}unknown{RESET}"
        return f"{BOLD_WHITE}{name}{RESET}"

    elif to_fmt == "rgb":
        return f"{BOLD_WHITE}{format_colorspace('rgb', int(round(r)), int(round(g)), int(round(b)))}{RESET}"
    elif to_fmt == "hsl":
        return f"{BOLD_WHITE}{format_colorspace('hsl', *rgb_to_hsl(r, g, b))}{RESET}"
    elif to_fmt == "hsv":
        return f"{BOLD_WHITE}{format_colorspace('hsv', *rgb_to_hsv(r, g, b))}{RESET}"
    elif to_fmt == "hwb":
        return f"{BOLD_WHITE}{format_colorspace('hwb', *rgb_to_hwb(r, g, b))}{RESET}"
    elif to_fmt == "cmyk":
        return f"{BOLD_WHITE}{format_colorspace('cmyk', *rgb_to_cmyk(r, g, b))}{RESET}"
    elif to_fmt == "xyz":
        return f"{BOLD_WHITE}{format_colorspace('xyz', *rgb_to_xyz(r, g, b))}{RESET}"
    elif to_fmt == "lab":
        return f"{BOLD_WHITE}{format_colorspace('lab', *rgb_to_lab(r, g, b))}{RESET}"
    elif to_fmt == "lch":
        return f"{BOLD_WHITE}{format_colorspace('lch', *rgb_to_lch(r, g, b))}{RESET}"
    elif to_fmt == "oklab":
        return f"{BOLD_WHITE}{format_colorspace('oklab', *rgb_to_oklab(r, g, b))}{RESET}"
    elif to_fmt == "oklch":
        return f"{BOLD_WHITE}{format_colorspace('oklch', *rgb_to_oklch(r, g, b))}{RESET}"
    elif to_fmt == "luv":
        return f"{BOLD_WHITE}{format_colorspace('luv', *rgb_to_luv(r, g, b))}{RESET}"

    return ""


def handle_convert_command(args: argparse.Namespace) -> None:
    """Handle the convert command logic.

    Processes input, converts formats, and outputs result.

    Args:
        args (argparse.Namespace): Parsed arguments.
    """
    if args.seed is not None:
        random.seed(args.seed)
    try:
        from_fmt = FORMAT_ALIASES[args.from_format]
        to_fmt = FORMAT_ALIASES[args.to_format]
    except KeyError as e:
        log("error", f"invalid format specified: {e}")
        log("info", "use 'hexlab convert -h' to see all formats")
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
        print(f"{input_value_str} {MSG_BOLD_COLORS['info']}->{RESET} {output_value_str}")
    else:
        print(output_value_str)


def get_convert_parser() -> argparse.ArgumentParser:
    """Create argument parser for convert command.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
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


def main() -> None:
    """Main entry point for convert command."""
    parser = get_convert_parser()
    args = parser.parse_args(sys.argv[1:])
    handle_convert_command(args)


if __name__ == "__main__":
    main()