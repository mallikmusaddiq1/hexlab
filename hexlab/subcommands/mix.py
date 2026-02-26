# File: mix.py
#!/usr/bin/env python3

import argparse
import random
import sys
from typing import List, Tuple

from ..color_math.conversions import (
    _linear_to_srgb,
    _srgb_to_linear,
    hex_to_rgb,
    lab_to_rgb,
    luv_to_rgb,
    oklab_to_rgb,
    rgb_to_hex,
    rgb_to_lab,
    rgb_to_luv,
    rgb_to_oklab,
)
from ..constants.constants import (
    MAX_DEC,
    MAX_COUNT,
    MSG_BOLD_COLORS,
    RESET,
)
from ..utils.color_names_handler import resolve_color_name_or_exit
from ..utils.hexlab_logger import log, HexlabArgumentParser
from ..utils.input_handler import INPUT_HANDLERS
from ..utils.print_color_block import print_color_block
from ..utils.truecolor import ensure_truecolor


def _convert_rgb_to_space(r: int, g: int, b: int, colorspace: str) -> Tuple[float, ...]:
    """Convert RGB values to components in the specified colorspace.

    Args:
        r (int): Red component (0-255).
        g (int): Green component (0-255).
        b (int): Blue component (0-255).
        colorspace (str): Target colorspace for conversion.

    Returns:
        Tuple[float, ...]: Color components in the target space.
    """
    if colorspace == "srgb":
        return (r, g, b)
    if colorspace == "srgblinear":
        return (_srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b))
    if colorspace == "lab":
        return rgb_to_lab(r, g, b)
    if colorspace == "oklab":
        return rgb_to_oklab(r, g, b)
    if colorspace == "luv":
        return rgb_to_luv(r, g, b)
    return (r, g, b)


def handle_mix_command(args: argparse.Namespace) -> None:
    """Handle the mix command logic.

    Mixes multiple colors by averaging their components in the specified colorspace.
    For two colors, uses the provided amount as mix ratio; for more, averages equally.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    if args.seed is not None:
        random.seed(args.seed)

    colorspace = args.colorspace
    amount = args.amount

    colors_hex = []
    if args.random:
        num_hex = args.count
        if num_hex == 0:
            num_hex = 2
        num_hex = max(2, min(MAX_COUNT, num_hex))
        colors_hex = [f"{random.randint(0, MAX_DEC):06X}" for _ in range(num_hex)]
    else:
        input_list: List[str] = []
        if args.hex:
            input_list.extend(args.hex)
        if args.color_name:
            for nm in args.color_name:
                hexv = resolve_color_name_or_exit(nm)
                input_list.append(hexv)
        if getattr(args, "decimal_index", None):
            for di in args.decimal_index:
                hexv = di
                input_list.append(hexv)

        if len(input_list) < 2:
            log(
                "error",
                "at least two hex codes, color names, decimal indexes are required for a mix",
            )
            log("info", "use -H HEX, -cn NAME, -di INDEX multiple times or -r")
            sys.exit(2)

        colors_hex = input_list

    colors_rgb = [hex_to_rgb(h) for h in colors_hex]

    colors_in_space = []
    for r_val, g_val, b_val in colors_rgb:
        colors_in_space.append(_convert_rgb_to_space(r_val, g_val, b_val, colorspace))

    final_c1, final_c2, final_c3 = 0.0, 0.0, 0.0

    if len(colors_in_space) == 2:
        c1 = colors_in_space[0]
        c2 = colors_in_space[1]

        t = amount / 100.0

        final_c1 = c1[0] * (1 - t) + c2[0] * t
        final_c2 = c1[1] * (1 - t) + c2[1] * t
        final_c3 = c1[2] * (1 - t) + c2[2] * t
    else:
        total_c1, total_c2, total_c3 = 0.0, 0.0, 0.0
        for c in colors_in_space:
            total_c1 += c[0]
            total_c2 += c[1]
            total_c3 += c[2]

        count = len(colors_in_space)
        final_c1 = total_c1 / count
        final_c2 = total_c2 / count
        final_c3 = total_c3 / count

    res_r_f, res_g_f, res_b_f = 0.0, 0.0, 0.0

    if colorspace == "srgb":
        res_r_f, res_g_f, res_b_f = final_c1, final_c2, final_c3
    elif colorspace == "srgblinear":
        res_r_f = _linear_to_srgb(final_c1) * 255
        res_g_f = _linear_to_srgb(final_c2) * 255
        res_b_f = _linear_to_srgb(final_c3) * 255
    elif colorspace == "lab":
        res_r_f, res_g_f, res_b_f = lab_to_rgb(final_c1, final_c2, final_c3)
    elif colorspace == "oklab":
        res_r_f, res_g_f, res_b_f = oklab_to_rgb(final_c1, final_c2, final_c3)
    elif colorspace == "luv":
        res_r_f, res_g_f, res_b_f = luv_to_rgb(final_c1, final_c2, final_c3)

    mixed_hex = rgb_to_hex(res_r_f, res_g_f, res_b_f)

    print()
    for i, hex_code in enumerate(colors_hex):
        print_color_block(hex_code, f"{MSG_BOLD_COLORS['info']}input{f'{i + 1}':>11}{RESET}")

    print()

    label_suffix = ""

    if len(colors_hex) == 2:
        label_suffix = f" {amount:g}%"

    print_color_block(mixed_hex, f"{MSG_BOLD_COLORS['info']}result{f'{label_suffix}':>10}{RESET}")
    print()


def get_mix_parser() -> argparse.ArgumentParser:
    """Create argument parser for mix command.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = HexlabArgumentParser(
        prog="hexlab mix",
        description="hexlab mix: mix multiple colors together by averaging them",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-H",
        "--hex",
        action="append",
        type=INPUT_HANDLERS["hex"],
        help="use -H HEX multiple times for inputs",
    )
    parser.add_argument(
        "-r",
        "--random",
        action="store_true",
        help="generate mix from random colors",
    )
    parser.add_argument(
        "-cn",
        "--color-name",
        action="append",
        type=INPUT_HANDLERS["color_name"],
        help="use -cn NAME multiple times for inputs by name",
    )
    parser.add_argument(
        "-di",
        "--decimal-index",
        action="append",
        type=INPUT_HANDLERS["decimal_index"],
        help="use -di INDEX multiple times for inputs by decimal index",
    )
    parser.add_argument(
        "-a",
        "--amount",
        type=INPUT_HANDLERS["float_0_100"],
        default=50.0,
        help="mix ratio for 2 colors (0 to 100%%, default: 50%%)",
    )
    parser.add_argument(
        "-cs",
        "--colorspace",
        default="lab",
        type=INPUT_HANDLERS["colorspace"],
        choices=["srgb", "srgblinear", "lab", "oklab", "luv"],
        help="colorspace for mixing (default: lab)",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=INPUT_HANDLERS["count"],
        default=2,
        help=f"number of random colors for input (default: 2, max: {MAX_COUNT})",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=INPUT_HANDLERS["seed"],
        default=None,
        help="seed for reproducibility of random",
    )
    return parser


def main() -> None:
    """Main entry point for mix command."""
    parser = get_mix_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    handle_mix_command(args)


if __name__ == "__main__":
    main()