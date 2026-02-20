# File: gradient.py
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
    lch_to_rgb,
    luv_to_rgb,
    oklab_to_rgb,
    oklch_to_rgb,
    rgb_to_hex,
    rgb_to_lab,
    rgb_to_lch,
    rgb_to_luv,
    rgb_to_oklab,
    rgb_to_oklch,
)
from ..constants.constants import (
    MAX_DEC,
    MAX_COUNT,
    MAX_STEPS,
    MSG_BOLD_COLORS,
    RESET
)
from ..utils.color_names_handler import get_title_for_hex, resolve_color_name_or_exit
from ..utils.hexlab_logger import log, HexlabArgumentParser
from ..utils.input_handler import INPUT_HANDLERS
from ..utils.print_color_block import print_color_block
from ..utils.truecolor import ensure_truecolor


def _get_interpolated_color(c1, c2, t: float, colorspace: str) -> Tuple[float, float, float]:
    if colorspace == 'srgb':
        r1, g1, b1 = c1
        r2, g2, b2 = c2
        r_new = r1 + t * (r2 - r1)
        g_new = g1 + t * (g2 - g1)
        b_new = b1 + t * (b2 - b1)
        return r_new, g_new, b_new

    if colorspace == 'srgblinear':
        r_lin1, g_lin1, b_lin1 = c1
        r_lin2, g_lin2, b_lin2 = c2
        r_lin_new = r_lin1 + t * (r_lin2 - r_lin1)
        g_lin_new = g_lin1 + t * (g_lin2 - g_lin1)
        b_lin_new = b_lin1 + t * (b_lin2 - b_lin1)
        r = _linear_to_srgb(r_lin_new) * 255
        g = _linear_to_srgb(g_lin_new) * 255
        b = _linear_to_srgb(b_lin_new) * 255
        return r, g, b

    if colorspace == 'lab':
        l1, a1, b1 = c1
        l2, a2, b2 = c2
        l_new = l1 + t * (l2 - l1)
        a_new = a1 + t * (a2 - a1)
        b_new = b1 + t * (b2 - b1)
        return lab_to_rgb(l_new, a_new, b_new)

    if colorspace == 'oklab':
        l1, a1, b1 = c1
        l2, a2, b2 = c2
        l_new = l1 + t * (l2 - l1)
        a_new = a1 + t * (a2 - a1)
        b_new = b1 + t * (b2 - b1)
        return oklab_to_rgb(l_new, a_new, b_new)

    if colorspace == 'lch':
        l1, c1, h1 = c1
        l2, c2, h2 = c2
        h1, h2 = h1 % 360, h2 % 360
        h_diff = h2 - h1
        if h_diff > 180:
            h2 -= 360
        elif h_diff < -180:
            h2 += 360
        l_new = l1 + t * (l2 - l1)
        c_new = c1 + t * (c2 - c1)
        h_new = (h1 + t * (h2 - h1)) % 360
        return lch_to_rgb(l_new, c_new, h_new)

    if colorspace == 'oklch':
        l1, c1, h1 = c1
        l2, c2, h2 = c2
        h1, h2 = h1 % 360, h2 % 360
        h_diff = h2 - h1
        if h_diff > 180:
            h2 -= 360
        elif h_diff < -180:
            h2 += 360
        l_new = l1 + t * (l2 - l1)
        c_new = c1 + t * (c2 - c1)
        h_new = (h1 + t * (h2 - h1)) % 360
        return oklch_to_rgb(l_new, c_new, h_new)

    if colorspace == 'luv':
        l1, u1, v1 = c1
        l2, u2, v2 = c2
        l_new = l1 + t * (l2 - l1)
        u_new = u1 + t * (u2 - u1)
        v_new = v1 + t * (v2 - v1)
        return luv_to_rgb(l_new, u_new, v_new)

    return 0, 0, 0


def _convert_rgb_to_space(r: int, g: int, b: int, colorspace: str) -> Tuple[float, ...]:
    if colorspace == 'srgb':
        return (r, g, b)
    if colorspace == 'srgblinear':
        return (_srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b))
    if colorspace == 'lab':
        return rgb_to_lab(r, g, b)
    if colorspace == 'oklab':
        return rgb_to_oklab(r, g, b)
    if colorspace == 'lch':
        return rgb_to_lch(r, g, b)
    if colorspace == 'oklch':
        return rgb_to_oklch(r, g, b)
    if colorspace == 'luv':
        return rgb_to_luv(r, g, b)
    return (r, g, b)


def handle_gradient_command(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)

    colorspace = args.colorspace

    colors_hex: List[str] = []
    if args.random:
        num_hex = args.count
        if num_hex == 0:
            num_hex = random.randint(2, 3)
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
                'error',
                "at least two hex codes, color names, decimal indexes are required for a gradient"
               )
            log('info', "use -H HEX, -cn NAME, -di INDEX multiple times or -r")
            sys.exit(2)

        colors_hex = input_list

    num_steps = max(1, min(MAX_STEPS, args.steps))

    print()
    if num_steps == 1:
        print_color_block(colors_hex[0], f"{MSG_BOLD_COLORS['info']}step{f'1':>11}{RESET}")
        return

    colors_rgb = [hex_to_rgb(h) for h in colors_hex]

    colors_in_space = []
    for r_val, g_val, b_val in colors_rgb:
        colors_in_space.append(_convert_rgb_to_space(r_val, g_val, b_val, colorspace))

    num_segments = len(colors_in_space) - 1
    total_intervals = num_steps - 1
    gradient_colors: List[str] = []

    for i in range(total_intervals + 1):
        t_global = (i / total_intervals) if total_intervals > 0 else 0
        t_segment_scaled = t_global * num_segments
        segment_index = min(int(t_segment_scaled), num_segments - 1)
        t_local = t_segment_scaled - segment_index

        c1 = colors_in_space[segment_index]
        c2 = colors_in_space[segment_index + 1]

        r_f, g_f, b_f = _get_interpolated_color(c1, c2, t_local, colorspace)
        gradient_colors.append(rgb_to_hex(r_f, g_f, b_f))

    for i, hex_code in enumerate(gradient_colors):
        print_color_block(hex_code, f"{MSG_BOLD_COLORS['info']}step{f'{i + 1}':>11}{RESET}")

    print()

def get_gradient_parser() -> argparse.ArgumentParser:
    parser = HexlabArgumentParser(
        prog="hexlab gradient",
        description="hexlab gradient: generate color gradients between multiple hex codes",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-H", "--hex",
        action="append",
        type=INPUT_HANDLERS["hex"],
        help="use -H HEX multiple times for inputs"
    )
    parser.add_argument(
        "-r", "--random",
        action="store_true",
        help="generate gradient from random colors"
    )
    parser.add_argument(
        "-cn", "--color-name",
        action="append",
        type=INPUT_HANDLERS["color_name"],
        help="use -cn NAME multiple times for inputs by name"
    )
    parser.add_argument(
        "-di", "--decimal-index",
        action="append",
        type=INPUT_HANDLERS["decimal_index"],
        help="use -di INDEX multiple times for inputs by decimal index"
    )
    parser.add_argument(
        "-S", "--steps",
        type=INPUT_HANDLERS["steps"],
        default=10,
        help=f"total steps in gradient (default: 10, max: {MAX_STEPS})"
    )
    parser.add_argument(
        "-cs", "--colorspace",
        default="lab",
        type=INPUT_HANDLERS["colorspace"],
        choices=['srgb', 'srgblinear', 'lab', 'lch', 'oklab', 'oklch', 'luv'],
        help="colorspace for interpolation (default: lab)"
    )
    parser.add_argument(
        "-c", "--count",
        type=INPUT_HANDLERS["count"],
        default=0,
        help=f"number of random colors for input (default: 2-3, max: {MAX_COUNT})"
    )
    parser.add_argument(
        "-s", "--seed",
        type=INPUT_HANDLERS["seed"],
        default=None,
        help="seed for reproducibility of random"
    )
    return parser


def main() -> None:
    parser = get_gradient_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    handle_gradient_command(args)


if __name__ == "__main__":
    main()
