# File: main.py
#!/usr/bin/env python3

import argparse
import math
import random
import sys
from typing import Tuple

from .color_math.conversions import (
    hex_to_rgb,
    lab_to_lch,
    oklab_to_oklch,
    rgb_to_cmyk,
    rgb_to_hsl,
    rgb_to_hsv,
    rgb_to_hwb,
    rgb_to_luv,
    rgb_to_oklab,
    rgb_to_xyz,
    xyz_to_lab,
)
from .color_math.luminance import get_luminance
from .color_math.wcag_contrast import get_wcag_contrast
from .constants.constants import (
    EPS,
    LINEAR_TO_SRGB_TH,
    MAX_DEC,
    SRGB_TO_LINEAR_TH,
    TECH_INFO_KEYS,
    __version__,
)
from .subcommands.registry import SUBCOMMANDS
from .utils.color_names_handler import (
    get_title_for_hex,
    handle_list_color_names_action,
    resolve_color_name_or_exit,
)
from .utils.formatting import format_colorspace
from .utils.hexlab_logger import log
from .utils.input_handler import INPUT_HANDLERS, HexlabArgumentParser
from .utils.print_color_block import print_color_block
from .utils.truecolor import ensure_truecolor


def _zero_small(v: float, threshold: float = 1e-4) -> float:
    return 0.0 if abs(v) <= threshold else v


def _draw_bar(val: float, max_val: float, r_c: int, g_c: int, b_c: int) -> str:
    total_len = 16
    abs_val = abs(val)
    if abs_val > max_val:
        abs_val = max_val
    percent = abs_val / max_val
    filled = int(total_len * percent)
    filled = max(0, min(total_len, filled))
    empty = total_len - filled

    color_ansi = f"\033[38;2;{r_c};{g_c};{b_c}m"
    reset_ansi = "\033[0m"
    empty_ansi = "\033[90m"

    block_char = "█"
    empty_char = "░"

    if val < 0:
        bar_str = (
            f"{empty_ansi}{empty_char * empty}{reset_ansi}"
            f"{color_ansi}{block_char * filled}{reset_ansi}"
        )
    else:
        bar_str = (
            f"{color_ansi}{block_char * filled}{reset_ansi}"
            f"{empty_ansi}{empty_char * empty}{reset_ansi}"
        )

    return bar_str


def print_color_and_info(
    hex_code: str,
    title: str,
    args: argparse.Namespace,
    *,
    neighbors=None,
) -> None:
    print()
    print_color_block(hex_code, title)

    hide_bars = getattr(args, 'hide_bars', False)

    if neighbors:
        nxt = neighbors.get("next")
        prv = neighbors.get("previous")
        neg = neighbors.get("negative")
        if nxt is not None:
            print_color_block(nxt, "next")
        if prv is not None:
            print_color_block(prv, "previous")
        if neg is not None:
            print_color_block(neg, "negative")

    r, g, b = hex_to_rgb(hex_code)

    x, y, z, l_lab, a_lab, b_lab = (0.0,) * 6
    l_ok, a_ok, b_ok = (0.0,) * 3

    arg_xyz = getattr(args, 'xyz', False)
    arg_lab = getattr(args, 'lab', False)
    arg_lch = getattr(args, 'lch', False)
    arg_cieluv = getattr(args, 'cieluv', False)
    arg_oklab = getattr(args, 'oklab', False)
    arg_oklch = getattr(args, 'oklch', False)

    needs_xyz = arg_xyz or arg_lab or arg_lch or arg_cieluv
    needs_lab = arg_lab or arg_lch
    needs_oklab = arg_oklab or arg_oklch

    if needs_xyz:
        x, y, z = rgb_to_xyz(r, g, b)
    if needs_lab:
        l_lab, a_lab, b_lab = xyz_to_lab(x, y, z)
    if needs_oklab:
        l_ok, a_ok, b_ok = rgb_to_oklab(r, g, b)

    if arg_cieluv:
        l_uv, u_uv, v_uv = rgb_to_luv(r, g, b)
        u_comp_luv = _zero_small(u_uv)
        v_comp_luv = _zero_small(v_uv)

    arg_lum = getattr(args, 'luminance', False)
    arg_contrast = getattr(args, 'contrast', False)

    if getattr(args, 'index', False):
        print(f"\n\nindex             : {int(hex_code, 16)} / {MAX_DEC}")
    if getattr(args, 'name', False):
        name_or_hex = get_title_for_hex(hex_code)
        if not name_or_hex.startswith("#") and name_or_hex.lower() != "unknown":
            print(f"\nname              : {name_or_hex}")

    if arg_lum or arg_contrast:
        l_rel = get_luminance(r, g, b)
        if arg_lum:
            print(f"\nluminance         : {l_rel:.6f}")
            if not hide_bars:
                print(f"                    L {_draw_bar(l_rel, 1.0, 200, 200, 200)}")

    if getattr(args, 'rgb', False):
        print(f"\nrgb               : {format_colorspace('rgb', r, g, b)}")
        if not hide_bars:
            print(f"                    R {_draw_bar(r, 255, 255, 60, 60)} {(r / 255) * 100:6.2f}%")
            print(f"                    G {_draw_bar(g, 255, 60, 255, 60)} {(g / 255) * 100:6.2f}%")
            print(f"                    B {_draw_bar(b, 255, 60, 80, 255)} {(b / 255) * 100:6.2f}%")

    if getattr(args, 'hsl', False):
        h, s, l_hsl = rgb_to_hsl(r, g, b)
        print(f"\nhsl               : {format_colorspace('hsl', h, s, l_hsl)}")
        if not hide_bars:
            print(f"                    H {_draw_bar(h, 360, 255, 200, 0)}")
            print(f"                    S {_draw_bar(s, 1.0, 0, 200, 255)}")
            print(f"                    L {_draw_bar(l_hsl, 1.0, 200, 200, 200)}")

    if getattr(args, 'hsv', False):
        h, s, v = rgb_to_hsv(r, g, b)
        print(f"\nhsv               : {format_colorspace('hsv', h, s, v)}")
        if not hide_bars:
            print(f"                    H {_draw_bar(h, 360, 255, 200, 0)}")
            print(f"                    S {_draw_bar(s, 1.0, 0, 200, 255)}")
            print(f"                    V {_draw_bar(v, 1.0, 200, 200, 200)}")

    if getattr(args, 'hwb', False):
        h, w, b_hwb = rgb_to_hwb(r, g, b)
        print(f"\nhwb               : {format_colorspace('hwb', h, w, b_hwb)}")
        if not hide_bars:
            print(f"                    H {_draw_bar(h, 360, 255, 200, 0)}")
            print(f"                    W {_draw_bar(w, 1.0, 200, 200, 200)}")
            print(f"                    B {_draw_bar(b_hwb, 1.0, 100, 100, 100)}")

    if getattr(args, 'cmyk', False):
        c, m, y_cmyk, k = rgb_to_cmyk(r, g, b)
        print(f"\ncmyk              : {format_colorspace('cmyk', c, m, y_cmyk, k)}")
        if not hide_bars:
            print(f"                    C {_draw_bar(c, 1.0, 0, 255, 255)}")
            print(f"                    M {_draw_bar(m, 1.0, 255, 0, 255)}")
            print(f"                    Y {_draw_bar(y_cmyk, 1.0, 255, 255, 0)}")
            print(f"                    K {_draw_bar(k, 1.0, 100, 100, 100)}")

    if arg_xyz:
        print(f"\nxyz               : {format_colorspace('xyz', x, y, z)}")
        if not hide_bars:
            print(f"                    X {_draw_bar(x / 100.0, 1.0, 255, 60, 60)}")
            print(f"                    Y {_draw_bar(y / 100.0, 1.0, 60, 255, 60)}")
            print(f"                    Z {_draw_bar(z / 100.0, 1.0, 60, 80, 255)}")

    if arg_lab:
        a_comp_lab = _zero_small(a_lab)
        b_comp_lab = _zero_small(b_lab)
        print(f"\nlab               : {format_colorspace('lab', l_lab, a_comp_lab, b_comp_lab)}")
        if not hide_bars:
            print(f"                    L {_draw_bar(l_lab / 100.0, 1.0, 200, 200, 200)}")
            print(f"                    A {_draw_bar(a_comp_lab, 128.0, 60, 255, 60)}")
            print(f"                    B {_draw_bar(b_comp_lab, 128.0, 60, 60, 255)}")

    if arg_lch:
        l_lch, c_lch, h_lch = lab_to_lch(l_lab, a_lab, b_lab)
        print(f"\nlch               : {format_colorspace('lch', l_lch, c_lch, h_lch)}")
        if not hide_bars:
            print(f"                    L {_draw_bar(l_lch / 100.0, 1.0, 200, 200, 200)}")
            print(f"                    C {_draw_bar(c_lch / 150.0, 1.0, 255, 60, 255)}")
            print(f"                    H {_draw_bar(h_lch, 360, 255, 200, 0)}")

    if arg_cieluv:
        print(f"\nluv               : {format_colorspace('luv', l_uv, u_comp_luv, v_comp_luv)}")
        if not hide_bars:
            print(f"                    L {_draw_bar(l_uv / 100.0, 1.0, 200, 200, 200)}")
            print(f"                    U {_draw_bar(u_comp_luv, 100.0, 60, 255, 60)}")
            print(f"                    V {_draw_bar(v_comp_luv, 100.0, 60, 60, 255)}")

    if arg_oklab:
        a_comp_ok = _zero_small(a_ok)
        b_comp_ok = _zero_small(b_ok)
        print(f"\noklab             : {format_colorspace('oklab', l_ok, a_comp_ok, b_comp_ok)}")
        if not hide_bars:
            print(f"                    L {_draw_bar(l_ok, 1.0, 200, 200, 200)}")
            print(f"                    A {_draw_bar(a_comp_ok, 0.4, 60, 255, 60)}")
            print(f"                    B {_draw_bar(b_comp_ok, 0.4, 60, 60, 255)}")

    if arg_oklch:
        l_oklch, c_oklch, h_oklch = oklab_to_oklch(l_ok, a_ok, b_ok)
        print(f"\noklch             : {format_colorspace('oklch', l_oklch, c_oklch, h_oklch)}")
        if not hide_bars:
            print(f"                    L {_draw_bar(l_oklch, 1.0, 200, 200, 200)}")
            print(f"                    C {_draw_bar(c_oklch / 0.4, 1.0, 255, 60, 255)}")
            print(f"                    H {_draw_bar(h_oklch, 360, 255, 200, 0)}")

    if arg_contrast:
        if not arg_lum:
            l_rel = get_luminance(r, g, b)
        wcag = get_wcag_contrast(l_rel)

        bg_ansi = f"\033[48;2;{r};{g};{b}m"
        fg_white = "\033[38;2;255;255;255m"
        fg_black = "\033[38;2;0;0;0m"
        reset = "\033[0m"

        line_1_block = f"{bg_ansi}{fg_white}{'white':^16}{reset}"
        line_2_block = f"{bg_ansi}{'ㅤ' * 8}{reset}"
        line_3_block = f"{bg_ansi}{fg_black}{'black':^16}{reset}"

        white_ratio = wcag['white']['ratio']
        white_lvls = wcag['white']['levels']
        black_ratio = wcag['black']['ratio']
        black_lvls = wcag['black']['levels']

        s_white = f"{white_ratio:.2f}:1 (AA:{white_lvls['AA']}, AAA:{white_lvls['AAA']})"
        s_black = f"{black_ratio:.2f}:1 (AA:{black_lvls['AA']}, AAA:{black_lvls['AAA']})"

        print(f"\n                      {line_1_block}  {s_white}")
        print(f"contrast          :   {line_2_block}")
        print(f"                      {line_3_block}  {s_black}")
    print()


def handle_color_command(args: argparse.Namespace) -> None:
    if args.all_tech_infos:
        for key in TECH_INFO_KEYS:
            setattr(args, key, True)

    clean_hex = None
    title = "current"
    if args.seed is not None:
        random.seed(args.seed)

    if args.random:
        current_dec = random.randint(0, MAX_DEC)
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
            'error',
            "one of the arguments -H/--hex -r/--random -cn/--color-name -di/--decimal-index is required"
        )
        log('info', "use 'hexlab --help' for more information")
        sys.exit(2)

    current_dec = int(clean_hex, 16)

    neighbors = {}
    if args.next:
        next_dec = (current_dec + 1) % (MAX_DEC + 1)
        neighbors["next"] = f"{next_dec:06X}"
    if args.previous:
        prev_dec = (current_dec - 1) % (MAX_DEC + 1)
        neighbors["previous"] = f"{prev_dec:06X}"
    if args.negative:
        neg_dec = MAX_DEC - current_dec
        neighbors["negative"] = f"{neg_dec:06X}"

    if not neighbors:
        neighbors = None

    print_color_and_info(clean_hex, title, args, neighbors=neighbors)


def main() -> None:
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd in SUBCOMMANDS:
            sys.argv.pop(1)
            ensure_truecolor()
            SUBCOMMANDS[cmd].main()
            sys.exit(0)

    parser = HexlabArgumentParser(
        prog="hexlab",
        description="hexlab: a feature-rich color exploration and manipulation tool",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False
    )

    parser.add_argument(
        '-h', '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='show this help message and exit'
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"hexlab {__version__}",
        help="show program version and exit"
    )
    parser.add_argument(
        "-hf", "--help-full",
        action="store_true",
        help="show full help message including subcommands"
    )

    parser.add_argument(
        "--list-color-names",
        nargs='?',
        const='text',
        default=None,
        choices=['text', 'json', 'prettyjson'],
        type=INPUT_HANDLERS["color_name"],
        help="list available color names and exit"
    )

    color_input_group = parser.add_mutually_exclusive_group()
    color_input_group.add_argument(
        "-H", "--hex",
        dest="hex",
        type=INPUT_HANDLERS["hex"],
        help="6-digit hex color code without # sign"
    )
    color_input_group.add_argument(
        "-r", "--random",
        action="store_true",
        help="generate a random hex color"
    )
    color_input_group.add_argument(
        "-cn", "--color-name",
        type=INPUT_HANDLERS["color_name"],
        help="color names from 'hexlab --list-color-names'"
    )
    color_input_group.add_argument(
        "-di", "--decimal-index",
        dest="decimal_index",
        type=INPUT_HANDLERS["decimal_index"],
        help=f"decimal index of the color (0 to {MAX_DEC})"
    )

    parser.add_argument(
        "-s", "--seed",
        type=INPUT_HANDLERS["seed"],
        default=None,
        help="seed for reproducibility of random"
    )

    mod_group = parser.add_argument_group("color modifications")
    mod_group.add_argument(
        "-n", "--next",
        action="store_true",
        help="show the next color"
    )
    mod_group.add_argument(
        "-p", "--previous",
        action="store_true",
        help="show the previous color"
    )
    mod_group.add_argument(
        "-N", "--negative",
        action="store_true",
        help="show the inverse color"
    )

    info_group = parser.add_argument_group("technical information flags")
    info_group.add_argument(
        '-all', '--all-tech-infos',
        action="store_true",
        help="show all technical information"
    )
    info_group.add_argument(
        "-hb", "--hide-bars",
        action="store_true",
        help="hide visual color bars"
    )

    info_group.add_argument(
        "-i", "--index",
        action="store_true",
        help="show decimal index"
    )
    info_group.add_argument(
        "-rgb", "--red-green-blue",
        action="store_true",
        dest="rgb",
        help="show RGB values"
    )
    info_group.add_argument(
        "-l", "--luminance",
        action="store_true",
        help="show relative luminance"
    )
    info_group.add_argument(
        "-hsl", "--hue-saturation-lightness",
        action="store_true",
        dest="hsl",
        help="show HSL values"
    )
    info_group.add_argument(
        "-hsv", "--hue-saturation-value",
        action="store_true",
        dest="hsv",
        help="show HSV values"
    )
    info_group.add_argument(
        "-hwb", "--hue-whiteness-blackness",
        action="store_true",
        dest="hwb",
        help="show HWB values"
    )
    info_group.add_argument(
        "-cmyk", "--cyan-magenta-yellow-key",
        action="store_true",
        dest="cmyk",
        help="show CMYK values"
    )
    info_group.add_argument(
        "-xyz", "--ciexyz",
        dest="xyz",
        action="store_true",
        help="show CIE 1931 XYZ values"
    )
    info_group.add_argument(
        "-lab", "--cielab",
        dest="lab",
        action="store_true",
        help="show CIE 1976 LAB values"
    )
    info_group.add_argument(
        "-lch", "--lightness-chroma-hue",
        action="store_true",
        dest="lch",
        help="show CIE 1976 LCH values"
    )
    info_group.add_argument(
        "--cieluv", "-luv",
        action="store_true",
        dest="cieluv",
        help="show CIE 1976 LUV values"
    )
    info_group.add_argument(
        "--oklab",
        action="store_true",
        dest="oklab",
        help="show OKLAB values"
    )
    info_group.add_argument(
        "--oklch",
        action="store_true",
        dest="oklch",
        help="show OKLCH values"
    )
    info_group.add_argument(
        "-wcag", "--contrast",
        action="store_true",
        help="show WCAG contrast ratio"
    )
    info_group.add_argument(
        "--name",
        action="store_true",
        help="show color name if available in --list-color-names"
    )

    parser.add_argument("command", nargs='?', help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.list_color_names:
        handle_list_color_names_action(args.list_color_names)

    if args.help_full:
        parser.print_help()
        for name, module in SUBCOMMANDS.items():
            print("\n" * 2)
            try:
                getter = getattr(module, f"get_{name}_parser")
                getter().print_help()
            except AttributeError:
                log('info', f"help for '{name}' not available")
        sys.exit(0)

    if args.command:
        if args.command.lower() in SUBCOMMANDS:
            log('error', f"the '{args.command}' command must be the first argument")
        else:
            log('error', f"unrecognized command or argument: '{args.command}'")
        sys.exit(2)

    ensure_truecolor()
    handle_color_command(args)


if __name__ == "__main__":
    main()
