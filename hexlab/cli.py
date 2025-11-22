#!/usr/bin/env python3

import argparse
import os
import sys
import random
import math
import json
import re
from typing import Tuple

from .input_utils import INPUT_HANDLERS, log, HexlabArgumentParser

from . import gradient
from . import mix
from . import scheme
from . import vision
from . import convert
from . import similar
from . import adjust

from .constants import (
    COLOR_NAMES as COLOR_NAMES_RAW, MAX_DEC, __version__,
    TECH_INFO_KEYS, SRGB_TO_LINEAR_TH, LINEAR_TO_SRGB_TH, EPS
)

from .convert import (
    hex_to_rgb as conv_hex_to_rgb,
    rgb_to_xyz as conv_rgb_to_xyz,
    xyz_to_lab as conv_xyz_to_lab,
    lab_to_lch as conv_lab_to_lch,
    rgb_to_hsl as conv_rgb_to_hsl,
    rgb_to_hsv as conv_rgb_to_hsv,
    rgb_to_hwb as conv_rgb_to_hwb,
    rgb_to_cmyk as conv_rgb_to_cmyk,
    rgb_to_oklab as conv_rgb_to_oklab,
    oklab_to_oklch as conv_oklab_to_oklch,
    rgb_to_luv as conv_rgb_to_luv,
    _clamp01 as conv_clamp01,
    _srgb_to_linear as conv_srgb_to_linear,
)

def _normalize_hex_val(v: str) -> str:
    if not isinstance(v, str):
        return ''
    vv = v.replace('#', '').strip().upper()
    if len(vv) == 3:
        vv = ''.join([c * 2 for c in vv])
    return vv

def _norm_name_key(s: str) -> str:
    return re.sub(r'[^0-9a-z]', '', str(s).lower())

COLOR_NAMES = {k: _normalize_hex_val(v) for k, v in COLOR_NAMES_RAW.items()}

HEX_TO_NAME = {v.upper(): k for k, v in COLOR_NAMES.items()}

_norm_map = {}
for k, v in COLOR_NAMES.items():
    key = _norm_name_key(k)
    if key in _norm_map and _norm_map.get(key) != v:
        original_hex = _norm_map.get(key)
        original_name = HEX_TO_NAME.get(original_hex.upper(), '???')
        log(
            'warn',
            f"Color name collision on key '{key}': '{original_name}' and "
            f"'{k}' both normalize to the same key. '{k}' will be used."
        )
    _norm_map[key] = v
COLOR_NAMES_LOOKUP = _norm_map

def _get_color_name_hex(sanitized_name: str) -> str:
    if not sanitized_name:
        return None
    return COLOR_NAMES_LOOKUP.get(sanitized_name)

def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    return conv_hex_to_rgb(hex_code)

def _clamp01(v: float) -> float:
    return conv_clamp01(v)

def lum_comp(c: int) -> float:
    return conv_srgb_to_linear(c)

def get_luminance(r: int, g: int, b: int) -> float:
    return 0.2126 * lum_comp(r) + 0.7152 * lum_comp(g) + 0.0722 * lum_comp(b)

def rgb_to_hsl(r: int, g: int, b: int) -> Tuple[float, float, float]:
    return conv_rgb_to_hsl(r, g, b)

def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    return conv_rgb_to_hsv(r, g, b)

def rgb_to_hwb(r: int, g: int, b: int) -> Tuple[float, float, float]:
    return conv_rgb_to_hwb(r, g, b)

def rgb_to_cmyk(r: int, g: int, b: int) -> Tuple[float, float, float, float]:
    return conv_rgb_to_cmyk(r, g, b)

def rgb_to_xyz(r: int, g: int, b: int) -> Tuple[float, float, float]:
    return conv_rgb_to_xyz(r, g, b)

def xyz_to_lab(x: float, y: float, z: float) -> Tuple[float, float, float]:
    return conv_xyz_to_lab(x, y, z)

def lab_to_lch(l: float, a: float, b: float) -> Tuple[float, float, float]:
    return conv_lab_to_lch(l, a, b)

def rgb_to_oklab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    return conv_rgb_to_oklab(r, g, b)

def oklab_to_oklch(l: float, a: float, b: float) -> Tuple[float, float, float]:
    return conv_oklab_to_oklch(l, a, b)

def rgb_to_luv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    return conv_rgb_to_luv(r, g, b)

def get_wcag_contrast(lum: float) -> dict:
    contrast_white = (1.0 + 0.05) / (lum + 0.05)
    contrast_black = (lum + 0.05) / (0.0 + 0.05)

    def get_pass_fail(ratio: float) -> dict:
        return {
            "AA-Large": "Pass" if ratio >= 3 else "Fail",
            "AA": "Pass" if ratio >= 4.5 else "Fail",
            "AAA-Large": "Pass" if ratio >= 4.5 else "Fail",
            "AAA": "Pass" if ratio >= 7 else "Fail",
        }

    return {
        "white": {"ratio": contrast_white, "levels": get_pass_fail(contrast_white)},
        "black": {"ratio": contrast_black, "levels": get_pass_fail(contrast_black)},
    }

def print_color_block(hex_code: str, title: str = "Color") -> None:
    r, g, b = hex_to_rgb(hex_code)
    print(f"{title:<18}: \033[48;2;{r};{g};{b}m        \033[0m #{hex_code}")

def _zero_small(v: float, threshold: float = 1e-4) -> float:
    return 0.0 if abs(v) <= threshold else v

def _draw_bar(val: float, max_val: float, r_c: int, g_c: int, b_c: int) -> str:
    total_len = 15
    
    if val < 0: val = 0
    if val > max_val: val = max_val
        
    percent = val / max_val
    filled = int(total_len * percent)
    filled = max(0, min(total_len, filled))
    empty = total_len - filled
    
    color_ansi = f"\033[38;2;{r_c};{g_c};{b_c}m"
    reset_ansi = "\033[0m"
    empty_ansi = "\033[90m"
    
    bar_str = f"{color_ansi}{'█' * filled}{reset_ansi}{empty_ansi}{'░' * empty}{reset_ansi}"
    
    if max_val == 1.0:
        val_str = f"{val*100:>6.2f}%"
    elif max_val == 360:
        val_str = f"{val:>6.1f}°"
    elif max_val == 100:
        val_str = f"{val:>6.1f}%"
    else:
        val_str = f"{val/max_val*100:>6.2f}%"
        
    return f"{bar_str} {val_str}"

def print_color_and_info(hex_code: str, title: str, args: argparse.Namespace) -> None:
    print_color_block(hex_code, title)
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

    print()

    arg_lum = getattr(args, 'luminance', False)
    arg_contrast = getattr(args, 'contrast', False)

    if getattr(args, 'index', False):
        print(f"\n   index      : {int(hex_code, 16)} / {MAX_DEC}")
    if getattr(args, 'name', False):
        name = HEX_TO_NAME.get(hex_code.upper())
        if name:
            print(f"\n   name       : {name}")

    if arg_lum or arg_contrast:
        l_rel = get_luminance(r, g, b)
        if arg_lum:
            print(f"\n   luminance  : {l_rel:.6f}")
            print(f"                L {_draw_bar(l_rel, 1.0, 200, 200, 200)}")

    if getattr(args, 'red_green_blue', False):
        print(f"\n   rgb        : rgb({r}, {g}, {b})")
        print(f"                R {_draw_bar(r, 255, 255, 60, 60)}")
        print(f"                G {_draw_bar(g, 255, 60, 255, 60)}")
        print(f"                B {_draw_bar(b, 255, 60, 80, 255)}")

    if getattr(args, 'hue_saturation_lightness', False):
        h, s, l_hsl = rgb_to_hsl(r, g, b)
        print(f"\n   hsl        : hsl({h:.1f}°, {s * 100:.1f}%, {l_hsl * 100:.1f}%)")
        print(f"                H {_draw_bar(h, 360, 255, 200, 0)}")
        print(f"                S {_draw_bar(s, 1.0, 0, 200, 255)}")
        print(f"                L {_draw_bar(l_hsl, 1.0, 200, 200, 200)}")

    if getattr(args, 'hsv', False):
        h, s, v = rgb_to_hsv(r, g, b)
        print(f"\n   hsv        : hsv({h:.1f}°, {s * 100:.1f}%, {v * 100:.1f}%)")
        print(f"                H {_draw_bar(h, 360, 255, 200, 0)}")
        print(f"                S {_draw_bar(s, 1.0, 0, 200, 255)}")
        print(f"                V {_draw_bar(v, 1.0, 200, 200, 200)}")

    if getattr(args, 'hue_whiteness_blackness', False):
        h, w, b_hwb = rgb_to_hwb(r, g, b)
        print(f"\n   hwb        : hwb({h:.1f}°, {w * 100:.1f}%, {b_hwb * 100:.1f}%)")
        print(f"                H {_draw_bar(h, 360, 255, 200, 0)}")
        print(f"                W {_draw_bar(w, 1.0, 200, 200, 200)}")
        print(f"                B {_draw_bar(b_hwb, 1.0, 100, 100, 100)}")

    if getattr(args, 'cmyk', False):
        c, m, y_cmyk, k = rgb_to_cmyk(r, g, b)
        print(f"\n   cmyk       : cmyk({c * 100:.1f}%, {m * 100:.1f}%, {y_cmyk * 100:.1f}%, {k * 100:.1f}%)")
        print(f"                C {_draw_bar(c, 1.0, 0, 255, 255)}")
        print(f"                M {_draw_bar(m, 1.0, 255, 0, 255)}")
        print(f"                Y {_draw_bar(y_cmyk, 1.0, 255, 255, 0)}")
        print(f"                K {_draw_bar(k, 1.0, 100, 100, 100)}")

    if arg_xyz:
        print(f"\n   xyz        : xyz({x:.4f}, {y:.4f}, {z:.4f})")
        print(f"                X {_draw_bar(x / 100.0, 1.0, 255, 60, 60)}")
        print(f"                Y {_draw_bar(y / 100.0, 1.0, 60, 255, 60)}")
        print(f"                Z {_draw_bar(z / 100.0, 1.0, 60, 80, 255)}")

    if arg_lab:
        a_comp_lab = _zero_small(a_lab)
        b_comp_lab = _zero_small(b_lab)
        print(f"\n   lab        : lab({l_lab:.4f}, {a_comp_lab:.4f}, {b_comp_lab:.4f})")
        print(f"                L {_draw_bar(l_lab / 100.0, 1.0, 200, 200, 200)}")
        print(f"                A {_draw_bar(a_comp_lab + 128.0, 255.0, 60, 255, 60)}")
        print(f"                B {_draw_bar(b_comp_lab + 128.0, 255.0, 60, 60, 255)}")

    if arg_lch:
        l_lch, c_lch, h_lch = lab_to_lch(l_lab, a_lab, b_lab)
        print(f"\n   lch        : lch({l_lch:.4f}, {c_lch:.4f}, {h_lch:.4f}°)")
        print(f"                L {_draw_bar(l_lch / 100.0, 1.0, 200, 200, 200)}")
        print(f"                C {_draw_bar(c_lch / 150.0, 1.0, 255, 60, 255)}")
        print(f"                H {_draw_bar(h_lch, 360, 255, 200, 0)}")

    if arg_cieluv:
        print(f"\n   luv        : luv({l_uv:.4f}, {u_comp_luv:.4f}, {v_comp_luv:.4f})")
        print(f"                L {_draw_bar(l_uv / 100.0, 1.0, 200, 200, 200)}")
        print(f"                U {_draw_bar(u_comp_luv + 100.0, 200.0, 60, 255, 60)}")
        print(f"                V {_draw_bar(v_comp_luv + 100.0, 200.0, 60, 60, 255)}")

    if arg_oklab:
        a_comp_ok = _zero_small(a_ok)
        b_comp_ok = _zero_small(b_ok)
        print(f"\n   oklab      : oklab({l_ok:.4f}, {a_comp_ok:.4f}, {b_comp_ok:.4f})")
        print(f"                L {_draw_bar(l_ok, 1.0, 200, 200, 200)}")
        print(f"                A {_draw_bar(a_comp_ok + 0.4, 0.8, 60, 255, 60)}")
        print(f"                B {_draw_bar(b_comp_ok + 0.4, 0.8, 60, 60, 255)}")

    if arg_oklch:
        l_oklch, c_oklch, h_oklch = oklab_to_oklch(l_ok, a_ok, b_ok)
        print(f"\n   oklch      : oklch({l_oklch:.4f}, {c_oklch:.4f}, {h_oklch:.4f}°)")
        print(f"                L {_draw_bar(l_oklch, 1.0, 200, 200, 200)}")
        print(f"                C {_draw_bar(c_oklch / 0.4, 1.0, 255, 60, 255)}")
        print(f"                H {_draw_bar(h_oklch, 360, 255, 200, 0)}")

    if arg_contrast:
        if not arg_lum:
            l_rel = get_luminance(r, g, b)
        wcag = get_wcag_contrast(l_rel)
        
        bg_ansi = f"\033[48;2;{r};{g};{b}m"
        fg_white = "\033[38;2;255;255;255m"
        fg_black = "\033[38;2;0;0;0m"
        reset = "\033[0m"
        
        line_1_block = f"{bg_ansi}{fg_white}{'white':^15}{reset}"
        line_2_block = f"{bg_ansi}{' ' * 15}{reset}"
        line_3_block = f"{bg_ansi}{fg_black}{'black':^15}{reset}"
        
        status_white = f"{wcag['white']['ratio']:.2f}:1 (AA:{wcag['white']['levels']['AA']}, AAA:{wcag['white']['levels']['AAA']})"
        status_black = f"{wcag['black']['ratio']:.2f}:1 (AA:{wcag['black']['levels']['AA']}, AAA:{wcag['black']['levels']['AAA']})"

        print(f"\n                  {line_1_block}  {status_white}")
        print(f"   contrast   :   {line_2_block}")
        print(f"                  {line_3_block}  {status_black}")
    print()

def handle_color_command(args: argparse.Namespace) -> None:
    if args.all_tech_infos:
        for key in TECH_INFO_KEYS:
            if key != 'similar':
                setattr(args, key, True)

    clean_hex = None
    title = "current"
    if args.seed is not None:
        random.seed(args.seed)

    if args.random_hex:
        current_dec = random.randint(0, MAX_DEC)
        clean_hex = f"{current_dec:06X}"
        title = "random"
    elif args.color_name:
        hex_val = _get_color_name_hex(args.color_name)
        if not hex_val:
            log('error', f"unknown color name '{args.color_name}'")
            log('info', "use 'hexlab --list-color-names' to see all options")
            sys.exit(2)
        clean_hex = hex_val
        title = HEX_TO_NAME.get(clean_hex.upper())
        if not title:
            title = args.color_name.title()
    elif args.hexcode:
        clean_hex = args.hexcode
        title = HEX_TO_NAME.get(clean_hex.upper(), f"#{clean_hex}")
    elif getattr(args, "decimal_index", None) is not None:
        clean_hex = args.decimal_index
        idx = int(clean_hex, 16)
        title = HEX_TO_NAME.get(clean_hex.upper(), f"index {idx}")
    else:
        log('error', "one of the arguments -H/--hex, -rh/--random-hex, -di/--decimal-index, or -cn/--color-name is required")
        log('info', "use 'hexlab --help' for more information")
        sys.exit(2)

    current_dec = int(clean_hex, 16)
    print_color_and_info(clean_hex, title, args)

    if args.next:
        next_dec = (current_dec + 1) % (MAX_DEC + 1)
        print_color_and_info(f"{next_dec:06X}", "next", args)
    if args.previous:
        prev_dec = (current_dec - 1) % (MAX_DEC + 1)
        print_color_and_info(f"{prev_dec:06X}", "previous", args)
    if args.negative:
        neg_dec = MAX_DEC - current_dec
        print_color_and_info(f"{neg_dec:06X}", "negative", args)


def ensure_truecolor() -> None:
    if sys.platform != "win32" and os.environ.get("COLORTERM") != "truecolor":
        os.environ["COLORTERM"] = "truecolor"


SUBCOMMANDS = {
    'gradient': gradient,
    'mix': mix,
    'scheme': scheme,
    'vision': vision,
    'convert': convert,
    'similar': similar,
    'adjust': adjust,
}


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
        description="hexlab: 24-bit hex color exploration and manipulation tool",
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
        choices=['text', 'json', 'pretty-json'],
        help="list available color names and exit"
    )

    color_input_group = parser.add_mutually_exclusive_group()
    color_input_group.add_argument(
        "-H", "--hex",
        dest="hexcode",
        type=INPUT_HANDLERS["hex"],
        help="6-digit hex color code without # sign"
    )
    color_input_group.add_argument(
        "-rh", "--random-hex",
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
        help=f"decimal index of the color: 0 to {MAX_DEC}"
    )

    parser.add_argument(
        "-s", "--seed",
        type=INPUT_HANDLERS["seed"],
        default=None,
        help="random seed for reproducibility"
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
        "-i", "--index",
        action="store_true",
        help="show decimal index"
    )
    info_group.add_argument(
        "-rgb", "--red-green-blue",
        action="store_true",
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
        fmt = args.list_color_names
        color_keys = sorted(list(COLOR_NAMES.keys()))
        if fmt == 'text':
            for name in color_keys:
                print(name)
        elif fmt == 'json':
            print(json.dumps(color_keys))
        elif fmt == 'pretty-json':
            print(json.dumps(color_keys, indent=4))
        sys.exit(0)

    if args.help_full:
        parser.print_help()
        for name, module in SUBCOMMANDS.items():
            print("\n"*2)
            try:
                getter = getattr(module, f"get_{name}_parser")
                getter().print_help()
            except AttributeError:
                print(f"(Help for '{name}' not available)")
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