#!/usr/bin/env python3

import argparse
import os
import sys
import random
import math
import re
from typing import Tuple, List

from .input_utils import INPUT_HANDLERS, log, HexlabArgumentParser
from .constants import (
    COLOR_NAMES, MAX_DEC, CB_MATRICES, SIMULATE_KEYS,
    SRGB_TO_LINEAR_TH, LINEAR_TO_SRGB_TH
)

def _norm_name_key(s: str) -> str:
    return re.sub(r'[^0-9a-z]', '', str(s).lower())

HEX_TO_NAME = {}
for name, hexv in COLOR_NAMES.items():
    HEX_TO_NAME[hexv.upper()] = name

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

def ensure_truecolor() -> None:
    if sys.platform == "win32":
        return
    if os.environ.get("COLORTERM") != "truecolor":
        os.environ["COLORTERM"] = "truecolor"

def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    return tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))

def rgb_to_hex(r, g, b):
    return f"{int(round(r)):02X}{int(round(g)):02X}{int(round(b)):02X}"

def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v)) if v == v else 0.0

def lum_comp(c: int) -> float:
    c_norm = _clamp01(c / 255.0)
    return c_norm / 12.92 if c_norm <= SRGB_TO_LINEAR_TH else ((c_norm + 0.055) / 1.055) ** 2.4

def get_luminance(r: int, g: int, b: int) -> float:
    return 0.2126 * lum_comp(r) + 0.7152 * lum_comp(g) + 0.0722 * lum_comp(b)

def _srgb_to_linear(c: int) -> float:
    c_norm = c / 255.0
    c_norm = _clamp01(c_norm)
    return c_norm / 12.92 if c_norm <= SRGB_TO_LINEAR_TH else ((c_norm + 0.055) / 1.055) ** 2.4

def _linear_to_srgb(l: float) -> float:
    l = max(l, 0.0)
    return 12.92 * l if l <= LINEAR_TO_SRGB_TH else 1.055 * (l ** (1 / 2.4)) - 0.055

def _finalize_rgb_vals(r: float, g: float, b: float) -> Tuple[int, int, int]:
    r_i = int(round(r))
    g_i = int(round(g))
    b_i = int(round(b))
    r_i = max(0, min(255, r_i))
    g_i = max(0, min(255, g_i))
    b_i = max(0, min(255, b_i))
    return r_i, g_i, b_i

def print_color_block(hex_code: str, title: str = "color") -> None:
    r, g, b = hex_to_rgb(hex_code)
    print(f"{title:<18}:   \033[48;2;{r};{g};{b}m                \033[0m  #{hex_code}")


def handle_simulate_command(args: argparse.Namespace) -> None:
    if args.all_simulates:
        for key in SIMULATE_KEYS:
            setattr(args, key, True)
    if args.seed is not None:
        random.seed(args.seed)

    base_hex = None
    title = "Base Color"

    if args.random:
        base_hex = f"{random.randint(0, MAX_DEC):06X}"
        title = "random"
    elif args.color_name:
        hexv = _get_color_name_hex(args.color_name)
        if not hexv:
            log('error', f"unknown color name '{args.color_name}'")
            log('info', "use 'hexlab --list-color-names' to see all options")
            sys.exit(2)
        base_hex = hexv
        title = HEX_TO_NAME.get(base_hex.upper())
        if not title:
            title = args.color_name.title()
    elif args.hex:
        base_hex = args.hex
        title = HEX_TO_NAME.get(base_hex.upper(), f"#{base_hex}")
    elif getattr(args, "decimal_index", None) is not None:
        base_hex = args.decimal_index
        idx = int(base_hex, 16)
        title = HEX_TO_NAME.get(base_hex.upper(), f"index {idx}")

    print()
    print_color_block(base_hex, title)
    print()
    r, g, b = hex_to_rgb(base_hex)

    def apply_matrix(r: int, g: int, b: int, m: List[List[float]]) -> Tuple[int, int, int]:
        r_lin = _srgb_to_linear(r)
        g_lin = _srgb_to_linear(g)
        b_lin = _srgb_to_linear(b)
        rr_lin = r_lin * m[0][0] + g_lin * m[0][1] + b_lin * m[0][2]
        gg_lin = r_lin * m[1][0] + g_lin * m[1][1] + b_lin * m[1][2]
        bb_lin = r_lin * m[2][0] + g_lin * m[2][1] + b_lin * m[2][2]
        rr_srgb_norm = _linear_to_srgb(rr_lin)
        gg_srgb_norm = _linear_to_srgb(gg_lin)
        bb_srgb_norm = _linear_to_srgb(bb_lin)
        rr, gg, bb = _finalize_rgb_vals(rr_srgb_norm * 255, gg_srgb_norm * 255, bb_srgb_norm * 255)
        return rr, gg, bb

    no_specific_flag = not (
        args.protanopia or
        args.deuteranopia or
        args.tritanopia or
        args.achromatopsia or
        args.all_simulates
    )
    if args.protanopia or no_specific_flag or args.all_simulates:
        rr, gg, bb = apply_matrix(r, g, b, CB_MATRICES["Protanopia"])
        sim_hex = rgb_to_hex(rr, gg, bb)
        print_color_block(sim_hex, "protanopia")
    if args.deuteranopia or args.all_simulates:
        rr, gg, bb = apply_matrix(r, g, b, CB_MATRICES["Deuteranopia"])
        sim_hex = rgb_to_hex(rr, gg, bb)
        print_color_block(sim_hex, "deuteranopia")
    if args.tritanopia or args.all_simulates:
        rr, gg, bb = apply_matrix(r, g, b, CB_MATRICES["Tritanopia"])
        sim_hex = rgb_to_hex(rr, gg, bb)
        print_color_block(sim_hex, "tritanopia")
    if args.achromatopsia or args.all_simulates:
        l_lin = get_luminance(r, g, b)
        gray_srgb_norm = _linear_to_srgb(l_lin)
        gray_255 = max(0, min(255, int(round(gray_srgb_norm * 255))))
        sim_hex = rgb_to_hex(gray_255, gray_255, gray_255)
        print_color_block(sim_hex, "achromatopsia")
    print()


def get_vision_parser() -> argparse.ArgumentParser:
    parser = HexlabArgumentParser(
        prog="hexlab vision",
        description="hexlab vision: simulate color blindness",
        formatter_class=argparse.RawTextHelpFormatter
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-H", "--hex",
        type=INPUT_HANDLERS["hex"],
        help="base hex code for simulation"
    )
    input_group.add_argument(
        "-r", "--random",
        action="store_true",
        help="simulate with a random color"
    )
    input_group.add_argument(
        "-cn", "--color-name",
        type=INPUT_HANDLERS["color_name"],
        help="base color name from --list-color-names"
    )
    input_group.add_argument(
        "-di", "--decimal-index",
        type=INPUT_HANDLERS["decimal_index"],
        help="base color decimal index for simulation"
    )
    parser.add_argument(
        "-s", "--seed",
        type=INPUT_HANDLERS["seed"],
        default=None,
        help="seed for reproducibility of random"
    )
    simulate_group = parser.add_argument_group("simulation types")
    simulate_group.add_argument(
        '-all', '--all-simulates',
        action="store_true",
        help="show all simulation types"
    )
    simulate_group.add_argument(
        '-p', '--protanopia',
        action="store_true",
        help="simulate protanopia red-blind"
    )
    simulate_group.add_argument(
        '-d', '--deuteranopia',
        action="store_true",
        help="simulate deuteranopia green-blind"
    )
    simulate_group.add_argument(
        '-t', '--tritanopia',
        action="store_true",
        help="simulate tritanopia blue-blind"
    )
    simulate_group.add_argument(
        '-a', '--achromatopsia',
        action="store_true",
        help="simulate achromatopsia total-blind"
    )
    return parser


def main() -> None:
    parser = get_vision_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    handle_simulate_command(args)


if __name__ == "__main__":
    main()
