#!/usr/bin/env python3

import argparse
import sys
import random
from typing import List

from ..utils.input_handler import INPUT_HANDLERS, HexlabArgumentParser
from ..utils.hexlab_logger import log
from ..utils.truecolor import ensure_truecolor
from ..utils.print_color_block import print_color_block
from ..utils.color_names_handler import resolve_color_name_or_exit, get_title_for_hex
from ..constants.constants import MAX_DEC, CB_MATRICES, SIMULATE_KEYS
from ..color_math.luminance import get_luminance
from ..color_math.conversions import (
    hex_to_rgb, rgb_to_hex,
    _srgb_to_linear, _linear_to_srgb
)

def handle_vision_command(args: argparse.Namespace) -> None:
    if args.all_simulates:
        for key in SIMULATE_KEYS:
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

    def get_simulated_hex(r: int, g: int, b: int, matrix: List[List[float]]) -> str:
        r_lin = _srgb_to_linear(r)
        g_lin = _srgb_to_linear(g)
        b_lin = _srgb_to_linear(b)
        
        rr_lin = r_lin * matrix[0][0] + g_lin * matrix[0][1] + b_lin * matrix[0][2]
        gg_lin = r_lin * matrix[1][0] + g_lin * matrix[1][1] + b_lin * matrix[1][2]
        bb_lin = r_lin * matrix[2][0] + g_lin * matrix[2][1] + b_lin * matrix[2][2]
        
        rr_srgb = _linear_to_srgb(rr_lin) * 255
        gg_srgb = _linear_to_srgb(gg_lin) * 255
        bb_srgb = _linear_to_srgb(bb_lin) * 255
        
        return rgb_to_hex(rr_srgb, gg_srgb, bb_srgb)

    no_specific_flag = not (
        args.protanopia or
        args.deuteranopia or
        args.tritanopia or
        args.achromatopsia or
        args.all_simulates
    )

    if args.protanopia or no_specific_flag or args.all_simulates:
        sim_hex = get_simulated_hex(r, g, b, CB_MATRICES["Protanopia"])
        print_color_block(sim_hex, "protanopia")
    
    if args.deuteranopia or args.all_simulates:
        sim_hex = get_simulated_hex(r, g, b, CB_MATRICES["Deuteranopia"])
        print_color_block(sim_hex, "deuteranopia")
    
    if args.tritanopia or args.all_simulates:
        sim_hex = get_simulated_hex(r, g, b, CB_MATRICES["Tritanopia"])
        print_color_block(sim_hex, "tritanopia")
    
    if args.achromatopsia or args.all_simulates:
        l_lin = get_luminance(r, g, b)
        gray_val = _linear_to_srgb(l_lin) * 255
        sim_hex = rgb_to_hex(gray_val, gray_val, gray_val)
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
    handle_vision_command(args)


if __name__ == "__main__":
    main()
