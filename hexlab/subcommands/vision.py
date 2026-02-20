# File: vision.py
#!/usr/bin/env python3

import argparse
import random
import sys
from typing import List

from ..color_math.conversions import (
    _linear_to_srgb,
    _srgb_to_linear,
    hex_to_rgb,
    rgb_to_hex,
)
from ..color_math.luminance import get_luminance
from ..constants.constants import (
    CB_MATRICES,
    MAX_DEC,
    SIMULATE_KEYS,
    MSG_BOLD_COLORS,
    BOLD_WHITE,
    RESET
)
from ..utils.color_names_handler import get_title_for_hex, resolve_color_name_or_exit
from ..utils.hexlab_logger import log, HexlabArgumentParser
from ..utils.input_handler import INPUT_HANDLERS
from ..utils.print_color_block import print_color_block
from ..utils.truecolor import ensure_truecolor


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
    print_color_block(base_hex, f"{BOLD_WHITE}{title}{RESET}")

    any_sim = any([
        args.protanopia, 
        args.deuteranopia, 
        args.tritanopia, 
        args.achromatopsia, 
        args.all_simulates
    ])

    if any_sim:
        print()

    r, g, b = hex_to_rgb(base_hex)
    
    intensity = getattr(args, 'intensity', 100)
    factor = max(0, min(100, intensity)) / 100.0
    perc_str = f"{intensity}%"

    def get_simulated_hex(r: int, g: int, b: int, matrix: List[List[float]], f: float) -> str:
        r_lin, g_lin, b_lin = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)

        rr_sim = r_lin * matrix[0][0] + g_lin * matrix[0][1] + b_lin * matrix[0][2]
        gg_sim = r_lin * matrix[1][0] + g_lin * matrix[1][1] + b_lin * matrix[1][2]
        bb_sim = r_lin * matrix[2][0] + g_lin * matrix[2][1] + b_lin * matrix[2][2]

        rr_lin = (1 - f) * r_lin + f * rr_sim
        gg_lin = (1 - f) * g_lin + f * gg_sim
        bb_lin = (1 - f) * b_lin + f * bb_sim

        return rgb_to_hex(
            _linear_to_srgb(rr_lin) * 255,
            _linear_to_srgb(gg_lin) * 255,
            _linear_to_srgb(bb_lin) * 255
        )

    if args.protanopia or args.all_simulates:
        sim_hex = get_simulated_hex(r, g, b, CB_MATRICES["Protanopia"], factor)
        label = f"{MSG_BOLD_COLORS['info']}protan{perc_str:>10}{RESET}"
        print_color_block(sim_hex, label)

    if args.deuteranopia or args.all_simulates:
        sim_hex = get_simulated_hex(r, g, b, CB_MATRICES["Deuteranopia"], factor)
        label = f"{MSG_BOLD_COLORS['info']}deuter{perc_str:>10}{RESET}"
        print_color_block(sim_hex, label)

    if args.tritanopia or args.all_simulates:
        sim_hex = get_simulated_hex(r, g, b, CB_MATRICES["Tritanopia"], factor)
        label = f"{MSG_BOLD_COLORS['info']}tritan{perc_str:>10}{RESET}"
        print_color_block(sim_hex, label)

    if args.achromatopsia or args.all_simulates:
        l_lin = get_luminance(r, g, b)
        r_lin, g_lin, b_lin = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)
        
        rr_lin = (1 - factor) * r_lin + factor * l_lin
        gg_lin = (1 - factor) * g_lin + factor * l_lin
        bb_lin = (1 - factor) * b_lin + factor * l_lin
        
        sim_hex = rgb_to_hex(
            _linear_to_srgb(rr_lin) * 255,
            _linear_to_srgb(gg_lin) * 255,
            _linear_to_srgb(bb_lin) * 255
        )
        label = f"{MSG_BOLD_COLORS['info']}achroma{perc_str:>9}{RESET}"
        print_color_block(sim_hex, label)

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
        help="base hex code"
    )
    input_group.add_argument(
        "-r", "--random",
        action="store_true",
        help="use a random base"
    )
    input_group.add_argument(
        "-cn", "--color-name",
        type=INPUT_HANDLERS["color_name"],
        help="base color name"
    )
    input_group.add_argument(
        "-di", "--decimal-index",
        type=INPUT_HANDLERS["decimal_index"],
        help="base decimal index"
    )
    parser.add_argument(
        "-s", "--seed",
        type=INPUT_HANDLERS["seed"],
        default=None,
        help="seed for reproducibility of random"
    )
    parser.add_argument(
        "-i", "--intensity",
        type=INPUT_HANDLERS["intensity"],
        default=100,
        help="simulation intensity: 0 to 100 (default: 100)"
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