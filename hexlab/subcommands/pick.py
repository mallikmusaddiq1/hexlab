#!/usr/bin/env python3

import argparse
import sys
from typing import List

from ..utils.input_handler import HexlabArgumentParser
from ..utils.hexlab_logger import log
from ..utils.truecolor import ensure_truecolor
from ..color_math.conversions import hex_to_rgb, hsl_to_rgb, rgb_to_hex

BLOCK_W = 36
BLOCK_H = 32

def _generate_spectrum_grid() -> List[str]:
    colors = []
    rows_visual = BLOCK_H * 2
    cols_visual = BLOCK_W

    for char_y in range(BLOCK_H):
        for char_x in range(BLOCK_W):
            visual_y_top = char_y * 2
            h_top = (char_x / cols_visual) * 360
            l_top = 0.98 - (visual_y_top / rows_visual) * 0.96
            rt, gt, bt = hsl_to_rgb(h_top, 1.0, l_top)
            colors.append(rgb_to_hex(rt, gt, bt))

            visual_y_bottom = visual_y_top + 1
            h_bot = (char_x / cols_visual) * 360
            l_bot = 0.98 - (visual_y_bottom / rows_visual) * 0.96
            rb, gb, bb = hsl_to_rgb(h_bot, 1.0, l_bot)
            colors.append(rgb_to_hex(rb, gb, bb))

    return colors

def _print_terminal_block(colors: List[str]) -> None:
    idx = 0
    total_needed = BLOCK_W * BLOCK_H * 2

    if len(colors) < total_needed:
        log('error', "internal error: not enough colors generated")
        sys.exit(3)

    char_block = "▄"
    reset = "\033[0m"

    print()

    for _ in range(BLOCK_H):
        line_buffer = ["    "]
        for _ in range(BLOCK_W):
            hex_top = colors[idx]
            rt, gt, bt = hex_to_rgb(hex_top)

            hex_bottom = colors[idx + 1]
            rb, gb, bb = hex_to_rgb(hex_bottom)

            ansi = f"\033[48;2;{rt};{gt};{bt}m\033[38;2;{rb};{gb};{bb}m{char_block}"
            line_buffer.append(ansi)

            idx += 2

        line_buffer.append(reset)
        print("".join(line_buffer))

    print()

def handle_pick_command(args: argparse.Namespace) -> None:
    colors = _generate_spectrum_grid()
    _print_terminal_block(colors)

def get_pick_parser() -> argparse.ArgumentParser:
    parser = HexlabArgumentParser(
        prog="hexlab pick",
        description="hexlab pick: show a color spectrum block",
        formatter_class=argparse.RawTextHelpFormatter
    )
    return parser

def main() -> None:
    parser = get_pick_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    handle_pick_command(args)

if __name__ == "__main__":
    main()
