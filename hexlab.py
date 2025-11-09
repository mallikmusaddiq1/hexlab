#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
import random

MAX_DEC = 16777215
__version__ = "0.0.1"

HEX_REGEX = re.compile(r"([0-9A-Fa-f]{6}|[0-9A-Fa-f]{3})")

def log(level, message):
    level = level.lower()
    level_map = {
        'info': sys.stdout,
        'warn': sys.stderr,
        'error': sys.stderr
    }
    stream = level_map.get(level, sys.stderr)
    print(f"[hexlab][{level}] {message}", file=stream)

def ensure_truecolor():
    if sys.platform == "win32":
        return

    if os.environ.get("COLORTERM") != "truecolor":
        os.environ["COLORTERM"] = "truecolor"

class HexlabArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        log('error', message)
        sys.exit(2)

def hex_to_rgb(hex_code):
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def is_valid_hex(h):
    return HEX_REGEX.fullmatch(h) is not None

def lum_comp(c):
    c_norm = c / 255.0
    return c_norm / 12.92 if c_norm <= 0.03928 else ((c_norm + 0.055) / 1.055) ** 2.4

def print_color_block(hex_code, title="Color"):
    r, g, b = hex_to_rgb(hex_code)
    print(f"{title:<18}: \033[48;2;{r};{g};{b}m        \033[0m #{hex_code}")

def show_tech_info(hex_code):
    r, g, b = hex_to_rgb(hex_code)
    l = 0.2126 * lum_comp(r) + 0.7152 * lum_comp(g) + 0.0722 * lum_comp(b)
    index = int(hex_code, 16)
    
    print(f"   Index      : {index} / {MAX_DEC}")
    print(f"   RGB        : {r}, {g}, {b}")
    print(f"   Luminance  : {l:.6f}")
    print()

def main():
    parser = HexlabArgumentParser(
        description="hexlab: A CLI tool for 24-bit hex color exploration",
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"hexlab {__version__}",
        help="show program version and exit"
    )
    
    color_input_group = parser.add_mutually_exclusive_group()
    color_input_group.add_argument(
        "-H", "--hex",
        dest="hexcode",
        help="6-digit hex color code without # symbol",
    )
    color_input_group.add_argument(
        "-rh", "--random-hex",
        action="store_true",
        help="generate a random hex color"
    )
    
    parser.add_argument(
        "-n", "--next",
        action="store_true",
        help="show the next color"
    )
    parser.add_argument(
        "-p", "--previous",
        action="store_true",
        help="show the previous color"
    )
    parser.add_argument(
        "-N", "--negative",
        action="store_true",
        help="show the negative color"
    )

    args = parser.parse_args()

    ensure_truecolor()

    clean_hex = None
    title = "Current Color"

    if args.random_hex:
        current_dec = random.randint(0, MAX_DEC)
        clean_hex = f"{current_dec:06X}"
        title = "Random Color"
    elif args.hexcode:
        input_hex = args.hexcode
        clean_hex = input_hex.lstrip("#").upper()
        
        if not is_valid_hex(clean_hex):
            parser.error(f"'{input_hex}' is not a valid 6-digit hex code")
        
        if len(clean_hex) == 3:
            clean_hex = "".join([c*2 for c in clean_hex])
            
    else:
        parser.error("hex code is required. use -H/--hex <HEXCODE> or -rh/--random-hex")
    
    current_dec = int(clean_hex, 16)

    print_color_block(clean_hex, title)
    show_tech_info(clean_hex)

    if args.next:
        next_dec = (current_dec + 1) % (MAX_DEC + 1)
        next_hex = f"{next_dec:06X}"
        print_color_block(next_hex, "Next Color")
        show_tech_info(next_hex)

    if args.previous:
        prev_dec = (current_dec - 1) % (MAX_DEC + 1)
        prev_hex = f"{prev_dec:06X}"
        print_color_block(prev_hex, "Previous Color")
        show_tech_info(prev_hex)

    if args.negative:
        neg_dec = MAX_DEC - current_dec
        neg_hex = f"{neg_dec:06X}"
        print_color_block(neg_hex, "Negative Color")
        show_tech_info(neg_hex)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[hexlab][info] exiting.", file=sys.stderr)
        sys.exit(0)