#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
import random

MAX_DEC = 16777215

SHELL_CONFIG_FILES = [
    os.path.expanduser("~/.zshrc"),
    os.path.expanduser("~/.bashrc"),
    os.path.expanduser("~/.profile")
]

if sys.platform == "darwin":
    SHELL_CONFIG_FILES.insert(1, os.path.expanduser("~/.bash_profile"))

TRUECOLOR_LINES = [
    "export COLORTERM=truecolor\n",
    "export TERM=xterm-256color\n"
]
SCRIPT_COMMENT = "# added by hexlab for truecolor support\n"

HEX_REGEX = re.compile(r"[0-9A-Fa-f]{6}")

def log(level, message):
    level = level.lower()
    level_map = {
        'info': sys.stdout,
        'warn': sys.stderr,
        'error': sys.stderr
    }
    stream = level_map.get(level, sys.stderr)
    print(f"[hexlab][{level}] {message}", file=stream)

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

def manage_truecolor(enable=True):
    if sys.platform == "win32":
        log('error', "this feature is only for unix-like shells (bash, zsh, etc.).")
        log('info', "truecolor on windows depends on your terminal emulator")
        return

    target_file = next((f for f in SHELL_CONFIG_FILES if os.path.exists(f)), None)
    
    if not target_file:
        log('error', f"could not find any shell config file to modify.")
        log('info', "Please create one of: ~/.zshrc, ~/.bash_profile, ~/.bashrc, or ~/.profile.")
        return

    log('info', f"detected target shell config: {target_file}")
    
    try:
        with open(target_file, "r") as f:
            lines = f.readlines()

        has_colorterm = any(TRUECOLOR_LINES[0].strip() in line for line in lines)
        has_term = any(TRUECOLOR_LINES[1].strip() in line for line in lines)
        is_enabled = has_colorterm and has_term

        if enable:
            if is_enabled:
                log('info', f"truecolor is already enabled in {target_file}")
                return
            
            log('info', f"enabling truecolor in {target_file}")
            with open(target_file, "a") as f:
                f.write(f"\n{SCRIPT_COMMENT}")
                if not has_colorterm:
                    f.write(TRUECOLOR_LINES[0])
                if not has_term:
                    f.write(TRUECOLOR_LINES[1])
            log('info', f"enabled! please reload your shell")

        else:
            if not is_enabled and not any(SCRIPT_COMMENT in line for line in lines):
                log('info', f"truecolor is already disabled in {target_file}")
                return

            log('info', f"disabling truecolor in {target_file}")
            new_lines = [
                line for line in lines 
                if TRUECOLOR_LINES[0].strip() not in line and
                   TRUECOLOR_LINES[1].strip() not in line and
                   SCRIPT_COMMENT.strip() not in line
            ]
            with open(target_file, "w") as f:
                f.writelines(new_lines)
            log('info', f"disabled! please reload your shell")

    except (IOError, OSError) as e:
        log('error', f"Could not process file {target_file} (check permissions?): {e}")
    except Exception as e:
        log('error', f"an unexpected error occurred with {target_file}: {e}")

def main():
    parser = HexlabArgumentParser(
        description="hexlab: A CLI tool for 24-bit hex color exploration",
    )
    
    color_input_group = parser.add_mutually_exclusive_group()
    color_input_group.add_argument(
        "-H", "--hex",
        dest="hexcode",
        help="6-digit hex code without # symbol",
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
    
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--enable-truecolor",
        action="store_true",
        help="adds truecolor export lines to your shell config file"
    )
    config_group.add_argument(
        "--disable-truecolor",
        action="store_true",
        help="removes truecolor export lines from your shell config file"
    )

    args = parser.parse_args()

    if args.enable_truecolor:
        manage_truecolor(enable=True)
        sys.exit(0)
    
    if args.disable_truecolor:
        manage_truecolor(enable=False)
        sys.exit(0)

    clean_hex = None
    title = "Current Color"

    if args.random_hex:
        current_dec = random.randint(0, MAX_DEC)
        clean_hex = f"{current_dec:06X}"
        title = "Random Color"
    elif args.hexcode:
        input_hex = args.hexcode
        clean_hex = input_hex.upper()
        
        if not is_valid_hex(clean_hex):
            if input_hex.startswith("#") and is_valid_hex(input_hex.lstrip("#")):
                 parser.error(f"invalid input '{input_hex}'. do not include '#'. use: -H {input_hex.lstrip('#').upper()}")
            else:
                 parser.error(f"'{input_hex}' is not a valid 6-digit hex code.")
    else:
        parser.error("a hex code is required. use -H <HEXCODE> or -rh/--random-hex")

    os.environ["COLORTERM"] = "truecolor"
    os.environ["TERM"] = "xterm-256color"
    
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