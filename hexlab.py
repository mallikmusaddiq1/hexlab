#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
import random
import math
import json
from typing import Tuple
from color_names import WEB_COLORS

MAX_DEC = 16777215
__version__ = "0.0.1"

HEX_REGEX = re.compile(r"([0-9A-Fa-f]{6}|[0-9A-Fa-f]{3})")

def log(level: str, message: str) -> None:
    level = level.lower()
    level_map = {
        'info': sys.stdout,
        'warn': sys.stderr,
        'error': sys.stderr
    }
    stream = level_map.get(level, sys.stderr)
    print(f"[hexlab][{level}] {message}", file=stream)

def ensure_truecolor() -> None:
    if sys.platform == "win32":
        return
    if os.environ.get("COLORTERM") != "truecolor":
        os.environ["COLORTERM"] = "truecolor"

class HexlabArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        log('error', message)
        self.exit(2)

def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(r: float, g: float, b: float) -> str:
    return f"{int(r):02X}{int(g):02X}{int(b):02X}"

def is_valid_hex(h: str) -> bool:
    return HEX_REGEX.fullmatch(h) is not None

def clean_hex_input(hex_str: str) -> str:
    clean_hex = hex_str.lstrip("#").upper()
    if not is_valid_hex(clean_hex):
        log('error', f"'{hex_str}' is not a valid 6-digit hex code")
        sys.exit(2)
    if len(clean_hex) == 3:
        clean_hex = "".join([c*2 for c in clean_hex])
    return clean_hex

def lum_comp(c: int) -> float:
    c_norm = c / 255.0
    return c_norm / 12.92 if c_norm <= 0.03928 else ((c_norm + 0.055) / 1.055) ** 2.4

def get_luminance(r: int, g: int, b: int) -> float:
    return 0.2126 * lum_comp(r) + 0.7152 * lum_comp(g) + 0.0722 * lum_comp(b)

def rgb_to_hsl(r: int, g: int, b: int) -> Tuple[float, float, float]:
    r, g, b = r/255.0, g/255.0, b/255.0
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    l = (cmax + cmin) / 2
    
    if delta == 0:
        h = 0
        s = 0
    else:
        s = delta / (1 - abs(2*l - 1))
        if cmax == r:
            h = 60 * (((g - b) / delta) % 6)
        elif cmax == g:
            h = 60 * ((b - r) / delta + 2)
        else:
            h = 60 * ((r - g) / delta + 4)
        if h < 0:
            h += 360
            
    return (h, s, l)

def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    r, g, b = r/255.0, g/255.0, b/255.0
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    v = cmax
    
    if delta == 0:
        h = 0
        s = 0
    else:
        s = delta / v
        if cmax == r:
            h = 60 * (((g - b) / delta) % 6)
        elif cmax == g:
            h = 60 * ((b - r) / delta + 2)
        else:
            h = 60 * ((r - g) / delta + 4)
        if h < 0:
            h += 360
            
    return (h, s, v)

def rgb_to_cmyk(r: int, g: int, b: int) -> Tuple[float, float, float, float]:
    if r == 0 and g == 0 and b == 0:
        return 0, 0, 0, 1
    
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    k = 1.0 - max(r_norm, g_norm, b_norm)
    if k == 1.0:
        return 0, 0, 0, 1
    c = (1.0 - r_norm - k) / (1.0 - k)
    m = (1.0 - g_norm - k) / (1.0 - k)
    y = (1.0 - b_norm - k) / (1.0 - k)
    return (c, m, y, k)

def get_wcag_contrast(lum: float) -> dict:
    lum_white = 1.0
    lum_black = 0.0
    
    contrast_white = (lum_white + 0.05) / (lum + 0.05)
    contrast_black = (lum + 0.05) / (lum_black + 0.05)
    
    results = {
        "white": {"ratio": contrast_white},
        "black": {"ratio": contrast_black}
    }

    def get_pass_fail(ratio: float) -> dict:
        return {
            "AA-Large": "Pass" if ratio >= 3 else "Fail",
            "AA": "Pass" if ratio >= 4.5 else "Fail",
            "AAA-Large": "Pass" if ratio >= 4.5 else "Fail",
            "AAA": "Pass" if ratio >= 7 else "Fail",
        }
        
    results["white"]["levels"] = get_pass_fail(contrast_white)
    results["black"]["levels"] = get_pass_fail(contrast_black)
    
    return results

def linear_interpolate(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int], t: float) -> Tuple[float, float, float]:
    r1, g1, b1 = rgb1
    r2, g2, b2 = rgb2
    
    r = r1 + t * (r2 - r1)
    g = g1 + t * (g2 - g1)
    b = b1 + t * (b2 - b1)
    
    r = max(0.0, min(255.0, r))
    g = max(0.0, min(255.0, g))
    b = max(0.0, min(255.0, b))
    
    return (r, g, b)

def print_color_block(hex_code: str, title: str = "Color") -> None:
    r, g, b = hex_to_rgb(hex_code)
    print(f"{title:<18}: \033[48;2;{r};{g};{b}m        \033[0m #{hex_code}")

def print_color_and_info(hex_code: str, title: str, args: argparse.Namespace) -> None:
    print_color_block(hex_code, title)
    
    r, g, b = hex_to_rgb(hex_code)
    
    if args.index:
        index = int(hex_code, 16)
        print(f"   Index      : {index} / {MAX_DEC}")
        
    if args.red_green_blue:
        print(f"   RGB        : {r}, {g}, {b}")
        
    if args.luminance or args.contrast:
        l = get_luminance(r, g, b)
        if args.luminance:
            print(f"   Luminance  : {l:.6f}")
            
    if args.hue_saturation_lightness:
        h, s, l_hsl = rgb_to_hsl(r, g, b)
        print(f"   HSL        : {h:.1f}°, {s*100:.1f}%, {l_hsl*100:.1f}%")
        
    if args.hsv:
        h, s, v = rgb_to_hsv(r, g, b)
        print(f"   HSV        : {h:.1f}°, {s*100:.1f}%, {v*100:.1f}%")
        
    if args.cmyk:
        c, m, y, k = rgb_to_cmyk(r, g, b)
        print(f"   CMYK       : {c*100:.1f}%, {m*100:.1f}%, {y*100:.1f}%, {k*100:.1f}%")

    if args.contrast:
        wcag = get_wcag_contrast(l)
        print( "   Contrast White: "
            f"{wcag['white']['ratio']:.2f}:1 "
            f"(AA-Large: {wcag['white']['levels']['AA-Large']}, "
            f"AA: {wcag['white']['levels']['AA']}, "
            f"AAA-Large: {wcag['white']['levels']['AAA-Large']}, "
            f"AAA: {wcag['white']['levels']['AAA']})"
        )
        print( "   Contrast Black: "
            f"{wcag['black']['ratio']:.2f}:1 "
            f"(AA-Large: {wcag['black']['levels']['AA-Large']}, "
            f"AA: {wcag['black']['levels']['AA']}, "
            f"AAA-Large: {wcag['black']['levels']['AAA-Large']}, "
            f"AAA: {wcag['black']['levels']['AAA']})"
        )
        
    print()

def handle_color_command(args: argparse.Namespace) -> None:
    clean_hex = None
    title = "Current Color"

    if args.seed is not None:
        random.seed(args.seed)
        
    if args.random_hex:
        current_dec = random.randint(0, MAX_DEC)
        clean_hex = f"{current_dec:06X}"
        title = "Random Color"
    elif args.color_name:
        named_lower = args.color_name.lower()
        if named_lower not in WEB_COLORS:
            log('error', f"unknown color name '{args.color_name}'")
            log('info', "use 'hexlab --list-color-names' to see all options")
            sys.exit(2)
        clean_hex = WEB_COLORS[named_lower]
        title = f"{args.color_name}"
    elif args.hexcode:
        clean_hex = clean_hex_input(args.hexcode)
    else:
        log('error', "one of the arguments -H/--hex, -rh/--random-hex, or -cn/--color-name is required")
        log('info', "use 'hexlab --help' for more information")
        sys.exit(2)
    
    current_dec = int(clean_hex, 16)

    print_color_and_info(clean_hex, title, args)

    if args.next:
        next_dec = (current_dec + 1) % (MAX_DEC + 1)
        next_hex = f"{next_dec:06X}"
        print_color_and_info(next_hex, "Next Color", args)

    if args.previous:
        prev_dec = (current_dec - 1) % (MAX_DEC + 1)
        prev_hex = f"{prev_dec:06X}"
        print_color_and_info(prev_hex, "Previous Color", args)

    if args.negative:
        neg_dec = MAX_DEC - current_dec
        neg_hex = f"{neg_dec:06X}"
        print_color_and_info(neg_hex, "Negative Color", args)

def handle_gradient_command(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)
        
    colors_hex = []
    
    if args.random_gradient:
        num_hex = args.total_random_hex
        if num_hex == 0:
            num_hex = random.randint(2, 5)
        if num_hex < 2:
            log('error', "--total-random-hex must be at least 2")
            sys.exit(2)
        
        colors_hex = [f"{random.randint(0, MAX_DEC):06X}" for _ in range(num_hex)]
    else:
        if not args.hex or len(args.hex) < 2:
            log('error', "at least 2 hex codes are required for a gradient. use -H <HEXCODE> -H ... -H ...")
            sys.exit(2)
        
        colors_hex = [clean_hex_input(h) for h in args.hex]
    
    num_steps = args.steps
    if num_steps < 1:
        log('error', "--steps must be at least 1")
        sys.exit(2)
    
    if num_steps == 1:
        print_color_block(colors_hex[0], "Step 1")
        return

    colors_rgb = [hex_to_rgb(h) for h in colors_hex]
    num_segments = len(colors_rgb) - 1
    total_intervals = num_steps - 1

    gradient_colors = []
    for i in range(total_intervals + 1):
        if total_intervals > 0:
            t_global = i / total_intervals
        else:
            t_global = 0
        t_segment_scaled = t_global * num_segments
        segment_index = min(int(t_segment_scaled), num_segments - 1)
        
        t_local = t_segment_scaled - segment_index
        
        rgb1 = colors_rgb[segment_index]
        rgb2 = colors_rgb[segment_index + 1]
        
        r, g, b = linear_interpolate(rgb1, rgb2, t_local)
        gradient_colors.append(rgb_to_hex(r, g, b))

    for i, hex_code in enumerate(gradient_colors):
        print_color_block(hex_code, f"Step {i+1}")

def handle_mix_command(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)
        
    colors_hex = []
    
    if args.random_mix:
        num_hex = args.total_random_hex
        if num_hex == 0:
            num_hex = 2
        if num_hex < 2:
            log('error', "--total-random-hex must be at least 2")
            sys.exit(2)
        
        colors_hex = [f"{random.randint(0, MAX_DEC):06X}" for _ in range(num_hex)]
    else:
        if not args.hex or len(args.hex) < 2:
            log('error', "at least 2 hex codes are required for mixing. use -H <HEXCODE> -H ... -H ...")
            sys.exit(2)
        
        colors_hex = [clean_hex_input(h) for h in args.hex]

    colors_rgb = [hex_to_rgb(h) for h in colors_hex]
    
    total_r, total_g, total_b = 0, 0, 0
    for r_val, g_val, b_val in colors_rgb:
        total_r += r_val
        total_g += g_val
        total_b += b_val
    
    count = len(colors_rgb)
    avg_r = int(round(total_r / count))
    avg_g = int(round(total_g / count))
    avg_b = int(round(total_b / count))
    
    mixed_hex = rgb_to_hex(avg_r, avg_g, avg_b)
    
    print()
    for i, hex_code in enumerate(colors_hex):
        print_color_block(hex_code, f"Input {i+1}")
    
    print("-" * 18)
    print_color_block(mixed_hex, "Mixed Result")
    print()

def get_gradient_parser() -> argparse.ArgumentParser:
    parser = HexlabArgumentParser(
        prog="hexlab gradient",
        description="hexlab gradient: generate color gradients between multiple hex codes",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-H", "--hex",
        action="append",
        help="use -H <HEX> multiple times for inputs"
    )
    input_group.add_argument(
        "-rg", "--random-gradient",
        action="store_true",
        help="generate gradient from random colors"
    )
    
    parser.add_argument(
        "-s", "--steps",
        type=int,
        default=10,
        help="total number of steps in the gradient (default: 10)"
    )
    parser.add_argument(
        "-trh", "--total-random-hex",
        type=int,
        default=0,
        help="number of random colors to use (default: 2-5)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="random seed for reproducibility"
    )
    return parser

def get_mix_parser() -> argparse.ArgumentParser:
    parser = HexlabArgumentParser(
        prog="hexlab mix",
        description="hexlab mix: mix multiple colors together by averaging them",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-H", "--hex",
        action="append",
        help="use -H <HEX> multiple times for inputs"
    )
    input_group.add_argument(
        "-rm", "--random-mix",
        action="store_true",
        help="generate mix from random colors"
    )
    
    parser.add_argument(
        "-trh", "--total-random-hex",
        type=int,
        default=0,
        help="number of random colors to use (default: 2)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="random seed for reproducibility"
    )
    return parser

def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == 'gradient':
        parser = get_gradient_parser()
        args = parser.parse_args(sys.argv[2:])
        ensure_truecolor()
        handle_gradient_command(args)

    elif len(sys.argv) > 1 and sys.argv[1] == 'mix':
        parser = get_mix_parser()
        args = parser.parse_args(sys.argv[2:])
        ensure_truecolor()
        handle_mix_command(args)

    else:
        parser = HexlabArgumentParser(
            description="hexlab: 24-bit hex color exploration tool",
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
            "-lcn", "--list-color-names",
            nargs='?',
            const='text',
            default=None,
            choices=['text', 'json', 'pretty-json'],
            help="list all web color names and exit"
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
        color_input_group.add_argument(
            "-cn", "--color-name",
            help="web color names from --list-color-names"
        )
        
        parser.add_argument(
            "-s", "--seed",
            type=int,
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
            "-cmyk", "--cyan-magenta-yellow-key",
            action="store_true",
            dest="cmyk",
            help="show CMYK values"
        )
        info_group.add_argument(
            "-wcag", "--contrast",
            action="store_true",
            help="show WCAG contrast ratio"
        )
        
        parser.add_argument(
            "command",
            nargs='?',
            help=argparse.SUPPRESS
        )
        
        args = parser.parse_args()
        
        if args.list_color_names:
            format = args.list_color_names
            color_keys = sorted(list(WEB_COLORS.keys()))
            if format == 'text':
                for name in color_keys:
                    print(name)
            elif format == 'json':
                print(json.dumps(color_keys))
            elif format == 'pretty-json':
                print(json.dumps(color_keys, indent=4))
            sys.exit(0)
        
        if args.help_full:
            parser.print_help()
            
            gradient_parser = get_gradient_parser()
            print("\n")
            gradient_parser.print_help()
            
            mix_parser = get_mix_parser()
            print("\n")
            mix_parser.print_help()
            
            sys.exit(0)
        
        if args.command == 'gradient':
            log('error', "the 'gradient' command must be the first argument.")
            log('info', "usage: hexlab gradient -H ... -H ...")
            sys.exit(2)
            
        if args.command == 'mix':
            log('error', "the 'mix' command must be the first argument.")
            log('info', "usage: hexlab mix -H ... -H ...")
            sys.exit(2)
        
        ensure_truecolor()
        handle_color_command(args)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log('info', 'exiting.')
        sys.exit(0)
    except Exception as e:
        log('error', f"an unexpected error occurred: {e}")
        sys.exit(1)