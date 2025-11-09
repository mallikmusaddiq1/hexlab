#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
import random
import math

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
        self.exit(2)

def hex_to_rgb(hex_code):
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(r, g, b):
    return f"{int(r):02X}{int(g):02X}{int(b):02X}"

def is_valid_hex(h):
    return HEX_REGEX.fullmatch(h) is not None

def clean_hex_input(hex_str):
    clean_hex = hex_str.lstrip("#").upper()
    if not is_valid_hex(clean_hex):
        log('error', f"'{hex_str}' is not a valid 6-digit hex code")
        sys.exit(2)
    if len(clean_hex) == 3:
        clean_hex = "".join([c*2 for c in clean_hex])
    return clean_hex

def lum_comp(c):
    c_norm = c / 255.0
    return c_norm / 12.92 if c_norm <= 0.03928 else ((c_norm + 0.055) / 1.055) ** 2.4

def get_luminance(r, g, b):
    return 0.2126 * lum_comp(r) + 0.7152 * lum_comp(g) + 0.0722 * lum_comp(b)

def rgb_to_hsl(r, g, b):
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
            
    return (h, s, l)

def rgb_to_cmyk(r, g, b):
    if r == 0 and g == 0 and b == 0:
        return 0, 0, 0, 1
    
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    k = 1.0 - max(r_norm, g_norm, b_norm)
    c = (1.0 - r_norm - k) / (1.0 - k)
    m = (1.0 - g_norm - k) / (1.0 - k)
    y = (1.0 - b_norm - k) / (1.0 - k)
    return (c, m, y, k)

def get_wcag_contrast(lum):
    lum_white = 1.0
    lum_black = 0.0
    
    contrast_white = (lum_white + 0.05) / (lum + 0.05)
    contrast_black = (lum + 0.05) / (lum_black + 0.05)
    
    results = {
        "white": {"ratio": contrast_white},
        "black": {"ratio": contrast_black}
    }

    def get_pass_fail(ratio):
        return {
            "AA-Large": "Pass" if ratio >= 3 else "Fail",
            "AA": "Pass" if ratio >= 4.5 else "Fail",
            "AAA-Large": "Pass" if ratio >= 4.5 else "Fail",
            "AAA": "Pass" if ratio >= 7 else "Fail",
        }
        
    results["white"]["levels"] = get_pass_fail(contrast_white)
    results["black"]["levels"] = get_pass_fail(contrast_black)
    
    return results

def linear_interpolate(rgb1, rgb2, t):
    r1, g1, b1 = rgb1
    r2, g2, b2 = rgb2
    
    r = r1 + t * (r2 - r1)
    g = g1 + t * (g2 - g1)
    b = b1 + t * (b2 - b1)
    
    return (r, g, b)

def print_color_block(hex_code, title="Color"):
    r, g, b = hex_to_rgb(hex_code)
    print(f"{title:<18}: \033[48;2;{r};{g};{b}m        \033[0m #{hex_code}")

def print_color_and_info(hex_code, title, args):
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
        
    if args.cmyk:
        c, m, y, k = rgb_to_cmyk(r, g, b)
        print(f"   CMYK       : {c*100:.1f}%, {m*100:.1f}%, {y*100:.1f}%, {k*100:.1f}%")

    if args.contrast:
        wcag = get_wcag_contrast(l)
        print( "   Contrast (White): "
            f"{wcag['white']['ratio']:.2f}:1 "
            f"(AA: {wcag['white']['levels']['AA']}, "
            f"AAA: {wcag['white']['levels']['AAA']})"
        )
        print( "   Contrast (Black): "
            f"{wcag['black']['ratio']:.2f}:1 "
            f"(AA: {wcag['black']['levels']['AA']}, "
            f"AAA: {wcag['black']['levels']['AAA']})"
        )
        
    print()

def handle_color_command(args):
    clean_hex = None
    title = "Current Color"

    if args.random_hex:
        current_dec = random.randint(0, MAX_DEC)
        clean_hex = f"{current_dec:06X}"
        title = "Random Color"
    elif args.hexcode:
        clean_hex = clean_hex_input(args.hexcode)
    else:
        log('error', "hex code is required. use -H/--hex <HEXCODE> or -rh/--random-hex")
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

def handle_gradient_command(args):
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
        t_global = i / total_intervals if total_intervals > 0 else 0
        t_segment_scaled = t_global * num_segments
        segment_index = min(int(t_segment_scaled), num_segments - 1)
        
        t_local = t_segment_scaled - segment_index
        
        rgb1 = colors_rgb[segment_index]
        rgb2 = colors_rgb[segment_index + 1]
        
        r, g, b = linear_interpolate(rgb1, rgb2, t_local)
        gradient_colors.append(rgb_to_hex(r, g, b))

    for i, hex_code in enumerate(gradient_colors):
        print_color_block(hex_code, f"Step {i+1}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'gradient':
        parser = HexlabArgumentParser(
            prog="hexlab gradient",
            description="hexlab gradient: Generate color gradients between multiple hex codes",
            formatter_class=argparse.RawTextHelpFormatter
        )
        
        input_group = parser.add_mutually_exclusive_group(required=True)
        input_group.add_argument(
            "-H", "--hex",
            action="append",
            help="6-digit hex code without # symbol, use -H <HEXCODE> multiple times for anchors"
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
        
        args = parser.parse_args(sys.argv[2:])
        ensure_truecolor()
        handle_gradient_command(args)

    else:
        parser = HexlabArgumentParser(
            description="hexlab: A CLI tool for 24-bit hex color exploration",
            formatter_class=argparse.RawTextHelpFormatter,
            add_help=False
        )
        
        # Re-add default help action
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
            help="show full help message"
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
        
        if args.help_full:
            parser.print_help()
            
            gradient_parser = HexlabArgumentParser(
                prog="hexlab gradient",
                description="hexlab gradient: Generate color gradients between multiple hex codes",
                formatter_class=argparse.RawTextHelpFormatter,
                add_help=False
            )
            
            grad_input_group = gradient_parser.add_mutually_exclusive_group(required=True)
            grad_input_group.add_argument(
                "-H", "--hex",
                action="append",
                help="6-digit hex code without # symbol, use -H <HEXCODE> multiple times for anchors"
            )
            grad_input_group.add_argument(
                "-rg", "--random-gradient",
                action="store_true",
                help="generate gradient from random colors"
            )
            
            gradient_parser.add_argument(
                "-s", "--steps",
                type=int,
                default=10,
                help="total number of steps in the gradient (default: 10)"
            )
            gradient_parser.add_argument(
                "-trh", "--total-random-hex",
                type=int,
                default=0,
                help="number of random colors to use (default: 2-5)"
            )
            print("\n")
            gradient_parser.print_help()
            sys.exit(0)
        
        if args.command == 'gradient':
            log('error', "the 'gradient' command must be the first argument")
            log('info', "usage: hexlab gradient -H ... -H ...")
            sys.exit(2)
        
        ensure_truecolor()
        handle_color_command(args)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[hexlab][info] exiting.", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        log('error', f"an unexpected error occurred: {e}")
        sys.exit(1))