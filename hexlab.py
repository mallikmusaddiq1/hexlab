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
from web_color_names import WEB_COLORS

MAX_DEC = 16777215
__version__ = "0.0.1"

HEX_REGEX = re.compile(r"([0-9A-Fa-f]{6}|[0-9A-Fa-f]{3})")

TECH_INFO_KEYS = [
    'index', 'red_green_blue', 'luminance', 'hue_saturation_lightness',
    'hsv', 'cmyk', 'contrast', 'xyz', 'lab', 'lightness_chroma_hue',
    'hue_whiteness_blackness'
]

SCHEME_KEYS = [
    'complementary', 'split_complementary', 'analogous', 'triadic',
    'tetradic_square', 'tetradic_rectangular', 'monochromatic'
]

SIMULATE_KEYS = [
    'protanopia', 'deuteranopia', 'tritanopia', 'achromatopsia'
]

CB_MATRICES = {
    "Protanopia": [
        [0.56667, 0.43333, 0],
        [0.55833, 0.44167, 0],
        [0, 0.24167, 0.75833]
    ],
    "Deuteranopia": [
        [0.625, 0.375, 0],
        [0.70, 0.30, 0],
        [0, 0.30, 0.70]
    ],
    "Tritanopia": [
        [0.95, 0.05, 0],
        [0, 0.43333, 0.56667],
        [0, 0.475, 0.525]
    ],
}

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
    hex_str = hex_str.replace(" ", "")
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

def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
    if s == 0:
        r = g = b = l
    else:
        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs(((h / 60) % 2) - 1))
        m = l - c / 2
        
        if 0 <= h < 60:
            r_p, g_p, b_p = c, x, 0
        elif 60 <= h < 120:
            r_p, g_p, b_p = x, c, 0
        elif 120 <= h < 180:
            r_p, g_p, b_p = 0, c, x
        elif 180 <= h < 240:
            r_p, g_p, b_p = 0, x, c
        elif 240 <= h < 300:
            r_p, g_p, b_p = x, 0, c
        else:
            r_p, g_p, b_p = c, 0, x
        
        r, g, b = (r_p + m), (g_p + m), (b_p + m)
    
    return r * 255, g * 255, b * 255

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

def _srgb_to_linear(c: int) -> float:
    c_norm = c / 255.0
    return c_norm / 12.92 if c_norm <= 0.03928 else ((c_norm + 0.055) / 1.055) ** 2.4

def rgb_to_xyz(r: int, g: int, b: int) -> Tuple[float, float, float]:
    r_lin = _srgb_to_linear(r)
    g_lin = _srgb_to_linear(g)
    b_lin = _srgb_to_linear(b)
    
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041
    
    return x * 100, y * 100, z * 100

def _xyz_f(t: float) -> float:
    return t ** (1/3) if t > 0.008856 else (7.787 * t) + (16 / 116)

def xyz_to_lab(x: float, y: float, z: float) -> Tuple[float, float, float]:
    ref_x, ref_y, ref_z = 95.047, 100.0, 108.883
    
    x_r = _xyz_f(x / ref_x)
    y_r = _xyz_f(y / ref_y)
    z_r = _xyz_f(z / ref_z)
    
    l = (116 * y_r) - 16
    a = 500 * (x_r - y_r)
    b = 200 * (y_r - z_r)
    
    return l, a, b

def lab_to_lch(l: float, a: float, b: float) -> Tuple[float, float, float]:
    c = math.hypot(a, b)
    h = math.degrees(math.atan2(b, a))
    h = h % 360
    return l, c, h

def rgb_to_hwb(r: int, g: int, b: int) -> Tuple[float, float, float]:
    h, s, v = rgb_to_hsv(r, g, b)
    w = (1 - s) * v
    b_hwb = 1 - v
    return h, w, b_hwb

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

    if args.hue_whiteness_blackness:
        h, w, b_hwb = rgb_to_hwb(r, g, b)
        print(f"   HWB        : {h:.1f}°, {w*100:.1f}%, {b_hwb*100:.1f}%")
        
    if args.cmyk:
        c, m, y, k = rgb_to_cmyk(r, g, b)
        print(f"   CMYK       : {c*100:.1f}%, {m*100:.1f}%, {y*100:.1f}%, {k*100:.1f}%")

    if args.xyz or args.lab or args.lightness_chroma_hue:
        x, y, z = rgb_to_xyz(r, g, b)
        if args.xyz:
            print(f"   XYZ        : {x:.4f}, {y:.4f}, {z:.4f}")
        
    if args.lab or args.lightness_chroma_hue:
        l_lab, a_lab, b_lab = xyz_to_lab(x, y, z)
        if args.lab:
            print(f"   LAB        : {l_lab:.4f}, {a_lab:.4f}, {b_lab:.4f}")
            
    if args.lightness_chroma_hue:
        l_lch, c_lch, h_lch = lab_to_lch(l_lab, a_lab, b_lab)
        print(f"   LCH        : {l_lch:.4f}, {c_lch:.4f}, {h_lch:.4f}°")

    if args.contrast:
        if 'l' not in locals():
            l = get_luminance(r, g, b)
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
    if args.all_tech_infos:
        for key in TECH_INFO_KEYS:
            setattr(args, key, True)
            
    clean_hex = None
    title = "Current Color"

    if args.seed is not None:
        random.seed(args.seed)
        
    if args.random_hex:
        current_dec = random.randint(0, MAX_DEC)
        clean_hex = f"{current_dec:06X}"
        title = "Random Color"
    elif args.color_name:
        named_lower = args.color_name.strip().lower().replace(" ", "")
        if named_lower not in WEB_COLORS:
            log('error', f"unknown color name '{named_lower}''")
            log('info', "use 'hexlab --list-color-names' to see all options")
            sys.exit(2)
        clean_hex = WEB_COLORS[named_lower]
        title = named_lower
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
            log('error', "at least 2 hex codes are required for a gradient")
            log('info', "usage: -H HEX -H ...")
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
        hex_no_spaces = args.hex.replace(" ", "")
        hex_strings = hex_no_spaces.split('+')

        if len(hex_strings) < 2:
            log('error', "at least 2 hex codes are required for mixing")
            log('info', "usage: input multiple hex codes separated by + symbol")
            sys.exit(2)
        
        colors_hex = [clean_hex_input(h) for h in hex_strings]

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

def handle_scheme_command(args: argparse.Namespace) -> None:
    if args.all_schemes:
        for key in SCHEME_KEYS:
            setattr(args, key, True)

    if args.seed is not None:
        random.seed(args.seed)

    if args.random_scheme:
        base_hex = f"{random.randint(0, MAX_DEC):06X}"
        title = "Random Base"
    else:
        base_hex = clean_hex_input(args.hex)
        title = "Base Color"
    
    print()
    print_color_block(base_hex, title)
    print()

    r, g, b = hex_to_rgb(base_hex)
    h, s, l = rgb_to_hsl(r, g, b)

    def get_scheme_hex(hue_shift: float) -> str:
        new_h = (h + hue_shift) % 360
        new_r, new_g, new_b = hsl_to_rgb(new_h, s, l)
        return rgb_to_hex(new_r, new_g, new_b)

    def get_mono_hex(l_shift: float) -> str:
        new_l = max(0.0, min(1.0, l + l_shift))
        new_r, new_g, new_b = hsl_to_rgb(h, s, new_l)
        return rgb_to_hex(new_r, new_g, new_b)

    any_specific_flag = (
        args.complementary or 
        args.split_complementary or 
        args.analogous or 
        args.triadic or 
        args.tetradic_square or
        args.tetradic_rectangular or
        args.monochromatic
    )

    if not any_specific_flag:
        print_color_block(get_scheme_hex(180), "Comp        180°")
    else:
        if args.complementary:
            print_color_block(get_scheme_hex(180), "Comp        180°")
        
        if args.split_complementary:
            print_color_block(get_scheme_hex(150), "Split Comp  150°")
            print_color_block(get_scheme_hex(210), "Split Comp  210°")

        if args.analogous:
            print_color_block(get_scheme_hex(-30), "Analog      -30°")
            print_color_block(get_scheme_hex(30), "Analog       30°")

        if args.triadic:
            print_color_block(get_scheme_hex(120), "Tria        120°")
            print_color_block(get_scheme_hex(240), "Tria        240°")

        if args.tetradic_square:
            print_color_block(get_scheme_hex(90), "Tetra Sq     90°")
            print_color_block(get_scheme_hex(180), "Tetra Sq    180°")
            print_color_block(get_scheme_hex(270), "Tetra Sq    270°")
        
        if args.tetradic_rectangular:
            print_color_block(get_scheme_hex(60), "Tetra Rec    60°")
            print_color_block(get_scheme_hex(180), "Tetra Rec   180°")
            print_color_block(get_scheme_hex(240), "Tetra Rec   240°")
            
        if args.monochromatic:
            print_color_block(get_mono_hex(-0.2), "Mono       -20%L")
            print_color_block(get_mono_hex(0.2), "Mono       +20%L")
    
    print()

def _linear_to_srgb(l: float) -> float:
    return 12.92 * l if l <= 0.0031308 else 1.055 * (l ** (1/2.4)) - 0.055

def handle_simulate_command(args: argparse.Namespace) -> None:
    if args.all_simulates:
        for key in SIMULATE_KEYS:
            setattr(args, key, True)

    if args.seed is not None:
        random.seed(args.seed)

    if args.random_simulate:
        base_hex = f"{random.randint(0, MAX_DEC):06X}"
        title = "Random Base"
    else:
        base_hex = clean_hex_input(args.hex)
        title = "Base Color"
    
    print()
    print_color_block(base_hex, title)
    print()

    r, g, b = hex_to_rgb(base_hex)

    def apply_matrix(r, g, b, m):
        rr = r * m[0][0] + g * m[0][1] + b * m[0][2]
        gg = r * m[1][0] + g * m[1][1] + b * m[1][2]
        bb = r * m[2][0] + g * m[2][1] + b * m[2][2]
        rr = max(0, min(255, int(rr)))
        gg = max(0, min(255, int(gg)))
        bb = max(0, min(255, int(bb)))
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
        print_color_block(sim_hex, "Protanopia")

    if args.deuteranopia or args.all_simulates:
        rr, gg, bb = apply_matrix(r, g, b, CB_MATRICES["Deuteranopia"])
        sim_hex = rgb_to_hex(rr, gg, bb)
        print_color_block(sim_hex, "Deuteranopia")
    
    if args.tritanopia or args.all_simulates:
        rr, gg, bb = apply_matrix(r, g, b, CB_MATRICES["Tritanopia"])
        sim_hex = rgb_to_hex(rr, gg, bb)
        print_color_block(sim_hex, "Tritanopia")
    
    if args.achromatopsia or args.all_simulates:
        l_lin = get_luminance(r, g, b)
        gray_srgb_norm = _linear_to_srgb(l_lin)
        gray_255 = max(0, min(255, int(gray_srgb_norm * 255)))
        sim_hex = rgb_to_hex(gray_255, gray_255, gray_255)
        print_color_block(sim_hex, "Achromatopsia")
    
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
        help="use -H HEX multiple times for inputs"
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
        "--seed",
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
        help="input multiple hex codes separated by + symbol"
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

def get_scheme_parser() -> argparse.ArgumentParser:
    parser = HexlabArgumentParser(
        prog="hexlab scheme",
        description="hexlab scheme: generate color harmonies",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-H", "--hex",
        help="base hex code for the scheme"
    )
    input_group.add_argument(
        "-rs", "--random-scheme",
        action="store_true",
        help="generate a scheme from a random color"
    )
    
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="random seed for reproducibility"
    )
    
    scheme_group = parser.add_argument_group("scheme types")
    scheme_group.add_argument(
        '-all', '--all-schemes',
        action="store_true",
        help="show all color schemes"
    )
    scheme_group.add_argument(
        '-co', '--complementary',
        action="store_true",
        help="show complementary color 180°"
    )
    scheme_group.add_argument(
        '-sco', '--split-complementary',
        action="store_true",
        help="show split-complementary colors 150° 210°"
    )
    scheme_group.add_argument(
        '-an', '--analogous',
        action="store_true",
        help="show analogous colors -30° +30°"
    )
    scheme_group.add_argument(
        '-tr', '--triadic',
        action="store_true",
        help="show triadic colors 120° 240°"
    )
    scheme_group.add_argument(
        '-tsq', '--tetradic-square',
        action="store_true",
        help="show tetradic square colors 90° 180° 270°"
    )
    scheme_group.add_argument(
        '-trc', '--tetradic-rectangular',
        action="store_true",
        help="show tetradic rectangular colors 60° 180° 240°"
    )
    scheme_group.add_argument(
        '-mch', '--monochromatic',
        action="store_true",
        help="show monochromatic colors -20%%L +20%%L"
    )
    return parser

def get_simulate_parser() -> argparse.ArgumentParser:
    parser = HexlabArgumentParser(
        prog="hexlab simulate",
        description="hexlab simulate: simulate color blindness",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-H", "--hex",
        help="base hex code for simulation"
    )
    input_group.add_argument(
        "-rs", "--random-simulate",
        action="store_true",
        help="simulate with a random color"
    )
    
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="random seed for reproducibility"
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
    
    elif len(sys.argv) > 1 and sys.argv[1] == 'scheme':
        parser = get_scheme_parser()
        args = parser.parse_args(sys.argv[2:])
        ensure_truecolor()
        handle_scheme_command(args)
    
    elif len(sys.argv) > 1 and sys.argv[1] == 'simulate':
        parser = get_simulate_parser()
        args = parser.parse_args(sys.argv[2:])
        ensure_truecolor()
        handle_simulate_command(args)

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
            "--list-color-names",
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
            help="show CIELCH values"
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
            
            scheme_parser = get_scheme_parser()
            print("\n")
            scheme_parser.print_help()
            
            simulate_parser = get_simulate_parser()
            print("\n")
            simulate_parser.print_help()
            
            sys.exit(0)
        
        if args.command == 'gradient':
            log('error', "the 'gradient' command must be the first argument")
            sys.exit(2)
            
        if args.command == 'mix':
            log('error', "the 'mix' command must be the first argument")
            sys.exit(2)
            
        if args.command == 'scheme':
            log('error', "the 'scheme' command must be the first argument")
            sys.exit(2)
            
        if args.command == 'simulate':
            log('error', "the 'simulate' command must be the first argument")
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