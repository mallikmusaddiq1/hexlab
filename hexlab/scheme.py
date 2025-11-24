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
    COLOR_NAMES, MAX_DEC, SCHEME_KEYS,
    SRGB_TO_LINEAR_TH, LINEAR_TO_SRGB_TH, EPS
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
        original_name = HEX_TO_NAME.get(original_hex, '???')
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
    if v != v:
        return 0.0
    return max(0.0, min(1.0, v))

def _srgb_to_linear(c: int) -> float:
    c_norm = c / 255.0
    c_norm = _clamp01(c_norm)
    return c_norm / 12.92 if c_norm <= SRGB_TO_LINEAR_TH else ((c_norm + 0.055) / 1.055) ** 2.4

def rgb_to_hsl(r: int, g: int, b: int) -> Tuple[float, float, float]:
    r_f, g_f, b_f = r / 255.0, g / 255.0, b / 255.0
    cmax = max(r_f, g_f, b_f)
    cmin = min(r_f, g_f, b_f)
    delta = cmax - cmin
    l = (cmax + cmin) / 2
    if delta == 0:
        h = 0.0
        s = 0.0
    else:
        denom = 1 - abs(2 * l - 1)
        s = 0.0 if abs(denom) < EPS else delta / denom
        if cmax == r_f:
            h = 60 * (((g_f - b_f) / delta) % 6)
        elif cmax == g_f:
            h = 60 * ((b_f - r_f) / delta + 2)
        else:
            h = 60 * ((r_f - g_f) / delta + 4)
        h = (h + 360) % 360
    return (h, s, l)

def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
    h = h % 360
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
    return _clamp01(r) * 255, _clamp01(g) * 255, _clamp01(b) * 255

def rgb_to_xyz(r: int, g: int, b: int) -> Tuple[float, float, float]:
    r_lin = _srgb_to_linear(r)
    g_lin = _srgb_to_linear(g)
    b_lin = _srgb_to_linear(b)
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041
    return x * 100.0, y * 100.0, z * 100.0

def _xyz_f(t: float) -> float:
    return t ** (1 / 3) if t > 0.008856 else (7.787 * t) + (16.0 / 116.0)

def _xyz_f_inv(t: float) -> float:
    return t ** 3 if t > 0.20689655 else (t - 16.0 / 116.0) / 7.787

def xyz_to_lab(x: float, y: float, z: float) -> Tuple[float, float, float]:
    ref_x, ref_y, ref_z = 95.047, 100.0, 108.883
    x_r = _xyz_f(x / ref_x)
    y_r = _xyz_f(y / ref_y)
    z_r = _xyz_f(z / ref_z)
    l = (116.0 * y_r) - 16.0
    a = 500.0 * (x_r - y_r)
    b = 200.0 * (y_r - z_r)
    return l, a, b

def lab_to_xyz(l: float, a: float, b: float) -> Tuple[float, float, float]:
    ref_x, ref_y, ref_z = 95.047, 100.0, 108.883
    y_r = (l + 16.0) / 116.0
    x_r = a / 500.0 + y_r
    z_r = y_r - b / 200.0
    x = _xyz_f_inv(x_r) * ref_x
    y = _xyz_f_inv(y_r) * ref_y
    z = _xyz_f_inv(z_r) * ref_z
    return x, y, z

def _linear_to_srgb(l: float) -> float:
    l = max(l, 0.0)
    return 12.92 * l if l <= LINEAR_TO_SRGB_TH else 1.055 * (l ** (1 / 2.4)) - 0.055

def xyz_to_rgb(x: float, y: float, z: float) -> Tuple[float, float, float]:
    x_n, y_n, z_n = x / 100.0, y / 100.0, z / 100.0
    r_lin = x_n * 3.2404542 + y_n * -1.5371385 + z_n * -0.4985314
    g_lin = x_n * -0.9692660 + y_n * 1.8760108 + z_n * 0.0415560
    b_lin = x_n * 0.0556434 + y_n * -0.2040259 + z_n * 1.0572252
    r = _linear_to_srgb(r_lin)
    g = _linear_to_srgb(g_lin)
    b = _linear_to_srgb(b_lin)
    return _clamp01(r) * 255, _clamp01(g) * 255, _clamp01(b) * 255

def lab_to_lch(l: float, a: float, b: float) -> Tuple[float, float, float]:
    c = math.hypot(a, b)
    h = math.degrees(math.atan2(b, a)) % 360
    return l, c, h

def lch_to_lab(l: float, c: float, h: float) -> Tuple[float, float, float]:
    a = c * math.cos(math.radians(h))
    b = c * math.sin(math.radians(h))
    return l, a, b

def rgb_to_oklab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    r_lin = _srgb_to_linear(r)
    g_lin = _srgb_to_linear(g)
    b_lin = _srgb_to_linear(b)

    l = 0.4122214708 * r_lin + 0.5363325363 * g_lin + 0.0514459929 * b_lin
    m = 0.2119034982 * r_lin + 0.6806995451 * g_lin + 0.1073969566 * b_lin
    s = 0.0883024619 * r_lin + 0.2817188376 * g_lin + 0.6299787005 * b_lin

    l_ = (l + EPS) ** (1 / 3) if l >= 0 else -((-l + EPS) ** (1 / 3))
    m_ = (m + EPS) ** (1 / 3) if m >= 0 else -((-m + EPS) ** (1 / 3))
    s_ = (s + EPS) ** (1 / 3) if s >= 0 else -((-s + EPS) ** (1 / 3))

    ok_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    ok_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    ok_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return ok_l, ok_a, ok_b

def oklab_to_rgb(l: float, a: float, b: float) -> Tuple[float, float, float]:
    l_ = l + 0.3963377774 * a + 0.2158037573 * b
    m_ = l - 0.1055613458 * a - 0.0638541728 * b
    s_ = l - 0.0894841775 * a - 1.2914855480 * b

    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3

    r_lin = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g_lin = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_lin = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    r = _linear_to_srgb(r_lin)
    g = _linear_to_srgb(g_lin)
    b = _linear_to_srgb(b_lin)

    return _clamp01(r) * 255, _clamp01(g) * 255, _clamp01(b) * 255

def oklab_to_oklch(l: float, a: float, b: float) -> Tuple[float, float, float]:
    c = math.hypot(a, b)
    h = math.degrees(math.atan2(b, a)) % 360
    return l, c, h

def oklch_to_oklab(l: float, c: float, h: float) -> Tuple[float, float, float]:
    a = c * math.cos(math.radians(h))
    b = c * math.sin(math.radians(h))
    return l, a, b

def rgb_to_oklch(r: int, g: int, b: int) -> Tuple[float, float, float]:
    l, a, b_ok = rgb_to_oklab(r, g, b)
    return oklab_to_oklch(l, a, b_ok)

def oklch_to_rgb(l: float, c: float, h: float) -> Tuple[float, float, float]:
    l, a, b_ok = oklch_to_oklab(l, c, h)
    return oklab_to_rgb(l, a, b_ok)

def print_color_block(hex_code: str, title: str = "Color") -> None:
    r, g, b = hex_to_rgb(hex_code)
    print(f"{title:<18}:   \033[48;2;{r};{g};{b}m                \033[0m  #{hex_code}")


def handle_scheme_command(args: argparse.Namespace) -> None:
    if args.all_schemes:
        for key in SCHEME_KEYS:
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

    if base_hex is None:
        log('error', "no valid color provided for scheme")
        sys.exit(2)

    print()
    print_color_block(base_hex, title)
    print()
    r, g, b = hex_to_rgb(base_hex)

    h, s, l, c = (0.0,) * 4
    model = args.harmony_model if args.harmony_model else 'hsl'

    if model == 'hsl':
        h, s, l = rgb_to_hsl(r, g, b)
    elif model == 'lch':
        x, y, z = rgb_to_xyz(r, g, b)
        l_lab, a_lab, b_lab = xyz_to_lab(x, y, z)
        l, c, h = lab_to_lch(l_lab, a_lab, b_lab)
    elif model == 'oklch':
        l, c, h = rgb_to_oklch(r, g, b)

    def get_scheme_hex(hue_shift: float) -> str:
        new_h = (h + hue_shift) % 360
        new_r, new_g, new_b = 0.0, 0.0, 0.0
        if model == 'hsl':
            new_r, new_g, new_b = hsl_to_rgb(new_h, s, l)
        elif model == 'lch':
            nl, na, nb = lch_to_lab(l, c, new_h)
            nx, ny, nz = lab_to_xyz(nl, na, nb)
            new_r, new_g, new_b = xyz_to_rgb(nx, ny, nz)
        elif model == 'oklch':
            new_r, new_g, new_b = oklch_to_rgb(l, c, new_h)
        return rgb_to_hex(new_r, new_g, new_b)

    def get_mono_hex(l_shift: float) -> str:
        new_r, new_g, new_b = 0.0, 0.0, 0.0
        if model == 'hsl':
            new_l = max(0.0, min(1.0, l + l_shift))
            new_r, new_g, new_b = hsl_to_rgb(h, s, new_l)
        elif model == 'lch':
            new_l = max(0.0, min(100.0, l + (l_shift * 100)))
            nl, na, nb = lch_to_lab(new_l, c, h)
            nx, ny, nz = lab_to_xyz(nl, na, nb)
            new_r, new_g, new_b = xyz_to_rgb(nx, ny, nz)
        elif model == 'oklch':
            new_l = max(0.0, min(1.0, l + l_shift))
            new_r, new_g, new_b = oklch_to_rgb(new_l, c, h)
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
        print_color_block(get_scheme_hex(180), "comp        180°")
    else:
        if args.complementary:
            print_color_block(get_scheme_hex(180), "comp        180°")
        if args.split_complementary:
            print_color_block(get_scheme_hex(150), "split comp  150°")
            print_color_block(get_scheme_hex(210), "split comp  210°")
        if args.analogous:
            print_color_block(get_scheme_hex(-30), "analog      -30°")
            print_color_block(get_scheme_hex(30), "analog       30°")
        if args.triadic:
            print_color_block(get_scheme_hex(120), "tria        120°")
            print_color_block(get_scheme_hex(240), "tria        240°")
        if args.tetradic_square:
            print_color_block(get_scheme_hex(90), "tetra sq     90°")
            print_color_block(get_scheme_hex(180), "tetra sq    180°")
            print_color_block(get_scheme_hex(270), "tetra sq    270°")
        if args.tetradic_rectangular:
            print_color_block(get_scheme_hex(60), "tetra rec    60°")
            print_color_block(get_scheme_hex(180), "tetra rec   180°")
            print_color_block(get_scheme_hex(240), "tetra rec   240°")
        if args.monochromatic:
            print_color_block(get_mono_hex(-0.2), "mono       -20%L")
            print_color_block(get_mono_hex(0.2), "mono       +20%L")
    print()


def get_scheme_parser() -> argparse.ArgumentParser:
    parser = HexlabArgumentParser(
        prog="hexlab scheme",
        description="hexlab scheme: generate color harmonies",
        formatter_class=argparse.RawTextHelpFormatter
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-H", "--hex",
        type=INPUT_HANDLERS["hex"],
        help="base hex code for the scheme"
    )
    input_group.add_argument(
        "-r", "--random",
        action="store_true",
        help="generate a scheme from a random color"
    )
    input_group.add_argument(
        "-cn", "--color-name",
        type=INPUT_HANDLERS["color_name"],
        help="base color name from --list-color-names"
    )
    input_group.add_argument(
        "-di", "--decimal-index",
        type=INPUT_HANDLERS["decimal_index"],
        help="base color decimal index for the scheme"
    )
    parser.add_argument(
        "-s", "--seed",
        type=INPUT_HANDLERS["seed"],
        default=None,
        help="random seed for reproducibility"
    )
    parser.add_argument(
        "-hm", "--harmony-model",
        type=INPUT_HANDLERS["harmony_model"],
        default='hsl',
        help="harmony model: hsl lch oklch (default: hsl)"
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


def main() -> None:
    parser = get_scheme_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    handle_scheme_command(args)


if __name__ == "__main__":
    main()
