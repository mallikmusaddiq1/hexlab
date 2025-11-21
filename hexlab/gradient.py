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
    COLOR_NAMES, MAX_DEC, MAX_STEPS, MAX_RANDOM_COLORS,
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
    if v != v:
        return 0.0
    return max(0.0, min(1.0, v))

def _srgb_to_linear(c: int) -> float:
    c_norm = c / 255.0
    c_norm = _clamp01(c_norm)
    return c_norm / 12.92 if c_norm <= SRGB_TO_LINEAR_TH else ((c_norm + 0.055) / 1.055) ** 2.4

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

def _finalize_rgb_vals(r: float, g: float, b: float) -> Tuple[int, int, int]:
    r_i = int(round(r))
    g_i = int(round(g))
    b_i = int(round(b))
    r_i = max(0, min(255, r_i))
    g_i = max(0, min(255, g_i))
    b_i = max(0, min(255, b_i))
    return r_i, g_i, b_i

def print_color_block(hex_code: str, title: str = "Color") -> None:
    r, g, b = hex_to_rgb(hex_code)
    print(f"{title:<18}: \033[48;2;{r};{g};{b}m        \033[0m #{hex_code}")

def rgb_to_luv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    X, Y, Z = rgb_to_xyz(r, g, b)
    ref_X, ref_Y, ref_Z = 95.047, 100.0, 108.883
    denom = (X + 15 * Y + 3 * Z)
    if denom == 0:
        u_prime = 0.0
        v_prime = 0.0
    else:
        u_prime = (4 * X) / denom
        v_prime = (9 * Y) / denom

    denom_n = (ref_X + 15 * ref_Y + 3 * ref_Z)
    u_prime_n = (4 * ref_X) / denom_n
    v_prime_n = (9 * ref_Y) / denom_n

    y_r = Y / ref_Y
    if y_r > 0.008856:
        L = (116.0 * (y_r ** (1.0 / 3.0))) - 16.0
    else:
        L = 903.3 * y_r

    if L == 0:
        u = 0.0
        v = 0.0
    else:
        u = 13.0 * L * (u_prime - u_prime_n)
        v = 13.0 * L * (v_prime - v_prime_n)

    return L, u, v

def luv_to_rgb(L: float, u: float, v: float) -> Tuple[float, float, float]:
    ref_X, ref_Y, ref_Z = 95.047, 100.0, 108.883
    denom_n = (ref_X + 15 * ref_Y + 3 * ref_Z)
    u_prime_n = (4 * ref_X) / denom_n
    v_prime_n = (9 * ref_Y) / denom_n

    if L == 0:
        X = 0.0
        Y = 0.0
        Z = 0.0
        return xyz_to_rgb(X, Y, Z)

    u_prime = (u / (13.0 * L)) + u_prime_n
    v_prime = (v / (13.0 * L)) + v_prime_n

    if L > 8.0:
        Y = ref_Y * (((L + 16.0) / 116.0) ** 3)
    else:
        Y = ref_Y * (L / 903.3)

    if v_prime == 0:
        X = 0.0
        Z = 0.0
    else:
        X = Y * (9.0 * u_prime) / (4.0 * v_prime)
        Z = Y * (12.0 - 3.0 * u_prime - 20.0 * v_prime) / (4.0 * v_prime)

    return xyz_to_rgb(X, Y, Z)

def _get_interpolated_color(c1, c2, t: float, colorspace: str) -> Tuple[float, float, float]:
    if colorspace == 'srgb':
        r1, g1, b1 = c1
        r2, g2, b2 = c2
        r_new = r1 + t * (r2 - r1)
        g_new = g1 + t * (g2 - g1)
        b_new = b1 + t * (b2 - b1)
        return r_new, g_new, b_new

    if colorspace == 'srgb-linear':
        r_lin1, g_lin1, b_lin1 = c1
        r_lin2, g_lin2, b_lin2 = c2
        r_lin_new = r_lin1 + t * (r_lin2 - r_lin1)
        g_lin_new = g_lin1 + t * (g_lin2 - g_lin1)
        b_lin_new = b_lin1 + t * (b_lin2 - b_lin1)
        r = _linear_to_srgb(r_lin_new) * 255
        g = _linear_to_srgb(g_lin_new) * 255
        b = _linear_to_srgb(b_lin_new) * 255
        return r, g, b

    if colorspace == 'lab':
        l1, a1, b1 = c1
        l2, a2, b2 = c2
        l_new = l1 + t * (l2 - l1)
        a_new = a1 + t * (a2 - a1)
        b_new = b1 + t * (b2 - b1)
        x, y, z = lab_to_xyz(l_new, a_new, b_new)
        return xyz_to_rgb(x, y, z)

    if colorspace == 'oklab':
        l1, a1, b1 = c1
        l2, a2, b2 = c2
        l_new = l1 + t * (l2 - l1)
        a_new = a1 + t * (a2 - a1)
        b_new = b1 + t * (b2 - b1)
        return oklab_to_rgb(l_new, a_new, b_new)

    if colorspace == 'lch':
        l1, c1, h1 = c1
        l2, c2, h2 = c2
        h1, h2 = h1 % 360, h2 % 360
        h_diff = h2 - h1
        if h_diff > 180:
            h2 -= 360
        elif h_diff < -180:
            h2 += 360
        l_new = l1 + t * (l2 - l1)
        c_new = c1 + t * (c2 - c1)
        h_new = (h1 + t * (h2 - h1)) % 360
        l_lab, a_lab, b_lab = lch_to_lab(l_new, c_new, h_new)
        x, y, z = lab_to_xyz(l_lab, a_lab, b_lab)
        return xyz_to_rgb(x, y, z)

    if colorspace == 'oklch':
        l1, c1, h1 = c1
        l2, c2, h2 = c2
        h1, h2 = h1 % 360, h2 % 360
        h_diff = h2 - h1
        if h_diff > 180:
            h2 -= 360
        elif h_diff < -180:
            h2 += 360
        l_new = l1 + t * (l2 - l1)
        c_new = c1 + t * (c2 - c1)
        h_new = (h1 + t * (h2 - h1)) % 360
        return oklch_to_rgb(l_new, c_new, h_new)

    if colorspace == 'luv':
        l1, u1, v1 = c1
        l2, u2, v2 = c2
        l_new = l1 + t * (l2 - l1)
        u_new = u1 + t * (u2 - u1)
        v_new = v1 + t * (v2 - v1)
        return luv_to_rgb(l_new, u_new, v_new)

    return 0, 0, 0

def _convert_rgb_to_space(r: int, g: int, b: int, colorspace: str) -> Tuple[float, ...]:
    if colorspace == 'srgb':
        return (r, g, b)
    if colorspace == 'srgb-linear':
        return (_srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b))
    if colorspace == 'lab':
        x, y, z = rgb_to_xyz(r, g, b)
        return xyz_to_lab(x, y, z)
    if colorspace == 'oklab':
        return rgb_to_oklab(r, g, b)
    if colorspace == 'lch':
        x, y, z = rgb_to_xyz(r, g, b)
        l, a, b_lab = xyz_to_lab(x, y, z)
        return lab_to_lch(l, a, b_lab)
    if colorspace == 'oklch':
        return rgb_to_oklch(r, g, b)
    if colorspace == 'luv':
        return rgb_to_luv(r, g, b)
    return (r, g, b)


def handle_gradient_command(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)

    if (
        not args.random_gradient
        and not args.hex
        and not args.color_name
        and not getattr(args, "decimal_index", None)
    ):
        log('error', "at least one -H/--hex, -cn/--color-name, -di/--decimal-index, or -rg/--random-gradient is required")
        log('info', "use 'hexlab gradient -h' for more information")
        sys.exit(2)

    colorspace = args.colorspace

    colors_hex: List[str] = []
    if args.random_gradient:
        num_hex = args.total_random_hex
        if num_hex == 0:
            num_hex = random.randint(2, 5)
        num_hex = max(2, min(MAX_RANDOM_COLORS, num_hex))
        colors_hex = [f"{random.randint(0, MAX_DEC):06X}" for _ in range(num_hex)]
    else:
        input_list: List[str] = []
        if args.hex:
            input_list.extend(args.hex)
        if args.color_name:
            for nm in args.color_name:
                hexv = _get_color_name_hex(nm)
                if not hexv:
                    log('error', f"unknown color name '{nm}'")
                    log('info', "use 'hexlab --list-color-names' to see all options")
                    sys.exit(2)
                input_list.append(hexv)
        if getattr(args, "decimal_index", None):
            for di in args.decimal_index:
                hexv = di
                input_list.append(hexv)

        if len(input_list) < 2:
            log('error', "at least 2 hex codes, color names, or decimal indexes are required for a gradient")
            log('info', "usage: use -H HEX, -cn NAME, or -di INDEX multiple times")
            sys.exit(2)
        
        colors_hex = input_list

    num_steps = max(1, min(MAX_STEPS, args.steps))

    if num_steps == 1:
        print_color_block(colors_hex[0], "step 1")
        return

    colors_rgb = [hex_to_rgb(h) for h in colors_hex]

    colors_in_space = []
    for r_val, g_val, b_val in colors_rgb:
        colors_in_space.append(_convert_rgb_to_space(r_val, g_val, b_val, colorspace))

    num_segments = len(colors_in_space) - 1
    total_intervals = num_steps - 1
    gradient_colors: List[str] = []

    for i in range(total_intervals + 1):
        t_global = (i / total_intervals) if total_intervals > 0 else 0
        t_segment_scaled = t_global * num_segments
        segment_index = min(int(t_segment_scaled), num_segments - 1)
        t_local = t_segment_scaled - segment_index

        c1 = colors_in_space[segment_index]
        c2 = colors_in_space[segment_index + 1]

        r_f, g_f, b_f = _get_interpolated_color(c1, c2, t_local, colorspace)
        r_i, g_i, b_i = _finalize_rgb_vals(r_f, g_f, b_f)
        gradient_colors.append(rgb_to_hex(r_i, g_i, b_i))

    for i, hex_code in enumerate(gradient_colors):
        print_color_block(hex_code, f"step {i + 1}")


def get_gradient_parser() -> argparse.ArgumentParser:
    parser = HexlabArgumentParser(
        prog="hexlab gradient",
        description="hexlab gradient: generate color gradients between multiple hex codes",
        formatter_class=argparse.RawTextHelpFormatter
    )
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "-H", "--hex",
        action="append",
        type=INPUT_HANDLERS["hex"],
        help="use -H HEX multiple times for inputs"
    )
    input_group.add_argument(
        "-rg", "--random-gradient",
        action="store_true",
        help="generate gradient from random colors"
    )
    input_group.add_argument(
        "-cn", "--color-name",
        action="append",
        type=INPUT_HANDLERS["color_name"],
        help="use -cn NAME multiple times for inputs by name from --list-color-names"
    )
    input_group.add_argument(
        "-di", "--decimal-index",
        action="append",
        type=INPUT_HANDLERS["decimal_index"],
        help="use -di INDEX multiple times for inputs by decimal index"
    )
    parser.add_argument(
        "-S", "--steps",
        type=INPUT_HANDLERS["steps"],
        default=10,
        help=f"total steps in gradient (default: 10, max: {MAX_STEPS})"
    )
    parser.add_argument(
        "-cs", "--colorspace",
        default="lab",
        type=INPUT_HANDLERS["colorspace"],
        choices=['srgb', 'srgb-linear', 'lab', 'lch', 'oklab', 'oklch', 'luv'],
        help="colorspace for interpolation (default: lab)"
    )
    parser.add_argument(
        "-trh", "--total-random-hex",
        type=INPUT_HANDLERS["total_random"],
        default=0,
        help=f"number of random colors (default: 2-5, max: {MAX_RANDOM_COLORS})"
    )
    parser.add_argument(
        "-s", "--seed",
        type=INPUT_HANDLERS["seed"],
        default=None,
        help="random seed for reproducibility"
    )
    return parser


def main() -> None:
    parser = get_gradient_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    handle_gradient_command(args)


if __name__ == "__main__":
    main()
