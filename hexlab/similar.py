#!/usr/bin/env python3

import argparse
import sys
import random
import math
import re
import os
from typing import Tuple, List

from .input_utils import INPUT_HANDLERS, log, HexlabArgumentParser
from .constants import (
    COLOR_NAMES as COLOR_NAMES_RAW, MAX_DEC,
    DEDUP_DELTA_E_LAB, DEDUP_DELTA_E_OKLAB, DEDUP_DELTA_E_RGB,
    SRGB_TO_LINEAR_TH, EPS
)

def _normalize_hex_value(v: str) -> str:
    if not isinstance(v, str):
        return ''
    vv = v.replace('#', '').strip().upper()
    if len(vv) == 3:
        vv = ''.join([c * 2 for c in vv])
    return vv

COLOR_NAMES = {k: _normalize_hex_value(v) for k, v in COLOR_NAMES_RAW.items()}

def _norm_name_key(s: str) -> str:
    return re.sub(r'[^a-z]', '', str(s).lower())

HEX_TO_NAME = {}
for name, hexv in COLOR_NAMES.items():
    HEX_TO_NAME[hexv.upper()] = name

_norm_map = {}
for k, v in COLOR_NAMES.items():
    key = _norm_name_key(k)
    if key in _norm_map and _norm_map.get(key) != v:
        original_hex = _norm_map.get(key)
        original_name = HEX_TO_NAME.get(original_hex, '???')
        log('warn', f"color name collision on key '{key}': '{original_name}' and '{k}' both normalize to the same key. '{k}' will be used.")
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

def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    h = h % 360
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
    
    r_fin = (r_p + m) * 255.0
    g_fin = (g_p + m) * 255.0
    b_fin = (b_p + m) * 255.0
    
    return (
        max(0, min(255, int(round(r_fin)))),
        max(0, min(255, int(round(g_fin)))),
        max(0, min(255, int(round(b_fin))))
    )

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

def delta_e_ciede2000(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    C1 = math.sqrt(a1 ** 2 + b1 ** 2)
    C2 = math.sqrt(a2 ** 2 + b2 ** 2)
    C_bar = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt((C_bar ** 7) / (C_bar ** 7 + 25 ** 7 + EPS)))

    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2
    C1_prime = math.sqrt(a1_prime ** 2 + b1 ** 2)
    C2_prime = math.sqrt(a2_prime ** 2 + b2 ** 2)

    h1_prime_rad = math.atan2(b1, a1_prime)
    h1_prime_rad += 2 * math.pi if h1_prime_rad < 0 else 0
    h1_prime_deg = math.degrees(h1_prime_rad)

    h2_prime_rad = math.atan2(b2, a2_prime)
    h2_prime_rad += 2 * math.pi if h2_prime_rad < 0 else 0
    h2_prime_deg = math.degrees(h2_prime_rad)

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    C_prime_bar = (C1_prime + C2_prime) / 2

    if C1_prime * C2_prime == 0:
        delta_h_prime_deg = 0
    elif abs(h2_prime_deg - h1_prime_deg) <= 180:
        delta_h_prime_deg = h2_prime_deg - h1_prime_deg
    elif h2_prime_deg - h1_prime_deg > 180:
        delta_h_prime_deg = (h2_prime_deg - h1_prime_deg) - 360
    else:
        delta_h_prime_deg = (h2_prime_deg - h1_prime_deg) + 360

    delta_H_prime = 2 * math.sqrt(max(0.0, C1_prime * C2_prime)) * math.sin(math.radians(delta_h_prime_deg) / 2)

    L_prime_bar = (L1 + L2) / 2
    if C1_prime * C2_prime == 0:
        h_prime_bar_deg = h1_prime_deg + h2_prime_deg
    elif abs(h2_prime_deg - h1_prime_deg) <= 180:
        h_prime_bar_deg = (h1_prime_deg + h2_prime_deg) / 2
    elif (h1_prime_deg + h2_prime_deg) < 360:
        h_prime_bar_deg = (h1_prime_deg + h2_prime_deg + 360) / 2
    else:
        h_prime_bar_deg = (h1_prime_deg + h2_prime_deg - 360) / 2

    T = (
        1
        - 0.17 * math.cos(math.radians(h_prime_bar_deg - 30))
        + 0.24 * math.cos(math.radians(2 * h_prime_bar_deg))
        + 0.32 * math.cos(math.radians(3 * h_prime_bar_deg + 6))
        - 0.20 * math.cos(math.radians(4 * h_prime_bar_deg - 63))
    )

    S_L = 1 + (0.015 * (L_prime_bar - 50) ** 2) / math.sqrt(20 + (L_prime_bar - 50) ** 2 + EPS)
    S_C = 1 + 0.045 * C_prime_bar
    S_H = 1 + 0.015 * C_prime_bar * T

    delta_theta_deg = 30 * math.exp(-(((h_prime_bar_deg - 275) / 25) ** 2))
    R_C = 2 * math.sqrt((C_prime_bar ** 7) / (C_prime_bar ** 7 + 25 ** 7 + EPS))
    R_T = -R_C * math.sin(math.radians(2 * delta_theta_deg))

    k_L, k_C, k_H = 1, 1, 1

    delta_E = math.sqrt(
        (delta_L_prime / (k_L * S_L)) ** 2 +
        (delta_C_prime / (k_C * S_C)) ** 2 +
        (delta_H_prime / (k_H * S_H)) ** 2 +
        R_T * (delta_C_prime / (k_C * S_C)) * (delta_H_prime / (k_H * S_H))
    )

    return delta_E

def delta_e_euclidean_rgb(r1: int, g1: int, b1: int, r2: int, g2: int, b2: int) -> float:
    return math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)

def delta_e_euclidean_oklab(oklab1: Tuple[float, float, float], oklab2: Tuple[float, float, float]) -> float:
    l1, a1, b1 = oklab1
    l2, a2, b2 = oklab2
    return math.sqrt((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)

def print_color_block(hex_code: str, title: str = "color") -> None:
    r, g, b = hex_to_rgb(hex_code)
    print(f"{title:<18}:   \033[48;2;{r};{g};{b}m                \033[0m  #{hex_code}", end="")

def _generate_search_cloud(base_rgb: Tuple[int, int, int], count: int = 5000) -> List[Tuple[int, int, int]]:
    r, g, b = base_rgb
    h, s, l = rgb_to_hsl(r, g, b)
    
    candidates = set()

    for _ in range(count):
        h_delta = random.uniform(-20, 20)
        s_delta = random.uniform(-0.15, 0.15)
        l_delta = random.uniform(-0.15, 0.15)

        new_h = (h + h_delta) % 360
        new_s = max(0.0, min(1.0, s + s_delta))
        new_l = max(0.0, min(1.0, l + l_delta))

        nr, ng, nb = hsl_to_rgb(new_h, new_s, new_l)
        
        candidates.add((nr, ng, nb))

    return list(candidates)

def find_similar_colors_dynamic(base_rgb: Tuple[int, int, int], n: int = 5, metric: str = 'lab', dedup_val: float = 7.7) -> List[Tuple[str, float]]:
    base_r_i, base_g_i, base_b_i = base_rgb

    base_lab, base_oklab = None, None

    if metric == 'lab':
        x, y, z = rgb_to_xyz(base_r_i, base_g_i, base_b_i)
        base_lab = xyz_to_lab(x, y, z)
    elif metric == 'oklab':
        base_oklab = rgb_to_oklab(base_r_i, base_g_i, base_b_i)

    pool_size = max(5000, n * 10)
    candidate_pool = _generate_search_cloud(base_rgb, count=pool_size)

    valid_similar = []

    for cand_rgb in candidate_pool:
        r, g, b = cand_rgb
        if r == base_r_i and g == base_g_i and b == base_b_i:
            continue

        diff = 0.0

        if metric == 'lab':
            x, y, z = rgb_to_xyz(r, g, b)
            cand_lab = xyz_to_lab(x, y, z)
            diff = delta_e_ciede2000(base_lab, cand_lab)
        elif metric == 'oklab':
            cand_oklab = rgb_to_oklab(r, g, b)
            diff = delta_e_euclidean_oklab(base_oklab, cand_oklab)
        elif metric == 'rgb':
            diff = delta_e_euclidean_rgb(base_r_i, base_g_i, base_b_i, r, g, b)

        if diff >= dedup_val:
            valid_similar.append((rgb_to_hex(r, g, b), diff))

    valid_similar.sort(key=lambda x: x[1])

    return valid_similar[:n]


def handle_similar_command(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)

    clean_hex = None
    title = "base color"

    if args.random:
        current_dec = random.randint(0, MAX_DEC)
        clean_hex = f"{current_dec:06X}"
        title = "random"
    elif args.color_name:
        hex_val = _get_color_name_hex(args.color_name)
        if not hex_val:
            log('error', f"unknown color name '{args.color_name}'")
            log('info', "use 'hexlab --list-color-names' to see all options")
            sys.exit(2)
        clean_hex = hex_val
        title = HEX_TO_NAME.get(clean_hex.upper())
        if not title:
            title = args.color_name.title()
    elif args.hex:
        clean_hex = args.hex
        title = HEX_TO_NAME.get(clean_hex.upper(), f"#{clean_hex}")
    elif args.decimal_index:
        clean_hex = args.decimal_index
        idx = int(clean_hex, 16)
        title = HEX_TO_NAME.get(clean_hex.upper(), f"index {idx}")
    else:
        log('error', "one of the arguments -H/--hex, -r/--random, -di/--decimal-index, or -cn/--color-name is required")
        log('info', "use 'hexlab similar --help' for more information")
        sys.exit(2)

    print()
    print_color_block(clean_hex, title)
    print()
    print()

    base_rgb = hex_to_rgb(clean_hex)
    metric = args.distance_metric

    dedup_val = 0.0
    if args.dedup_value is not None:
        dedup_val = args.dedup_value
    else:
        if metric == 'rgb':
            dedup_val = DEDUP_DELTA_E_RGB
        elif metric == 'oklab':
            dedup_val = DEDUP_DELTA_E_OKLAB
        else:
            dedup_val = DEDUP_DELTA_E_LAB

    num_results = args.number

    similar_list = find_similar_colors_dynamic(
        base_rgb,
        n=num_results,
        metric=metric,
        dedup_val=dedup_val
    )

    metric_map = {'lab': 'ΔE2000', 'oklab': 'ΔE(OKLAB)', 'rgb': 'ΔE(RGB)'}
    metric_label = metric_map.get(metric, 'ΔE')

    if not similar_list:
        log('info', "no similar colors found within parameters")
    else:
        for i, (hex_val, diff) in enumerate(similar_list):
            label = f"similar {i+1}"
            print_color_block(hex_val, label)
            print(f"  ({metric_label}: {diff:.2f})")

    print()


def get_similar_parser() -> argparse.ArgumentParser:
    parser = HexlabArgumentParser(
        prog="hexlab similar",
        description="hexlab similar: find perceptually similar colors from the full 24-bit spectrum",
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
        help="base color name from --list-color-names"
    )
    input_group.add_argument(
        "-di", "--decimal-index",
        type=INPUT_HANDLERS["decimal_index"],
        help="base decimal index"
    )
    parser.add_argument(
        "-dm", "--distance-metric",
        type=INPUT_HANDLERS["distance_metric"],
        default='lab',
        help="distance metric: lab oklab rgb (default: lab)",
        choices=['lab', 'oklab', 'rgb']
    )
    parser.add_argument(
        "-dv", "--dedup-value",
        type=INPUT_HANDLERS["dedup_value"],
        default=None,
        help=f"custom deduplication threshold (defaults: lab={DEDUP_DELTA_E_LAB}, oklab={DEDUP_DELTA_E_OKLAB}, rgb={DEDUP_DELTA_E_RGB})"
    )
    parser.add_argument(
        "-n", "--number",
        type=INPUT_HANDLERS["number"],
        default=5,
        help="number of similar colors to find (min: 2, max: 1000, default: 5)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=INPUT_HANDLERS["seed"],
        default=None,
        help="seed for reproducibility of random"
    )
    return parser


def main() -> None:
    parser = get_similar_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    handle_similar_command(args)


if __name__ == "__main__":
    main()