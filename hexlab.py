#!/usr/bin/env python3

import argparse
import os
import re
import sys
import random
import math
import json
from typing import Tuple, List
from constants import COLOR_NAMES as COLOR_NAMES_RAW

MAX_DEC = 16777215
MAX_STEPS = 10000
MAX_RANDOM_COLORS = 100

__version__ = "0.0.1"

HEX_REGEX = re.compile(r"^(?:[0-9A-Fa-f]{6}|[0-9A-Fa-f]{3})$")

TECH_INFO_KEYS = [
    'index', 'red_green_blue', 'luminance', 'hue_saturation_lightness',
    'hsv', 'cmyk', 'contrast', 'xyz', 'lab', 'lightness_chroma_hue',
    'hue_whiteness_blackness', 'oklab', 'cieluv', 'similar', 'name'
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

FORMAT_ALIASES = {
    'hex': 'hex',
    'index': 'index',
    'rgb': 'rgb',
    'redgreenblue': 'rgb',
    'hsl': 'hsl',
    'huesaturationlightness': 'hsl',
    'hsv': 'hsv',
    'huesaturationvalue': 'hsv',
    'hwb': 'hwb',
    'huewhitenessblackness': 'hwb',
    'cmyk': 'cmyk',
    'cyanmagentayellowkey': 'cmyk',
    'xyz': 'xyz',
    'ciexyz': 'xyz',
    'lab': 'lab',
    'cielab': 'lab',
    'lch': 'lch',
    'cielch': 'lch',
    'oklab': 'oklab',
    'name': 'name',
    'luv': 'luv',
    'cieluv': 'luv'
}

def _strip_and_unquote(s: str) -> str:
    if s is None:
        return ''
    s = str(s).strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'`":
        s = s[1:-1].strip()
    return s

def _extract_hex_chars(s: str) -> str:
    s = _strip_and_unquote(s)
    chars = re.findall(r'[0-9A-Fa-f]', s, re.IGNORECASE)
    return ''.join(chars).upper()

def type_hex(value: str) -> str:
    h = _extract_hex_chars(value)
    if not h:
        raise argparse.ArgumentTypeError(f"invalid hex value: '{value}' (no hex characters found)")
    return h

def type_color_name(value: str) -> str:
    if not isinstance(value, str):
        return ""

    s = _strip_and_unquote(value)
    return re.sub(r'[^0-9a-z]', '', s.lower())

def type_general_string(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return _strip_and_unquote(value)

def type_format_alias(value: str) -> str:
    if not isinstance(value, str):
        return ""
    s = _strip_and_unquote(value)
    return re.sub(r'[^0-9a-z]', '', s.lower())

def type_list_format(value: str) -> str:
    if not isinstance(value, str):
        return ""

    s = _strip_and_unquote(value).lower()
    s = re.sub(r'[\s_]+', '-', s)
    s = re.sub(r'[^0-9a-z-]', '', s)
    return s

def type_colorspace(value: str) -> str:
    if not isinstance(value, str):
        return ""
    s = _strip_and_unquote(value)
    return s.lower().replace(" ", "")

def type_int_safe(value) -> int:
    try:
        s = _strip_and_unquote(str(value))
        num_str = re.findall(r'[-+]?\d+', s)
        if not num_str:
            raise ValueError("No digits found")
        return int(num_str[0])
    except Exception:
        raise argparse.ArgumentTypeError(f"invalid integer value: '{value}'")

def _normalize_hex_value(v: str) -> str:
    if not isinstance(v, str):
        return ''
    vv = v.replace('#', '').strip().upper()
    if len(vv) == 3:
        vv = ''.join([c*2 for c in vv])
    return vv

def log(level: str, message: str) -> None:
    level = level.lower()
    level_map = {
        'info': sys.stdout,
        'warn': sys.stderr,
        'error': sys.stderr
    }
    stream = level_map.get(level, sys.stderr)
    print(f"[hexlab][{level}] {message}", file=stream)

COLOR_NAMES = {k: _normalize_hex_value(v) for k, v in COLOR_NAMES_RAW.items()}

def _norm_name_key(s: str) -> str:
    return re.sub(r'[^0-9a-z]', '', s.lower())

HEX_TO_NAME = {}
for name, hexv in COLOR_NAMES.items():
    HEX_TO_NAME[hexv.upper()] = name

_norm_map = {}
for k, v in COLOR_NAMES.items():
    key = _norm_name_key(k)
    if key in _norm_map and _norm_map.get(key) != v:
        original_hex = _norm_map.get(key)
        original_name = HEX_TO_NAME.get(original_hex, '???')
        log('warn', f"Color name collision on key '{key}': '{original_name}' and '{k}' both normalize to the same key. '{k}' will be used.")
    _norm_map[key] = v
COLOR_NAMES_LOOKUP = _norm_map

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
    h = _extract_hex_chars(hex_code)
    if len(h) == 3:
        h = ''.join([c*2 for c in h])
    if len(h) != 6:
        h = h.ljust(6, '0')[:6]
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(r, g, b):
    return f"{int(round(r)):02X}{int(round(g)):02X}{int(round(b)):02X}"

def fmt_hex_for_output(hex_str: str) -> str:
    return f"#{hex_str.upper()}"

def is_valid_hex(h: str) -> bool:
    return HEX_REGEX.fullmatch(h) is not None

def clean_hex_input(hex_str: str) -> str:
    if not hex_str:
        log('error', "input does not contain any hex characters")
        sys.exit(2)

    clean_hex = hex_str
    if len(clean_hex) == 3:
        clean_hex = "".join([c*2 for c in clean_hex])

    if len(clean_hex) < 6:
        padded_hex = clean_hex.ljust(6, '0')
        log('warn', f"hex '{hex_str}' is too short, padding to '{padded_hex}'")
        clean_hex = padded_hex
    elif len(clean_hex) > 6:
        truncated_hex = clean_hex[:6]
        log('warn', f"hex '{hex_str}' is too long, truncating to '{truncated_hex}'")
        clean_hex = truncated_hex

    if not is_valid_hex(clean_hex):
        log('error', f"'{hex_str}' (processed to '{clean_hex}') is not valid")
        sys.exit(2)
    return clean_hex

SRGB_TO_LINEAR_TH = 0.04045
LINEAR_TO_SRGB_TH = 0.0031308
EPS = 1e-12

def _clamp01(v: float) -> float:
    if v != v:
        return 0.0
    return max(0.0, min(1.0, v))

def lum_comp(c: int) -> float:
    c_norm = c / 255.0
    c_norm = _clamp01(c_norm)
    return c_norm / 12.92 if c_norm <= SRGB_TO_LINEAR_TH else ((c_norm + 0.055) / 1.055) ** 2.4

def get_luminance(r: int, g: int, b: int) -> float:
    return 0.2126 * lum_comp(r) + 0.7152 * lum_comp(g) + 0.0722 * lum_comp(b)

def rgb_to_hsl(r: int, g: int, b: int) -> Tuple[float, float, float]:
    r_f, g_f, b_f = r/255.0, g/255.0, b/255.0
    cmax = max(r_f, g_f, b_f)
    cmin = min(r_f, g_f, b_f)
    delta = cmax - cmin
    l = (cmax + cmin) / 2
    if delta == 0:
        h = 0.0
        s = 0.0
    else:
        denom = 1 - abs(2*l - 1)
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

def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    r_f, g_f, b_f = r/255.0, g/255.0, b/255.0
    cmax = max(r_f, g_f, b_f)
    cmin = min(r_f, g_f, b_f)
    delta = cmax - cmin
    v = cmax
    if delta == 0:
        h = 0.0
        s = 0.0
    else:
        s = delta / v if v != 0 else 0.0
        if cmax == r_f:
            h = 60 * (((g_f - b_f) / delta) % 6)
        elif cmax == g_f:
            h = 60 * ((b_f - r_f) / delta + 2)
        else:
            h = 60 * ((r_f - g_f) / delta + 4)
        h = (h + 360) % 360
    return (h, s, v)

def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    h = h % 360
    c = v * s
    x = c * (1 - abs(((h / 60) % 2) - 1))
    m = v - c
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

def rgb_to_cmyk(r: int, g: int, b: int) -> Tuple[float, float, float, float]:
    if r == 0 and g == 0 and b == 0:
        return 0.0, 0.0, 0.0, 1.0
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    k = 1.0 - max(r_norm, g_norm, b_norm)
    if k >= 1.0:
        return 0.0, 0.0, 0.0, 1.0
    denom = (1.0 - k)
    c = (1.0 - r_norm - k) / denom
    m = (1.0 - g_norm - k) / denom
    y = (1.0 - b_norm - k) / denom
    return (c, m, y, k)

def cmyk_to_rgb(c: float, m: float, y: float, k: float) -> Tuple[float, float, float]:
    r = 255 * (1 - _clamp01(c)) * (1 - _clamp01(k))
    g = 255 * (1 - _clamp01(m)) * (1 - _clamp01(k))
    b = 255 * (1 - _clamp01(y)) * (1 - _clamp01(k))
    return r, g, b

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
    return t ** (1/3) if t > 0.008856 else (7.787 * t) + (16.0 / 116.0)

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
    return 12.92 * l if l <= LINEAR_TO_SRGB_TH else 1.055 * (l ** (1/2.4)) - 0.055

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

    l_ = (l + EPS) ** (1/3) if l >= 0 else -((-l + EPS) ** (1/3))
    m_ = (m + EPS) ** (1/3) if m >= 0 else -((-m + EPS) ** (1/3))
    s_ = (s + EPS) ** (1/3) if s >= 0 else -((-s + EPS) ** (1/3))

    ok_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    ok_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    ok_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return ok_l, ok_a, ok_b

def oklab_to_rgb(l: float, a: float, b: float) -> Tuple[float, float, float]:
    l_ = l + 0.3963377774 * a + 0.2158037573 * b
    m_ = l - 0.1055613458 * a - 0.0638541728 * b
    s_ = l - 0.0894841775 * a - 1.2914855480 * b

    l = l_**3
    m = m_**3
    s = s_**3

    r_lin =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g_lin = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_lin = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    r = _linear_to_srgb(r_lin)
    g = _linear_to_srgb(g_lin)
    b = _linear_to_srgb(b_lin)

    return _clamp01(r) * 255, _clamp01(g) * 255, _clamp01(b) * 255

def rgb_to_hwb(r: int, g: int, b: int) -> Tuple[float, float, float]:
    h, s, v = rgb_to_hsv(r, g, b)
    w = (1 - s) * v
    b_hwb = 1 - v
    return h, w, b_hwb

def hwb_to_rgb(h: float, w: float, b: float) -> Tuple[float, float, float]:
    v = 1 - b
    s = 1 - (w / v) if v != 0 else 0
    return hsv_to_rgb(h, s, v)

def delta_e_ciede2000(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt((C_bar**7) / (C_bar**7 + 25**7 + EPS)))

    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2
    C1_prime = math.sqrt(a1_prime**2 + b1**2)
    C2_prime = math.sqrt(a2_prime**2 + b2**2)

    h1_prime_rad = math.atan2(b1, a1_prime)
    h1_prime_rad += 2 * math.pi if h1_prime_rad < 0 else 0
    h1_prime_deg = math.degrees(h1_prime_rad)

    h2_prime_rad = math.atan2(b2, a2_prime)
    h2_prime_rad += 2 * math.pi if h2_prime_rad < 0 else 0
    h2_prime_deg = math.degrees(h2_prime_rad)

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    C_prime_bar = (C1_prime + C2_prime) / 2

    delta_h_prime_deg = 0
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
    h_prime_bar_deg = 0
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

    S_L = 1 + (0.015 * (L_prime_bar - 50)**2) / math.sqrt(20 + (L_prime_bar - 50)**2 + EPS)
    S_C = 1 + 0.045 * C_prime_bar
    S_H = 1 + 0.015 * C_prime_bar * T

    delta_theta_deg = 30 * math.exp(-(((h_prime_bar_deg - 275) / 25)**2))
    R_C = 2 * math.sqrt((C_prime_bar**7) / (C_prime_bar**7 + 25**7 + EPS))
    R_T = -R_C * math.sin(math.radians(2 * delta_theta_deg))

    k_L, k_C, k_H = 1, 1, 1

    delta_E = math.sqrt(
        (delta_L_prime / (k_L * S_L))**2 +
        (delta_C_prime / (k_C * S_C))**2 +
        (delta_H_prime / (k_H * S_H))**2 +
        R_T * (delta_C_prime / (k_C * S_C)) * (delta_H_prime / (k_H * S_H))
    )

    return delta_E

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

def _normalize_value_string(s: str) -> str:
    s = _strip_and_unquote(s)
    s = s.replace('°', ' ')
    s = s.replace('–', '-')
    s = re.sub(r'deg', ' ', s, flags=re.IGNORECASE)

    if s.lower().startswith(('rgb(', 'hsl(', 'hsv(', 'hwb(', 'cmyk(', 'xyz(', 'lab(', 'lch(', 'oklab(', 'luv(')):
        s = re.sub(r'^[a-zA-Z]+\s*\(', '', s, flags=re.IGNORECASE)
        s = s.rstrip(')')

    s = s.replace(',', ' ')
    s = s.replace('/', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def _parse_numerical_string(s: str) -> List[float]:
    s = _normalize_value_string(s)
    try:
        tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        vals = [float(t) for t in tokens]
        for v in vals:
            if not math.isfinite(v):
                raise ValueError
        return vals
    except Exception:
        raise ValueError(f"could not parse numerical values from '{s}'")

def _safe_float(s: str) -> float:
    try:
        v = float(s)
    except Exception:
        log('error', f"invalid numeric value '{s}'")
        sys.exit(2)
    if not math.isfinite(v):
        log('error', f"non-finite numeric value '{s}'")
        sys.exit(2)
    return v

def _finalize_rgb_vals(r: float, g: float, b: float) -> Tuple[int, int, int]:
    r_i = int(round(r))
    g_i = int(round(g))
    b_i = int(round(b))
    r_i = max(0, min(255, r_i))
    g_i = max(0, min(255, g_i))
    b_i = max(0, min(255, b_i))
    return r_i, g_i, b_i

def parse_rgb_string(s: str) -> Tuple[int, int, int]:
    try:
        nums = _parse_numerical_string(s)
    except ValueError:
        log('error', f"invalid rgb string: {s}")
        sys.exit(2)
    if len(nums) < 3:
        log('error', f"invalid rgb string: {s}")
        sys.exit(2)
    def _to_8bit(val: float) -> int:
        if 0.0 <= val <= 1.0:
            v = val * 255.0
        else:
            v = val
        return max(0, min(255, int(round(v))))
    r = _to_8bit(nums[0])
    g = _to_8bit(nums[1])
    b = _to_8bit(nums[2])
    return r, g, b

def _parse_h_ss_string(s: str, model_name: str) -> Tuple[float, float, float]:
    s_norm = _normalize_value_string(s)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?%?", s_norm)
    if len(nums) < 3:
        log('error', f"invalid {model_name} string: {s}")
        sys.exit(2)
    
    h = _safe_float(re.sub(r'%', '', nums[0])) % 360
    
    s_val = nums[1]
    v_val = nums[2]
    s_f = _safe_float(re.sub(r'%', '', s_val))
    v_f = _safe_float(re.sub(r'%', '', v_val))
    s_f = s_f / 100.0 if '%' in s_val or s_f > 1.0 else s_f
    v_f = v_f / 100.0 if '%' in v_val or v_f > 1.0 else v_f
    return h, _clamp01(s_f), _clamp01(v_f)

def parse_hsl_string(s: str) -> Tuple[float, float, float]:
    return _parse_h_ss_string(s, "hsl")

def parse_hsv_string(s: str) -> Tuple[float, float, float]:
    return _parse_h_ss_string(s, "hsv")

def parse_hwb_string(s: str) -> Tuple[float, float, float]:
    return _parse_h_ss_string(s, "hwb")

def parse_cmyk_string(s: str) -> Tuple[float, float, float, float]:
    s_norm = _normalize_value_string(s)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?%?", s_norm)
    if len(nums) < 4:
        log('error', f"invalid cmyk string: {s}")
        sys.exit(2)
    vals = []
    for n in nums[:4]:
        v = _safe_float(re.sub(r'%', '', n))
        v = v / 100.0 if '%' in n or v > 1.0 else v
        vals.append(_clamp01(v))
    return tuple(vals)

def _parse_3_floats(s: str, model_name: str) -> Tuple[float, float, float]:
    try:
        nums = _parse_numerical_string(s)
    except ValueError:
        log('error', f"invalid {model_name} string: {s}")
        sys.exit(2)
    if len(nums) < 3:
        log('error', f"invalid {model_name} string: {s}")
        sys.exit(2)
    return float(nums[0]), float(nums[1]), float(nums[2])

def parse_xyz_string(s: str) -> Tuple[float, float, float]:
    return _parse_3_floats(s, "xyz")

def parse_lab_string(s: str) -> Tuple[float, float, float]:
    return _parse_3_floats(s, "lab")

def parse_lch_string(s: str) -> Tuple[float, float, float]:
    return _parse_3_floats(s, "lch")

def parse_oklab_string(s: str) -> Tuple[float, float, float]:
    return _parse_3_floats(s, "oklab")

def parse_luv_string(s: str) -> Tuple[float, float, float]:
    # expects "L U V" where L in [0..100], u and v are floats (can be negative)
    return _parse_3_floats(s, "luv")

def find_similar_colors(base_lab: Tuple[float, float, float], n: int = 5) -> List[Tuple[str, str, float]]:

    similar = []
    try:
        base_x, base_y, base_z = lab_to_xyz(*base_lab)
        base_r, base_g, base_b = xyz_to_rgb(base_x, base_y, base_z)
        base_r_i, base_g_i, base_b_i = _finalize_rgb_vals(base_r, base_g, base_b)
        base_hex = rgb_to_hex(base_r_i, base_g_i, base_b_i)
    except Exception:
        base_hex = None

    for name, hex_code in COLOR_NAMES.items():
        if base_hex and hex_code.upper() == base_hex.upper():
            continue

        r, g, b = hex_to_rgb(hex_code)
        x, y, z = rgb_to_xyz(r, g, b)
        lab = xyz_to_lab(x, y, z)

        diff = delta_e_ciede2000(base_lab, lab)
        similar.append((name, hex_code, diff))

    similar.sort(key=lambda x: x[2])
    return similar[:n]

def print_color_block(hex_code: str, title: str = "Color") -> None:
    r, g, b = hex_to_rgb(hex_code)
    print(f"{title:<18}: \033[48;2;{r};{g};{b}m        \033[0m #{hex_code}")

def rgb_to_luv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    # Convert sRGB -> XYZ -> Luv
    X, Y, Z = rgb_to_xyz(r, g, b)
    # reference white D65 same as used in lab functions
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

    # compute L*
    y_r = Y / ref_Y
    if y_r > 0.008856:
        L = (116.0 * (y_r ** (1.0/3.0))) - 16.0
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
    # Convert Luv -> XYZ -> sRGB
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

    # avoid division by zero
    if v_prime == 0:
        X = 0.0
        Z = 0.0
    else:
        X = Y * (9.0 * u_prime) / (4.0 * v_prime)
        Z = Y * (12.0 - 3.0 * u_prime - 20.0 * v_prime) / (4.0 * v_prime)

    return xyz_to_rgb(X, Y, Z)

def print_color_and_info(hex_code: str, title: str, args: argparse.Namespace) -> None:
    print_color_block(hex_code, title)
    r, g, b = hex_to_rgb(hex_code)

    x, y, z, l_lab, a_lab, b_lab = (0.0,) * 6

    needs_xyz = args.xyz or args.lab or args.lightness_chroma_hue or args.oklab or args.similar or args.cieluv
    needs_lab = args.lab or args.lightness_chroma_hue or args.similar

    if needs_xyz:
        x, y, z = rgb_to_xyz(r, g, b)
    if needs_lab:
        if not needs_xyz:
            x, y, z = rgb_to_xyz(r, g, b)
        l_lab, a_lab, b_lab = xyz_to_lab(x, y, z)

    if args.index:
        index = int(hex_code, 16)
        print(f"   Index      : {index} / {MAX_DEC}")
    if args.red_green_blue:
        print(f"   RGB        : {r}, {g}, {b}")
    if args.name:
        name = HEX_TO_NAME.get(hex_code.upper())
        if name:
            print(f"   Name       : {name}")
        else:
            print(f"   Name       : unknown")
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
        c, m, y_cmyk, k = rgb_to_cmyk(r, g, b)
        print(f"   CMYK       : {c*100:.1f}%, {m*100:.1f}%, {y_cmyk*100:.1f}%, {k*100:.1f}%")
    if args.xyz:
        print(f"   XYZ        : {x:.4f}, {y:.4f}, {z:.4f}")
    if args.lab:
        print(f"   LAB        : {l_lab:.4f}, {a_lab:.4f}, {b_lab:.4f}")
    if args.lightness_chroma_hue:
        l_lch, c_lch, h_lch = lab_to_lch(l_lab, a_lab, b_lab)
        print(f"   LCH        : {l_lch:.4f}, {c_lch:.4f}, {h_lch:.4f}°")
    if args.oklab:
        l_ok, a_ok, b_ok = rgb_to_oklab(r, g, b)
        print(f"   OKLAB      : {l_ok:.4f}, {a_ok:.4f}, {b_ok:.4f}")
    if args.cieluv:
        l_uv, u_uv, v_uv = rgb_to_luv(r, g, b)
        print(f"   LUV        : {l_uv:.4f}, {u_uv:.4f}, {v_uv:.4f}")
    if args.contrast:
        if not (args.luminance):
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

    if args.similar:
        print("   Similar Colors:")
        similar_colors = find_similar_colors((l_lab, a_lab, b_lab))
        if not similar_colors:
            print("     (No similar colors found in list)")
        for name, hex_val, diff in similar_colors:
            s_r, s_g, s_b = hex_to_rgb(hex_val)
            print(f"     \033[48;2;{s_r};{s_g};{s_b}m  \033[0m #{hex_val} {name:<18} (ΔE: {diff:.2f})")

    print()

def _get_color_name_hex(sanitized_name: str) -> str:
    if not sanitized_name:
        return None
    return COLOR_NAMES_LOOKUP.get(sanitized_name)

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
        hex_val = _get_color_name_hex(args.color_name)
        if not hex_val:
            log('error', f"unknown color name '{args.color_name}'")
            log('info', "use 'hexlab --list-color-names' to see all options")
            sys.exit(2)
        clean_hex = hex_val
        title = HEX_TO_NAME.get(clean_hex.upper())
        if not title:
            title = args.color_name.title()
    elif args.hexcode:
        clean_hex = clean_hex_input(args.hexcode)
        title = HEX_TO_NAME.get(clean_hex.upper(), f"#{clean_hex}")
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

def _get_interpolated_color(c1, c2, t: float, colorspace: str) -> Tuple[float, float, float]:
    if colorspace == 'srgb':
        r1, g1, b1 = c1
        r2, g2, b2 = c2
        r_new = r1 + t * (r2 - r1)
        g_new = g1 + t * (g2 - g1)
        b_new = b1 + t * (b2 - b1)
        return r_new, g_new, b_new

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

    return 0, 0, 0

def _convert_rgb_to_space(r: int, g: int, b: int, colorspace: str) -> Tuple[float, ...]:
    if colorspace == 'srgb':
        return (r, g, b)
    if colorspace == 'lab':
        x, y, z = rgb_to_xyz(r, g, b)
        return xyz_to_lab(x, y, z)
    if colorspace == 'oklab':
        return rgb_to_oklab(r, g, b)
    if colorspace == 'lch':
        x, y, z = rgb_to_xyz(r, g, b)
        l, a, b_lab = xyz_to_lab(x, y, z)
        return lab_to_lch(l, a, b_lab)
    return (r, g, b)

def handle_gradient_command(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)

    if not args.random_gradient and not args.hex and not args.color_name:
        log('error', "at least one -H/--hex, -cn/--color-name, or -rg/--random-gradient is required")
        log('info', "use 'hexlab gradient -h' for more information")
        sys.exit(2)

    colorspace = args.colorspace

    colors_hex = []
    if args.random_gradient:
        num_hex = args.total_random_hex
        if num_hex == 0:
            num_hex = random.randint(2, 5)
        num_hex = max(2, min(MAX_RANDOM_COLORS, num_hex))
        colors_hex = [f"{random.randint(0, MAX_DEC):06X}" for _ in range(num_hex)]
    else:
        input_list = []
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
        
        if len(input_list) < 2:
            log('error', "at least 2 hex codes or color names are required for a gradient")
            log('info', "usage: use -H HEX or -cn NAME multiple times")
            sys.exit(2)
        colors_hex = [clean_hex_input(h) for h in input_list]

    num_steps = max(1, min(MAX_STEPS, args.steps))

    if num_steps == 1:
        print_color_block(colors_hex[0], "Step 1")
        return

    colors_rgb = [hex_to_rgb(h) for h in colors_hex]

    colors_in_space = []
    for r_val, g_val, b_val in colors_rgb:
        colors_in_space.append(_convert_rgb_to_space(r_val, g_val, b_val, colorspace))

    num_segments = len(colors_in_space) - 1
    total_intervals = num_steps - 1
    gradient_colors = []

    for i in range(total_intervals + 1):
        t_global = (i / total_intervals) if total_intervals > 0 else 0
        t_segment_scaled = t_global * num_segments
        segment_index = min(int(t_segment_scaled), num_segments - 1)
        t_local = t_segment_scaled - segment_index

        c1 = colors_in_space[segment_index]
        c2 = colors_in_space[segment_index + 1]

        r_f, g_f, b_f = _get_interpolated_color(c1, c2, t_local, colorspace)
        r_f, g_f, b_f = _finalize_rgb_vals(r_f, g_f, b_f)
        gradient_colors.append(rgb_to_hex(r_f, g_f, b_f))

    for i, hex_code in enumerate(gradient_colors):
        print_color_block(hex_code, f"Step {i+1}")

def handle_mix_command(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)

    if not args.random_mix and not args.hex and not args.color_name:
        log('error', "at least one -H/--hex, -cn/--color-name, or -rm/--random-mix is required")
        log('info', "use 'hexlab mix -h' for more information")
        sys.exit(2)

    colorspace = args.colorspace

    colors_hex = []
    if args.random_mix:
        num_hex = args.total_random_hex
        if num_hex == 0:
            num_hex = 2
        num_hex = max(2, min(MAX_RANDOM_COLORS, num_hex))
        colors_hex = [f"{random.randint(0, MAX_DEC):06X}" for _ in range(num_hex)]
    else:
        input_list = []
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

        if len(input_list) < 2:
            log('error', "at least 2 hex codes or color names are required for mixing")
            log('info', "usage: use -H HEX or -cn NAME multiple times")
            sys.exit(2)
        colors_hex = [clean_hex_input(h) for h in input_list]

    colors_rgb = [hex_to_rgb(h) for h in colors_hex]

    colors_in_space = []
    for r_val, g_val, b_val in colors_rgb:
        colors_in_space.append(_convert_rgb_to_space(r_val, g_val, b_val, colorspace))

    total_c1, total_c2, total_c3 = 0.0, 0.0, 0.0
    for c in colors_in_space:
        total_c1 += c[0]
        total_c2 += c[1]
        total_c3 += c[2]

    count = len(colors_in_space)
    avg_c1 = total_c1 / count
    avg_c2 = total_c2 / count
    avg_c3 = total_c3 / count

    avg_r_f, avg_g_f, avg_b_f = 0.0, 0.0, 0.0

    if colorspace == 'srgb':
        avg_r_f, avg_g_f, avg_b_f = avg_c1, avg_c2, avg_c3
    elif colorspace == 'lab':
        avg_x, avg_y, avg_z = lab_to_xyz(avg_c1, avg_c2, avg_c3)
        avg_r_f, avg_g_f, avg_b_f = xyz_to_rgb(avg_x, avg_y, avg_z)
    elif colorspace == 'oklab':
        avg_r_f, avg_g_f, avg_b_f = oklab_to_rgb(avg_c1, avg_c2, avg_c3)

    r_i, g_i, b_i = _finalize_rgb_vals(avg_r_f, avg_g_f, avg_b_f)
    mixed_hex = rgb_to_hex(r_i, g_i, b_i)

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
    
    base_hex = None
    title = "Base Color"
        
    if args.random_scheme:
        base_hex = f"{random.randint(0, MAX_DEC):06X}"
        title = "Random Base"
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
        base_hex = clean_hex_input(args.hex)
        title = HEX_TO_NAME.get(base_hex.upper(), f"#{base_hex}")
    
    if base_hex is None:
        log('error', "no valid color provided for scheme")
        sys.exit(2)

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

def handle_simulate_command(args: argparse.Namespace) -> None:
    if args.all_simulates:
        for key in SIMULATE_KEYS:
            setattr(args, key, True)
    if args.seed is not None:
        random.seed(args.seed)

    base_hex = None
    title = "Base Color"

    if args.random_simulate:
        base_hex = f"{random.randint(0, MAX_DEC):06X}"
        title = "Random Base"
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
        base_hex = clean_hex_input(args.hex)
        title = HEX_TO_NAME.get(base_hex.upper(), f"#{base_hex}")
    
    if base_hex is None:
        log('error', "no valid color provided for simulate")
        sys.exit(2)
            
    print()
    print_color_block(base_hex, title)
    print()
    r, g, b = hex_to_rgb(base_hex)
    def apply_matrix(r: int, g: int, b: int, m: List[List[float]]) -> Tuple[int, int, int]:
        r_lin = _srgb_to_linear(r)
        g_lin = _srgb_to_linear(g)
        b_lin = _srgb_to_linear(b)
        rr_lin = r_lin * m[0][0] + g_lin * m[0][1] + b_lin * m[0][2]
        gg_lin = r_lin * m[1][0] + g_lin * m[1][1] + b_lin * m[1][2]
        bb_lin = r_lin * m[2][0] + g_lin * m[2][1] + b_lin * m[2][2]
        rr_srgb_norm = _linear_to_srgb(rr_lin)
        gg_srgb_norm = _linear_to_srgb(gg_lin)
        bb_srgb_norm = _linear_to_srgb(bb_lin)
        rr, gg, bb = _finalize_rgb_vals(rr_srgb_norm * 255, gg_srgb_norm * 255, bb_srgb_norm * 255)
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
        gray_255 = max(0, min(255, int(round(gray_srgb_norm * 255))))
        sim_hex = rgb_to_hex(gray_255, gray_255, gray_255)
        print_color_block(sim_hex, "Achromatopsia")
    print()

def _parse_value_to_rgb(clean_val: str, from_fmt: str) -> Tuple[int, int, int]:
    r_f, g_f, b_f = 0.0, 0.0, 0.0
    if from_fmt == 'hex':
        hex_val = clean_hex_input(type_hex(clean_val))
        r_f, g_f, b_f = hex_to_rgb(hex_val)
    elif from_fmt == 'rgb':
        r_i, g_i, b_i = parse_rgb_string(clean_val)
        return r_i, g_i, b_i
    elif from_fmt == 'hsl':
        h, s, l = parse_hsl_string(clean_val)
        r_f, g_f, b_f = hsl_to_rgb(h, s, l)
    elif from_fmt == 'hsv':
        h, s, v = parse_hsv_string(clean_val)
        r_f, g_f, b_f = hsv_to_rgb(h, s, v)
    elif from_fmt == 'hwb':
        h, w, b_hwb = parse_hwb_string(clean_val)
        r_f, g_f, b_f = hwb_to_rgb(h, w, b_hwb)
    elif from_fmt == 'cmyk':
        c, m, y, k = parse_cmyk_string(clean_val)
        r_f, g_f, b_f = cmyk_to_rgb(c, m, y, k)
    elif from_fmt == 'xyz':
        x, y, z = parse_xyz_string(clean_val)
        r_f, g_f, b_f = xyz_to_rgb(x, y, z)
    elif from_fmt == 'lab':
        l, a, b_lab = parse_lab_string(clean_val)
        x, y, z = lab_to_xyz(l, a, b_lab)
        r_f, g_f, b_f = xyz_to_rgb(x, y, z)
    elif from_fmt == 'lch':
        l, c, h = parse_lch_string(clean_val)
        l, a, b_lab = lch_to_lab(l, c, h)
        x, y, z = lab_to_xyz(l, a, b_lab)
        r_f, g_f, b_f = xyz_to_rgb(x, y, z)
    elif from_fmt == 'oklab':
        l, a, b_ok = parse_oklab_string(clean_val)
        r_f, g_f, b_f = oklab_to_rgb(l, a, b_ok)
    elif from_fmt == 'luv':
        L, u, v = parse_luv_string(clean_val)
        r_f, g_f, b_f = luv_to_rgb(L, u, v)
    elif from_fmt == 'index':
        try:
            dec_str = re.findall(r'[-+]?\d+', str(clean_val))[0]
            dec_val = int(dec_str)
        except Exception:
            log('error', f"invalid index value '{clean_val}'")
            sys.exit(2)
        dec_val = max(0, min(MAX_DEC, dec_val))
        hex_val = f"{dec_val:06X}"
        r_f, g_f, b_f = hex_to_rgb(hex_val)
    elif from_fmt == 'name':
        sanitized_name = type_color_name(clean_val)
        hex_val = _get_color_name_hex(sanitized_name)
        if not hex_val:
            log('error', f"unknown color name '{clean_val}' (sanitized to '{sanitized_name}')")
            log('info', "use 'hexlab --list-color-names' to see all options")
            sys.exit(2)
        r_f, g_f, b_f = hex_to_rgb(hex_val)
    r_i, g_i, b_i = _finalize_rgb_vals(r_f, g_f, b_f)
    return r_i, g_i, b_i

def _format_value_from_rgb(r: int, g: int, b: int, to_fmt: str) -> str:
    output_value = ""
    if to_fmt == 'hex':
        output_value = fmt_hex_for_output(rgb_to_hex(r, g, b))
    elif to_fmt == 'rgb':
        output_value = f"rgb({r}, {g}, {b})"
    elif to_fmt == 'hsl':
        h, s, l = rgb_to_hsl(r, g, b)
        output_value = f"hsl({h:.1f}°, {s*100:.1f}%, {l*100:.1f}%)"
    elif to_fmt == 'hsv':
        h, s, v = rgb_to_hsv(r, g, b)
        output_value = f"hsv({h:.1f}°, {s*100:.1f}%, {v*100:.1f}%)"
    elif to_fmt == 'hwb':
        h, w, b_hwb = rgb_to_hwb(r, g, b)
        output_value = f"hwb({h:.1f}°, {w*100:.1f}%, {b_hwb*100:.1f}%)"
    elif to_fmt == 'cmyk':
        c, m, y, k = rgb_to_cmyk(r, g, b)
        output_value = f"cmyk({c*100:.1f}%, {m*100:.1f}%, {y*100:.1f}%, {k*100:.1f}%)"
    elif to_fmt == 'index':
        hex_val = rgb_to_hex(r, g, b)
        output_value = str(int(hex_val, 16))
    elif to_fmt == 'xyz':
        x, y, z = rgb_to_xyz(r, g, b)
        output_value = f"xyz({x:.4f}, {y:.4f}, {z:.4f})"
    elif to_fmt == 'lab':
        x, y, z = rgb_to_xyz(r, g, b)
        l, a, b_lab = xyz_to_lab(x, y, z)
        output_value = f"lab({l:.4f}, {a:.4f}, {b_lab:.4f})"
    elif to_fmt == 'lch':
        x, y, z = rgb_to_xyz(r, g, b)
        l, a, b_lab = xyz_to_lab(x, y, z)
        l, c, h = lab_to_lch(l, a, b_lab)
        output_value = f"lch({l:.4f}, {c:.4f}, {h:.4f}°)"
    elif to_fmt == 'oklab':
        l, a, b_ok = rgb_to_oklab(r, g, b)
        output_value = f"oklab({l:.4f}, {a:.4f}, {b_ok:.4f})"
    elif to_fmt == 'luv':
        L, u, v = rgb_to_luv(r, g, b)
        output_value = f"luv({L:.4f}, {u:.4f}, {v:.4f})"
    elif to_fmt == 'name':
        hex_val = rgb_to_hex(r, g, b)
        found = False
        for name, hex_code in COLOR_NAMES.items():
            if hex_code.upper() == hex_val.upper():
                output_value = name
                found = True
                break
        if not found:
            output_value = f"unknown #{hex_val}"
    return output_value

def handle_convert_command(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)
    try:
        from_fmt = FORMAT_ALIASES[args.from_format]
        to_fmt = FORMAT_ALIASES[args.to_format]
    except KeyError as e:
        log('error', f"invalid format specified: {e} (original input: '{vars(args).get('from_format') or vars(args).get('to_format')}')")
        log('info', f"use 'hexlab convert -h' to see all formats")
        sys.exit(2)
    r, g, b = (0, 0, 0)
    if args.random_value:
        dec_val = random.randint(0, MAX_DEC)
        r, g, b = hex_to_rgb(f"{dec_val:06X}")
    else:
        raw_value = args.value
        clean_val = raw_value
        r, g, b = _parse_value_to_rgb(clean_val, from_fmt)
    output_value_str = _format_value_from_rgb(r, g, b, to_fmt)
    if args.verbose:
        input_value_str = _format_value_from_rgb(r, g, b, from_fmt)
        print(f"{input_value_str} -> {output_value_str}")
    else:
        print(output_value_str)

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
        type=type_hex,
        help="use -H HEX multiple times for inputs"
    )
    input_group.add_argument(
        "-rg", "--random-gradient",
        action="store_true",
        help="generate gradient from random colors"
    )
    parser.add_argument(
        "-cn", "--color-name",
        action="append",
        type=type_color_name,
        help="use -cn NAME multiple times for inputs by name from --list-color-names"
    )
    parser.add_argument(
        "-S", "--steps",
        type=type_int_safe,
        default=10,
        help=f"total steps in gradient (default: 10, max: {MAX_STEPS})"
    )
    parser.add_argument(
        "-cs", "--colorspace",
        default="lab",
        type=type_colorspace,
        choices=['srgb', 'lab', 'lch', 'oklab'],
        help="colorspace for interpolation (default: lab)"
    )
    parser.add_argument(
        "-trh", "--total-random-hex",
        type=type_int_safe,
        default=0,
        help=f"number of random colors (default: 2-5, max: {MAX_RANDOM_COLORS})"
    )
    parser.add_argument(
        "-s", "--seed",
        type=type_int_safe,
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
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "-H", "--hex",
        action="append",
        type=type_hex,
        help="use -H HEX multiple times for inputs"
    )
    input_group.add_argument(
        "-rm", "--random-mix",
        action="store_true",
        help="generate mix from random colors"
    )
    parser.add_argument(
        "-cn", "--color-name",
        action="append",
        type=type_color_name,
        help="use -cn NAME multiple times for inputs by name from --list-color-names"
    )
    parser.add_argument(
        "-cs", "--colorspace",
        default="lab",
        type=type_colorspace,
        choices=['srgb', 'lab', 'oklab'],
        help="colorspace for mixing (default: lab)"
    )
    parser.add_argument(
        "-trh", "--total-random-hex",
        type=type_int_safe,
        default=0,
        help=f"number of random colors (default: 2, max: {MAX_RANDOM_COLORS})"
    )
    parser.add_argument(
        "-s", "--seed",
        type=type_int_safe,
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
        type=type_hex,
        help="base hex code for the scheme"
    )
    input_group.add_argument(
        "-rs", "--random-scheme",
        action="store_true",
        help="generate a scheme from a random color"
    )
    input_group.add_argument(
        "-cn", "--color-name",
        type=type_color_name,
        help="base color name from --list-color-names"
    )
    parser.add_argument(
        "-s", "--seed",
        type=type_int_safe,
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
        type=type_hex,
        help="base hex code for simulation"
    )
    input_group.add_argument(
        "-rs", "--random-simulate",
        action="store_true",
        help="simulate with a random color"
    )
    input_group.add_argument(
        "-cn", "--color-name",
        type=type_color_name,
        help="base color name from --list-color-names"
    )
    parser.add_argument(
        "-s", "--seed",
        type=type_int_safe,
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

def get_convert_parser() -> argparse.ArgumentParser:
    parser = HexlabArgumentParser(
        prog="hexlab convert",
        description="hexlab convert: convert a color value from one format to another",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False
    )
    formats_list = "hex rgb hsl hsv hwb cmyk xyz lab lch oklab luv index name"
    parser.add_argument(
        '-h', '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='show this help message and exit'
    )
    parser.add_argument(
        "-f", "--from-format",
        required=True,
        type=type_format_alias,
        help="the format to convert from\n"
             f"all formats: {formats_list}\n"
             f"use quotes for better UX"
    )
    parser.add_argument(
        "-t", "--to-format",
        required=True,
        type=type_format_alias,
        help="the format to convert to\n"
             f"all formats: {formats_list}\n"
             f"use quotes for better UX"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-v", "--value",
        type=type_general_string,
        help="write value to convert in quotes, if it contains spaces\n"
             "examples:\n"
             '  -v "000000"\n'
             '  -v "0"\n'
             '  -v "black"\n'
             '  -v "rgb(0, 0, 0)"\n'
             '  -v "hsl(0°, 0%%, 0%%)"\n'
             '  -v "hsv(0°, 0%%, 0%%)"\n'
             '  -v "hwb(0°, 0%%, 100%%)"\n'
             '  -v "cmyk(0%%, 0%%, 0%%, 100%%)"\n'
             '  -v "xyz(0, 0, 0)"\n'
             '  -v "lab(0, 0, 0)"\n'
             '  -v "lch(0, 0, 0°)"\n'
             '  -v "oklab(0, 0, 0)"\n'
             '  -v "luv(0, 0, 0)"'
    )
    input_group.add_argument(
        "-rv", "--random-value",
        action="store_true",
        help="generate a random value for the --from-format"
    )
    parser.add_argument(
        "-s", "--seed",
        type=type_int_safe,
        default=None,
        help="random seed for reproducibility"
    )
    parser.add_argument(
        "-V", "--verbose",
        action="store_true",
        help="print the conversion verbosely"
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
    elif len(sys.argv) > 1 and sys.argv[1] == 'convert':
        parser = get_convert_parser()
        args = parser.parse_args(sys.argv[2:])
        ensure_truecolor()
        handle_convert_command(args)
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
            type=type_list_format,
            choices=['text', 'json', 'pretty-json'],
            help="list available color names and exit"
        )
        color_input_group = parser.add_mutually_exclusive_group()
        color_input_group.add_argument(
            "-H", "--hex",
            dest="hexcode",
            type=type_hex,
            help="6-digit hex color code without # symbol",
        )
        color_input_group.add_argument(
            "-rh", "--random-hex",
            action="store_true",
            help="generate a random hex color"
        )
        color_input_group.add_argument(
            "-cn", "--color-name",
            type=type_color_name,
            help="color names from 'hexlab --list-color-names'"
        )
        parser.add_argument(
            "-s", "--seed",
            type=type_int_safe,
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
            help="show CIE 1976 LCH values"
        )
        info_group.add_argument(
            "--oklab",
            action="store_true",
            help="show OKLAB values"
        )
        info_group.add_argument(
            "--cieluv", "-luv",
            action="store_true",
            dest="cieluv",
            help="show CIE L*u*v* (LUV) values"
        )
        info_group.add_argument(
            "-wcag", "--contrast",
            action="store_true",
            help="show WCAG contrast ratio"
        )
        info_group.add_argument(
            "-S", "--similar",
            action="store_true",
            help="find similar colors from the color name list"
        )
        info_group.add_argument(
            "--name",
            action="store_true",
            help="show color name if available in --list-color-names otherwise 'unknown'"
        )
        parser.add_argument(
            "command",
            nargs='?',
            help=argparse.SUPPRESS
        )
        args = parser.parse_args()
        if args.list_color_names:
            fmt = args.list_color_names
            color_keys = sorted(list(COLOR_NAMES.keys()))
            if fmt == 'text':
                for name in color_keys:
                    print(name)
            elif fmt == 'json':
                print(json.dumps(color_keys))
            elif fmt == 'pretty-json':
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
            convert_parser = get_convert_parser()
            print("\n")
            convert_parser.print_help()
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
        if args.command == 'convert':
            log('error', "the 'convert' command must be the first argument")
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