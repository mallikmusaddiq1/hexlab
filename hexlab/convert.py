#!/usr/bin/env python3

import argparse
import os
import sys
import random
import math
import re
from typing import Tuple

from .input_utils import INPUT_HANDLERS, log, HexlabArgumentParser
from .constants import (
    COLOR_NAMES, MAX_DEC, FORMAT_ALIASES,
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
    h = hex_code.replace(" ", "").upper()
    if len(h) == 3:
        h = ''.join([c * 2 for c in h])
    if len(h) > 6: h = h[:6]
    if len(h) < 6: h = h.zfill(6)
    
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

def rgb_to_hex(r, g, b):
    return f"{int(round(r)):02X}{int(round(g)):02X}{int(round(b)):02X}"

def fmt_hex_for_output(hex_str: str) -> str:
    return f"#{hex_str.upper()}"

def _clamp01(v: float) -> float:
    if v != v:
        return 0.0
    return max(0.0, min(1.0, v))

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

def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    r_f, g_f, b_f = r / 255.0, g / 255.0, b / 255.0
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

    l_lin = l_ ** 3
    m_lin = m_ ** 3
    s_lin = s_ ** 3

    r_lin = 4.0767416621 * l_lin - 3.3077115913 * m_lin + 0.2309699292 * s_lin
    g_lin = -1.2684380046 * l_lin + 2.6097574011 * m_lin - 0.3413193965 * s_lin
    b_lin = -0.0041960863 * l_lin - 0.7034186147 * m_lin + 1.7076147010 * s_lin

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

def rgb_to_hwb(r: int, g: int, b: int) -> Tuple[float, float, float]:
    h, s, v = rgb_to_hsv(r, g, b)
    w = (1 - s) * v
    b_hwb = 1 - v
    return h, w, b_hwb

def hwb_to_rgb(h: float, w: float, b: float) -> Tuple[float, float, float]:
    w = _clamp01(w)
    b = _clamp01(b)
    if w + b > 1.0:
        total = w + b
        if total > 0.0:
            w = w / total
            b = b / total
    v = 1.0 - b
    if v <= 0.0:
        return 0.0, 0.0, 0.0
    s = 1.0 - (w / v)
    s = _clamp01(s)
    return hsv_to_rgb(h, s, v)

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

def _normalize_value_string(s: str) -> str:
    if not s: return ""
    s = s.strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'`":
        s = s[1:-1].strip()
    s = s.replace('°', ' ')
    s = s.replace('–', '-')
    s = re.sub(r'deg', ' ', s, flags=re.IGNORECASE)

    if s.lower().startswith(('rgb(', 'hsl(', 'hsv(', 'hwb(', 'cmyk(', 'xyz(', 'lab(', 'lch(', 'oklab(', 'luv(', 'oklch(')):
        s = re.sub(r'^[a-zA-Z]+\s*\(', '', s, flags=re.IGNORECASE)
        s = s.rstrip(')')

    s = s.replace(',', ' ')
    s = s.replace('/', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def _parse_numerical_string(s: str):
    s = _normalize_value_string(s)
    pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    try:
        matches = list(re.finditer(pattern, s))
        if not matches:
            raise ValueError

        tokens = []
        i = 0
        while i < len(matches):
            m = matches[i]
            token = m.group()

            if (
                i + 1 < len(matches)
                and '.' not in token
            ):
                m_next = matches[i + 1]
                next_token = m_next.group()
                if next_token.startswith('.'):
                    between = s[m.end():m_next.start()]
                    if between and re.fullmatch(r'[A-Za-z]+', between):
                        token = token + next_token
                        i += 1

            val = float(token)
            if not math.isfinite(val):
                raise ValueError
            tokens.append(val)
            i += 1

        return tokens
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
        if 0.0 < val < 1.0:
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

def parse_oklch_string(s: str) -> Tuple[float, float, float]:
    return _parse_3_floats(s, "oklch")

def parse_luv_string(s: str) -> Tuple[float, float, float]:
    return _parse_3_floats(s, "luv")

def _parse_value_to_rgb(clean_val: str, from_fmt: str) -> Tuple[int, int, int]:
    r_f, g_f, b_f = 0.0, 0.0, 0.0
    if from_fmt == 'hex':
        hex_val = clean_val.replace("#", "").replace(" ", "").upper()
        if len(hex_val) == 3: hex_val = "".join([c*2 for c in hex_val])
        if len(hex_val) < 6: hex_val = hex_val.zfill(6)
        if len(hex_val) > 6: hex_val = hex_val[:6]
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
    elif from_fmt == 'oklch':
        l, c, h = parse_oklch_string(clean_val)
        r_f, g_f, b_f = oklch_to_rgb(l, c, h)
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
        sanitized_name = re.sub(r'[^a-z]', '', clean_val.lower())
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
        output_value = f"hsl({h:.1f}°, {s * 100:.1f}%, {l * 100:.1f}%)"
    elif to_fmt == 'hsv':
        h, s, v = rgb_to_hsv(r, g, b)
        output_value = f"hsv({h:.1f}°, {s * 100:.1f}%, {v * 100:.1f}%)"
    elif to_fmt == 'hwb':
        h, w, b_hwb = rgb_to_hwb(r, g, b)
        output_value = f"hwb({h:.1f}°, {w * 100:.1f}%, {b_hwb * 100:.1f}%)"
    elif to_fmt == 'cmyk':
        c, m, y, k = rgb_to_cmyk(r, g, b)
        output_value = f"cmyk({c * 100:.1f}%, {m * 100:.1f}%, {y * 100:.1f}%, {k * 100:.1f}%)"
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
    elif to_fmt == 'oklch':
        l, c, h = rgb_to_oklch(r, g, b)
        output_value = f"oklch({l:.4f}, {c:.4f}, {h:.4f}°)"
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
        log('error', f"invalid format specified: {e}")
        log('info', "use 'hexlab convert -h' to see all formats")
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


def get_convert_parser() -> argparse.ArgumentParser:
    parser = HexlabArgumentParser(
        prog="hexlab convert",
        description="hexlab convert: convert a color value from one format to another",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False
    )
    formats_list = "hex rgb hsl hsv hwb cmyk xyz lab lch oklab oklch luv index name"
    parser.add_argument(
        '-h', '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='show this help message and exit'
    )
    parser.add_argument(
        "-f", "--from-format",
        required=True,
        type=INPUT_HANDLERS["from_format"],
        help="the format to convert from\n"
             f"all formats: {formats_list}\n"
             f"use quotes for better UX"
    )
    parser.add_argument(
        "-t", "--to-format",
        required=True,
        type=INPUT_HANDLERS["to_format"],
        help="the format to convert to\n"
             f"all formats: {formats_list}\n"
             f"use quotes for better UX"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-v", "--value",
        type=str, 
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
             '  -v "luv(0, 0, 0)"\n'
             '  -v "oklab(0, 0, 0)"\n'
             '  -v "oklch(0, 0, 0°)"'
    )
    input_group.add_argument(
        "-rv", "--random-value",
        action="store_true",
        help="generate a random value for the --from-format"
    )
    parser.add_argument(
        "-s", "--seed",
        type=INPUT_HANDLERS["seed"],
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
    parser = get_convert_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    handle_convert_command(args)


if __name__ == "__main__":
    main()
