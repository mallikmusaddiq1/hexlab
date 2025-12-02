# File: conversions.py
import math
from typing import Tuple

from ..constants.constants import EPS, LINEAR_TO_SRGB_TH, SRGB_TO_LINEAR_TH
from ..utils.clamping import _clamp01
from ..utils.input_handler import normalize_hex


def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    h = normalize_hex(hex_code)
    if not h:
        return (0, 0, 0)
    try:
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return (0, 0, 0)


def rgb_to_hex(r: float, g: float, b: float) -> str:
    r_clamped = max(0, min(255, int(round(r))))
    g_clamped = max(0, min(255, int(round(g))))
    b_clamped = max(0, min(255, int(round(b))))
    return f"{r_clamped:02X}{g_clamped:02X}{b_clamped:02X}"


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
        x = c * (1 - abs(((h / 60.0) % 2) - 1))
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
    x = c * (1 - abs(((h / 60.0) % 2) - 1))
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


def rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    x, y, z = rgb_to_xyz(r, g, b)
    return xyz_to_lab(x, y, z)


def lab_to_lch(l: float, a: float, b: float) -> Tuple[float, float, float]:
    c = math.hypot(a, b)
    h = math.degrees(math.atan2(b, a)) % 360
    return l, c, h


def rgb_to_lch(r: int, g: int, b: int) -> Tuple[float, float, float]:
    L, a_, b_ = rgb_to_lab(r, g, b)
    return lab_to_lch(L, a_, b_)


def lch_to_rgb(l: float, c: float, h: float) -> Tuple[float, float, float]:
    L, a, b_ = lch_to_lab(l, c, h)
    return lab_to_rgb(L, a, b_)


def lch_to_lab(l: float, c: float, h: float) -> Tuple[float, float, float]:
    a = c * math.cos(math.radians(h))
    b = c * math.sin(math.radians(h))
    return l, a, b


def lab_to_rgb(l: float, a: float, b: float) -> Tuple[float, float, float]:
    x, y, z = lab_to_xyz(l, a, b)
    return xyz_to_rgb(x, y, z)


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

# ==========================================
# Direct Conversion Wrappers
# ==========================================

def hex_to_hsl(hex_code: str) -> Tuple[float, float, float]:
    return rgb_to_hsl(*hex_to_rgb(hex_code))


def hsl_to_hex(h: float, s: float, l: float) -> str:
    return rgb_to_hex(*hsl_to_rgb(h, s, l))


def hex_to_hsv(hex_code: str) -> Tuple[float, float, float]:
    return rgb_to_hsv(*hex_to_rgb(hex_code))


def hsv_to_hex(h: float, s: float, v: float) -> str:
    return rgb_to_hex(*hsv_to_rgb(h, s, v))


def hex_to_hwb(hex_code: str) -> Tuple[float, float, float]:
    return rgb_to_hwb(*hex_to_rgb(hex_code))


def hwb_to_hex(h: float, w: float, b: float) -> str:
    return rgb_to_hex(*hwb_to_rgb(h, w, b))


def hex_to_cmyk(hex_code: str) -> Tuple[float, float, float, float]:
    return rgb_to_cmyk(*hex_to_rgb(hex_code))


def cmyk_to_hex(c: float, m: float, y: float, k: float) -> str:
    return rgb_to_hex(*cmyk_to_rgb(c, m, y, k))


def hex_to_xyz(hex_code: str) -> Tuple[float, float, float]:
    return rgb_to_xyz(*hex_to_rgb(hex_code))


def xyz_to_hex(x: float, y: float, z: float) -> str:
    return rgb_to_hex(*xyz_to_rgb(x, y, z))


def hex_to_lab(hex_code: str) -> Tuple[float, float, float]:
    return rgb_to_lab(*hex_to_rgb(hex_code))


def lab_to_hex(l: float, a: float, b: float) -> str:
    return rgb_to_hex(*lab_to_rgb(l, a, b))


def hex_to_lch(hex_code: str) -> Tuple[float, float, float]:
    return rgb_to_lch(*hex_to_rgb(hex_code))


def lch_to_hex(l: float, c: float, h: float) -> str:
    return rgb_to_hex(*lch_to_rgb(l, c, h))


def hex_to_oklab(hex_code: str) -> Tuple[float, float, float]:
    return rgb_to_oklab(*hex_to_rgb(hex_code))


def oklab_to_hex(l: float, a: float, b: float) -> str:
    return rgb_to_hex(*oklab_to_rgb(l, a, b))


def hex_to_oklch(hex_code: str) -> Tuple[float, float, float]:
    return rgb_to_oklch(*hex_to_rgb(hex_code))


def oklch_to_hex(l: float, c: float, h: float) -> str:
    return rgb_to_hex(*oklch_to_rgb(l, c, h))


def hex_to_luv(hex_code: str) -> Tuple[float, float, float]:
    return rgb_to_luv(*hex_to_rgb(hex_code))


def luv_to_hex(l: float, u: float, v: float) -> str:
    return rgb_to_hex(*luv_to_rgb(l, u, v))
