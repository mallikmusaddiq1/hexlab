import functools
import math
from typing import Tuple

from ..constants.constants import (
    # Floating-point precision and division-by-zero safety
    EPS,

    # Standard Scaling Constants
    RGB_MAX,
    HUE_MAX,
    HUE_SECTOR,

    # sRGB Transfer Function Constants
    SRGB_SLOPE,
    SRGB_OFFSET,
    SRGB_DIVISOR,
    SRGB_GAMMA,
    LINEAR_TO_SRGB_TH,
    SRGB_TO_LINEAR_TH,

    # XYZ D65 Reference White
    D65_X,
    D65_Y,
    D65_Z,

    # sRGB to XYZ Matrix
    M_SRGB_XYZ_X,
    M_SRGB_XYZ_Y,
    M_SRGB_XYZ_Z,

    # XYZ to sRGB Matrix
    M_XYZ_SRGB_R,
    M_XYZ_SRGB_G,
    M_XYZ_SRGB_B,

    # CIELAB Constants
    LAB_E,
    LAB_K,
    LAB_OFFSET,
    LAB_L_MULT,
    LAB_L_SUB,
    LAB_A_MULT,
    LAB_B_MULT,
    LAB_POW,
    LAB_INV_THR,

    # OKLab Matrices
    M1_OKLAB,
    M2_OKLAB,
    M2_INV_OKLAB,
    M1_INV_OKLAB,

    # CIELUV Constants
    LUV_U_V_MULT,
    LUV_KAPPA,
    LUV_U_NUM,
    LUV_V_NUM,
    LUV_DENOM_Y,
    LUV_DENOM_Z,
    LUV_Z_CONST,
    LUV_Z_U_MULT,
    LUV_Z_V_MULT,
    LUV_L_THR
)
from ..utils.clamping import _clamp01
from ..utils.input_handler import normalize_hex


def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    """Convert hex string to RGB tuple."""
    h = normalize_hex(hex_code)
    if not h:
        return (0, 0, 0)
    try:
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return (0, 0, 0)


def rgb_to_hex(r: float, g: float, b: float) -> str:
    """Convert RGB components to hex string."""
    r_clamped = max(0, min(int(RGB_MAX), int(round(r))))
    g_clamped = max(0, min(int(RGB_MAX), int(round(g))))
    b_clamped = max(0, min(int(RGB_MAX), int(round(b))))
    return f"{r_clamped:02X}{g_clamped:02X}{b_clamped:02X}"


def rgb_to_hsl(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB to HSL."""
    r_f, g_f, b_f = r / RGB_MAX, g / RGB_MAX, b / RGB_MAX
    cmax = max(r_f, g_f, b_f)
    cmin = min(r_f, g_f, b_f)
    delta = cmax - cmin
    L = (cmax + cmin) / 2
    if delta == 0:
        h = 0.0
        s = 0.0
    else:
        denom = 1 - abs(2 * L - 1)
        s = 0.0 if abs(denom) < EPS else delta / denom
        if cmax == r_f:
            h = HUE_SECTOR * (((g_f - b_f) / delta) % 6)
        elif cmax == g_f:
            h = HUE_SECTOR * ((b_f - r_f) / delta + 2)
        else:
            h = HUE_SECTOR * ((r_f - g_f) / delta + 4)
        h = (h + HUE_MAX) % HUE_MAX
    return (h, s, L)


def hsl_to_rgb(h: float, s: float, L: float) -> Tuple[float, float, float]:
    """Convert HSL to RGB."""
    h = h % HUE_MAX
    if s == 0:
        r = g = b = L
    else:
        c = (1 - abs(2 * L - 1)) * s
        x = c * (1 - abs(((h / HUE_SECTOR) % 2) - 1))
        m = L - c / 2
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
    return _clamp01(r) * RGB_MAX, _clamp01(g) * RGB_MAX, _clamp01(b) * RGB_MAX


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB to HSV."""
    r_f, g_f, b_f = r / RGB_MAX, g / RGB_MAX, b / RGB_MAX
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
            h = HUE_SECTOR * (((g_f - b_f) / delta) % 6)
        elif cmax == g_f:
            h = HUE_SECTOR * ((b_f - r_f) / delta + 2)
        else:
            h = HUE_SECTOR * ((r_f - g_f) / delta + 4)
        h = (h + HUE_MAX) % HUE_MAX
    return (h, s, v)


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    """Convert HSV to RGB."""
    h = h % HUE_MAX
    c = v * s
    x = c * (1 - abs(((h / HUE_SECTOR) % 2) - 1))
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
    return _clamp01(r) * RGB_MAX, _clamp01(g) * RGB_MAX, _clamp01(b) * RGB_MAX


def rgb_to_cmyk(r: int, g: int, b: int) -> Tuple[float, float, float, float]:
    """Convert RGB to CMYK."""
    if r == 0 and g == 0 and b == 0:
        return 0.0, 0.0, 0.0, 1.0
    r_norm, g_norm, b_norm = r / RGB_MAX, g / RGB_MAX, b / RGB_MAX
    k = 1.0 - max(r_norm, g_norm, b_norm)
    if k >= 1.0:
        return 0.0, 0.0, 0.0, 1.0
    denom = 1.0 - k
    c = (1.0 - r_norm - k) / denom
    m = (1.0 - g_norm - k) / denom
    y = (1.0 - b_norm - k) / denom
    return (c, m, y, k)


def cmyk_to_rgb(c: float, m: float, y: float, k: float) -> Tuple[float, float, float]:
    """Convert CMYK to RGB."""
    r = RGB_MAX * (1 - _clamp01(c)) * (1 - _clamp01(k))
    g = RGB_MAX * (1 - _clamp01(m)) * (1 - _clamp01(k))
    b = RGB_MAX * (1 - _clamp01(y)) * (1 - _clamp01(k))
    return r, g, b


def _srgb_to_linear(c: int) -> float:
    """Linearize sRGB component."""
    c_norm = c / RGB_MAX
    c_norm = _clamp01(c_norm)
    return (
        c_norm / SRGB_SLOPE
        if c_norm <= SRGB_TO_LINEAR_TH
        else ((c_norm + SRGB_OFFSET) / SRGB_DIVISOR) ** SRGB_GAMMA
    )


def rgb_to_xyz(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB to CIE XYZ."""
    r_lin = _srgb_to_linear(r)
    g_lin = _srgb_to_linear(g)
    b_lin = _srgb_to_linear(b)
    x = r_lin * M_SRGB_XYZ_X[0] + g_lin * M_SRGB_XYZ_X[1] + b_lin * M_SRGB_XYZ_X[2]
    y = r_lin * M_SRGB_XYZ_Y[0] + g_lin * M_SRGB_XYZ_Y[1] + b_lin * M_SRGB_XYZ_Y[2]
    z = r_lin * M_SRGB_XYZ_Z[0] + g_lin * M_SRGB_XYZ_Z[1] + b_lin * M_SRGB_XYZ_Z[2]
    return x * 100.0, y * 100.0, z * 100.0


def _xyz_f(t: float) -> float:
    """Helper function for XYZ to LAB."""
    return t ** LAB_POW if t > LAB_E else (LAB_K * t) + LAB_OFFSET


def _xyz_f_inv(t: float) -> float:
    """Helper function for LAB to XYZ."""
    return t ** 3 if t > LAB_INV_THR else (t - LAB_OFFSET) / LAB_K


def xyz_to_lab(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert XYZ to CIE LAB."""
    x_r = _xyz_f(x / D65_X)
    y_r = _xyz_f(y / D65_Y)
    z_r = _xyz_f(z / D65_Z)
    L = (LAB_L_MULT * y_r) - LAB_L_SUB
    a = LAB_A_MULT * (x_r - y_r)
    b = LAB_B_MULT * (y_r - z_r)
    return L, a, b


def lab_to_xyz(L: float, a: float, b: float) -> Tuple[float, float, float]:
    """Convert LAB to CIE XYZ."""
    y_r = (L + LAB_L_SUB) / LAB_L_MULT
    x_r = a / LAB_A_MULT + y_r
    z_r = y_r - b / LAB_B_MULT
    x = _xyz_f_inv(x_r) * D65_X
    y = _xyz_f_inv(y_r) * D65_Y
    z = _xyz_f_inv(z_r) * D65_Z
    return x, y, z


def _linear_to_srgb(l_val: float) -> float:
    """Apply sRGB gamma to linear component."""
    l_val = max(l_val, 0.0)
    return (
        SRGB_SLOPE * l_val
        if l_val <= LINEAR_TO_SRGB_TH
        else SRGB_DIVISOR * (l_val ** (1 / SRGB_GAMMA)) - SRGB_OFFSET
    )


def xyz_to_rgb(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert CIE XYZ to RGB."""
    x_n, y_n, z_n = x / 100.0, y / 100.0, z / 100.0
    r_lin = x_n * M_XYZ_SRGB_R[0] + y_n * M_XYZ_SRGB_R[1] + z_n * M_XYZ_SRGB_R[2]
    g_lin = x_n * M_XYZ_SRGB_G[0] + y_n * M_XYZ_SRGB_G[1] + z_n * M_XYZ_SRGB_G[2]
    b_lin = x_n * M_XYZ_SRGB_B[0] + y_n * M_XYZ_SRGB_B[1] + z_n * M_XYZ_SRGB_B[2]
    r = _linear_to_srgb(r_lin)
    g = _linear_to_srgb(g_lin)
    b = _linear_to_srgb(b_lin)
    return _clamp01(r) * RGB_MAX, _clamp01(g) * RGB_MAX, _clamp01(b) * RGB_MAX


def rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Direct RGB to LAB conversion."""
    x, y, z = rgb_to_xyz(r, g, b)
    return xyz_to_lab(x, y, z)


def lab_to_lch(L: float, a: float, b: float) -> Tuple[float, float, float]:
    """Convert LAB to LCH."""
    c = math.hypot(a, b)
    h = math.degrees(math.atan2(b, a)) % HUE_MAX
    return L, c, h


def rgb_to_lch(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Direct RGB to LCH conversion."""
    L, a_, b_ = rgb_to_lab(r, g, b)
    return lab_to_lch(L, a_, b_)


def lch_to_rgb(L: float, c: float, h: float) -> Tuple[float, float, float]:
    """Direct LCH to RGB conversion."""
    L_val, a, b_ = lch_to_lab(L, c, h)
    return lab_to_rgb(L_val, a, b_)


def lch_to_lab(L: float, c: float, h: float) -> Tuple[float, float, float]:
    """Convert LCH to LAB."""
    a = c * math.cos(math.radians(h))
    b = c * math.sin(math.radians(h))
    return L, a, b


def lab_to_rgb(L: float, a: float, b: float) -> Tuple[float, float, float]:
    """Direct LAB to RGB conversion."""
    x, y, z = lab_to_xyz(L, a, b)
    return xyz_to_rgb(x, y, z)


def rgb_to_oklab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB to OKLab."""
    r_lin = _srgb_to_linear(r)
    g_lin = _srgb_to_linear(g)
    b_lin = _srgb_to_linear(b)

    l_val = M1_OKLAB[0][0] * r_lin + M1_OKLAB[0][1] * g_lin + M1_OKLAB[0][2] * b_lin
    m = M1_OKLAB[1][0] * r_lin + M1_OKLAB[1][1] * g_lin + M1_OKLAB[1][2] * b_lin
    s = M1_OKLAB[2][0] * r_lin + M1_OKLAB[2][1] * g_lin + M1_OKLAB[2][2] * b_lin

    l_ = (abs(l_val)) ** LAB_POW if l_val >= 0 else -((abs(l_val)) ** LAB_POW)
    m_ = (abs(m)) ** LAB_POW if m >= 0 else -((abs(m)) ** LAB_POW)
    s_ = (abs(s)) ** LAB_POW if s >= 0 else -((abs(s)) ** LAB_POW)

    ok_l = M2_OKLAB[0][0] * l_ + M2_OKLAB[0][1] * m_ + M2_OKLAB[0][2] * s_
    ok_a = M2_OKLAB[1][0] * l_ + M2_OKLAB[1][1] * m_ + M2_OKLAB[1][2] * s_
    ok_b = M2_OKLAB[2][0] * l_ + M2_OKLAB[2][1] * m_ + M2_OKLAB[2][2] * s_

    return ok_l, ok_a, ok_b


def oklab_to_rgb(L: float, a: float, b: float) -> Tuple[float, float, float]:
    """Convert OKLab to RGB."""
    l_ = L * M2_INV_OKLAB[0][0] + a * M2_INV_OKLAB[0][1] + b * M2_INV_OKLAB[0][2]
    m_ = L * M2_INV_OKLAB[1][0] + a * M2_INV_OKLAB[1][1] + b * M2_INV_OKLAB[1][2]
    s_ = L * M2_INV_OKLAB[2][0] + a * M2_INV_OKLAB[2][1] + b * M2_INV_OKLAB[2][2]

    l_lin = l_ ** 3
    m_lin = m_ ** 3
    s_lin = s_ ** 3

    r_lin = M1_INV_OKLAB[0][0] * l_lin + M1_INV_OKLAB[0][1] * m_lin + M1_INV_OKLAB[0][2] * s_lin
    g_lin = M1_INV_OKLAB[1][0] * l_lin + M1_INV_OKLAB[1][1] * m_lin + M1_INV_OKLAB[1][2] * s_lin
    b_lin = M1_INV_OKLAB[2][0] * l_lin + M1_INV_OKLAB[2][1] * m_lin + M1_INV_OKLAB[2][2] * s_lin

    r = _linear_to_srgb(r_lin)
    g = _linear_to_srgb(g_lin)
    b = _linear_to_srgb(b_lin)

    return _clamp01(r) * RGB_MAX, _clamp01(g) * RGB_MAX, _clamp01(b) * RGB_MAX


def oklab_to_oklch(L: float, a: float, b: float) -> Tuple[float, float, float]:
    """Convert OKLab to OKLCH."""
    c = math.hypot(a, b)
    h = math.degrees(math.atan2(b, a)) % HUE_MAX
    return L, c, h


def oklch_to_oklab(L: float, c: float, h: float) -> Tuple[float, float, float]:
    """Convert OKLCH to OKLab."""
    a = c * math.cos(math.radians(h))
    b = c * math.sin(math.radians(h))
    return L, a, b


def rgb_to_oklch(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Direct RGB to OKLCH conversion."""
    L, a, b_ok = rgb_to_oklab(r, g, b)
    return oklab_to_oklch(L, a, b_ok)


def oklch_to_rgb(L: float, c: float, h: float) -> Tuple[float, float, float]:
    """Direct OKLCH to RGB conversion."""
    L_val, a, b_ok = oklch_to_oklab(L, c, h)
    return oklab_to_rgb(L_val, a, b_ok)


def rgb_to_hwb(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB to HWB."""
    h, s, v = rgb_to_hsv(r, g, b)
    w = (1 - s) * v
    b_hwb = 1 - v
    return h, w, b_hwb


def hwb_to_rgb(h: float, w: float, b: float) -> Tuple[float, float, float]:
    """Convert HWB to RGB."""
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
    """Convert RGB to CIE LUV."""
    X, Y, Z = rgb_to_xyz(r, g, b)
    denom = X + 15 * Y + 3 * Z
    if denom == 0:
        u_prime = 0.0
        v_prime = 0.0
    else:
        u_prime = (LUV_U_NUM * X) / denom
        v_prime = (LUV_V_NUM * Y) / denom

    denom_n = D65_X + 15 * D65_Y + 3 * D65_Z
    u_prime_n = (LUV_U_NUM * D65_X) / denom_n
    v_prime_n = (LUV_V_NUM * D65_Y) / denom_n

    y_r = Y / D65_Y
    if y_r > LAB_E:
        L = (LAB_L_MULT * (y_r ** LAB_POW)) - LAB_L_SUB
    else:
        L = LUV_KAPPA * y_r

    if L == 0:
        u = 0.0
        v = 0.0
    else:
        u = LUV_U_V_MULT * L * (u_prime - u_prime_n)
        v = LUV_U_V_MULT * L * (v_prime - v_prime_n)

    return L, u, v


def luv_to_rgb(L: float, u: float, v: float) -> Tuple[float, float, float]:
    """Convert CIE LUV to RGB."""
    denom_n = D65_X + 15 * D65_Y + 3 * D65_Z
    u_prime_n = (LUV_U_NUM * D65_X) / denom_n
    v_prime_n = (LUV_V_NUM * D65_Y) / denom_n

    if L == 0:
        return xyz_to_rgb(0.0, 0.0, 0.0)

    u_prime = (u / (LUV_U_V_MULT * L)) + u_prime_n
    v_prime = (v / (LUV_U_V_MULT * L)) + v_prime_n

    if L > LUV_L_THR:
        Y = D65_Y * (((L + LAB_L_SUB) / LAB_L_MULT) ** 3)
    else:
        Y = D65_Y * (L / LUV_KAPPA)

    if v_prime == 0:
        X = 0.0
        Z = 0.0
    else:
        X = Y * (LUV_V_NUM * u_prime) / (LUV_U_NUM * v_prime)
        Z = Y * (LUV_Z_CONST - LUV_Z_U_MULT * u_prime - LUV_Z_V_MULT * v_prime) / (LUV_U_NUM * v_prime)

    return xyz_to_rgb(X, Y, Z)


# ==========================================
# Direct Conversion Wrappers
# ==========================================


def hex_to_hsl(hex_code: str) -> Tuple[float, float, float]:
    """Direct Hex to HSL."""
    return rgb_to_hsl(*hex_to_rgb(hex_code))


def hsl_to_hex(h: float, s: float, L: float) -> str:
    """Direct HSL to Hex."""
    return rgb_to_hex(*hsl_to_rgb(h, s, L))


def hex_to_hsv(hex_code: str) -> Tuple[float, float, float]:
    """Direct Hex to HSV."""
    return rgb_to_hsv(*hex_to_rgb(hex_code))


def hsv_to_hex(h: float, s: float, v: float) -> str:
    """Direct HSV to Hex."""
    return rgb_to_hex(*hsv_to_rgb(h, s, v))


def hex_to_hwb(hex_code: str) -> Tuple[float, float, float]:
    """Direct Hex to HWB."""
    return rgb_to_hwb(*hex_to_rgb(hex_code))


def hwb_to_hex(h: float, w: float, b: float) -> str:
    """Direct HWB to Hex."""
    return rgb_to_hex(*hwb_to_rgb(h, w, b))


def hex_to_cmyk(hex_code: str) -> Tuple[float, float, float, float]:
    """Direct Hex to CMYK."""
    return rgb_to_cmyk(*hex_to_rgb(hex_code))


def cmyk_to_hex(c: float, m: float, y: float, k: float) -> str:
    """Direct CMYK to Hex."""
    return rgb_to_hex(*cmyk_to_rgb(c, m, y, k))


def hex_to_xyz(hex_code: str) -> Tuple[float, float, float]:
    """Direct Hex to XYZ."""
    return rgb_to_xyz(*hex_to_rgb(hex_code))


def xyz_to_hex(x: float, y: float, z: float) -> str:
    """Direct XYZ to Hex."""
    return rgb_to_hex(*xyz_to_rgb(x, y, z))


def hex_to_lab(hex_code: str) -> Tuple[float, float, float]:
    """Direct Hex to LAB."""
    return rgb_to_lab(*hex_to_rgb(hex_code))


def lab_to_hex(L: float, a: float, b: float) -> str:
    """Direct LAB to Hex."""
    return rgb_to_hex(*lab_to_rgb(L, a, b))


def hex_to_lch(hex_code: str) -> Tuple[float, float, float]:
    """Direct Hex to LCH."""
    return rgb_to_lch(*hex_to_rgb(hex_code))


def lch_to_hex(L: float, c: float, h: float) -> str:
    """Direct LCH to Hex."""
    return rgb_to_hex(*lch_to_rgb(L, c, h))


def hex_to_oklab(hex_code: str) -> Tuple[float, float, float]:
    """Direct Hex to OKLab."""
    return rgb_to_oklab(*hex_to_rgb(hex_code))


def oklab_to_hex(L: float, a: float, b: float) -> str:
    """Direct OKLab to Hex."""
    return rgb_to_hex(*oklab_to_rgb(L, a, b))


def hex_to_oklch(hex_code: str) -> Tuple[float, float, float]:
    """Direct Hex to OKLCH."""
    return rgb_to_oklch(*hex_to_rgb(hex_code))


def oklch_to_hex(L: float, c: float, h: float) -> str:
    """Direct OKLCH to Hex."""
    return rgb_to_hex(*oklch_to_rgb(L, c, h))


def hex_to_luv(hex_code: str) -> Tuple[float, float, float]:
    """Direct Hex to LUV."""
    return rgb_to_luv(*hex_to_rgb(hex_code))


def luv_to_hex(L: float, u: float, v: float) -> str:
    """Direct LUV to Hex."""
    return rgb_to_hex(*luv_to_rgb(L, u, v))


# Apply LRU caching to all functions in this module
for _name, _obj in list(globals().items()):
    if callable(_obj) and _obj.__module__ == __name__:
        globals()[_name] = functools.lru_cache(maxsize=1024)(_obj)