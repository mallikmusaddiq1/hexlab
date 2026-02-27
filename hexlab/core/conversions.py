#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/core/conversions.py

import functools
import math
from typing import Tuple

from . import config as c
from hexlab.shared.clamping import _clamp01
from hexlab.shared.sanitizer import normalize_hex


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
    r_clamped = max(0, min(int(c.RGB_MAX), int(round(r))))
    g_clamped = max(0, min(int(c.RGB_MAX), int(round(g))))
    b_clamped = max(0, min(int(c.RGB_MAX), int(round(b))))
    return f"{r_clamped:02X}{g_clamped:02X}{b_clamped:02X}"


def rgb_to_hsl(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB to HSL."""
    r_f, g_f, b_f = r / c.RGB_MAX, g / c.RGB_MAX, b / c.RGB_MAX
    cmax = max(r_f, g_f, b_f)
    cmin = min(r_f, g_f, b_f)
    delta = cmax - cmin
    L = (cmax + cmin) / c.DIV_2
    if delta == 0:
        h = 0.0
        s = 0.0
    else:
        denom = c.UNIT - abs(c.DIV_2 * L - c.UNIT)
        s = 0.0 if abs(denom) < c.EPS else delta / denom
        if cmax == r_f:
            h = c.HUE_SECTOR * (((g_f - b_f) / delta) % c.HSL_HUE_MOD)
        elif cmax == g_f:
            h = c.HUE_SECTOR * ((b_f - r_f) / delta + c.DIV_2)
        else:
            h = c.HUE_SECTOR * ((r_f - g_f) / delta + c.LUV_U_NUM) # 4.0
        h = (h + c.HUE_MAX) % c.HUE_MAX
    return (h, s, L)


def hsl_to_rgb(h: float, s: float, L: float) -> Tuple[float, float, float]:
    """Convert HSL to RGB."""
    h = h % c.HUE_MAX
    if s == 0:
        r = g = b = L
    else:
        chroma = (c.UNIT - abs(c.DIV_2 * L - c.UNIT)) * s
        x = chroma * (c.UNIT - abs(((h / c.HUE_SECTOR) % c.DIV_2) - c.UNIT))
        m = L - chroma / c.DIV_2
        if 0 <= h < 60:
            r_p, g_p, b_p = chroma, x, 0
        elif 60 <= h < 120:
            r_p, g_p, b_p = x, chroma, 0
        elif 120 <= h < 180:
            r_p, g_p, b_p = 0, chroma, x
        elif 180 <= h < 240:
            r_p, g_p, b_p = 0, x, chroma
        elif 240 <= h < 300:
            r_p, g_p, b_p = x, 0, chroma
        else:
            r_p, g_p, b_p = chroma, 0, x
        r, g, b = (r_p + m), (g_p + m), (b_p + m)
    return _clamp01(r) * c.RGB_MAX, _clamp01(g) * c.RGB_MAX, _clamp01(b) * c.RGB_MAX


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB to HSV."""
    r_f, g_f, b_f = r / c.RGB_MAX, g / c.RGB_MAX, b / c.RGB_MAX
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
            h = c.HUE_SECTOR * (((g_f - b_f) / delta) % c.HSL_HUE_MOD)
        elif cmax == g_f:
            h = c.HUE_SECTOR * ((b_f - r_f) / delta + c.DIV_2)
        else:
            h = c.HUE_SECTOR * ((r_f - g_f) / delta + c.LUV_U_NUM)
        h = (h + c.HUE_MAX) % c.HUE_MAX
    return (h, s, v)


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    """Convert HSV to RGB."""
    h = h % c.HUE_MAX
    chroma = v * s
    x = chroma * (c.UNIT - abs(((h / c.HUE_SECTOR) % c.DIV_2) - c.UNIT))
    m = v - chroma
    if 0 <= h < 60:
        r_p, g_p, b_p = chroma, x, 0
    elif 60 <= h < 120:
        r_p, g_p, b_p = x, chroma, 0
    elif 120 <= h < 180:
        r_p, g_p, b_p = 0, chroma, x
    elif 180 <= h < 240:
        r_p, g_p, b_p = 0, x, chroma
    elif 240 <= h < 300:
        r_p, g_p, b_p = x, 0, chroma
    else:
        r_p, g_p, b_p = chroma, 0, x
    r, g, b = (r_p + m), (g_p + m), (b_p + m)
    return _clamp01(r) * c.RGB_MAX, _clamp01(g) * c.RGB_MAX, _clamp01(b) * c.RGB_MAX


def rgb_to_cmyk(r: int, g: int, b: int) -> Tuple[float, float, float, float]:
    """Convert RGB to CMYK."""
    if r == 0 and g == 0 and b == 0:
        return 0.0, 0.0, 0.0, c.UNIT
    r_norm, g_norm, b_norm = r / c.RGB_MAX, g / c.RGB_MAX, b / c.RGB_MAX
    k = c.UNIT - max(r_norm, g_norm, b_norm)
    if k >= c.UNIT:
        return 0.0, 0.0, 0.0, c.UNIT
    denom = c.UNIT - k
    cy = (c.UNIT - r_norm - k) / denom
    m = (c.UNIT - g_norm - k) / denom
    y = (c.UNIT - b_norm - k) / denom
    return (cy, m, y, k)


def cmyk_to_rgb(cy: float, m: float, y: float, k: float) -> Tuple[float, float, float]:
    """Convert CMYK to RGB."""
    r = c.RGB_MAX * (c.UNIT - _clamp01(cy)) * (c.UNIT - _clamp01(k))
    g = c.RGB_MAX * (c.UNIT - _clamp01(m)) * (c.UNIT - _clamp01(k))
    b = c.RGB_MAX * (c.UNIT - _clamp01(y)) * (c.UNIT - _clamp01(k))
    return r, g, b


def _srgb_to_linear(color_comp: int) -> float:
    """Linearize sRGB component."""
    c_norm = _clamp01(color_comp / c.RGB_MAX)
    if c_norm <= c.SRGB_TO_LINEAR_TH:
        return c_norm / c.SRGB_SLOPE
    return ((c_norm + c.SRGB_OFFSET) / c.SRGB_DIVISOR) ** c.SRGB_GAMMA


def rgb_to_xyz(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB to CIE XYZ."""
    r_lin = _srgb_to_linear(r)
    g_lin = _srgb_to_linear(g)
    b_lin = _srgb_to_linear(b)
    x = r_lin * c.M_SRGB_XYZ_X[0] + g_lin * c.M_SRGB_XYZ_X[1] + b_lin * c.M_SRGB_XYZ_X[2]
    y = r_lin * c.M_SRGB_XYZ_Y[0] + g_lin * c.M_SRGB_XYZ_Y[1] + b_lin * c.M_SRGB_XYZ_Y[2]
    z = r_lin * c.M_SRGB_XYZ_Z[0] + g_lin * c.M_SRGB_XYZ_Z[1] + b_lin * c.M_SRGB_XYZ_Z[2]
    return x * c.XYZ_SCALING, y * c.XYZ_SCALING, z * c.XYZ_SCALING


def _xyz_f(t: float) -> float:
    """Helper function for XYZ to LAB."""
    return t**c.LAB_POW if t > c.LAB_E else (c.LAB_K * t) + c.LAB_OFFSET


def _xyz_f_inv(t: float) -> float:
    """Helper function for LAB to XYZ."""
    return t**3 if t > c.LAB_INV_THR else (t - c.LAB_OFFSET) / c.LAB_K


def xyz_to_lab(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert XYZ to CIE LAB."""
    x_r = _xyz_f(x / c.D65_X)
    y_r = _xyz_f(y / c.D65_Y)
    z_r = _xyz_f(z / c.D65_Z)
    L = (c.LAB_L_MULT * y_r) - c.LAB_L_SUB
    a = c.LAB_A_MULT * (x_r - y_r)
    b = c.LAB_B_MULT * (y_r - z_r)
    return L, a, b


def lab_to_xyz(L: float, a: float, b: float) -> Tuple[float, float, float]:
    """Convert LAB to CIE XYZ."""
    y_r = (L + c.LAB_L_SUB) / c.LAB_L_MULT
    x_r = a / c.LAB_A_MULT + y_r
    z_r = y_r - b / c.LAB_B_MULT
    x = _xyz_f_inv(x_r) * c.D65_X
    y = _xyz_f_inv(y_r) * c.D65_Y
    z = _xyz_f_inv(z_r) * c.D65_Z
    return x, y, z


def _linear_to_srgb(l_val: float) -> float:
    """Apply sRGB gamma to linear component."""
    l_val = max(l_val, 0.0)
    if l_val <= c.LINEAR_TO_SRGB_TH:
        return c.SRGB_SLOPE * l_val
    return c.SRGB_DIVISOR * (l_val ** (c.UNIT / c.SRGB_GAMMA)) - c.SRGB_OFFSET


def xyz_to_rgb(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert CIE XYZ to RGB."""
    x_n, y_n, z_n = x / c.XYZ_SCALING, y / c.XYZ_SCALING, z / c.XYZ_SCALING
    r_lin = x_n * c.M_XYZ_SRGB_R[0] + y_n * c.M_XYZ_SRGB_R[1] + z_n * c.M_XYZ_SRGB_R[2]
    g_lin = x_n * c.M_XYZ_SRGB_G[0] + y_n * c.M_XYZ_SRGB_G[1] + z_n * c.M_XYZ_SRGB_G[2]
    b_lin = x_n * c.M_XYZ_SRGB_B[0] + y_n * c.M_XYZ_SRGB_B[1] + z_n * c.M_XYZ_SRGB_B[2]
    r = _linear_to_srgb(r_lin)
    g = _linear_to_srgb(g_lin)
    b = _linear_to_srgb(b_lin)
    return _clamp01(r) * c.RGB_MAX, _clamp01(g) * c.RGB_MAX, _clamp01(b) * c.RGB_MAX


def rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Direct RGB to LAB conversion."""
    x, y, z = rgb_to_xyz(r, g, b)
    return xyz_to_lab(x, y, z)


def lab_to_lch(L: float, a: float, b: float) -> Tuple[float, float, float]:
    """Convert LAB to LCH."""
    chroma = math.hypot(a, b)
    hue = math.degrees(math.atan2(b, a)) % c.HUE_MAX
    return L, chroma, hue


def rgb_to_lch(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Direct RGB to LCH conversion."""
    L, a_val, b_val = rgb_to_lab(r, g, b)
    return lab_to_lch(L, a_val, b_val)


def lch_to_rgb(L: float, chroma: float, hue: float) -> Tuple[float, float, float]:
    """Direct LCH to RGB conversion."""
    L_val, a_val, b_val = lch_to_lab(L, chroma, hue)
    return lab_to_rgb(L_val, a_val, b_val)


def lch_to_lab(L: float, chroma: float, hue: float) -> Tuple[float, float, float]:
    """Convert LCH to LAB."""
    a = chroma * math.cos(math.radians(hue))
    b = chroma * math.sin(math.radians(hue))
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

    l_val = c.OKLAB_XYZ_TO_LMS_RR * r_lin + c.OKLAB_XYZ_TO_LMS_RG * g_lin + c.OKLAB_XYZ_TO_LMS_RB * b_lin
    m = c.OKLAB_XYZ_TO_LMS_GR * r_lin + c.OKLAB_XYZ_TO_LMS_GG * g_lin + c.OKLAB_XYZ_TO_LMS_GB * b_lin
    s = c.OKLAB_XYZ_TO_LMS_BR * r_lin + c.OKLAB_XYZ_TO_LMS_BG * g_lin + c.OKLAB_XYZ_TO_LMS_BB * b_lin

    l_ = (abs(l_val)) ** c.OKLAB_CUBE_ROOT_EXP if l_val >= 0 else -((abs(l_val)) ** c.OKLAB_CUBE_ROOT_EXP)
    m_ = (abs(m)) ** c.OKLAB_CUBE_ROOT_EXP if m >= 0 else -((abs(m)) ** c.OKLAB_CUBE_ROOT_EXP)
    s_ = (abs(s)) ** c.OKLAB_CUBE_ROOT_EXP if s >= 0 else -((abs(s)) ** c.OKLAB_CUBE_ROOT_EXP)

    ok_l = c.OKLAB_LMS_TO_LAB_LL * l_ + c.OKLAB_LMS_TO_LAB_LM * m_ + c.OKLAB_LMS_TO_LAB_LS * s_
    # Note: ok_a and ok_b coefficients would ideally be defined in config as M2_OKLAB for full mapping
    # but using existing config pattern for consistency
    ok_a = c.M2_OKLAB[1][0] * l_ + c.M2_OKLAB[1][1] * m_ + c.M2_OKLAB[1][2] * s_
    ok_b = c.M2_OKLAB[2][0] * l_ + c.M2_OKLAB[2][1] * m_ + c.M2_OKLAB[2][2] * s_

    return ok_l, ok_a, ok_b


def oklab_to_rgb(L: float, a: float, b: float) -> Tuple[float, float, float]:
    """Convert OKLab to RGB."""
    l_ = (L + c.OKLAB_TO_LMS_PRIME_LA * a + c.OKLAB_TO_LMS_PRIME_LB * b)
    m_ = (L + c.OKLAB_TO_LMS_PRIME_MA * a + c.OKLAB_TO_LMS_PRIME_MB * b)
    s_ = (L + c.OKLAB_TO_LMS_PRIME_SA * a + c.OKLAB_TO_LMS_PRIME_SB * b)

    l_lin = l_**3
    m_lin = m_**3
    s_lin = s_**3

    r_lin = (c.OKLAB_LMS_PRIME_TO_XYZ_RL * l_lin + c.OKLAB_LMS_PRIME_TO_XYZ_RM * m_lin + c.OKLAB_LMS_PRIME_TO_XYZ_RS * s_lin)
    g_lin = (c.OKLAB_LMS_PRIME_TO_XYZ_GL * l_lin + c.OKLAB_LMS_PRIME_TO_XYZ_GM * m_lin + c.OKLAB_LMS_PRIME_TO_XYZ_GS * s_lin)
    b_lin = (c.OKLAB_LMS_PRIME_TO_XYZ_BL * l_lin + c.OKLAB_LMS_PRIME_TO_XYZ_BM * m_lin + c.OKLAB_LMS_PRIME_TO_XYZ_BS * s_lin)

    r = _linear_to_srgb(r_lin)
    g = _linear_to_srgb(g_lin)
    b = _linear_to_srgb(b_lin)

    return _clamp01(r) * c.RGB_MAX, _clamp01(g) * c.RGB_MAX, _clamp01(b) * c.RGB_MAX


def oklab_to_oklch(L: float, a: float, b: float) -> Tuple[float, float, float]:
    """Convert OKLab to OKLCH."""
    chroma = math.hypot(a, b)
    hue = math.degrees(math.atan2(b, a)) % c.HUE_MAX
    return L, chroma, hue


def oklch_to_oklab(L: float, chroma: float, hue: float) -> Tuple[float, float, float]:
    """Convert OKLCH to OKLab."""
    a = chroma * math.cos(math.radians(hue))
    b = chroma * math.sin(math.radians(hue))
    return L, a, b


def rgb_to_oklch(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Direct RGB to OKLCH conversion."""
    L, a_val, b_val = rgb_to_oklab(r, g, b)
    return oklab_to_oklch(L, a_val, b_val)


def oklch_to_rgb(L: float, chroma: float, hue: float) -> Tuple[float, float, float]:
    """Direct OKLCH to RGB conversion."""
    L_val, a_val, b_val = oklch_to_oklab(L, chroma, hue)
    return oklab_to_rgb(L_val, a_val, b_val)


def rgb_to_hwb(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB to HWB."""
    h, s, v = rgb_to_hsv(r, g, b)
    w = (c.UNIT - s) * v
    blk = c.UNIT - v
    return h, w, blk


def hwb_to_rgb(h: float, w: float, b: float) -> Tuple[float, float, float]:
    """Convert HWB to RGB."""
    w = _clamp01(w)
    b = _clamp01(b)
    if w + b > c.UNIT:
        total = w + b
        if total > 0.0:
            w = w / total
            b = b / total
    v = c.UNIT - b
    if v <= 0.0:
        return 0.0, 0.0, 0.0
    s = c.UNIT - (w / v)
    s = _clamp01(s)
    return hsv_to_rgb(h, s, v)


def rgb_to_luv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB to CIE LUV."""
    X, Y, Z = rgb_to_xyz(r, g, b)
    denom = X + c.LUV_DENOM_Y * Y + c.LUV_DENOM_Z * Z
    if denom == 0:
        u_prime = 0.0
        v_prime = 0.0
    else:
        u_prime = (c.LUV_U_NUM * X) / denom
        v_prime = (c.LUV_V_NUM * Y) / denom

    denom_n = c.D65_X + c.LUV_DENOM_Y * c.D65_Y + c.LUV_DENOM_Z * c.D65_Z
    u_prime_n = (c.LUV_U_NUM * c.D65_X) / denom_n
    v_prime_n = (c.LUV_V_NUM * c.D65_Y) / denom_n

    y_r = Y / c.D65_Y
    if y_r > c.LAB_E:
        L = (c.LAB_L_MULT * (y_r**c.LAB_POW)) - c.LAB_L_SUB
    else:
        L = c.LUV_KAPPA * y_r

    if L == 0:
        u = 0.0
        v = 0.0
    else:
        u = c.LUV_U_V_MULT * L * (u_prime - u_prime_n)
        v = c.LUV_U_V_MULT * L * (v_prime - v_prime_n)

    return L, u, v


def luv_to_rgb(L: float, u: float, v: float) -> Tuple[float, float, float]:
    """Convert CIE LUV to RGB."""
    denom_n = c.D65_X + c.LUV_DENOM_Y * c.D65_Y + c.LUV_DENOM_Z * c.D65_Z
    u_prime_n = (c.LUV_U_NUM * c.D65_X) / denom_n
    v_prime_n = (c.LUV_V_NUM * c.D65_Y) / denom_n

    if L == 0:
        return xyz_to_rgb(0.0, 0.0, 0.0)

    u_prime = (u / (c.LUV_U_V_MULT * L)) + u_prime_n
    v_prime = (v / (c.LUV_U_V_MULT * L)) + v_prime_n

    if L > c.LUV_L_THR:
        Y = c.D65_Y * (((L + c.LAB_L_SUB) / c.LAB_L_MULT) ** 3)
    else:
        Y = c.D65_Y * (L / c.LUV_KAPPA)

    if v_prime == 0:
        X = 0.0
        Z = 0.0
    else:
        X = Y * (c.LUV_V_NUM * u_prime) / (c.LUV_U_NUM * v_prime)
        Z = (
            Y
            * (c.LUV_Z_CONST - c.LUV_Z_U_MULT * u_prime - c.LUV_Z_V_MULT * v_prime)
            / (c.LUV_U_NUM * v_prime)
        )

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


def hwb_to_hex(h: float, w: float, blk: float) -> str:
    """Direct HWB to Hex."""
    return rgb_to_hex(*hwb_to_rgb(h, w, blk))


def hex_to_cmyk(hex_code: str) -> Tuple[float, float, float, float]:
    """Direct Hex to CMYK."""
    return rgb_to_cmyk(*hex_to_rgb(hex_code))


def cmyk_to_hex(cy: float, m: float, y: float, k: float) -> str:
    """Direct CMYK to Hex."""
    return rgb_to_hex(*cmyk_to_rgb(cy, m, y, k))


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


def lch_to_hex(L: float, chroma: float, hue: float) -> str:
    """Direct LCH to Hex."""
    return rgb_to_hex(*lch_to_rgb(L, chroma, hue))


def hex_to_oklab(hex_code: str) -> Tuple[float, float, float]:
    """Direct Hex to OKLab."""
    return rgb_to_oklab(*hex_to_rgb(hex_code))


def oklab_to_hex(L: float, a: float, b: float) -> str:
    """Direct OKLab to Hex."""
    return rgb_to_hex(*oklab_to_rgb(L, a, b))


def hex_to_oklch(hex_code: str) -> Tuple[float, float, float]:
    """Direct Hex to OKLCH."""
    return rgb_to_oklch(*hex_to_rgb(hex_code))


def oklch_to_hex(L: float, chroma: float, hue: float) -> str:
    """Direct OKLCH to Hex."""
    return rgb_to_hex(*oklch_to_oklch(L, chroma, hue)) # Using nested conversion consistent with flow


def hex_to_luv(hex_code: str) -> Tuple[float, float, float]:
    """Direct Hex to LUV."""
    return rgb_to_luv(*hex_to_rgb(hex_code))


def luv_to_hex(L: float, u: float, v: float) -> str:
    """Direct LUV to Hex."""
    return rgb_to_hex(*luv_to_rgb(L, u, v))


# Apply LRU caching to all functions in this module
for _name, _obj in list(globals().items()):
    if callable(_obj) and _obj.__module__ == __name__:
        globals()[_name] = functools.lru_cache(maxsize=c.LRU_CACHE_SIZE)(_obj)