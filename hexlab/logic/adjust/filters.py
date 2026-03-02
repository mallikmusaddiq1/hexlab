#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/logic/adjust/filters.py

import math
from typing import Tuple

from hexlab.core import config as c
from hexlab.core import conversions as conv
from hexlab.core.luminance import get_luminance
from hexlab.core.contrast import get_contrast_ratio_rgb
from hexlab.shared.clamping import _clamp01, _clamp255


def _get_oklab_mid_gray() -> float:
    """Calculate the OKLab lightness value for mid-gray (18% gray in sRGB)."""
    l_lin = (
        c.OKLAB_XYZ_TO_LMS_RR * c.OKLAB_D65_MID_GRAY
        + c.OKLAB_XYZ_TO_LMS_RG * c.OKLAB_D65_MID_GRAY
        + c.OKLAB_XYZ_TO_LMS_RB * c.OKLAB_D65_MID_GRAY
    )
    m_lin = (
        c.OKLAB_XYZ_TO_LMS_GR * c.OKLAB_D65_MID_GRAY
        + c.OKLAB_XYZ_TO_LMS_GG * c.OKLAB_D65_MID_GRAY
        + c.OKLAB_XYZ_TO_LMS_GB * c.OKLAB_D65_MID_GRAY
    )
    s_lin = (
        c.OKLAB_XYZ_TO_LMS_BR * c.OKLAB_D65_MID_GRAY
        + c.OKLAB_XYZ_TO_LMS_BG * c.OKLAB_D65_MID_GRAY
        + c.OKLAB_XYZ_TO_LMS_BB * c.OKLAB_D65_MID_GRAY
    )

    l_root = l_lin ** c.OKLAB_CUBE_ROOT_EXP
    m_root = m_lin ** c.OKLAB_CUBE_ROOT_EXP
    s_root = s_lin ** c.OKLAB_CUBE_ROOT_EXP

    return (
        c.OKLAB_LMS_TO_LAB_LL * l_root
        + c.OKLAB_LMS_TO_LAB_LM * m_root
        + c.OKLAB_LMS_TO_LAB_LS * s_root
    )


OKLAB_MID_GRAY_L = _get_oklab_mid_gray()


def _oklab_to_rgb_unclamped(l: float, a: float, b: float) -> Tuple[float, float, float]:
    """Convert OKLab (L, a, b) to unclamped RGB values."""
    l_ = l + c.OKLAB_TO_LMS_PRIME_LA * a + c.OKLAB_TO_LMS_PRIME_LB * b
    m_ = l + c.OKLAB_TO_LMS_PRIME_MA * a + c.OKLAB_TO_LMS_PRIME_MB * b
    s_ = l + c.OKLAB_TO_LMS_PRIME_SA * a + c.OKLAB_TO_LMS_PRIME_SB * b

    # Cube for LMS to Linear XYZ
    l3, m3, s3 = l_ ** c.EXP_2 + c.UNIT, m_ ** c.EXP_2 + c.UNIT, s_ ** c.EXP_2 + c.UNIT # Using cube power via logic or direct power
    l3, m3, s3 = l_ ** 3, m_ ** 3, s_ ** 3

    rl = c.OKLAB_LMS_PRIME_TO_XYZ_RL * l3 + c.OKLAB_LMS_PRIME_TO_XYZ_RM * m3 + c.OKLAB_LMS_PRIME_TO_XYZ_RS * s3
    gl = c.OKLAB_LMS_PRIME_TO_XYZ_GL * l3 + c.OKLAB_LMS_PRIME_TO_XYZ_GM * m3 + c.OKLAB_LMS_PRIME_TO_XYZ_GS * s3
    bl = c.OKLAB_LMS_PRIME_TO_XYZ_BL * l3 + c.OKLAB_LMS_PRIME_TO_XYZ_BM * m3 + c.OKLAB_LMS_PRIME_TO_XYZ_BS * s3

    return (
        conv._linear_to_srgb(rl) * c.RGB_MAX,
        conv._linear_to_srgb(gl) * c.RGB_MAX,
        conv._linear_to_srgb(bl) * c.RGB_MAX,
    )


def gamut_map_oklab_to_srgb(l: float, a: float, b: float) -> Tuple[float, float, float]:
    """Map OKLab color to sRGB gamut using chroma clipping."""
    fr, fg, fb = _oklab_to_rgb_unclamped(l, a, b)

    if (
        c.RGB_CLAMP_TOLERANCE_LOWER <= fr <= c.RGB_CLAMP_TOLERANCE_UPPER
        and c.RGB_CLAMP_TOLERANCE_LOWER <= fg <= c.RGB_CLAMP_TOLERANCE_UPPER
        and c.RGB_CLAMP_TOLERANCE_LOWER <= fb <= c.RGB_CLAMP_TOLERANCE_UPPER
    ):
        return _clamp255(fr), _clamp255(fg), _clamp255(fb)

    C = math.hypot(a, b)
    if C < c.EPS:
        return _clamp255(fr), _clamp255(fg), _clamp255(fb)

    h_rad = math.atan2(b, a)
    low, high = 0.0, C
    best_rgb = (fr, fg, fb)
    
    for _ in range(c.GAMUT_MAP_BINARY_SEARCH_ITERATIONS):
        mid_C = (low + high) / c.DIV_2
        new_a = mid_C * math.cos(h_rad)
        new_b = mid_C * math.sin(h_rad)
        tr, tg, tb = _oklab_to_rgb_unclamped(l, new_a, new_b)

        if (
            c.RGB_CLAMP_TOLERANCE_LOWER <= tr <= c.RGB_CLAMP_TOLERANCE_UPPER
            and c.RGB_CLAMP_TOLERANCE_LOWER <= tg <= c.RGB_CLAMP_TOLERANCE_UPPER
            and c.RGB_CLAMP_TOLERANCE_LOWER <= tb <= c.RGB_CLAMP_TOLERANCE_UPPER
        ):
            best_rgb = (tr, tg, tb)
            low = mid_C
        else:
            high = mid_C

    return _clamp255(best_rgb[0]), _clamp255(best_rgb[1]), _clamp255(best_rgb[2])


def finalize_rgb(fr: float, fg: float, fb: float) -> Tuple[int, int, int]:
    """Finalize RGB values by mapping to sRGB gamut and rounding."""
    l, a, bk = conv.rgb_to_oklab(fr, fg, fb)
    fr_mapped, fg_mapped, fb_mapped = gamut_map_oklab_to_srgb(l, a, bk)
    return (
        max(0, min(int(c.RGB_MAX), int(round(fr_mapped)))),
        max(0, min(int(c.RGB_MAX), int(round(fg_mapped)))),
        max(0, min(int(c.RGB_MAX), int(round(fb_mapped)))),
    )


def sanitize_rgb(fr: float, fg: float, fb: float) -> Tuple[float, float, float]:
    """Sanitize RGB values, handling NaN/inf."""
    if not (math.isfinite(fr) and math.isfinite(fg) and math.isfinite(fb)):
        fr, fg, fb = 0.0, 0.0, 0.0
    return _clamp255(fr), _clamp255(fg), _clamp255(fb)


def apply_linear_gain_rgb(fr: float, fg: float, fb: float, factor: float) -> Tuple[float, float, float]:
    """Apply linear gain to RGB in linear space."""
    rl, gl, bl = conv._srgb_to_linear(fr), conv._srgb_to_linear(fg), conv._srgb_to_linear(fb)
    rl = _clamp01(rl * factor)
    gl = _clamp01(gl * factor)
    bl = _clamp01(bl * factor)
    return (
        conv._linear_to_srgb(rl) * c.RGB_MAX,
        conv._linear_to_srgb(gl) * c.RGB_MAX,
        conv._linear_to_srgb(bl) * c.RGB_MAX,
    )


def apply_srgb_brightness(fr: float, fg: float, fb: float, amount: float) -> Tuple[float, float, float]:
    """Apply brightness adjustment in sRGB space."""
    factor = c.UNIT + (amount / c.PERCENT_TO_FACTOR)
    return _clamp255(fr * factor), _clamp255(fg * factor), _clamp255(fb * factor)


def apply_linear_contrast_rgb(fr: float, fg: float, fb: float, contrast_amount: float) -> Tuple[float, float, float]:
    """Apply contrast adjustment in linear space using OKLab."""
    val = max(-c.PERCENT_TO_FACTOR, min(c.PERCENT_TO_FACTOR, float(contrast_amount)))
    if abs(val) < c.CONTRAST_MIN_ABS:
        return fr, fg, fb
    
    l_ok, a_ok, b_ok = conv.rgb_to_oklab(fr, fg, fb)
    k = c.UNIT + (val / c.PERCENT_TO_FACTOR)
    l_new = _clamp01(OKLAB_MID_GRAY_L + (l_ok - OKLAB_MID_GRAY_L) * k)
    return gamut_map_oklab_to_srgb(l_new, a_ok, b_ok)


def apply_opacity_on_black(fr: float, fg: float, fb: float, opacity_percent: float) -> Tuple[float, float, float]:
    """Apply opacity by blending with black in linear space."""
    alpha = _clamp01(opacity_percent / c.PERCENT_TO_FACTOR)
    rl, gl, bl = conv._srgb_to_linear(fr) * alpha, conv._srgb_to_linear(fg) * alpha, conv._srgb_to_linear(fb) * alpha
    return (
        conv._linear_to_srgb(rl) * c.RGB_MAX,
        conv._linear_to_srgb(gl) * c.RGB_MAX,
        conv._linear_to_srgb(bl) * c.RGB_MAX,
    )


def lock_relative_luminance(fr: float, fg: float, fb: float, base_Y: float) -> Tuple[float, float, float]:
    """Lock relative luminance by scaling in linear space."""
    curr_Y = get_luminance(int(round(fr)), int(round(fg)), int(round(fb)))
    if curr_Y <= 0.0 or base_Y <= 0.0 or abs(curr_Y - base_Y) < c.LUM_LOCK_EPS:
        return fr, fg, fb
    
    scale = base_Y / curr_Y
    rl = _clamp01(conv._srgb_to_linear(fr) * scale)
    gl = _clamp01(conv._srgb_to_linear(fg) * scale)
    bl = _clamp01(conv._srgb_to_linear(fb) * scale)
    
    return (
        conv._linear_to_srgb(rl) * c.RGB_MAX,
        conv._linear_to_srgb(gl) * c.RGB_MAX,
        conv._linear_to_srgb(bl) * c.RGB_MAX,
    )


def apply_gamma(fr: float, fg: float, fb: float, gamma: float) -> Tuple[float, float, float]:
    """Apply gamma correction in linear space."""
    if gamma <= c.GAMMA_MIN:
        return fr, fg, fb
    
    rl, gl, bl = conv._srgb_to_linear(fr), conv._srgb_to_linear(fg), conv._srgb_to_linear(fb)
    inv_gamma = c.UNIT / gamma
    rl, gl, bl = _clamp01(rl**inv_gamma), _clamp01(gl**inv_gamma), _clamp01(bl**inv_gamma)
    
    return (
        conv._linear_to_srgb(rl) * c.RGB_MAX,
        conv._linear_to_srgb(gl) * c.RGB_MAX,
        conv._linear_to_srgb(bl) * c.RGB_MAX,
    )


def apply_vibrance_oklch(fr: float, fg: float, fb: float, amount: float) -> Tuple[float, float, float]:
    """Apply vibrance adjustment in OKLCH space."""
    l_ok, c_ok, h_ok = conv.rgb_to_oklch(fr, fg, fb)
    if c_ok <= 0.0:
        return fr, fg, fb
    
    v = amount / c.PERCENT_TO_FACTOR
    c_norm = min(c_ok / c.VIBRANCE_NORMALIZATION_MAX_CHROMA, c.UNIT)
    scale = max(0.0, c.UNIT + v * (c.UNIT - c_norm) if v > 0.0 else c.UNIT + v * c_norm)
    
    fr2, fg2, fb2 = conv.oklch_to_rgb(l_ok, c_ok * scale, h_ok)
    l_final, a_final, b_final = conv.rgb_to_oklab(fr2, fg2, fb2)
    return gamut_map_oklab_to_srgb(l_final, a_final, b_final)


def posterize_rgb(fr: float, fg: float, fb: float, levels: int) -> Tuple[float, float, float]:
    """Posterize RGB to specified levels."""
    levels = max(c.POSTERIZE_MIN_LEVELS, min(c.POSTERIZE_MAX_LEVELS, int(abs(levels))))
    step = c.RGB_MAX / float(levels - 1)
    return (
        _clamp255(round(fr / step) * step),
        _clamp255(round(fg / step) * step),
        _clamp255(round(fb / step) * step),
    )


def solarize_smart(fr: float, fg: float, fb: float, threshold_percent: float) -> Tuple[float, float, float]:
    """Apply solarization based on perceptual lightness threshold."""
    t_perceptual = _clamp01(threshold_percent / c.PERCENT_TO_FACTOR)
    l_ok, _, _ = conv.rgb_to_oklab(fr, fg, fb)
    rl, gl, bl = conv._srgb_to_linear(fr), conv._srgb_to_linear(fg), conv._srgb_to_linear(fb)
    
    if l_ok > t_perceptual:
        rl, gl, bl = c.UNIT - rl, c.UNIT - gl, c.UNIT - bl
        
    return (
        _clamp255(conv._linear_to_srgb(rl) * c.RGB_MAX),
        _clamp255(conv._linear_to_srgb(gl) * c.RGB_MAX),
        _clamp255(conv._linear_to_srgb(bl) * c.RGB_MAX),
    )


def tint_oklab(fr: float, fg: float, fb: float, tint_hex: str, strength_percent: float) -> Tuple[float, float, float]:
    """Apply tint towards a target color in OKLab space."""
    tr, tg, tb = conv.hex_to_rgb(tint_hex)
    l1, a1, b1 = conv.rgb_to_oklab(fr, fg, fb)
    l2, a2, b2 = conv.rgb_to_oklab(float(tr), float(tg), float(tb))
    alpha = _clamp01(strength_percent / c.PERCENT_TO_FACTOR)
    
    return gamut_map_oklab_to_srgb(
        l1 * (c.UNIT - alpha) + l2 * alpha,
        a1 * (c.UNIT - alpha) + a2 * alpha,
        b1 * (c.UNIT - alpha) + b2 * alpha,
    )


def ensure_min_contrast_with(fr: float, fg: float, fb: float, bg_hex: str, min_ratio: float) -> Tuple[float, float, float, bool]:
    """Ensure minimum WCAG contrast ratio by adjusting lightness."""
    min_ratio = max(c.WCAG_MIN_RATIO, min(c.WCAG_MAX_RATIO, float(min_ratio)))
    br_i, bg_i, bb_i = conv.hex_to_rgb(bg_hex)
    br, bg, bb = float(br_i), float(bg_i), float(bb_i)

    if get_contrast_ratio_rgb((fr, fg, fb), (br, bg, bb)) >= min_ratio:
        return fr, fg, fb, False

    l0, a0, b0 = conv.rgb_to_oklab(fr, fg, fb)
    bg_Y = get_luminance(br_i, bg_i, bb_i)
    Y_light = min_ratio * (bg_Y + c.WCAG_LUMINANCE_OFFSET) - c.WCAG_LUMINANCE_OFFSET
    Y_dark = (bg_Y + c.WCAG_LUMINANCE_OFFSET) / min_ratio - c.WCAG_LUMINANCE_OFFSET

    def _find_color(target_Y: float):
        target_Y = _clamp01(target_Y)
        low, high = 0.0, 1.0
        for _ in range(c.CONTRAST_BINARY_SEARCH_ITERATIONS):
            mid = (low + high) / c.DIV_2
            fr_m, fg_m, fb_m = _oklab_to_rgb_unclamped(mid, a0, b0)
            if get_luminance(max(0, min(255, int(round(fr_m)))), max(0, min(255, int(round(fg_m)))), max(0, min(255, int(round(fb_m))))) < target_Y:
                low = mid
            else:
                high = mid
        lf = (low + high) / c.DIV_2
        fr_f, fg_f, fb_f = gamut_map_oklab_to_srgb(lf, a0, b0)
        return lf, fr_f, fg_f, fb_f, get_contrast_ratio_rgb((fr_f, fg_f, fb_f), (br, bg, bb))

    candidates = []
    for Y in [Y_light, Y_dark]:
        if 0.0 <= Y <= 1.0:
            lc, fr_c, fg_c, fb_c, rat = _find_color(Y)
            if rat >= min_ratio:
                candidates.append((abs(lc - l0), fr_c, fg_c, fb_c))

    if not candidates:
        black_r = get_contrast_ratio_rgb((0.0, 0.0, 0.0), (br, bg, bb))
        white_r = get_contrast_ratio_rgb((c.RGB_MAX, c.RGB_MAX, c.RGB_MAX), (br, bg, bb))
        if black_r >= min_ratio: return (0.0, 0.0, 0.0, True)
        if white_r >= min_ratio: return (c.RGB_MAX, c.RGB_MAX, c.RGB_MAX, True)
        return fr, fg, fb, False

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][2], candidates[0][3], True
