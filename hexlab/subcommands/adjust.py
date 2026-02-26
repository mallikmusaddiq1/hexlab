# File: adjust.py
#!/usr/bin/env python3

import argparse
import math
import random
import sys
from typing import Tuple

from ..color_math.conversions import (
    _linear_to_srgb,
    _srgb_to_linear,
    hex_to_rgb,
    hsl_to_rgb,
    hsv_to_rgb,
    hwb_to_rgb,
    oklab_to_rgb,
    oklch_to_rgb,
    rgb_to_hex,
    rgb_to_hsl,
    rgb_to_hsv,
    rgb_to_hwb,
    rgb_to_oklab,
    rgb_to_oklch,
)
from ..color_math.luminance import get_luminance
from ..color_math.wcag_contrast import _wcag_contrast_ratio_from_rgb
from ..constants.constants import (

    MAX_DEC,

    # Floating-point precision and division-by-zero safety
    EPS,

    # Image processing and color adjustment sequence
    PIPELINE,

    # Standard ANSI Escape Codes for UI
    BOLD_WHITE,
    MSG_BOLD_COLORS,
    RESET,

    # Constants for OKLab color space conversions
    # Source: https://bottosson.github.io/posts/oklab/
    # D65 mid gray in sRGB (18% gray)
    OKLAB_D65_MID_GRAY,

    # XYZ to LMS matrix coefficients
    OKLAB_XYZ_TO_LMS_RR,
    OKLAB_XYZ_TO_LMS_RG,
    OKLAB_XYZ_TO_LMS_RB,
    OKLAB_XYZ_TO_LMS_GR,
    OKLAB_XYZ_TO_LMS_GG,
    OKLAB_XYZ_TO_LMS_GB,
    OKLAB_XYZ_TO_LMS_BR,
    OKLAB_XYZ_TO_LMS_BG,
    OKLAB_XYZ_TO_LMS_BB,

    # LMS to Lab matrix coefficients
    OKLAB_LMS_TO_LAB_LL,
    OKLAB_LMS_TO_LAB_LM,
    OKLAB_LMS_TO_LAB_LS,

    # OKLab to LMS' matrix coefficients
    OKLAB_TO_LMS_PRIME_LA,
    OKLAB_TO_LMS_PRIME_LB,
    OKLAB_TO_LMS_PRIME_MA,
    OKLAB_TO_LMS_PRIME_MB,
    OKLAB_TO_LMS_PRIME_SA,
    OKLAB_TO_LMS_PRIME_SB,

    # LMS' to XYZ matrix coefficients (inverse of XYZ to LMS)
    OKLAB_LMS_PRIME_TO_XYZ_RL,
    OKLAB_LMS_PRIME_TO_XYZ_RM,
    OKLAB_LMS_PRIME_TO_XYZ_RS,
    OKLAB_LMS_PRIME_TO_XYZ_GL,
    OKLAB_LMS_PRIME_TO_XYZ_GM,
    OKLAB_LMS_PRIME_TO_XYZ_GS,
    OKLAB_LMS_PRIME_TO_XYZ_BL,
    OKLAB_LMS_PRIME_TO_XYZ_BM,
    OKLAB_LMS_PRIME_TO_XYZ_BS,

    # Cube root exponent
    OKLAB_CUBE_ROOT_EXP,

    # RGB clamping tolerance for gamut check
    RGB_CLAMP_TOLERANCE_LOWER,
    RGB_CLAMP_TOLERANCE_UPPER,

    # Binary search iterations for gamut mapping
    GAMUT_MAP_BINARY_SEARCH_ITERATIONS,

    # Vibrance normalization max chroma
    VIBRANCE_NORMALIZATION_MAX_CHROMA,  # Empirical value for chroma scaling in OKLCH

    # Sepia filter matrix coefficients
    # Source: Standard sepia tone transformation matrix
    SEPIA_RR,
    SEPIA_RG,
    SEPIA_RB,
    SEPIA_GR,
    SEPIA_GG,
    SEPIA_GB,
    SEPIA_BR,
    SEPIA_BG,
    SEPIA_BB,

    # Warm/cool adjustment scales
    WARM_OKLAB_A_SCALE,  # Empirical denominator for warmth a-component
    WARM_OKLAB_B_SCALE,  # Empirical denominator for warmth b-component

    # Posterize min/max levels
    POSTERIZE_MIN_LEVELS,
    POSTERIZE_MAX_LEVELS,

    # Threshold default colors
    THRESHOLD_DEFAULT_LOW,
    THRESHOLD_DEFAULT_HIGH,

    # Tint default strength
    TINT_DEFAULT_STRENGTH,

    # Exposure stops scale
    EXPOSURE_STOPS_SCALE,  # 10% = 1 stop

    # WCAG contrast constants
    WCAG_LUMINANCE_OFFSET,
    WCAG_MIN_RATIO,
    WCAG_MAX_RATIO,

    # Binary search iterations for contrast adjustment
    CONTRAST_BINARY_SEARCH_ITERATIONS,

    # Channel adjustment min/max
    CHANNEL_MIN,
    CHANNEL_MAX,

    # Small epsilon for contrast check
    CONTRAST_EPS,

    # Average components divisor
    AVG_DIVISOR,

    # Contrast min abs for skip
    CONTRAST_MIN_ABS,

    # Gamma min value
    GAMMA_MIN,

    # Luminance lock epsilon
    LUM_LOCK_EPS,

    # Saturation epsilon
    SAT_EPS,

    # Percent to factor divisor
    PERCENT_TO_FACTOR,

    # RGB to sRGB scale
    RGB_TO_SRGB_SCALE,
)
from ..utils.clamping import _clamp01, _clamp255
from ..utils.color_names_handler import (
    get_title_for_hex,
    resolve_color_name_or_exit,
)
from ..utils.hexlab_logger import MSG_COLORS, log, HexlabArgumentParser
from ..utils.input_handler import INPUT_HANDLERS
from ..utils.print_color_block import print_color_block
from ..utils.truecolor import ensure_truecolor


def _get_oklab_mid_gray() -> float:
    """Calculate the OKLab lightness value for mid-gray (18% gray in sRGB).

    This function computes the perceptual lightness L in OKLab for a mid-gray point
    under D65 illuminant.

    Returns:
        float: OKLab L value for mid-gray.
    """
    # Apply XYZ to LMS transformation
    l_lin = (
        OKLAB_XYZ_TO_LMS_RR * OKLAB_D65_MID_GRAY
        + OKLAB_XYZ_TO_LMS_RG * OKLAB_D65_MID_GRAY
        + OKLAB_XYZ_TO_LMS_RB * OKLAB_D65_MID_GRAY
    )
    m_lin = (
        OKLAB_XYZ_TO_LMS_GR * OKLAB_D65_MID_GRAY
        + OKLAB_XYZ_TO_LMS_GG * OKLAB_D65_MID_GRAY
        + OKLAB_XYZ_TO_LMS_GB * OKLAB_D65_MID_GRAY
    )
    s_lin = (
        OKLAB_XYZ_TO_LMS_BR * OKLAB_D65_MID_GRAY
        + OKLAB_XYZ_TO_LMS_BG * OKLAB_D65_MID_GRAY
        + OKLAB_XYZ_TO_LMS_BB * OKLAB_D65_MID_GRAY
    )

    # Take cube roots
    l_root = l_lin**OKLAB_CUBE_ROOT_EXP
    m_root = m_lin**OKLAB_CUBE_ROOT_EXP
    s_root = s_lin**OKLAB_CUBE_ROOT_EXP

    # Compute L using LMS to Lab coefficients
    return (
        OKLAB_LMS_TO_LAB_LL * l_root
        + OKLAB_LMS_TO_LAB_LM * m_root
        + OKLAB_LMS_TO_LAB_LS * s_root
    )


OKLAB_MID_GRAY_L = _get_oklab_mid_gray()


def _oklab_to_rgb_unclamped(l: float, a: float, b: float) -> Tuple[float, float, float]:
    """Convert OKLab (L, a, b) to unclamped RGB values.

    This function applies the inverse transformation from OKLab to sRGB without clamping.

    Args:
        l (float): Lightness component.
        a (float): a component (green-red).
        b (float): b component (blue-yellow).

    Returns:
        Tuple[float, float, float]: Unclamped RGB values (0-255 scale).
    """
    # Transform to LMS'
    l_ = l + OKLAB_TO_LMS_PRIME_LA * a + OKLAB_TO_LMS_PRIME_LB * b
    m_ = l + OKLAB_TO_LMS_PRIME_MA * a + OKLAB_TO_LMS_PRIME_MB * b
    s_ = l + OKLAB_TO_LMS_PRIME_SA * a + OKLAB_TO_LMS_PRIME_SB * b

    # Cube the values
    l3 = l_**3
    m3 = m_**3
    s3 = s_**3

    # Transform to linear RGB
    rl = (
        OKLAB_LMS_PRIME_TO_XYZ_RL * l3
        + OKLAB_LMS_PRIME_TO_XYZ_RM * m3
        + OKLAB_LMS_PRIME_TO_XYZ_RS * s3
    )
    gl = (
        OKLAB_LMS_PRIME_TO_XYZ_GL * l3
        + OKLAB_LMS_PRIME_TO_XYZ_GM * m3
        + OKLAB_LMS_PRIME_TO_XYZ_GS * s3
    )
    bl = (
        OKLAB_LMS_PRIME_TO_XYZ_BL * l3
        + OKLAB_LMS_PRIME_TO_XYZ_BM * m3
        + OKLAB_LMS_PRIME_TO_XYZ_BS * s3
    )

    # Convert to sRGB and scale to 0-255
    return (
        _linear_to_srgb(rl) * RGB_TO_SRGB_SCALE,
        _linear_to_srgb(gl) * RGB_TO_SRGB_SCALE,
        _linear_to_srgb(bl) * RGB_TO_SRGB_SCALE,
    )


def _gamut_map_oklab_to_srgb(l: float, a: float, b: float) -> Tuple[float, float, float]:
    """Map OKLab color to sRGB gamut using chroma clipping.

    This function performs binary search to find the maximum chroma that fits within
    the sRGB gamut for the given lightness and hue.

    Args:
        l (float): Lightness.
        a (float): a component.
        b (float): b component.

    Returns:
        Tuple[float, float, float]: Clamped RGB values in sRGB gamut.
    """
    # Convert to unclamped RGB
    fr, fg, fb = _oklab_to_rgb_unclamped(l, a, b)

    # Check if already in gamut (with tolerance)
    if (
        RGB_CLAMP_TOLERANCE_LOWER <= fr <= RGB_CLAMP_TOLERANCE_UPPER
        and RGB_CLAMP_TOLERANCE_LOWER <= fg <= RGB_CLAMP_TOLERANCE_UPPER
        and RGB_CLAMP_TOLERANCE_LOWER <= fb <= RGB_CLAMP_TOLERANCE_UPPER
    ):
        return _clamp255(fr), _clamp255(fg), _clamp255(fb)

    # Compute chroma
    C = math.hypot(a, b)
    if C < EPS:
        return _clamp255(fr), _clamp255(fg), _clamp255(fb)

    # Compute hue angle
    h_rad = math.atan2(b, a)

    # Binary search for maximum in-gamut chroma
    low = 0.0
    high = C
    best_rgb = (fr, fg, fb)
    for _ in range(GAMUT_MAP_BINARY_SEARCH_ITERATIONS):
        mid_C = (low + high) / 2.0
        new_a = mid_C * math.cos(h_rad)
        new_b = mid_C * math.sin(h_rad)
        tr, tg, tb = _oklab_to_rgb_unclamped(l, new_a, new_b)

        if (
            RGB_CLAMP_TOLERANCE_LOWER <= tr <= RGB_CLAMP_TOLERANCE_UPPER
            and RGB_CLAMP_TOLERANCE_LOWER <= tg <= RGB_CLAMP_TOLERANCE_UPPER
            and RGB_CLAMP_TOLERANCE_LOWER <= tb <= RGB_CLAMP_TOLERANCE_UPPER
        ):
            best_rgb = (tr, tg, tb)
            low = mid_C
        else:
            high = mid_C

    return _clamp255(best_rgb[0]), _clamp255(best_rgb[1]), _clamp255(best_rgb[2])


def _finalize_rgb(fr: float, fg: float, fb: float) -> Tuple[int, int, int]:
    """Finalize RGB values by mapping to sRGB gamut and rounding to integers.

    Args:
        fr (float): Red component.
        fg (float): Green component.
        fb (float): Blue component.

    Returns:
        Tuple[int, int, int]: Final integer RGB values (0-255).
    """
    # Convert to OKLab
    l, a, bk = rgb_to_oklab(fr, fg, fb)
    # Map to gamut
    fr_mapped, fg_mapped, fb_mapped = _gamut_map_oklab_to_srgb(l, a, bk)
    # Round and clamp
    return (
        max(0, min(255, int(round(fr_mapped)))),
        max(0, min(255, int(round(fg_mapped)))),
        max(0, min(255, int(round(fb_mapped)))),
    )


def _apply_linear_gain_rgb(
    fr: float, fg: float, fb: float, factor: float
) -> Tuple[float, float, float]:
    """Apply linear gain to RGB in linear space.

    Args:
        fr (float): Red.
        fg (float): Green.
        fb (float): Blue.
        factor (float): Gain factor.

    Returns:
        Tuple[float, float, float]: Adjusted RGB.
    """
    # Convert to linear
    rl, gl, bl = _srgb_to_linear(fr), _srgb_to_linear(fg), _srgb_to_linear(fb)
    # Apply factor and clamp
    rl = _clamp01(rl * factor)
    gl = _clamp01(gl * factor)
    bl = _clamp01(bl * factor)
    # Back to sRGB
    return (
        _linear_to_srgb(rl) * RGB_TO_SRGB_SCALE,
        _linear_to_srgb(gl) * RGB_TO_SRGB_SCALE,
        _linear_to_srgb(bl) * RGB_TO_SRGB_SCALE,
    )


def _apply_srgb_brightness(
    fr: float, fg: float, fb: float, amount: float
) -> Tuple[float, float, float]:
    """Apply brightness adjustment in sRGB space.

    Args:
        fr (float): Red.
        fg (float): Green.
        fb (float): Blue.
        amount (float): Adjustment percentage.

    Returns:
        Tuple[float, float, float]: Adjusted RGB.
    """
    factor = 1.0 + (amount / PERCENT_TO_FACTOR)
    # Scale and clamp
    fr = _clamp255(fr * factor)
    fg = _clamp255(fg * factor)
    fb = _clamp255(fb * factor)
    return fr, fg, fb


def _apply_linear_contrast_rgb(
    fr: float, fg: float, fb: float, contrast_amount: float
) -> Tuple[float, float, float]:
    """Apply contrast adjustment in linear space using OKLab.

    Args:
        fr (float): Red.
        fg (float): Green.
        fb (float): Blue.
        contrast_amount (float): Contrast percentage (-100 to 100).

    Returns:
        Tuple[float, float, float]: Adjusted RGB.
    """
    # Clamp contrast
    c = max(-PERCENT_TO_FACTOR, min(PERCENT_TO_FACTOR, float(contrast_amount)))
    if abs(c) < CONTRAST_MIN_ABS:
        return fr, fg, fb
    # Convert to OKLab
    l_ok, a_ok, b_ok = rgb_to_oklab(fr, fg, fb)
    # Compute scale
    k = 1.0 + (c / PERCENT_TO_FACTOR)
    l_mid = OKLAB_MID_GRAY_L
    # Adjust lightness around mid-gray
    l_new = l_mid + (l_ok - l_mid) * k
    l_new = _clamp01(l_new)
    # Map back to RGB
    return _gamut_map_oklab_to_srgb(l_new, a_ok, b_ok)


def _apply_opacity_on_black(
    fr: float, fg: float, fb: float, opacity_percent: float
) -> Tuple[float, float, float]:
    """Apply opacity by blending with black in linear space.

    Args:
        fr (float): Red.
        fg (float): Green.
        fb (float): Blue.
        opacity_percent (float): Opacity (0-100).

    Returns:
        Tuple[float, float, float]: Adjusted RGB.
    """
    alpha = _clamp01(opacity_percent / PERCENT_TO_FACTOR)
    # To linear
    rl, gl, bl = _srgb_to_linear(fr), _srgb_to_linear(fg), _srgb_to_linear(fb)
    # Blend with black (multiply by alpha)
    rl *= alpha
    gl *= alpha
    bl *= alpha
    # Back to sRGB
    return (
        _linear_to_srgb(rl) * RGB_TO_SRGB_SCALE,
        _linear_to_srgb(gl) * RGB_TO_SRGB_SCALE,
        _linear_to_srgb(bl) * RGB_TO_SRGB_SCALE,
    )


def _lock_relative_luminance(
    fr: float, fg: float, fb: float, base_Y: float
) -> Tuple[float, float, float]:
    """Lock relative luminance by scaling in linear space.

    Args:
        fr (float): Red.
        fg (float): Green.
        fb (float): Blue.
        base_Y (float): Target relative luminance.

    Returns:
        Tuple[float, float, float]: Adjusted RGB.
    """
    # Compute current luminance
    curr_Y = get_luminance(int(round(fr)), int(round(fg)), int(round(fb)))
    if curr_Y <= 0.0 or base_Y <= 0.0 or abs(curr_Y - base_Y) < LUM_LOCK_EPS:
        return fr, fg, fb
    # Scale factor
    scale = base_Y / curr_Y
    # To linear, scale, clamp
    rl = _srgb_to_linear(fr) * scale
    gl = _srgb_to_linear(fg) * scale
    bl = _srgb_to_linear(fb) * scale
    rl = _clamp01(rl)
    gl = _clamp01(gl)
    bl = _clamp01(bl)
    # Back to sRGB
    return (
        _linear_to_srgb(rl) * RGB_TO_SRGB_SCALE,
        _linear_to_srgb(gl) * RGB_TO_SRGB_SCALE,
        _linear_to_srgb(bl) * RGB_TO_SRGB_SCALE,
    )


def _apply_gamma(
    fr: float, fg: float, fb: float, gamma: float
) -> Tuple[float, float, float]:
    """Apply gamma correction in linear space.

    Args:
        fr (float): Red.
        fg (float): Green.
        fb (float): Blue.
        gamma (float): Gamma value (>0).

    Returns:
        Tuple[float, float, float]: Adjusted RGB.
    """
    if gamma <= GAMMA_MIN:
        return fr, fg, fb
    # To linear
    rl = _srgb_to_linear(fr)
    gl = _srgb_to_linear(fg)
    bl = _srgb_to_linear(fb)
    # Apply inverse gamma
    inv_gamma = 1.0 / gamma
    rl = _clamp01(rl**inv_gamma)
    gl = _clamp01(gl**inv_gamma)
    bl = _clamp01(bl**inv_gamma)
    # Back to sRGB
    return (
        _linear_to_srgb(rl) * RGB_TO_SRGB_SCALE,
        _linear_to_srgb(gl) * RGB_TO_SRGB_SCALE,
        _linear_to_srgb(bl) * RGB_TO_SRGB_SCALE,
    )


def _apply_vibrance_oklch(
    fr: float, fg: float, fb: float, amount: float
) -> Tuple[float, float, float]:
    """Apply vibrance adjustment in OKLCH space.

    Vibrance boosts low-chroma colors more than high-chroma ones.

    Args:
        fr (float): Red.
        fg (float): Green.
        fb (float): Blue.
        amount (float): Vibrance percentage (-100 to 100).

    Returns:
        Tuple[float, float, float]: Adjusted RGB.
    """
    # To OKLCH
    l_ok, c_ok, h_ok = rgb_to_oklch(fr, fg, fb)
    if c_ok <= 0.0:
        return fr, fg, fb
    v = amount / PERCENT_TO_FACTOR
    # Normalize chroma
    c_norm = min(c_ok / VIBRANCE_NORMALIZATION_MAX_CHROMA, 1.0)
    # Compute scale based on sign of v
    if v > 0.0:
        scale = 1.0 + v * (1.0 - c_norm)
    else:
        scale = 1.0 + v * c_norm
    # Ensure non-negative
    if scale < 0.0:
        scale = 0.0
    # New chroma
    c_new = c_ok * scale
    # Back to RGB, then OKLab for gamut map
    fr2, fg2, fb2 = oklch_to_rgb(l_ok, c_new, h_ok)
    l_final, a_final, b_final = rgb_to_oklab(fr2, fg2, fb2)
    return _gamut_map_oklab_to_srgb(l_final, a_final, b_final)


def _posterize_rgb(
    fr: float, fg: float, fb: float, levels: int
) -> Tuple[float, float, float]:
    """Posterize RGB to specified levels.

    Args:
        fr (float): Red.
        fg (float): Green.
        fb (float): Blue.
        levels (int): Number of levels (2-256).

    Returns:
        Tuple[float, float, float]: Posterized RGB.
    """
    # Clamp levels
    levels = max(POSTERIZE_MIN_LEVELS, min(POSTERIZE_MAX_LEVELS, int(abs(levels))))
    # Step size
    step = RGB_TO_SRGB_SCALE / float(levels - 1)
    # Round to nearest step
    fr2 = round(fr / step) * step
    fg2 = round(fg / step) * step
    fb2 = round(fb / step) * step
    return _clamp255(fr2), _clamp255(fg2), _clamp255(fb2)


def _solarize_smart(
    fr: float, fg: float, fb: float, threshold_percent: float
) -> Tuple[float, float, float]:
    """Apply solarization based on perceptual lightness threshold.

    Inverts colors above the threshold in linear space.

    Args:
        fr (float): Red.
        fg (float): Green.
        fb (float): Blue.
        threshold_percent (float): Threshold (0-100).

    Returns:
        Tuple[float, float, float]: Solarized RGB.
    """
    t_perceptual = _clamp01(threshold_percent / PERCENT_TO_FACTOR)
    # Get OKLab L
    l_ok, _, _ = rgb_to_oklab(fr, fg, fb)
    # To linear
    rl, gl, bl = _srgb_to_linear(fr), _srgb_to_linear(fg), _srgb_to_linear(fb)
    # Invert if above threshold
    if l_ok > t_perceptual:
        rl = 1.0 - rl
        gl = 1.0 - gl
        bl = 1.0 - bl
    # Back to sRGB
    fr2 = _linear_to_srgb(rl) * RGB_TO_SRGB_SCALE
    fg2 = _linear_to_srgb(gl) * RGB_TO_SRGB_SCALE
    fb2 = _linear_to_srgb(bl) * RGB_TO_SRGB_SCALE
    return _clamp255(fr2), _clamp255(fg2), _clamp255(fb2)


def _tint_oklab(
    fr: float, fg: float, fb: float, tint_hex: str, strength_percent: float
) -> Tuple[float, float, float]:
    """Apply tint towards a target color in OKLab space.

    Linearly interpolates between current and tint color.

    Args:
        fr (float): Red.
        fg (float): Green.
        fb (float): Blue.
        tint_hex (str): Tint color hex.
        strength_percent (float): Strength (0-100).

    Returns:
        Tuple[float, float, float]: Tinted RGB.
    """
    tr, tg, tb = hex_to_rgb(tint_hex)
    # Current OKLab
    l1, a1, b1 = rgb_to_oklab(fr, fg, fb)
    # Tint OKLab
    l2, a2, b2 = rgb_to_oklab(float(tr), float(tg), float(tb))
    alpha = _clamp01(strength_percent / PERCENT_TO_FACTOR)
    # Lerp components
    l = l1 * (1.0 - alpha) + l2 * alpha
    a = a1 * (1.0 - alpha) + a2 * alpha
    b = b1 * (1.0 - alpha) + b2 * alpha
    # Map to gamut
    return _gamut_map_oklab_to_srgb(l, a, b)


def _ensure_min_contrast_with(
    fr: float, fg: float, fb: float, bg_hex: str, min_ratio: float
) -> Tuple[float, float, float, bool]:
    """Ensure minimum WCAG contrast ratio by adjusting lightness.

    Adjusts the color's OKLab L to meet the contrast ratio against background.

    Args:
        fr (float): Red.
        fg (float): Green.
        fb (float): Blue.
        bg_hex (str): Background hex.
        min_ratio (float): Minimum ratio (1-21).

    Returns:
        Tuple[float, float, float, bool]: Adjusted RGB and change flag.
    """
    # Clamp ratio
    min_ratio = max(WCAG_MIN_RATIO, min(WCAG_MAX_RATIO, float(min_ratio)))
    br_i, bg_i, bb_i = hex_to_rgb(bg_hex)
    br, bg, bb = float(br_i), float(bg_i), float(bb_i)

    # Current contrast
    current_ratio = _wcag_contrast_ratio_from_rgb(fr, fg, fb, br, bg, bb)
    if current_ratio >= min_ratio:
        return fr, fg, fb, False

    # Current OKLab
    l0, a0, b0 = rgb_to_oklab(fr, fg, fb)
    # Background luminance
    bg_Y = get_luminance(br_i, bg_i, bb_i)

    # Target luminances for light and dark
    Y_light = min_ratio * (bg_Y + WCAG_LUMINANCE_OFFSET) - WCAG_LUMINANCE_OFFSET
    Y_dark = (bg_Y + WCAG_LUMINANCE_OFFSET) / min_ratio - WCAG_LUMINANCE_OFFSET

    def _find_color_for_target_Y(target_Y: float):
        """Binary search for OKLab L that matches target Y."""
        target_Y = _clamp01(target_Y)
        low, high = 0.0, 1.0
        for _ in range(CONTRAST_BINARY_SEARCH_ITERATIONS):
            mid = (low + high) / 2.0
            fr_mid, fg_mid, fb_mid = _oklab_to_rgb_unclamped(mid, a0, b0)

            # Round for luminance calc
            r_check = max(0, min(255, int(round(fr_mid))))
            g_check = max(0, min(255, int(round(fg_mid))))
            b_check = max(0, min(255, int(round(fb_mid))))

            y_mid = get_luminance(r_check, g_check, b_check)
            if y_mid < target_Y:
                low = mid
            else:
                high = mid
        l_final = (low + high) / 2.0
        fr_fin, fg_fin, fb_fin = _gamut_map_oklab_to_srgb(l_final, a0, b0)
        ratio = _wcag_contrast_ratio_from_rgb(fr_fin, fg_fin, fb_fin, br, bg, bb)
        return l_final, fr_fin, fg_fin, fb_fin, ratio

    candidates = []

    # Light candidate
    if 0.0 <= Y_light <= 1.0:
        l_light, fr_light, fg_light, fb_light, ratio_light = _find_color_for_target_Y(
            Y_light
        )
        if ratio_light >= min_ratio:
            candidates.append(
                (abs(l_light - l0), l_light, fr_light, fg_light, fb_light, ratio_light)
            )

    # Dark candidate
    if 0.0 <= Y_dark <= 1.0:
        l_dark, fr_dark, fg_dark, fb_dark, ratio_dark = _find_color_for_target_Y(Y_dark)
        if ratio_dark >= min_ratio:
            candidates.append(
                (abs(l_dark - l0), l_dark, fr_dark, fg_dark, fb_dark, ratio_dark)
            )

    if not candidates:
        # Fallback to black or white
        black_ratio = _wcag_contrast_ratio_from_rgb(0.0, 0.0, 0.0, br, bg, bb)
        white_ratio = _wcag_contrast_ratio_from_rgb(
            RGB_TO_SRGB_SCALE,
            RGB_TO_SRGB_SCALE,
            RGB_TO_SRGB_SCALE,
            br,
            bg,
            bb,
        )
        best_rgb = (fr, fg, fb)
        best_ratio = current_ratio
        if black_ratio >= min_ratio and black_ratio >= best_ratio:
            best_rgb = (0.0, 0.0, 0.0)
            best_ratio = black_ratio
        if white_ratio >= min_ratio and white_ratio >= best_ratio:
            best_rgb = (RGB_TO_SRGB_SCALE, RGB_TO_SRGB_SCALE, RGB_TO_SRGB_SCALE)
            best_ratio = white_ratio
        if best_ratio > current_ratio:
            return best_rgb[0], best_rgb[1], best_rgb[2], True
        return fr, fg, fb, False

    # Select closest to original L
    candidates.sort(key=lambda x: x[0])
    _, _, fr_best, fg_best, fb_best, _ = candidates[0]
    return fr_best, fg_best, fb_best, True


def _format_steps(mods):
    """Format modification steps for logging.

    Args:
        mods: List of (label, value) tuples.

    Returns:
        List of formatted strings.
    """
    parts = []
    for label, val in mods:
        if val:
            parts.append(f"{label} {val}")
        else:
            parts.append(label)
    return parts


def _print_steps(mods, verbose: bool) -> None:
    """Print adjustment steps if verbose.

    Args:
        mods: List of modifications.
        verbose (bool): Whether to print.
    """
    if not verbose:
        return
    if not mods:
        log("info", "steps: no adjustments applied yet")
        return

    parts = _format_steps(mods)

    log("info", "steps:")
    for i, part in enumerate(parts, 1):
        print(f"{MSG_COLORS['info']}    {i}. {part}")


def _sanitize_rgb(fr: float, fg: float, fb: float) -> Tuple[float, float, float]:
    """Sanitize RGB values, handling NaN/inf.

    Args:
        fr (float): Red.
        fg (float): Green.
        fb (float): Blue.

    Returns:
        Tuple[float, float, float]: Sanitized RGB.
    """
    if not (math.isfinite(fr) and math.isfinite(fg) and math.isfinite(fb)):
        fr, fg, fb = 0.0, 0.0, 0.0
    return _clamp255(fr), _clamp255(fg), _clamp255(fb)


def _get_custom_pipeline_order(parser) -> list:
    """Get custom pipeline order from CLI arguments.

    Args:
        parser: Argument parser.

    Returns:
        list: Ordered operations.
    """
    flag_map = {}
    for action in parser._actions:
        for opt in action.option_strings:
            flag_map[opt] = action.dest

    dest_to_op = {
        "invert": "invert",
        "grayscale": "grayscale",
        "sepia": "sepia",
        "rotate": "rotate",
        "rotate_oklch": "rotate_oklch",
        "brightness": "brightness",
        "brightness_srgb": "brightness_srgb",
        "contrast": "contrast",
        "gamma": "gamma",
        "exposure": "exposure",
        "lighten": "lighten",
        "darken": "darken",
        "saturate": "saturate",
        "desaturate": "desaturate",
        "whiten_hwb": "whiten_hwb",
        "blacken_hwb": "blacken_hwb",
        "chroma_oklch": "chroma_oklch",
        "vibrance_oklch": "vibrance_oklch",
        "warm_oklab": "warm_oklab",
        "cool_oklab": "cool_oklab",
        "posterize": "posterize",
        "threshold": "threshold",
        "solarize": "solarize",
        "tint": "tint",
        "red_channel": "red_channel",
        "green_channel": "green_channel",
        "blue_channel": "blue_channel",
        "opacity": "opacity",
        "lock_luminance": "lock_luminance",
        "lock_rel_luminance": "lock_rel_luminance",
        "target_rel_lum": "target_rel_lum",
        "min_contrast_with": "min_contrast",
        "min_contrast": "min_contrast",
    }

    order = []
    seen = set()
    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            continue
        key = arg.split("=")[0]
        dest = flag_map.get(key)
        if not dest:
            continue
        op = dest_to_op.get(dest)
        if not op:
            continue
        if op not in seen:
            order.append(op)
            seen.add(op)

    return order


def handle_adjust_command(args: argparse.Namespace) -> None:
    """Handle the adjust command logic.

    Processes input color, applies adjustments in pipeline order, and outputs result.

    Args:
        args: Parsed arguments.
    """
    if args.seed is not None:
        random.seed(args.seed)

    locks = 0
    if getattr(args, "lock_luminance", False):
        locks += 1
    if getattr(args, "lock_rel_luminance", False):
        locks += 1
    if getattr(args, "target_rel_lum", None) is not None:
        locks += 1

    if locks > 1:
        log(
            "error",
            "conflicting luminance locks: use only one of --lock-luminance,"
            "--lock-rel-luminance or --target-rel-lum",
        )
        sys.exit(2)

    if getattr(args, "min_contrast_with", None) and locks > 0:
        log("warning", "--min-contrast-with will override previous luminance locks")

    pipeline = PIPELINE

    if getattr(args, "list_fixed_pipeline", False):
        for step in pipeline:
            print(step)
        return

    base_hex, title = None, "original"
    if args.random:
        base_hex, title = f"{random.randint(0, MAX_DEC):06X}", "random"
    elif args.color_name:
        base_hex = resolve_color_name_or_exit(args.color_name)
        title = get_title_for_hex(base_hex)
        if title.lower() == "unknown":
            title = args.color_name
    elif args.hex:
        base_hex, title = args.hex, get_title_for_hex(args.hex)
    elif getattr(args, "decimal_index", None):
        base_hex, title = args.decimal_index, f"index {int(args.decimal_index, 16)}"

    if not base_hex:
        log("error", "no input color")
        sys.exit(2)

    mc_with = getattr(args, "min_contrast_with", None)
    mc_val = getattr(args, "min_contrast", None)

    if (mc_with and mc_val is None) or (mc_with is None and mc_val is not None):
        log("error", "--min-contrast-with and --min-contrast must be used together")
        sys.exit(2)

    r, g, b = hex_to_rgb(base_hex)
    fr, fg, fb = float(r), float(g), float(b)
    base_l_oklab, _, _ = rgb_to_oklab(fr, fg, fb)
    base_rel_lum = get_luminance(r, g, b)

    mods = []

    if getattr(args, "custom_pipeline", False):
        parser = get_adjust_parser()
        custom_order = _get_custom_pipeline_order(parser)
        if custom_order:
            pipeline = custom_order

    fr, fg, fb = _sanitize_rgb(fr, fg, fb)

    for op in pipeline:
        # Capture current hex at start of each step
        curr_hex = rgb_to_hex(fr, fg, fb)
        src_info = f"from #{curr_hex}"

        if op == "invert" and args.invert:
            fr, fg, fb = RGB_TO_SRGB_SCALE - fr, RGB_TO_SRGB_SCALE - fg, RGB_TO_SRGB_SCALE - fb
            mods.append(("invert", src_info))

        elif op == "grayscale" and args.grayscale:
            l_ok, a_ok, b_ok = rgb_to_oklab(fr, fg, fb)
            fr, fg, fb = oklab_to_rgb(l_ok, 0.0, 0.0)
            avg = (fr + fg + fb) / AVG_DIVISOR
            fr = fg = fb = avg
            fr, fg, fb = _clamp255(fr), _clamp255(fg), _clamp255(fb)
            mods.append(("grayscale", src_info))

        elif op == "sepia" and args.sepia:
            tr = fr * SEPIA_RR + fg * SEPIA_RG + fb * SEPIA_RB
            tg = fr * SEPIA_GR + fg * SEPIA_GG + fb * SEPIA_GB
            tb = fr * SEPIA_BR + fg * SEPIA_BG + fb * SEPIA_BB
            fr, fg, fb = _clamp255(tr), _clamp255(tg), _clamp255(tb)
            mods.append(("sepia", src_info))

        elif op == "rotate" and args.rotate is not None:
            h, s, l_hsl = rgb_to_hsl(fr, fg, fb)
            fr, fg, fb = hsl_to_rgb(h + args.rotate, s, l_hsl)
            mods.append(("hue-rotate-hsl", f"{args.rotate:+.2f}deg {src_info}"))

        elif op == "rotate_oklch" and getattr(args, "rotate_oklch", None) is not None:
            l_ok, c_ok, h_ok = rgb_to_oklch(fr, fg, fb)
            fr, fg, fb = oklch_to_rgb(l_ok, c_ok, h_ok + args.rotate_oklch)
            fr, fg, fb = _clamp255(fr), _clamp255(fg), _clamp255(fb)
            mods.append(("hue-rotate-oklch", f"{args.rotate_oklch:+.2f}deg {src_info}"))

        elif op == "brightness" and args.brightness is not None:
            factor = 1.0 + (args.brightness / PERCENT_TO_FACTOR)
            fr, fg, fb = _apply_linear_gain_rgb(fr, fg, fb, factor)
            mods.append(("brightness-linear", f"{args.brightness:+.2f}% {src_info}"))

        elif op == "brightness_srgb" and getattr(args, "brightness_srgb", None) is not None:
            fr, fg, fb = _apply_srgb_brightness(fr, fg, fb, args.brightness_srgb)
            mods.append(("brightness-srgb", f"{args.brightness_srgb:+.2f}% {src_info}"))

        elif op == "contrast" and args.contrast is not None:
            fr, fg, fb = _apply_linear_contrast_rgb(fr, fg, fb, args.contrast)
            mods.append(("contrast", f"{args.contrast:+.2f}% {src_info}"))

        elif op == "gamma" and getattr(args, "gamma", None) is not None:
            fr, fg, fb = _apply_gamma(fr, fg, fb, args.gamma)
            mods.append(("gamma-linear", f"{args.gamma:.2f} {src_info}"))

        elif op == "exposure" and getattr(args, "exposure", None) is not None:
            factor = 2.0 ** (float(args.exposure) / EXPOSURE_STOPS_SCALE)
            fr, fg, fb = _apply_linear_gain_rgb(fr, fg, fb, factor)
            mods.append(("exposure-stops", f"{args.exposure:+.2f} {src_info}"))

        elif op == "lighten" and args.lighten is not None:
            h, s, l_hsl = rgb_to_hsl(fr, fg, fb)
            amount = args.lighten / PERCENT_TO_FACTOR
            l_new = _clamp01(l_hsl + (1.0 - l_hsl) * amount)
            fr, fg, fb = hsl_to_rgb(h, s, l_new)
            mods.append(("lighten", f"+{args.lighten:.2f}% {src_info}"))

        elif op == "darken" and args.darken is not None:
            h, s, l_hsl = rgb_to_hsl(fr, fg, fb)
            amount = args.darken / PERCENT_TO_FACTOR
            l_new = _clamp01(l_hsl * (1.0 - amount))
            fr, fg, fb = hsl_to_rgb(h, s, l_new)
            mods.append(("darken", f"-{args.darken:.2f}% {src_info}"))

        elif op == "saturate" and args.saturate is not None:
            h, s, l_hsl = rgb_to_hsl(fr, fg, fb)
            if s > SAT_EPS:
                amount = args.saturate / PERCENT_TO_FACTOR
                s_new = _clamp01(s + (1.0 - s) * amount)
                fr, fg, fb = hsl_to_rgb(h, s_new, l_hsl)
            mods.append(("saturate", f"+{args.saturate:.2f}% {src_info}"))

        elif op == "desaturate" and args.desaturate is not None:
            h, s, l_hsl = rgb_to_hsl(fr, fg, fb)
            amount = args.desaturate / PERCENT_TO_FACTOR
            s_new = _clamp01(s * (1.0 - amount))
            fr, fg, fb = hsl_to_rgb(h, s_new, l_hsl)
            mods.append(("desaturate", f"-{args.desaturate:.2f}% {src_info}"))

        elif op == "whiten_hwb" and getattr(args, "whiten_hwb", None) is not None:
            h, w, b_hwb = rgb_to_hwb(fr, fg, fb)
            w_new = _clamp01(w + args.whiten_hwb / PERCENT_TO_FACTOR)
            fr, fg, fb = hwb_to_rgb(h, w_new, b_hwb)
            mods.append(("whiten-hwb", f"+{args.whiten_hwb:.2f}% {src_info}"))

        elif op == "blacken_hwb" and getattr(args, "blacken_hwb", None) is not None:
            h, w, b_hwb = rgb_to_hwb(fr, fg, fb)
            b_new = _clamp01(b_hwb + args.blacken_hwb / PERCENT_TO_FACTOR)
            fr, fg, fb = hwb_to_rgb(h, w, b_new)
            mods.append(("blacken-hwb", f"+{args.blacken_hwb:.2f}% {src_info}"))

        elif op == "chroma_oklch" and getattr(args, "chroma_oklch", None) is not None:
            l_ok, c_ok, h_ok = rgb_to_oklch(fr, fg, fb)
            factor = 1.0 + (args.chroma_oklch / PERCENT_TO_FACTOR)
            c_new = max(0.0, c_ok * factor)
            fr, fg, fb = oklch_to_rgb(l_ok, c_new, h_ok)
            l_f, a_f, b_f = rgb_to_oklab(fr, fg, fb)
            fr, fg, fb = _gamut_map_oklab_to_srgb(l_f, a_f, b_f)
            mods.append(("chroma-oklch", f"{args.chroma_oklch:+.2f}% {src_info}"))

        elif op == "vibrance_oklch" and getattr(args, "vibrance_oklch", None) is not None:
            fr, fg, fb = _apply_vibrance_oklch(fr, fg, fb, args.vibrance_oklch)
            mods.append(("vibrance-oklch", f"{args.vibrance_oklch:+.2f}% {src_info}"))

        elif op == "warm_oklab" and getattr(args, "warm_oklab", None) is not None:
            l_ok, a_ok, b_ok = rgb_to_oklab(fr, fg, fb)
            fr, fg, fb = _gamut_map_oklab_to_srgb(
                l_ok,
                a_ok + args.warm_oklab / WARM_OKLAB_A_SCALE,
                b_ok + args.warm_oklab / WARM_OKLAB_B_SCALE,
            )
            mods.append(("warm-oklab", f"+{args.warm_oklab:.2f}% {src_info}"))

        elif op == "cool_oklab" and getattr(args, "cool_oklab", None) is not None:
            l_ok, a_ok, b_ok = rgb_to_oklab(fr, fg, fb)
            fr, fg, fb = _gamut_map_oklab_to_srgb(
                l_ok,
                a_ok - args.cool_oklab / WARM_OKLAB_A_SCALE,
                b_ok - args.cool_oklab / WARM_OKLAB_B_SCALE,
            )
            mods.append(("cool-oklab", f"+{args.cool_oklab:.2f}% {src_info}"))

        elif op == "posterize" and getattr(args, "posterize", None) is not None:
            fr, fg, fb = _posterize_rgb(fr, fg, fb, args.posterize)
            mods.append(
                (
                    "posterize-rgb",
                    f"{max(POSTERIZE_MIN_LEVELS, min(POSTERIZE_MAX_LEVELS, int(abs(args.posterize))))} {src_info}",
                )
            )

        elif op == "threshold" and getattr(args, "threshold", None) is not None:
            t = _clamp01(args.threshold / PERCENT_TO_FACTOR)
            y = get_luminance(int(round(fr)), int(round(fg)), int(round(fb)))
            low_hex = getattr(args, "threshold_low", None) or THRESHOLD_DEFAULT_LOW
            high_hex = getattr(args, "threshold_high", None) or THRESHOLD_DEFAULT_HIGH
            use_hex = low_hex if y < t else high_hex
            tr, tg, tb = hex_to_rgb(use_hex)
            fr, fg, fb = float(tr), float(tg), float(tb)
            mods.append(
                (
                    "threshold-luminance",
                    f"{args.threshold:.2f}% (result: #{use_hex.upper()}) {src_info}",
                )
            )

        elif op == "solarize" and getattr(args, "solarize", None) is not None:
            fr, fg, fb = _solarize_smart(fr, fg, fb, args.solarize)
            mods.append(("solarize", f"{args.solarize:.2f}% {src_info}"))

        elif op == "tint" and getattr(args, "tint", None) is not None:
            strength = getattr(args, "tint_strength", None)
            if strength is None:
                strength = TINT_DEFAULT_STRENGTH
            fr, fg, fb = _tint_oklab(fr, fg, fb, args.tint, strength)
            mods.append(
                ("tint-oklab", f"{strength:.2f}% from #{curr_hex} to #{args.tint.upper()}")
            )

        elif op == "red_channel" and args.red_channel is not None:
            fr = _clamp255(fr + args.red_channel)
            mods.append(("red-channel", f"{args.red_channel:+d} {src_info}"))

        elif op == "green_channel" and args.green_channel is not None:
            fg = _clamp255(fg + args.green_channel)
            mods.append(("green-channel", f"{args.green_channel:+d} {src_info}"))

        elif op == "blue_channel" and args.blue_channel is not None:
            fb = _clamp255(fb + args.blue_channel)
            mods.append(("blue-channel", f"{args.blue_channel:+d} {src_info}"))

        elif op == "opacity" and args.opacity is not None:
            fr, fg, fb = _apply_opacity_on_black(fr, fg, fb, args.opacity)
            mods.append(("opacity-on-black", f"{args.opacity:.2f}% {src_info}"))

        elif op == "lock_luminance" and getattr(args, "lock_luminance", False):
            l_ok, a_ok, b_ok = rgb_to_oklab(fr, fg, fb)
            fr, fg, fb = _gamut_map_oklab_to_srgb(base_l_oklab, a_ok, b_ok)
            mods.append(("lock-oklab-lightness", f"{src_info}"))

        elif op == "lock_rel_luminance" and getattr(args, "lock_rel_luminance", False):
            fr, fg, fb = _lock_relative_luminance(fr, fg, fb, base_rel_lum)
            mods.append(("lock-relative-luminance", f"{src_info}"))

        elif op == "target_rel_lum" and getattr(args, "target_rel_lum", None) is not None:
            target_Y = _clamp01(float(args.target_rel_lum))
            fr, fg, fb = _lock_relative_luminance(fr, fg, fb, target_Y)
            mods.append(("target-rel-luminance", f"{target_Y:.4f} {src_info}"))

        elif (
            op == "min_contrast"
            and getattr(args, "min_contrast_with", None)
            and getattr(args, "min_contrast", None) is not None
        ):
            fr, fg, fb, changed = _ensure_min_contrast_with(
                fr, fg, fb, args.min_contrast_with, args.min_contrast
            )
            if changed:
                mods.append(
                    (
                        "min-contrast",
                        f">={float(args.min_contrast):.2f} vs #{args.min_contrast_with.upper()} {src_info}",
                    )
                )

    ri, gi, bi = _finalize_rgb(fr, fg, fb)
    res_hex = rgb_to_hex(ri, gi, bi)
    base_hex_upper = base_hex.upper()
    is_hex_title = (
        isinstance(title, str)
        and title.startswith("#")
        and title[1:].upper() == base_hex_upper
    )

    print()
    label = "original" if is_hex_title else title
    print_color_block(base_hex, f"{BOLD_WHITE}{label}{RESET}")
    if mods:
        print()
        print_color_block(res_hex, f"{MSG_BOLD_COLORS['info']}adjusted{RESET}")

    if getattr(args, "verbose", False):
        print()

    mods_print = mods
    if getattr(args, "steps_compact", False):
        mods_print = [(label, None) for (label, val) in mods]

    _print_steps(mods_print, getattr(args, "verbose", False))
    print()


def get_adjust_parser() -> argparse.ArgumentParser:
    """Create argument parser for adjust command.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    p = HexlabArgumentParser(
        prog="hexlab adjust",
        description=(
            "hexlab adjust: advanced color manipulation\n\n"
            "by default, all operations are deterministic and applied in a fixed pipeline"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        usage=argparse.SUPPRESS,
    )
    p.add_argument(
        "usage_hack",
        nargs="?",
        help=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
    )
    original_print_help = p.print_help

    def custom_print_help(file=None):
        if file is None:
            file = sys.stdout
        print(
            "usage: hexlab adjust [-h] (-H HEX | -r | -cn NAME | -di INDEX) [OPTIONS...]",
            file=file,
        )
        print()
        original_print_help(file)

    p.print_help = custom_print_help
    input_group = p.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "-H",
        "--hex",
        type=INPUT_HANDLERS["hex"],
        help="base hex code",
    )
    input_group.add_argument(
        "-r",
        "--random",
        action="store_true",
        help="use a random base",
    )
    input_group.add_argument(
        "-cn",
        "--color-name",
        type=INPUT_HANDLERS["color_name"],
        help="base color name",
    )
    input_group.add_argument(
        "-di",
        "--decimal-index",
        type=INPUT_HANDLERS["decimal_index"],
        help=f"base decimal index",
    )
    p.add_argument(
        "-s",
        "--seed",
        type=INPUT_HANDLERS["seed"],
        help="seed for reproducibility of random",
    )
    p.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        help="log detailed pipeline steps",
    )
    p.add_argument(
        "--steps-compact",
        dest="steps_compact",
        action="store_true",
        help="show only operation names in verbose steps, hide numeric values",
    )
    p.add_argument(
        "-cp",
        "--custom-pipeline",
        action="store_true",
        help="disable fixed pipeline and apply adjustments in the order provided on CLI",
    )
    p.add_argument(
        "--list-fixed-pipeline",
        dest="list_fixed_pipeline",
        action="store_true",
        help="print the fixed pipeline order and exit",
    )

    ga = p.add_argument_group("hsl and hue")
    ga.add_argument(
        "-l",
        "--lighten",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="increase lightness (0 to 100%%)",
    )
    ga.add_argument(
        "-d",
        "--darken",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="decrease lightness (0 to 100%%)",
    )
    ga.add_argument(
        "-sat",
        "--saturate",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="increase saturation (0 to 100%%)",
    )
    ga.add_argument(
        "-des",
        "--desaturate",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="decrease saturation (0 to 100%%)",
    )
    ga.add_argument(
        "-rot",
        "--rotate",
        type=INPUT_HANDLERS["float_signed_360"],
        metavar="N",
        help="rotate hue in HSL (-360 to 360°)",
    )
    ga.add_argument(
        "-rotl",
        "--rotate-oklch",
        dest="rotate_oklch",
        type=INPUT_HANDLERS["float_signed_360"],
        metavar="N",
        help="rotate hue in OKLCH (-360 to 360°)",
    )
    adv_group = p.add_argument_group("tone and vividness")
    bgroup = adv_group.add_mutually_exclusive_group()
    bgroup.add_argument(
        "-br",
        "--brightness",
        type=INPUT_HANDLERS["float_signed_100"],
        metavar="N",
        help="adjust linear brightness (-100 to 100%%)",
    )
    bgroup.add_argument(
        "-brs",
        "--brightness-srgb",
        dest="brightness_srgb",
        type=INPUT_HANDLERS["float_signed_100"],
        metavar="N",
        help="adjust sRGB brightness (-100 to 100%%)",
    )
    adv_group.add_argument(
        "-ct",
        "--contrast",
        type=INPUT_HANDLERS["float_signed_100"],
        metavar="N",
        help="adjust contrast (-100 to 100%%)",
    )
    adv_group.add_argument(
        "-cb",
        "--chroma-oklch",
        dest="chroma_oklch",
        type=INPUT_HANDLERS["float_signed_100"],
        metavar="N",
        help="scale chroma in OKLCH (-100 to 100%%)",
    )
    adv_group.add_argument(
        "-whiten",
        "--whiten-hwb",
        dest="whiten_hwb",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="adjust white in HWB (0 to 100%%)",
    )
    adv_group.add_argument(
        "-blacken",
        "--blacken-hwb",
        dest="blacken_hwb",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="adjust black in HWB (0 to 100%%)",
    )
    adv_group.add_argument(
        "-warm",
        "--warm-oklab",
        dest="warm_oklab",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="adjust warmth (0 to 100%%)",
    )
    adv_group.add_argument(
        "-cool",
        "--cool-oklab",
        dest="cool_oklab",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="adjust coolness (0 to 100%%)",
    )
    adv_group.add_argument(
        "-ll",
        "--lock-luminance",
        action="store_true",
        help="preserve base OKLAB lightness perceptual-L",
    )
    adv_group.add_argument(
        "-lY",
        "--lock-rel-luminance",
        dest="lock_rel_luminance",
        action="store_true",
        help="preserve base relative luminance",
    )
    adv_group.add_argument(
        "--target-rel-lum",
        dest="target_rel_lum",
        type=INPUT_HANDLERS["float"],
        metavar="Y",
        help="set absolute target relative luminance (0.0 to 1.0)",
    )
    adv_group.add_argument(
        "--min-contrast-with",
        dest="min_contrast_with",
        type=INPUT_HANDLERS["hex"],
        metavar="HEX",
        help="target hex color to ensure contrast against",
    )
    adv_group.add_argument(
        "--min-contrast",
        dest="min_contrast",
        type=INPUT_HANDLERS["float"],
        metavar="RATIO",
        help=(
            "minimum WCAG contrast ratio with --min-contrast-with, "
            "best effort within srgb gamut"
        ),
    )
    adv_group.add_argument(
        "--gamma",
        dest="gamma",
        type=INPUT_HANDLERS["float"],
        metavar="N",
        help="gamma correction in linear space (>0, typical 0.5 to 3.0)",
    )
    adv_group.add_argument(
        "--exposure",
        dest="exposure",
        type=INPUT_HANDLERS["float_signed_100"],
        metavar="N",
        help="exposure adjustment in stops (-100 to 100%%, 10%% = 1 stop)",
    )
    adv_group.add_argument(
        "-vb",
        "--vibrance-oklch",
        dest="vibrance_oklch",
        type=INPUT_HANDLERS["float_signed_100"],
        metavar="N",
        help="adjust vibrance in OKLCH, boosting low chroma (-100 to 100%%)",
    )
    filter_group = p.add_argument_group("filters and channels")
    filter_group.add_argument(
        "-g",
        "--grayscale",
        action="store_true",
        help="convert to grayscale",
    )
    filter_group.add_argument(
        "-inv",
        "--invert",
        action="store_true",
        help="invert color",
    )
    filter_group.add_argument(
        "-sep",
        "--sepia",
        action="store_true",
        help="apply sepia filter",
    )
    filter_group.add_argument(
        "-red",
        "--red-channel",
        type=INPUT_HANDLERS["int_channel"],
        metavar="N",
        help="add or subtract red (-255 to 255)",
    )
    filter_group.add_argument(
        "-green",
        "--green-channel",
        type=INPUT_HANDLERS["int_channel"],
        metavar="N",
        help="add or subtract green (-255 to 255)",
    )
    filter_group.add_argument(
        "-blue",
        "--blue-channel",
        type=INPUT_HANDLERS["int_channel"],
        metavar="N",
        help="add or subtract blue (-255 to 255)",
    )
    filter_group.add_argument(
        "-op",
        "--opacity",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="opacity over black (0 to 100%%)",
    )
    filter_group.add_argument(
        "--posterize",
        dest="posterize",
        type=INPUT_HANDLERS["int_channel"],
        metavar="N",
        help="posterize RGB channels to N levels (2 to 256)",
    )
    filter_group.add_argument(
        "--threshold",
        dest="threshold",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="binarize by relative luminance threshold (0 to 100%%)",
    )
    filter_group.add_argument(
        "--threshold-low",
        dest="threshold_low",
        type=INPUT_HANDLERS["hex"],
        metavar="HEX",
        help="low output color for --threshold (default: 000000)",
    )
    filter_group.add_argument(
        "--threshold-high",
        dest="threshold_high",
        type=INPUT_HANDLERS["hex"],
        metavar="HEX",
        help="high output color for --threshold (default: FFFFFF)",
    )
    filter_group.add_argument(
        "--solarize",
        dest="solarize",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="solarize based on perceptual lightness OKLAB-L threshold (0 to 100%%)",
    )
    filter_group.add_argument(
        "--tint",
        dest="tint",
        type=INPUT_HANDLERS["hex"],
        metavar="HEX",
        help="tint result toward given hex color using OKLAB",
    )
    filter_group.add_argument(
        "--tint-strength",
        dest="tint_strength",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="tint strength (0 to 100%%, default: 20%%)",
    )
    return p


def main() -> None:
    """Main entry point."""
    parser = get_adjust_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    handle_adjust_command(args)


if __name__ == "__main__":
    main()