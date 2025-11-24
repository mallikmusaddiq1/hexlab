#!/usr/bin/env python3

import argparse
import sys
import random
import math
import re
import os
from typing import Tuple

from .input_utils import INPUT_HANDLERS, log, HexlabArgumentParser
from .constants import COLOR_NAMES, MAX_DEC, EPS, SRGB_TO_LINEAR_TH, LINEAR_TO_SRGB_TH


def _norm_name_key(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", str(s).lower())


_norm_map = {}
for k, v in COLOR_NAMES.items():
    key = _norm_name_key(k)
    _norm_map[key] = v
COLOR_NAMES_LOOKUP = _norm_map


def _get_oklab_mid_gray() -> float:
    g = 0.18
    l_lin = 0.4122214708 * g + 0.5363325363 * g + 0.0514459929 * g
    m_lin = 0.2119034982 * g + 0.6806995451 * g + 0.1073969566 * g
    s_lin = 0.0883024619 * g + 0.2817188376 * g + 0.6299787005 * g

    l_root = l_lin ** (1.0 / 3.0)
    m_root = m_lin ** (1.0 / 3.0)
    s_root = s_lin ** (1.0 / 3.0)

    return 0.2104542553 * l_root + 0.7936177850 * m_root - 0.0040720468 * s_root


OKLAB_MID_GRAY_L = _get_oklab_mid_gray()


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
    if len(hex_code) != 6:
        raise ValueError("hex code must be 6 characters")
    return tuple(int(hex_code[i: i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(r, g, b):
    return f"{int(round(r)):02X}{int(round(g)):02X}{int(round(b)):02X}"


def _clamp01(v: float) -> float:
    if v != v:
        return 0.0
    return max(0.0, min(1.0, v))


def _clamp255(v: float) -> float:
    if v != v:
        return 0.0
    return max(0.0, min(255.0, v))


def _srgb_to_linear(c: float) -> float:
    norm = c / 255.0
    return norm / 12.92 if norm <= SRGB_TO_LINEAR_TH else ((norm + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(l: float) -> float:
    return 12.92 * l if l <= LINEAR_TO_SRGB_TH else 1.055 * (l ** (1 / 2.4)) - 0.055


def rgb_to_hsl(r: float, g: float, b: float) -> Tuple[float, float, float]:
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    cmax, cmin = max(rf, gf, bf), min(rf, gf, bf)
    delta = cmax - cmin
    l = (cmax + cmin) / 2
    if delta == 0:
        h, s = 0.0, 0.0
    else:
        denom = 1 - abs(2 * l - 1)
        s = 0.0 if abs(denom) < EPS else delta / denom
        if cmax == rf:
            h = 60 * (((gf - bf) / delta) % 6)
        elif cmax == gf:
            h = 60 * ((bf - rf) / delta + 2)
        else:
            h = 60 * ((rf - gf) / delta + 4)
        h = (h + 360) % 360
    return h, s, l


def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
    h = h % 360
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs(((h / 60) % 2) - 1))
    m = l - c / 2
    if 0 <= h < 60:
        rp, gp, bp = c, x, 0
    elif 60 <= h < 120:
        rp, gp, bp = x, c, 0
    elif 120 <= h < 180:
        rp, gp, bp = 0, c, x
    elif 180 <= h < 240:
        rp, gp, bp = 0, x, c
    elif 240 <= h < 300:
        rp, gp, bp = x, 0, c
    else:
        rp, gp, bp = c, 0, x
    return (rp + m) * 255, (gp + m) * 255, (bp + m) * 255


def rgb_to_oklab(r: float, g: float, b: float) -> Tuple[float, float, float]:
    rl, gl, bl = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)
    l = 0.4122214708 * rl + 0.5363325363 * gl + 0.0514459929 * bl
    m = 0.2119034982 * rl + 0.6806995451 * gl + 0.1073969566 * bl
    s = 0.0883024619 * rl + 0.2817188376 * gl + 0.6299787005 * bl
    l_ = l ** (1 / 3) if l >= 0 else -((-l) ** (1 / 3))
    m_ = m ** (1 / 3) if m >= 0 else -((-m) ** (1 / 3))
    s_ = s ** (1 / 3) if s >= 0 else -((-s) ** (1 / 3))
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def oklab_to_rgb(l: float, a: float, b: float) -> Tuple[float, float, float]:
    l_ = l + 0.3963377774 * a + 0.2158037573 * b
    m_ = l - 0.1055613458 * a - 0.0638541728 * b
    s_ = l - 0.0894841775 * a - 1.2914855480 * b
    l3, m3, s3 = l_ ** 3, m_ ** 3, s_ ** 3
    rl = 4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
    gl = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
    bl = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3
    return _linear_to_srgb(rl) * 255, _linear_to_srgb(gl) * 255, _linear_to_srgb(bl) * 255


def _gamut_map_oklab_to_srgb(l: float, a: float, b: float) -> Tuple[float, float, float]:
    fr, fg, fb = oklab_to_rgb(l, a, b)
    if -0.5 <= fr <= 255.5 and -0.5 <= fg <= 255.5 and -0.5 <= fb <= 255.5:
        return _clamp255(fr), _clamp255(fg), _clamp255(fb)

    C = math.hypot(a, b)
    if C < EPS:
        return _clamp255(fr), _clamp255(fg), _clamp255(fb)

    h_rad = math.atan2(b, a)
    low = 0.0
    high = C
    best_rgb = (fr, fg, fb)

    for _ in range(10):
        mid_C = (low + high) / 2.0
        new_a = mid_C * math.cos(h_rad)
        new_b = mid_C * math.sin(h_rad)
        tr, tg, tb = oklab_to_rgb(l, new_a, new_b)
        if -0.5 <= tr <= 255.5 and -0.5 <= tg <= 255.5 and -0.5 <= tb <= 255.5:
            best_rgb = (tr, tg, tb)
            low = mid_C
        else:
            high = mid_C

    return _clamp255(best_rgb[0]), _clamp255(best_rgb[1]), _clamp255(best_rgb[2])


def rgb_to_oklch(r: float, g: float, b: float) -> Tuple[float, float, float]:
    l, a, bk = rgb_to_oklab(r, g, b)
    return l, math.hypot(a, bk), math.degrees(math.atan2(bk, a)) % 360


def oklch_to_rgb(l: float, c: float, h: float) -> Tuple[float, float, float]:
    hrad = math.radians(h)
    return oklab_to_rgb(l, c * math.cos(hrad), c * math.sin(hrad))


def rgb_to_hsv(r: float, g: float, b: float) -> Tuple[float, float, float]:
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    cmax, cmin = max(rf, gf, bf), min(rf, gf, bf)
    delta = cmax - cmin
    v = cmax
    s = (delta / v) if v != 0 else 0.0
    if delta == 0:
        h = 0.0
    elif cmax == rf:
        h = (60 * ((gf - bf) / delta) + 360) % 360
    elif cmax == gf:
        h = (60 * ((bf - rf) / delta) + 120) % 360
    else:
        h = (60 * ((rf - gf) / delta) + 240) % 360
    return h, s, v


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    h %= 360
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if 0 <= h < 60:
        rp, gp, bp = c, x, 0
    elif 60 <= h < 120:
        rp, gp, bp = x, c, 0
    elif 120 <= h < 180:
        rp, gp, bp = 0, c, x
    elif 180 <= h < 240:
        rp, gp, bp = 0, x, c
    elif 240 <= h < 300:
        rp, gp, bp = x, 0, c
    else:
        rp, gp, bp = c, 0, x
    return (rp + m) * 255, (gp + m) * 255, (bp + m) * 255


def rgb_to_hwb(r: float, g: float, b: float) -> Tuple[float, float, float]:
    h, s, v = rgb_to_hsv(r, g, b)
    w = (1 - s) * v
    bk = 1 - v
    return h, w, bk


def hwb_to_rgb(h: float, w: float, b: float) -> Tuple[float, float, float]:
    w, b = _clamp01(w), _clamp01(b)
    if w + b >= 1:
        gray = w / (w + b) * 255
        return gray, gray, gray
    v = 1 - b
    s = 1 - (w / v)
    return hsv_to_rgb(h, s, v)


def _finalize_rgb(r: float, g: float, b: float) -> Tuple[int, int, int]:
    l, a, bk = rgb_to_oklab(r, g, b)
    fr, fg, fb = _gamut_map_oklab_to_srgb(l, a, bk)

    return (
        max(0, min(255, int(round(fr)))),
        max(0, min(255, int(round(fg)))),
        max(0, min(255, int(round(fb)))),
    )


def print_block(hex_code: str, label: str) -> None:
    r, g, b = hex_to_rgb(hex_code)
    print(f"{label:<17} :   \033[48;2;{r};{g};{b}m                \033[0m  #{hex_code}")


def _apply_linear_gain_rgb(fr: float, fg: float, fb: float, factor: float) -> Tuple[float, float, float]:
    rl, gl, bl = _srgb_to_linear(fr), _srgb_to_linear(fg), _srgb_to_linear(fb)
    rl = _clamp01(rl * factor)
    gl = _clamp01(gl * factor)
    bl = _clamp01(bl * factor)
    return _linear_to_srgb(rl) * 255.0, _linear_to_srgb(gl) * 255.0, _linear_to_srgb(bl) * 255.0


def _apply_srgb_brightness(fr: float, fg: float, fb: float, amount: float) -> Tuple[float, float, float]:
    factor = 1.0 + (amount / 100.0)
    fr = _clamp255(fr * factor)
    fg = _clamp255(fg * factor)
    fb = _clamp255(fb * factor)
    return fr, fg, fb


def _apply_linear_contrast_rgb(fr: float, fg: float, fb: float, contrast_amount: float) -> Tuple[float, float, float]:
    c = max(-100.0, min(100.0, float(contrast_amount)))
    if abs(c) < 1e-8:
        return fr, fg, fb
    l_ok, a_ok, b_ok = rgb_to_oklab(fr, fg, fb)
    k = 1.0 + (c / 100.0)
    l_mid = OKLAB_MID_GRAY_L
    l_new = l_mid + (l_ok - l_mid) * k
    l_new = _clamp01(l_new)
    fr2, fg2, fb2 = oklab_to_rgb(l_new, a_ok, b_ok)
    return _clamp255(fr2), _clamp255(fg2), _clamp255(fb2)


def _apply_opacity_on_black(fr: float, fg: float, fb: float, opacity_percent: float) -> Tuple[float, float, float]:
    alpha = _clamp01(opacity_percent / 100.0)
    rl, gl, bl = _srgb_to_linear(fr), _srgb_to_linear(fg), _srgb_to_linear(fb)
    rl *= alpha
    gl *= alpha
    bl *= alpha
    return _linear_to_srgb(rl) * 255.0, _linear_to_srgb(gl) * 255.0, _linear_to_srgb(bl) * 255.0


def _relative_luminance(fr: float, fg: float, fb: float) -> float:
    rl = _srgb_to_linear(fr)
    gl = _srgb_to_linear(fg)
    bl = _srgb_to_linear(fb)
    return 0.2126 * rl + 0.7152 * gl + 0.0722 * bl


def _lock_relative_luminance(fr: float, fg: float, fb: float, base_Y: float) -> Tuple[float, float, float]:
    curr_Y = _relative_luminance(fr, fg, fb)
    if curr_Y <= 0.0 or base_Y <= 0.0 or abs(curr_Y - base_Y) < 1e-9:
        return fr, fg, fb
    scale = base_Y / curr_Y
    rl = _srgb_to_linear(fr) * scale
    gl = _srgb_to_linear(fg) * scale
    bl = _srgb_to_linear(fb) * scale
    rl = _clamp01(rl)
    gl = _clamp01(gl)
    bl = _clamp01(bl)
    fr_new = _linear_to_srgb(rl) * 255.0
    fg_new = _linear_to_srgb(gl) * 255.0
    fb_new = _linear_to_srgb(bl) * 255.0
    return _clamp255(fr_new), _clamp255(fg_new), _clamp255(fb_new)


def _apply_gamma(fr: float, fg: float, fb: float, gamma: float) -> Tuple[float, float, float]:
    if gamma <= 0.0:
        return fr, fg, fb
    rl = _srgb_to_linear(fr)
    gl = _srgb_to_linear(fg)
    bl = _srgb_to_linear(fb)
    inv_gamma = 1.0 / gamma
    rl = _clamp01(rl ** inv_gamma)
    gl = _clamp01(gl ** inv_gamma)
    bl = _clamp01(bl ** inv_gamma)
    return _linear_to_srgb(rl) * 255.0, _linear_to_srgb(gl) * 255.0, _linear_to_srgb(bl) * 255.0


def _apply_vibrance_oklch(fr: float, fg: float, fb: float, amount: float) -> Tuple[float, float, float]:
    l_ok, c_ok, h_ok = rgb_to_oklch(fr, fg, fb)
    if c_ok <= 0.0:
        return fr, fg, fb
    v = amount / 100.0
    c_norm = min(c_ok / 0.4, 1.0)
    if v > 0.0:
        scale = 1.0 + v * (1.0 - c_norm)
    else:
        scale = 1.0 + v * c_norm
    if scale < 0.0:
        scale = 0.0
    c_new = c_ok * scale
    fr2, fg2, fb2 = oklch_to_rgb(l_ok, c_new, h_ok)
    l_final, a_final, b_final = rgb_to_oklab(fr2, fg2, fb2)
    return _gamut_map_oklab_to_srgb(l_final, a_final, b_final)


def _posterize_rgb(fr: float, fg: float, fb: float, levels: int) -> Tuple[float, float, float]:
    levels = max(2, min(256, int(abs(levels))))
    step = 255.0 / float(levels - 1)
    fr2 = round(fr / step) * step
    fg2 = round(fg / step) * step
    fb2 = round(fb / step) * step
    return _clamp255(fr2), _clamp255(fg2), _clamp255(fb2)


def _solarize_smart(fr: float, fg: float, fb: float, threshold_percent: float) -> Tuple[float, float, float]:
    t_perceptual = _clamp01(threshold_percent / 100.0)
    l_ok, _, _ = rgb_to_oklab(fr, fg, fb)
    rl, gl, bl = _srgb_to_linear(fr), _srgb_to_linear(fg), _srgb_to_linear(fb)
    if l_ok > t_perceptual:
        rl = 1.0 - rl
        gl = 1.0 - gl
        bl = 1.0 - bl
    fr2 = _linear_to_srgb(rl) * 255.0
    fg2 = _linear_to_srgb(gl) * 255.0
    fb2 = _linear_to_srgb(bl) * 255.0
    return _clamp255(fr2), _clamp255(fg2), _clamp255(fb2)


def _tint_oklab(fr: float, fg: float, fb: float, tint_hex: str, strength_percent: float) -> Tuple[float, float, float]:
    tr, tg, tb = hex_to_rgb(tint_hex)
    l1, a1, b1 = rgb_to_oklab(fr, fg, fb)
    l2, a2, b2 = rgb_to_oklab(float(tr), float(tg), float(tb))
    alpha = _clamp01(strength_percent / 100.0)
    l = l1 * (1.0 - alpha) + l2 * alpha
    a = a1 * (1.0 - alpha) + a2 * alpha
    b = b1 * (1.0 - alpha) + b2 * alpha
    return _gamut_map_oklab_to_srgb(l, a, b)


def _wcag_contrast_ratio_from_rgb(fr: float, fg: float, fb: float, br: float, bg: float, bb: float) -> float:
    y1 = _relative_luminance(fr, fg, fb)
    y2 = _relative_luminance(br, bg, bb)
    if y1 < y2:
        y1, y2 = y2, y1
    return (y1 + 0.05) / (y2 + 0.05)


def _ensure_min_contrast_with(
    fr: float,
    fg: float,
    fb: float,
    bg_hex: str,
    min_ratio: float,
) -> Tuple[float, float, float, bool]:
    min_ratio = max(1.0, min(21.0, float(min_ratio)))
    br_i, bg_i, bb_i = hex_to_rgb(bg_hex)
    br = float(br_i)
    bg = float(bg_i)
    bb = float(bb_i)

    current_ratio = _wcag_contrast_ratio_from_rgb(fr, fg, fb, br, bg, bb)
    if current_ratio >= min_ratio:
        return fr, fg, fb, False

    l0, a0, b0 = rgb_to_oklab(fr, fg, fb)
    base_Y = _relative_luminance(fr, fg, fb)
    bg_Y = _relative_luminance(br, bg, bb)

    Y_light = min_ratio * (bg_Y + 0.05) - 0.05
    Y_dark = (bg_Y + 0.05) / min_ratio - 0.05

    def _find_color_for_target_Y(target_Y: float):
        target_Y = _clamp01(target_Y)
        low, high = 0.0, 1.0
        for _ in range(30):
            mid = (low + high) / 2.0
            fr_mid, fg_mid, fb_mid = oklab_to_rgb(mid, a0, b0)
            fr_mid = _clamp255(fr_mid)
            fg_mid = _clamp255(fg_mid)
            fb_mid = _clamp255(fb_mid)
            y_mid = _relative_luminance(fr_mid, fg_mid, fb_mid)
            if y_mid < target_Y:
                low = mid
            else:
                high = mid
        l_final = (low + high) / 2.0
        fr_fin, fg_fin, fb_fin = _gamut_map_oklab_to_srgb(l_final, a0, b0)
        ratio = _wcag_contrast_ratio_from_rgb(fr_fin, fg_fin, fb_fin, br, bg, bb)
        return l_final, fr_fin, fg_fin, fb_fin, ratio

    candidates = []

    if 0.0 <= Y_light <= 1.0:
        l_light, fr_light, fg_light, fb_light, ratio_light = _find_color_for_target_Y(Y_light)
        if ratio_light >= min_ratio:
            candidates.append((abs(l_light - l0), l_light, fr_light, fg_light, fb_light, ratio_light))

    if 0.0 <= Y_dark <= 1.0:
        l_dark, fr_dark, fg_dark, fb_dark, ratio_dark = _find_color_for_target_Y(Y_dark)
        if ratio_dark >= min_ratio:
            candidates.append((abs(l_dark - l0), l_dark, fr_dark, fg_dark, fb_dark, ratio_dark))

    if not candidates:
        black_ratio = _wcag_contrast_ratio_from_rgb(0.0, 0.0, 0.0, br, bg, bb)
        white_ratio = _wcag_contrast_ratio_from_rgb(255.0, 255.0, 255.0, br, bg, bb)
        best_rgb = (fr, fg, fb)
        best_ratio = current_ratio
        if black_ratio >= min_ratio and black_ratio >= best_ratio:
            best_rgb = (0.0, 0.0, 0.0)
            best_ratio = black_ratio
        if white_ratio >= min_ratio and white_ratio >= best_ratio:
            best_rgb = (255.0, 255.0, 255.0)
            best_ratio = white_ratio
        if best_ratio > current_ratio:
            return best_rgb[0], best_rgb[1], best_rgb[2], True
        return fr, fg, fb, False

    candidates.sort(key=lambda x: x[0])
    _, _, fr_best, fg_best, fb_best, _ = candidates[0]
    return fr_best, fg_best, fb_best, True


def _format_steps(mods):
    parts = []
    for label, val in mods:
        if val:
            parts.append(f"{label} {val}")
        else:
            parts.append(label)
    return parts


def _print_steps(mods, verbose: bool) -> None:
    if not verbose:
        return
    if not mods:
        log("info", "steps: no adjustments applied yet")
        return
    parts = _format_steps(mods)
    log("info", "steps: " + " \u2192 ".join(parts))


def handle_adjust_command(args: argparse.Namespace) -> None:
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
        log("error", "conflicting luminance locks: use only one of --lock-luminance, --lock-rel-luminance, or --target-rel-lum")
        sys.exit(2)

    if getattr(args, "min_contrast_with", None) and locks > 0:
        log("warning", "--min-contrast-with will override previous luminance locks")

    base_hex, title = None, "original"
    if args.random:
        base_hex, title = f"{random.randint(0, MAX_DEC):06X}", "random"
    elif args.color_name:
        base_hex = _get_color_name_hex(args.color_name)
        if not base_hex:
            log("error", f"unknown name '{args.color_name}'")
            sys.exit(2)
        title = args.color_name
    elif args.hex:
        base_hex, title = args.hex, f"#{args.hex}"
    elif getattr(args, "decimal_index", None):
        base_hex, title = args.decimal_index, f"idx {int(args.decimal_index, 16)}"

    if not base_hex:
        log("error", "no input color")
        sys.exit(2)

    if (getattr(args, "min_contrast_with", None) and getattr(args, "min_contrast", None) is None) or (
        getattr(args, "min_contrast_with", None) is None and getattr(args, "min_contrast", None) is not None
    ):
        log("error", "--min-contrast-with and --min-contrast must be used together")
        sys.exit(2)

    r, g, b = hex_to_rgb(base_hex)
    fr, fg, fb = float(r), float(g), float(b)
    base_l_oklab, _, _ = rgb_to_oklab(fr, fg, fb)
    base_rel_lum = _relative_luminance(fr, fg, fb)

    mods = []

    if args.invert:
        fr, fg, fb = 255.0 - fr, 255.0 - fg, 255.0 - fb
        mods.append(("invert", None))

    if args.grayscale:
        l, a, b_ok = rgb_to_oklab(fr, fg, fb)
        fr, fg, fb = oklab_to_rgb(l, 0.0, 0.0)
        fr, fg, fb = _clamp255(fr), _clamp255(fg), _clamp255(fb)
        mods.append(("grayscale", None))

    if args.sepia:
        tr = fr * 0.393 + fg * 0.769 + fb * 0.189
        tg = fr * 0.349 + fg * 0.686 + fb * 0.168
        tb = fr * 0.272 + fg * 0.534 + fb * 0.131
        fr, fg, fb = _clamp255(tr), _clamp255(tg), _clamp255(tb)
        mods.append(("sepia", None))

    if args.rotate is not None:
        h, s, l = rgb_to_hsl(fr, fg, fb)
        fr, fg, fb = hsl_to_rgb(h + args.rotate, s, l)
        mods.append(("hue-rotate-hsl", f"{args.rotate:+.2f}deg"))

    if getattr(args, "rotate_oklch", None) is not None:
        l_ok, c_ok, h_ok = rgb_to_oklch(fr, fg, fb)
        fr, fg, fb = oklch_to_rgb(l_ok, c_ok, h_ok + args.rotate_oklch)
        fr, fg, fb = _clamp255(fr), _clamp255(fg), _clamp255(fb)
        mods.append(("hue-rotate-oklch", f"{args.rotate_oklch:+.2f}deg"))

    if args.brightness is not None:
        factor = 1.0 + (args.brightness / 100.0)
        fr, fg, fb = _apply_linear_gain_rgb(fr, fg, fb, factor)
        mods.append(("brightness-linear", f"{args.brightness:+.2f}%%"))

    if getattr(args, "brightness_srgb", None) is not None:
        fr, fg, fb = _apply_srgb_brightness(fr, fg, fb, args.brightness_srgb)
        mods.append(("brightness-srgb", f"{args.brightness_srgb:+.2f}%%"))

    if args.contrast is not None:
        fr, fg, fb = _apply_linear_contrast_rgb(fr, fg, fb, args.contrast)
        mods.append(("contrast", f"{args.contrast:+.2f}%%"))

    if getattr(args, "gamma", None) is not None:
        fr, fg, fb = _apply_gamma(fr, fg, fb, args.gamma)
        mods.append(("gamma-linear", f"{args.gamma:.3f}"))

    if getattr(args, "exposure", None) is not None:
        factor = 2.0 ** float(args.exposure)
        fr, fg, fb = _apply_linear_gain_rgb(fr, fg, fb, factor)
        mods.append(("exposure-stops", f"{args.exposure:+.3f}"))

    if any(x is not None for x in [args.lighten, args.darken, args.saturate, args.desaturate]):
        h, s, l = rgb_to_hsl(fr, fg, fb)
        if args.lighten is not None:
            amount = args.lighten / 100.0
            l = _clamp01(l + (1.0 - l) * amount)
            mods.append(("lighten", f"+{args.lighten:.2f}%%"))
        if args.darken is not None:
            amount = args.darken / 100.0
            l = _clamp01(l * (1.0 - amount))
            mods.append(("darken", f"-{args.darken:.2f}%%"))
        if args.saturate is not None:
            amount = args.saturate / 100.0
            s = _clamp01(s + (1.0 - s) * amount)
            mods.append(("saturate", f"+{args.saturate:.2f}%%"))
        if args.desaturate is not None:
            amount = args.desaturate / 100.0
            s = _clamp01(s * (1.0 - amount))
            mods.append(("desaturate", f"-{args.desaturate:.2f}%%"))
        fr, fg, fb = hsl_to_rgb(h, s, l)

    if getattr(args, "whiten_hwb", None) is not None or getattr(args, "blacken_hwb", None) is not None:
        h, w, b_val = rgb_to_hwb(fr, fg, fb)
        if getattr(args, "whiten_hwb", None) is not None:
            w = _clamp01(w + args.whiten_hwb / 100.0)
            mods.append(("whiten-hwb", f"+{args.whiten_hwb:.2f}%%"))
        if getattr(args, "blacken_hwb", None) is not None:
            b_val = _clamp01(b_val + args.blacken_hwb / 100.0)
            mods.append(("blacken-hwb", f"+{args.blacken_hwb:.2f}%%"))
        fr, fg, fb = hwb_to_rgb(h, w, b_val)

    if getattr(args, "chroma_oklch", None) is not None:
        l_ok, c_ok, h_ok = rgb_to_oklch(fr, fg, fb)
        factor = 1.0 + (args.chroma_oklch / 100.0)
        c_ok = max(0.0, c_ok * factor)
        fr, fg, fb = oklch_to_rgb(l_ok, c_ok, h_ok)
        l_f, a_f, b_f = rgb_to_oklab(fr, fg, fb)
        fr, fg, fb = _gamut_map_oklab_to_srgb(l_f, a_f, b_f)
        mods.append(("chroma-oklch", f"{args.chroma_oklch:+.2f}%%"))

    if getattr(args, "vibrance_oklch", None) is not None:
        fr, fg, fb = _apply_vibrance_oklch(fr, fg, fb, args.vibrance_oklch)
        mods.append(("vibrance-oklch", f"{args.vibrance_oklch:+.2f}%%"))

    if getattr(args, "warm_oklab", None) is not None:
        l_ok, a_ok, b_ok = rgb_to_oklab(fr, fg, fb)
        fr, fg, fb = _gamut_map_oklab_to_srgb(
            l_ok,
            a_ok + args.warm_oklab / 2000.0,
            b_ok + args.warm_oklab / 1000.0,
        )
        mods.append(("warm-oklab", f"+{args.warm_oklab:.2f}%%"))

    if getattr(args, "cool_oklab", None) is not None:
        l_ok, a_ok, b_ok = rgb_to_oklab(fr, fg, fb)
        fr, fg, fb = _gamut_map_oklab_to_srgb(
            l_ok,
            a_ok - args.cool_oklab / 2000.0,
            b_ok - args.cool_oklab / 1000.0,
        )
        mods.append(("cool-oklab", f"+{args.cool_oklab:.2f}%%"))

    if getattr(args, "posterize", None) is not None:
        fr, fg, fb = _posterize_rgb(fr, fg, fb, args.posterize)
        mods.append(("posterize-rgb", f"{max(2, min(256, int(abs(args.posterize))))}"))

    if getattr(args, "threshold", None) is not None:
        t = _clamp01(args.threshold / 100.0)
        y = _relative_luminance(fr, fg, fb)
        low_hex = getattr(args, "threshold_low", None) or "000000"
        high_hex = getattr(args, "threshold_high", None) or "FFFFFF"
        use_hex = low_hex if y < t else high_hex
        tr, tg, tb = hex_to_rgb(use_hex)
        fr, fg, fb = float(tr), float(tg), float(tb)
        mods.append(("threshold-luminance", f"{args.threshold:.2f}%%"))

    if getattr(args, "solarize", None) is not None:
        fr, fg, fb = _solarize_smart(fr, fg, fb, args.solarize)
        mods.append(("solarize", f"{args.solarize:.2f}%%"))

    if getattr(args, "tint", None) is not None:
        strength = getattr(args, "tint_strength", None)
        if strength is None:
            strength = 20.0
        fr, fg, fb = _tint_oklab(fr, fg, fb, args.tint, strength)
        mods.append(("tint-oklab", f"{strength:.2f}%% to #{args.tint.upper()}"))

    if args.red_channel is not None:
        fr = _clamp255(fr + args.red_channel)
        mods.append(("red-channel", f"{args.red_channel:+d}"))

    if args.green_channel is not None:
        fg = _clamp255(fg + args.green_channel)
        mods.append(("green-channel", f"{args.green_channel:+d}"))

    if args.blue_channel is not None:
        fb = _clamp255(fb + args.blue_channel)
        mods.append(("blue-channel", f"{args.blue_channel:+d}"))

    if args.opacity is not None:
        fr, fg, fb = _apply_opacity_on_black(fr, fg, fb, args.opacity)
        mods.append(("opacity-on-black", f"{args.opacity:.2f}%%"))

    if getattr(args, "lock_luminance", False):
        l_ok, a_ok, b_ok = rgb_to_oklab(fr, fg, fb)
        fr, fg, fb = _gamut_map_oklab_to_srgb(base_l_oklab, a_ok, b_ok)
        mods.append(("lock-oklab-lightness", None))

    if getattr(args, "lock_rel_luminance", False):
        fr, fg, fb = _lock_relative_luminance(fr, fg, fb, base_rel_lum)
        mods.append(("lock-relative-luminance", None))

    if getattr(args, "target_rel_lum", None) is not None:
        target_Y = _clamp01(float(args.target_rel_lum))
        fr, fg, fb = _lock_relative_luminance(fr, fg, fb, target_Y)
        mods.append(("target-rel-luminance", f"{target_Y:.4f}"))

    if getattr(args, "min_contrast_with", None) and getattr(args, "min_contrast", None) is not None:
        fr, fg, fb, changed = _ensure_min_contrast_with(fr, fg, fb, args.min_contrast_with, args.min_contrast)
        if changed:
            mods.append(("min-contrast", f">={float(args.min_contrast):.2f} vs #{args.min_contrast_with.upper()}"))

    ri, gi, bi = _finalize_rgb(fr, fg, fb)
    res_hex = rgb_to_hex(ri, gi, bi)
    base_hex_upper = base_hex.upper()
    is_hex_title = isinstance(title, str) and title.startswith("#") and title[1:].upper() == base_hex_upper

    print()
    base_label = "original" if is_hex_title else title
    print_block(base_hex, base_label)
    if mods:
        print_block(res_hex, "adjusted")
    print()

    mods_print = mods
    if getattr(args, "steps_compact", False):
        mods_print = [(label, None) for (label, val) in mods]

    _print_steps(mods_print, getattr(args, "verbose", False))
    print()


def get_adjust_parser() -> argparse.ArgumentParser:
    p = HexlabArgumentParser(
        prog="hexlab adjust",
        description=(
            "hexlab adjust: advanced color manipulation\n\n"
            "all operations are deterministic and applied in a fixed pipeline"
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
        print("usage: hexlab adjust [-h] (-H HEX | -r | -cn NAME | -di INDEX) [OPTIONS...]", file=file)
        print("")
        original_print_help(file)

    p.print_help = custom_print_help
    input_group = p.add_mutually_exclusive_group(required=True)
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
        help="use a random base color",
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
        help="base decimal index (0 to MAX_DEC)",
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
    ga = p.add_argument_group("HSL and hue")
    ga.add_argument(
        "-l",
        "--lighten",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="increase lightness (0-100%%)",
    )
    ga.add_argument(
        "-d",
        "--darken",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="decrease lightness (0-100%%)",
    )
    ga.add_argument(
        "-sat",
        "--saturate",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="increase saturation (0-100%%)",
    )
    ga.add_argument(
        "-des",
        "--desaturate",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="decrease saturation (0-100%%)",
    )
    ga.add_argument(
        "-rot",
        "--rotate",
        type=INPUT_HANDLERS["float"],
        metavar="N",
        help="rotate hue in HSL (-360 to 360 degrees)",
    )
    ga.add_argument(
        "-rotl",
        "--rotate-oklch",
        dest="rotate_oklch",
        type=INPUT_HANDLERS["float"],
        metavar="N",
        help="rotate hue in OKLCH (-360 to 360 degrees)",
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
        help="adjust white in HWB (0-100%%)",
    )
    adv_group.add_argument(
        "-blacken",
        "--blacken-hwb",
        dest="blacken_hwb",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="adjust black in HWB (0-100%%)",
    )
    adv_group.add_argument(
        "-warm",
        "--warm-oklab",
        dest="warm_oklab",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="adjust warmth (0-100%%)",
    )
    adv_group.add_argument(
        "-cool",
        "--cool-oklab",
        dest="cool_oklab",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="adjust coolness (0-100%%)",
    )
    adv_group.add_argument(
        "-ll",
        "--lock-luminance",
        action="store_true",
        help="preserve base OKLAB lightness (perceptual L)",
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
        help="set absolute target relative luminance (0.0-1.0)",
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
        help="minimum WCAG contrast ratio with --min-contrast-with (best effort within sRGB gamut)",
    )
    adv_group.add_argument(
        "--gamma",
        dest="gamma",
        type=INPUT_HANDLERS["float"],
        metavar="N",
        help="gamma correction in linear space (>0, typical 0.5-3.0)",
    )
    adv_group.add_argument(
        "--exposure",
        dest="exposure",
        type=INPUT_HANDLERS["float"],
        metavar="N",
        help="exposure adjustment in stops (negative or positive)",
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
        help="opacity over black (0-100%%)",
    )
    filter_group.add_argument(
        "--posterize",
        dest="posterize",
        type=INPUT_HANDLERS["int_channel"],
        metavar="N",
        help="posterize RGB channels to N levels (2-256)",
    )
    filter_group.add_argument(
        "--threshold",
        dest="threshold",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="binarize by relative luminance threshold (0-100%%)",
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
        help="solarize based on perceptual lightness (OKLab L) threshold (0-100%%)",
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
        help="tint strength (0-100%%, default: 20%%)",
    )
    out_group = p.add_argument_group("output")
    out_group.add_argument(
        "--steps-compact",
        dest="steps_compact",
        action="store_true",
        help="show only operation names in verbose steps, hide numeric values",
    )
    return p


def main() -> None:
    parser = get_adjust_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    handle_adjust_command(args)


if __name__ == "__main__":
    main()