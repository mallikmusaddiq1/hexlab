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
from ..constants.constants import EPS, MAX_DEC
from ..utils.clamping import _clamp01, _clamp255
from ..utils.color_names_handler import (
    get_title_for_hex,
    resolve_color_name_or_exit,
)
from ..utils.hexlab_logger import log
from ..utils.input_handler import INPUT_HANDLERS, HexlabArgumentParser
from ..utils.print_color_block import print_color_block
from ..utils.truecolor import ensure_truecolor


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


def _oklab_to_rgb_unclamped(l: float, a: float, b: float) -> Tuple[float, float, float]:
    l_ = l + 0.3963377774 * a + 0.2158037573 * b
    m_ = l - 0.1055613458 * a - 0.0638541728 * b
    s_ = l - 0.0894841775 * a - 1.2914855480 * b

    l3 = l_ ** 3
    m3 = m_ ** 3
    s3 = s_ ** 3

    rl = 4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
    gl = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
    bl = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3

    return (
        _linear_to_srgb(rl) * 255.0,
        _linear_to_srgb(gl) * 255.0,
        _linear_to_srgb(bl) * 255.0,
    )


def _gamut_map_oklab_to_srgb(l: float, a: float, b: float) -> Tuple[float, float, float]:
    fr, fg, fb = _oklab_to_rgb_unclamped(l, a, b)

    if -0.5 <= fr <= 255.5 and -0.5 <= fg <= 255.5 and -0.5 <= fb <= 255.5:
        return _clamp255(fr), _clamp255(fg), _clamp255(fb)

    C = math.hypot(a, b)
    if C < EPS:
        return _clamp255(fr), _clamp255(fg), _clamp255(fb)

    h_rad = math.atan2(b, a)
    low = 0.0
    high = C
    best_rgb = (fr, fg, fb)

    for _ in range(20):
        mid_C = (low + high) / 2.0
        new_a = mid_C * math.cos(h_rad)
        new_b = mid_C * math.sin(h_rad)
        tr, tg, tb = _oklab_to_rgb_unclamped(l, new_a, new_b)

        if -0.5 <= tr <= 255.5 and -0.5 <= tg <= 255.5 and -0.5 <= tb <= 255.5:
            best_rgb = (tr, tg, tb)
            low = mid_C
        else:
            high = mid_C

    return _clamp255(best_rgb[0]), _clamp255(best_rgb[1]), _clamp255(best_rgb[2])


def _finalize_rgb(fr: float, fg: float, fb: float) -> Tuple[int, int, int]:
    l, a, bk = rgb_to_oklab(fr, fg, fb)
    fr_mapped, fg_mapped, fb_mapped = _gamut_map_oklab_to_srgb(l, a, bk)

    return (
        max(0, min(255, int(round(fr_mapped)))),
        max(0, min(255, int(round(fg_mapped)))),
        max(0, min(255, int(round(fb_mapped)))),
    )


def _apply_linear_gain_rgb(
    fr: float, fg: float, fb: float, factor: float
) -> Tuple[float, float, float]:
    rl, gl, bl = _srgb_to_linear(fr), _srgb_to_linear(fg), _srgb_to_linear(fb)
    rl = _clamp01(rl * factor)
    gl = _clamp01(gl * factor)
    bl = _clamp01(bl * factor)
    return (
        _linear_to_srgb(rl) * 255.0,
        _linear_to_srgb(gl) * 255.0,
        _linear_to_srgb(bl) * 255.0,
    )


def _apply_srgb_brightness(
    fr: float, fg: float, fb: float, amount: float
) -> Tuple[float, float, float]:
    factor = 1.0 + (amount / 100.0)
    fr = _clamp255(fr * factor)
    fg = _clamp255(fg * factor)
    fb = _clamp255(fb * factor)
    return fr, fg, fb


def _apply_linear_contrast_rgb(
    fr: float, fg: float, fb: float, contrast_amount: float
) -> Tuple[float, float, float]:
    c = max(-100.0, min(100.0, float(contrast_amount)))
    if abs(c) < 1e-8:
        return fr, fg, fb
    l_ok, a_ok, b_ok = rgb_to_oklab(fr, fg, fb)
    k = 1.0 + (c / 100.0)
    l_mid = OKLAB_MID_GRAY_L
    l_new = l_mid + (l_ok - l_mid) * k
    l_new = _clamp01(l_new)
    return _gamut_map_oklab_to_srgb(l_new, a_ok, b_ok)


def _apply_opacity_on_black(
    fr: float, fg: float, fb: float, opacity_percent: float
) -> Tuple[float, float, float]:
    alpha = _clamp01(opacity_percent / 100.0)
    rl, gl, bl = _srgb_to_linear(fr), _srgb_to_linear(fg), _srgb_to_linear(fb)
    rl *= alpha
    gl *= alpha
    bl *= alpha
    return (
        _linear_to_srgb(rl) * 255.0,
        _linear_to_srgb(gl) * 255.0,
        _linear_to_srgb(bl) * 255.0,
    )


def _lock_relative_luminance(
    fr: float, fg: float, fb: float, base_Y: float
) -> Tuple[float, float, float]:
    curr_Y = get_luminance(int(round(fr)), int(round(fg)), int(round(fb)))
    if curr_Y <= 0.0 or base_Y <= 0.0 or abs(curr_Y - base_Y) < 1e-9:
        return fr, fg, fb
    scale = base_Y / curr_Y
    rl = _srgb_to_linear(fr) * scale
    gl = _srgb_to_linear(fg) * scale
    bl = _srgb_to_linear(fb) * scale
    rl = _clamp01(rl)
    gl = _clamp01(gl)
    bl = _clamp01(bl)
    return (
        _linear_to_srgb(rl) * 255.0,
        _linear_to_srgb(gl) * 255.0,
        _linear_to_srgb(bl) * 255.0,
    )


def _apply_gamma(
    fr: float, fg: float, fb: float, gamma: float
) -> Tuple[float, float, float]:
    if gamma <= 0.0:
        return fr, fg, fb
    rl = _srgb_to_linear(fr)
    gl = _srgb_to_linear(fg)
    bl = _srgb_to_linear(fb)
    inv_gamma = 1.0 / gamma
    rl = _clamp01(rl ** inv_gamma)
    gl = _clamp01(gl ** inv_gamma)
    bl = _clamp01(bl ** inv_gamma)
    return (
        _linear_to_srgb(rl) * 255.0,
        _linear_to_srgb(gl) * 255.0,
        _linear_to_srgb(bl) * 255.0,
    )


def _apply_vibrance_oklch(
    fr: float, fg: float, fb: float, amount: float
) -> Tuple[float, float, float]:
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


def _posterize_rgb(
    fr: float, fg: float, fb: float, levels: int
) -> Tuple[float, float, float]:
    levels = max(2, min(256, int(abs(levels))))
    step = 255.0 / float(levels - 1)
    fr2 = round(fr / step) * step
    fg2 = round(fg / step) * step
    fb2 = round(fb / step) * step
    return _clamp255(fr2), _clamp255(fg2), _clamp255(fb2)


def _solarize_smart(
    fr: float, fg: float, fb: float, threshold_percent: float
) -> Tuple[float, float, float]:
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


def _tint_oklab(
    fr: float, fg: float, fb: float, tint_hex: str, strength_percent: float
) -> Tuple[float, float, float]:
    tr, tg, tb = hex_to_rgb(tint_hex)
    l1, a1, b1 = rgb_to_oklab(fr, fg, fb)
    l2, a2, b2 = rgb_to_oklab(float(tr), float(tg), float(tb))
    alpha = _clamp01(strength_percent / 100.0)
    l = l1 * (1.0 - alpha) + l2 * alpha
    a = a1 * (1.0 - alpha) + a2 * alpha
    b = b1 * (1.0 - alpha) + b2 * alpha
    return _gamut_map_oklab_to_srgb(l, a, b)


def _ensure_min_contrast_with(
    fr: float, fg: float, fb: float, bg_hex: str, min_ratio: float
) -> Tuple[float, float, float, bool]:
    min_ratio = max(1.0, min(21.0, float(min_ratio)))
    br_i, bg_i, bb_i = hex_to_rgb(bg_hex)
    br, bg, bb = float(br_i), float(bg_i), float(bb_i)

    current_ratio = _wcag_contrast_ratio_from_rgb(fr, fg, fb, br, bg, bb)
    if current_ratio >= min_ratio:
        return fr, fg, fb, False

    l0, a0, b0 = rgb_to_oklab(fr, fg, fb)
    bg_Y = get_luminance(br_i, bg_i, bb_i)

    Y_light = min_ratio * (bg_Y + 0.05) - 0.05
    Y_dark = (bg_Y + 0.05) / min_ratio - 0.05

    def _find_color_for_target_Y(target_Y: float):
        target_Y = _clamp01(target_Y)
        low, high = 0.0, 1.0
        for _ in range(30):
            mid = (low + high) / 2.0
            fr_mid, fg_mid, fb_mid = _oklab_to_rgb_unclamped(mid, a0, b0)

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

    if 0.0 <= Y_light <= 1.0:
        l_light, fr_light, fg_light, fb_light, ratio_light = _find_color_for_target_Y(Y_light)
        if ratio_light >= min_ratio:
            candidates.append(
                (abs(l_light - l0), l_light, fr_light, fg_light, fb_light, ratio_light)
            )

    if 0.0 <= Y_dark <= 1.0:
        l_dark, fr_dark, fg_dark, fb_dark, ratio_dark = _find_color_for_target_Y(Y_dark)
        if ratio_dark >= min_ratio:
            candidates.append(
                (abs(l_dark - l0), l_dark, fr_dark, fg_dark, fb_dark, ratio_dark)
            )

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
    log("info", "steps: " + " -> ".join(parts))


def _sanitize_rgb(fr: float, fg: float, fb: float) -> Tuple[float, float, float]:
    if not (math.isfinite(fr) and math.isfinite(fg) and math.isfinite(fb)):
        fr, fg, fb = 0.0, 0.0, 0.0
    return _clamp255(fr), _clamp255(fg), _clamp255(fb)


def _get_custom_pipeline_order(parser) -> list:
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
        # note: map min-contrast flags to "min_contrast" op (handled below)
        "min_contrast_with": "min_contrast",
        "min_contrast": "min_contrast",
        # include alias dests if any
    }

    order = []
    seen = set()
    # iterate through CLI args (skip program name)
    for arg in sys.argv[1:]:
        if not arg.startswith('-'):
            continue
        key = arg.split('=')[0]
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
            "conflicting luminance locks: use only one of --lock-luminance, "
            "--lock-rel-luminance, or --target-rel-lum"
        )
        sys.exit(2)

    if getattr(args, "min_contrast_with", None) and locks > 0:
        log("warning", "--min-contrast-with will override previous luminance locks")

    pipeline = [
        # tonal foundation (linear / OKLab-aware ops)
        "exposure",
        "gamma",
        "brightness",
        "contrast",

        # hue/lightness HSL & OKLCH
        "rotate",
        "rotate_oklch",
        "lighten",
        "darken",

        # saturation / chroma / vibrance
        "saturate",
        "desaturate",
        "chroma_oklch",
        "vibrance_oklch",

        # HWB adjustments
        "whiten_hwb",
        "blacken_hwb",

        # warm/cool/tint (OKLab)
        "warm_oklab",
        "cool_oklab",
        "tint",

        # channel arithmetic
        "red_channel",
        "green_channel",
        "blue_channel",

        # destructive / stylized ops (late)
        "posterize",
        "threshold",
        "solarize",
        "sepia",
        "grayscale",
        "invert",

        # luminance / contrast locks & accessibility (near-final)
        "lock_luminance",
        "lock_rel_luminance",
        "target_rel_lum",
        "min_contrast",

        # final compositing
        "opacity",
    ]

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
        if title.lower() == 'unknown':
            title = args.color_name
    elif args.hex:
        base_hex, title = args.hex, get_title_for_hex(args.hex)
    elif getattr(args, "decimal_index", None):
        base_hex, title = args.decimal_index, f"idx {int(args.decimal_index, 16)}"

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
        if op == "invert" and args.invert:
            fr, fg, fb = 255.0 - fr, 255.0 - fg, 255.0 - fb
            mods.append(("invert", None))

        elif op == "grayscale" and args.grayscale:
            l, a, b_ok = rgb_to_oklab(fr, fg, fb)
            fr, fg, fb = oklab_to_rgb(l, 0.0, 0.0)
            fr, fg, fb = _clamp255(fr), _clamp255(fg), _clamp255(fb)
            mods.append(("grayscale", None))

        elif op == "sepia" and args.sepia:
            tr = fr * 0.393 + fg * 0.769 + fb * 0.189
            tg = fr * 0.349 + fg * 0.686 + fb * 0.168
            tb = fr * 0.272 + fg * 0.534 + fb * 0.131
            fr, fg, fb = _clamp255(tr), _clamp255(tg), _clamp255(tb)
            mods.append(("sepia", None))

        elif op == "rotate" and args.rotate is not None:
            h, s, l = rgb_to_hsl(fr, fg, fb)
            fr, fg, fb = hsl_to_rgb(h + args.rotate, s, l)
            mods.append(("hue-rotate-hsl", f"{args.rotate:+.2f}deg"))

        elif op == "rotate_oklch" and getattr(args, "rotate_oklch", None) is not None:
            l_ok, c_ok, h_ok = rgb_to_oklch(fr, fg, fb)
            fr, fg, fb = oklch_to_rgb(l_ok, c_ok, h_ok + args.rotate_oklch)
            fr, fg, fb = _clamp255(fr), _clamp255(fg), _clamp255(fb)
            mods.append(("hue-rotate-oklch", f"{args.rotate_oklch:+.2f}deg"))

        elif op == "brightness" and args.brightness is not None:
            factor = 1.0 + (args.brightness / 100.0)
            fr, fg, fb = _apply_linear_gain_rgb(fr, fg, fb, factor)
            mods.append(("brightness-linear", f"{args.brightness:+.2f}%%"))

        elif op == "brightness_srgb" and getattr(args, "brightness_srgb", None) is not None:
            fr, fg, fb = _apply_srgb_brightness(fr, fg, fb, args.brightness_srgb)
            mods.append(("brightness-srgb", f"{args.brightness_srgb:+.2f}%%"))

        elif op == "contrast" and args.contrast is not None:
            fr, fg, fb = _apply_linear_contrast_rgb(fr, fg, fb, args.contrast)
            mods.append(("contrast", f"{args.contrast:+.2f}%%"))

        elif op == "gamma" and getattr(args, "gamma", None) is not None:
            fr, fg, fb = _apply_gamma(fr, fg, fb, args.gamma)
            mods.append(("gamma-linear", f"{args.gamma:.3f}"))

        elif op == "exposure" and getattr(args, "exposure", None) is not None:
            factor = 2.0 ** float(args.exposure)
            fr, fg, fb = _apply_linear_gain_rgb(fr, fg, fb, factor)
            mods.append(("exposure-stops", f"{args.exposure:+.3f}"))

        elif op == "lighten" and args.lighten is not None:
            h, s, l = rgb_to_hsl(fr, fg, fb)
            amount = args.lighten / 100.0
            l = _clamp01(l + (1.0 - l) * amount)
            fr, fg, fb = hsl_to_rgb(h, s, l)
            mods.append(("lighten", f"+{args.lighten:.2f}%%"))

        elif op == "darken" and args.darken is not None:
            h, s, l = rgb_to_hsl(fr, fg, fb)
            amount = args.darken / 100.0
            l = _clamp01(l * (1.0 - amount))
            fr, fg, fb = hsl_to_rgb(h, s, l)
            mods.append(("darken", f"-{args.darken:.2f}%%"))

        elif op == "saturate" and args.saturate is not None:
            h, s, l = rgb_to_hsl(fr, fg, fb)
            amount = args.saturate / 100.0
            s = _clamp01(s + (1.0 - s) * amount)
            fr, fg, fb = hsl_to_rgb(h, s, l)
            mods.append(("saturate", f"+{args.saturate:.2f}%%"))

        elif op == "desaturate" and args.desaturate is not None:
            h, s, l = rgb_to_hsl(fr, fg, fb)
            amount = args.desaturate / 100.0
            s = _clamp01(s * (1.0 - amount))
            fr, fg, fb = hsl_to_rgb(h, s, l)
            mods.append(("desaturate", f"-{args.desaturate:.2f}%%"))

        elif op == "whiten_hwb" and getattr(args, "whiten_hwb", None) is not None:
            h, w, b_val = rgb_to_hwb(fr, fg, fb)
            w = _clamp01(w + args.whiten_hwb / 100.0)
            fr, fg, fb = hwb_to_rgb(h, w, b_val)
            mods.append(("whiten-hwb", f"+{args.whiten_hwb:.2f}%%"))

        elif op == "blacken_hwb" and getattr(args, "blacken_hwb", None) is not None:
            h, w, b_val = rgb_to_hwb(fr, fg, fb)
            b_val = _clamp01(b_val + args.blacken_hwb / 100.0)
            fr, fg, fb = hwb_to_rgb(h, w, b_val)
            mods.append(("blacken-hwb", f"+{args.blacken_hwb:.2f}%%"))

        elif op == "chroma_oklch" and getattr(args, "chroma_oklch", None) is not None:
            l_ok, c_ok, h_ok = rgb_to_oklch(fr, fg, fb)
            factor = 1.0 + (args.chroma_oklch / 100.0)
            c_ok = max(0.0, c_ok * factor)
            fr, fg, fb = oklch_to_rgb(l_ok, c_ok, h_ok)
            l_f, a_f, b_f = rgb_to_oklab(fr, fg, fb)
            fr, fg, fb = _gamut_map_oklab_to_srgb(l_f, a_f, b_f)
            mods.append(("chroma-oklch", f"{args.chroma_oklch:+.2f}%%"))

        elif op == "vibrance_oklch" and getattr(args, "vibrance_oklch", None) is not None:
            fr, fg, fb = _apply_vibrance_oklch(fr, fg, fb, args.vibrance_oklch)
            mods.append(("vibrance-oklch", f"{args.vibrance_oklch:+.2f}%%"))

        elif op == "warm_oklab" and getattr(args, "warm_oklab", None) is not None:
            l_ok, a_ok, b_ok = rgb_to_oklab(fr, fg, fb)
            fr, fg, fb = _gamut_map_oklab_to_srgb(
                l_ok,
                a_ok + args.warm_oklab / 2000.0,
                b_ok + args.warm_oklab / 1000.0,
            )
            mods.append(("warm-oklab", f"+{args.warm_oklab:.2f}%%"))

        elif op == "cool_oklab" and getattr(args, "cool_oklab", None) is not None:
            l_ok, a_ok, b_ok = rgb_to_oklab(fr, fg, fb)
            fr, fg, fb = _gamut_map_oklab_to_srgb(
                l_ok,
                a_ok - args.cool_oklab / 2000.0,
                b_ok - args.cool_oklab / 1000.0,
            )
            mods.append(("cool-oklab", f"+{args.cool_oklab:.2f}%%"))

        elif op == "posterize" and getattr(args, "posterize", None) is not None:
            fr, fg, fb = _posterize_rgb(fr, fg, fb, args.posterize)
            mods.append(("posterize-rgb", f"{max(2, min(256, int(abs(args.posterize))))}"))

        elif op == "threshold" and getattr(args, "threshold", None) is not None:
            t = _clamp01(args.threshold / 100.0)
            y = get_luminance(int(round(fr)), int(round(fg)), int(round(fb)))
            low_hex = getattr(args, "threshold_low", None) or "000000"
            high_hex = getattr(args, "threshold_high", None) or "FFFFFF"
            use_hex = low_hex if y < t else high_hex
            tr, tg, tb = hex_to_rgb(use_hex)
            fr, fg, fb = float(tr), float(tg), float(tb)
            mods.append(("threshold-luminance", f"{args.threshold:.2f}%%"))

        elif op == "solarize" and getattr(args, "solarize", None) is not None:
            fr, fg, fb = _solarize_smart(fr, fg, fb, args.solarize)
            mods.append(("solarize", f"{args.solarize:.2f}%%"))

        elif op == "tint" and getattr(args, "tint", None) is not None:
            strength = getattr(args, "tint_strength", None)
            if strength is None:
                strength = 20.0
            fr, fg, fb = _tint_oklab(fr, fg, fb, args.tint, strength)
            mods.append(("tint-oklab", f"{strength:.2f}%% to #{args.tint.upper()}"))

        elif op == "red_channel" and args.red_channel is not None:
            fr = _clamp255(fr + args.red_channel)
            mods.append(("red-channel", f"{args.red_channel:+d}"))

        elif op == "green_channel" and args.green_channel is not None:
            fg = _clamp255(fg + args.green_channel)
            mods.append(("green-channel", f"{args.green_channel:+d}"))

        elif op == "blue_channel" and args.blue_channel is not None:
            fb = _clamp255(fb + args.blue_channel)
            mods.append(("blue-channel", f"{args.blue_channel:+d}"))

        elif op == "opacity" and args.opacity is not None:
            fr, fg, fb = _apply_opacity_on_black(fr, fg, fb, args.opacity)
            mods.append(("opacity-on-black", f"{args.opacity:.2f}%%"))

        elif op == "lock_luminance" and getattr(args, "lock_luminance", False):
            l_ok, a_ok, b_ok = rgb_to_oklab(fr, fg, fb)
            fr, fg, fb = _gamut_map_oklab_to_srgb(base_l_oklab, a_ok, b_ok)
            mods.append(("lock-oklab-lightness", None))

        elif op == "lock_rel_luminance" and getattr(args, "lock_rel_luminance", False):
            fr, fg, fb = _lock_relative_luminance(fr, fg, fb, base_rel_lum)
            mods.append(("lock-relative-luminance", None))

        elif op == "target_rel_lum" and getattr(args, "target_rel_lum", None) is not None:
            target_Y = _clamp01(float(args.target_rel_lum))
            fr, fg, fb = _lock_relative_luminance(fr, fg, fb, target_Y)
            mods.append(("target-rel-luminance", f"{target_Y:.4f}"))

        elif op == "min_contrast" and getattr(args, "min_contrast_with", None) and getattr(
            args, "min_contrast", None
        ) is not None:
            fr, fg, fb, changed = _ensure_min_contrast_with(
                fr, fg, fb, args.min_contrast_with, args.min_contrast
            )
            if changed:
                mods.append((
                    "min-contrast",
                    f">={float(args.min_contrast):.2f} vs #{args.min_contrast_with.upper()}"
                ))

    ri, gi, bi = _finalize_rgb(fr, fg, fb)
    res_hex = rgb_to_hex(ri, gi, bi)
    base_hex_upper = base_hex.upper()
    is_hex_title = (
        isinstance(title, str) and
        title.startswith("#") and
        title[1:].upper() == base_hex_upper
    )

    print()
    base_label = "original" if is_hex_title else title
    print_color_block(base_hex, base_label)
    if mods:
        print_color_block(res_hex, "adjusted")
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
        print(
            "usage: hexlab adjust [-h] (-H HEX | -r | -cn NAME | -di INDEX) [OPTIONS...]",
            file=file
        )
        print("")
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
        type=INPUT_HANDLERS["float_signed_360"],
        metavar="N",
        help="rotate hue in HSL (-360 to 360 degrees)",
    )
    ga.add_argument(
        "-rotl",
        "--rotate-oklch",
        dest="rotate_oklch",
        type=INPUT_HANDLERS["float_signed_360"],
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
        help="set absolute target relative luminance (0.0 - 1.0)",
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
        help=("minimum WCAG contrast ratio with --min-contrast-with, "
              "best effort within srgb gamut"),
    )
    adv_group.add_argument(
        "--gamma",
        dest="gamma",
        type=INPUT_HANDLERS["float"],
        metavar="N",
        help="gamma correction in linear space (>0, typical 0.5 - 3.0)",
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
        help="solarize based on perceptual lightness OKLAB-L threshold (0-100%%)",
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
    return p


def main() -> None:
    parser = get_adjust_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    handle_adjust_command(args)


if __name__ == "__main__":
    main()