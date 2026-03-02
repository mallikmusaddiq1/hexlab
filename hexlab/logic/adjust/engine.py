#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/logic/adjust/engine.py

import argparse
import random
import sys

from hexlab.core import config as c
from hexlab.core import conversions as conv
from . import filters as op
from .resolver import resolve_adjust_input
from .renderer import render_adjust_info
from hexlab.core.luminance import get_luminance
from hexlab.shared.clamping import _clamp01, _clamp255
from hexlab.shared.logger import log

def _get_custom_pipeline_order(parser) -> list:
    """Determine the order of operations based on the user's CLI arguments."""
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
        operation = dest_to_op.get(dest)
        if operation and operation not in seen:
            order.append(operation)
            seen.add(operation)

    return order

def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Main execution logic for the adjust command pipeline."""
    base_hex, title = resolve_adjust_input(args)

    locks = sum([
        getattr(args, "lock_luminance", False),
        getattr(args, "lock_rel_luminance", False),
        getattr(args, "target_rel_lum", None) is not None
    ])

    if locks > 1:
        log("error", "conflicting luminance locks: use only one of --lock-luminance, --lock-rel-luminance or --target-rel-lum")
        sys.exit(2)

    if getattr(args, "min_contrast_with", None) and locks > 0:
        log("warning", "--min-contrast-with will override previous luminance locks")

    pipeline = c.PIPELINE
    if getattr(args, "list_fixed_pipeline", False):
        for step in pipeline: print(step)
        return

    if getattr(args, "custom_pipeline", False):
        custom_order = _get_custom_pipeline_order(parser)
        if custom_order: pipeline = custom_order

    mc_with = getattr(args, "min_contrast_with", None)
    mc_val = getattr(args, "min_contrast", None)
    if (mc_with and mc_val is None) or (mc_with is None and mc_val is not None):
        log("error", "--min-contrast-with and --min-contrast must be used together")
        sys.exit(2)

    r, g, b = conv.hex_to_rgb(base_hex)
    fr, fg, fb = op.sanitize_rgb(float(r), float(g), float(b))
    base_l_oklab, _, _ = conv.rgb_to_oklab(fr, fg, fb)
    base_rel_lum = get_luminance(r, g, b)

    mods = []

    for op_name in pipeline:
        curr_hex = conv.rgb_to_hex(fr, fg, fb)
        src_info = f"from #{curr_hex}"

        if op_name == "invert" and args.invert:
            fr, fg, fb = c.RGB_MAX - fr, c.RGB_MAX - fg, c.RGB_MAX - fb
            mods.append(("invert", src_info))

        elif op_name == "grayscale" and args.grayscale:
            l_ok, _, _ = conv.rgb_to_oklab(fr, fg, fb)
            fr_g, fg_g, fb_g = conv.oklab_to_rgb(l_ok, 0.0, 0.0)
            avg = (fr_g + fg_g + fb_g) / c.AVG_DIVISOR
            fr, fg, fb = _clamp255(avg), _clamp255(avg), _clamp255(avg)
            mods.append(("grayscale", src_info))

        elif op_name == "sepia" and args.sepia:
            tr = fr * c.SEPIA_RR + fg * c.SEPIA_RG + fb * c.SEPIA_RB
            tg = fr * c.SEPIA_GR + fg * c.SEPIA_GG + fb * c.SEPIA_GB
            tb = fr * c.SEPIA_BR + fg * c.SEPIA_BG + fb * c.SEPIA_BB
            fr, fg, fb = _clamp255(tr), _clamp255(tg), _clamp255(tb)
            mods.append(("sepia", src_info))

        elif op_name == "rotate" and args.rotate is not None:
            h, s, l_hsl = conv.rgb_to_hsl(fr, fg, fb)
            fr, fg, fb = conv.hsl_to_rgb(h + args.rotate, s, l_hsl)
            mods.append(("hue-rotate-hsl", f"{args.rotate:+.2f}deg {src_info}"))

        elif op_name == "rotate_oklch" and getattr(args, "rotate_oklch", None) is not None:
            l_ok, c_ok, h_ok = conv.rgb_to_oklch(fr, fg, fb)
            fr, fg, fb = conv.oklch_to_rgb(l_ok, c_ok, h_ok + args.rotate_oklch)
            fr, fg, fb = _clamp255(fr), _clamp255(fg), _clamp255(fb)
            mods.append(("hue-rotate-oklch", f"{args.rotate_oklch:+.2f}deg {src_info}"))

        elif op_name == "brightness" and args.brightness is not None:
            fr, fg, fb = op.apply_linear_gain_rgb(fr, fg, fb, 1.0 + (args.brightness / c.PERCENT_TO_FACTOR))
            mods.append(("brightness-linear", f"{args.brightness:+.2f}% {src_info}"))

        elif op_name == "brightness_srgb" and getattr(args, "brightness_srgb", None) is not None:
            fr, fg, fb = op.apply_srgb_brightness(fr, fg, fb, args.brightness_srgb)
            mods.append(("brightness-srgb", f"{args.brightness_srgb:+.2f}% {src_info}"))

        elif op_name == "contrast" and args.contrast is not None:
            fr, fg, fb = op.apply_linear_contrast_rgb(fr, fg, fb, args.contrast)
            mods.append(("contrast", f"{args.contrast:+.2f}% {src_info}"))

        elif op_name == "gamma" and getattr(args, "gamma", None) is not None:
            fr, fg, fb = op.apply_gamma(fr, fg, fb, args.gamma)
            mods.append(("gamma-linear", f"{args.gamma:.2f} {src_info}"))

        elif op_name == "exposure" and getattr(args, "exposure", None) is not None:
            fr, fg, fb = op.apply_linear_gain_rgb(fr, fg, fb, 2.0 ** (float(args.exposure) / c.EXPOSURE_STOPS_SCALE))
            mods.append(("exposure-stops", f"{args.exposure:+.2f} {src_info}"))

        elif op_name == "lighten" and args.lighten is not None:
            h, s, l_hsl = conv.rgb_to_hsl(fr, fg, fb)
            l_new = _clamp01(l_hsl + (1.0 - l_hsl) * (args.lighten / c.PERCENT_TO_FACTOR))
            fr, fg, fb = conv.hsl_to_rgb(h, s, l_new)
            mods.append(("lighten", f"+{args.lighten:.2f}% {src_info}"))

        elif op_name == "darken" and args.darken is not None:
            h, s, l_hsl = conv.rgb_to_hsl(fr, fg, fb)
            l_new = _clamp01(l_hsl * (1.0 - (args.darken / c.PERCENT_TO_FACTOR)))
            fr, fg, fb = conv.hsl_to_rgb(h, s, l_new)
            mods.append(("darken", f"-{args.darken:.2f}% {src_info}"))

        elif op_name == "saturate" and args.saturate is not None:
            h, s, l_hsl = conv.rgb_to_hsl(fr, fg, fb)
            if s > c.EPS:
                s_new = _clamp01(s + (1.0 - s) * (args.saturate / c.PERCENT_TO_FACTOR))
                fr, fg, fb = conv.hsl_to_rgb(h, s_new, l_hsl)
            mods.append(("saturate", f"+{args.saturate:.2f}% {src_info}"))

        elif op_name == "desaturate" and args.desaturate is not None:
            h, s, l_hsl = conv.rgb_to_hsl(fr, fg, fb)
            s_new = _clamp01(s * (1.0 - (args.desaturate / c.PERCENT_TO_FACTOR)))
            fr, fg, fb = conv.hsl_to_rgb(h, s_new, l_hsl)
            mods.append(("desaturate", f"-{args.desaturate:.2f}% {src_info}"))

        elif op_name == "whiten_hwb" and getattr(args, "whiten_hwb", None) is not None:
            h, w, b_hwb = conv.rgb_to_hwb(fr, fg, fb)
            fr, fg, fb = conv.hwb_to_rgb(h, _clamp01(w + args.whiten_hwb / c.PERCENT_TO_FACTOR), b_hwb)
            mods.append(("whiten-hwb", f"+{args.whiten_hwb:.2f}% {src_info}"))

        elif op_name == "blacken_hwb" and getattr(args, "blacken_hwb", None) is not None:
            h, w, b_hwb = conv.rgb_to_hwb(fr, fg, fb)
            fr, fg, fb = conv.hwb_to_rgb(h, w, _clamp01(b_hwb + args.blacken_hwb / c.PERCENT_TO_FACTOR))
            mods.append(("blacken-hwb", f"+{args.blacken_hwb:.2f}% {src_info}"))

        elif op_name == "chroma_oklch" and getattr(args, "chroma_oklch", None) is not None:
            l_ok, c_ok, h_ok = conv.rgb_to_oklch(fr, fg, fb)
            c_new = max(0.0, c_ok * (1.0 + (args.chroma_oklch / c.PERCENT_TO_FACTOR)))
            fr_c, fg_c, fb_c = conv.oklch_to_rgb(l_ok, c_new, h_ok)
            l_f, a_f, b_f = conv.rgb_to_oklab(fr_c, fg_c, fb_c)
            fr, fg, fb = op.gamut_map_oklab_to_srgb(l_f, a_f, b_f)
            mods.append(("chroma-oklch", f"{args.chroma_oklch:+.2f}% {src_info}"))

        elif op_name == "vibrance_oklch" and getattr(args, "vibrance_oklch", None) is not None:
            fr, fg, fb = op.apply_vibrance_oklch(fr, fg, fb, args.vibrance_oklch)
            mods.append(("vibrance-oklch", f"{args.vibrance_oklch:+.2f}% {src_info}"))

        elif op_name == "warm_oklab" and getattr(args, "warm_oklab", None) is not None:
            l_ok, a_ok, b_ok = conv.rgb_to_oklab(fr, fg, fb)
            fr, fg, fb = op.gamut_map_oklab_to_srgb(l_ok, a_ok + args.warm_oklab / c.WARM_OKLAB_A_SCALE, b_ok + args.warm_oklab / c.WARM_OKLAB_B_SCALE)
            mods.append(("warm-oklab", f"+{args.warm_oklab:.2f}% {src_info}"))

        elif op_name == "cool_oklab" and getattr(args, "cool_oklab", None) is not None:
            l_ok, a_ok, b_ok = conv.rgb_to_oklab(fr, fg, fb)
            fr, fg, fb = op.gamut_map_oklab_to_srgb(l_ok, a_ok - args.cool_oklab / c.WARM_OKLAB_A_SCALE, b_ok - args.cool_oklab / c.WARM_OKLAB_B_SCALE)
            mods.append(("cool-oklab", f"+{args.cool_oklab:.2f}% {src_info}"))

        elif op_name == "posterize" and getattr(args, "posterize", None) is not None:
            fr, fg, fb = op.posterize_rgb(fr, fg, fb, args.posterize)
            mods.append(("posterize-rgb", f"{max(c.POSTERIZE_MIN_LEVELS, min(c.POSTERIZE_MAX_LEVELS, int(abs(args.posterize))))} {src_info}"))

        elif op_name == "threshold" and getattr(args, "threshold", None) is not None:
            y = get_luminance(int(round(fr)), int(round(fg)), int(round(fb)))
            use_hex = "000000" if y < (args.threshold / c.PERCENT_TO_FACTOR) else "FFFFFF"
            tr, tg, tb = conv.hex_to_rgb(use_hex)
            fr, fg, fb = float(tr), float(tg), float(tb)
            mods.append(("threshold-luminance", f"{args.threshold:.2f}% (result: #{use_hex.upper()}) {src_info}"))

        elif op_name == "solarize" and getattr(args, "solarize", None) is not None:
            fr, fg, fb = op.solarize_smart(fr, fg, fb, args.solarize)
            mods.append(("solarize", f"{args.solarize:.2f}% {src_info}"))

        elif op_name == "tint" and getattr(args, "tint", None) is not None:
            strength = getattr(args, "tint_strength", 20.0)
            fr, fg, fb = op.tint_oklab(fr, fg, fb, args.tint, strength)
            mods.append(("tint-oklab", f"{strength:.2f}% from #{curr_hex} to #{args.tint.upper()}"))

        elif op_name == "red_channel" and args.red_channel is not None:
            fr = _clamp255(fr + args.red_channel)
            mods.append(("red-channel", f"{args.red_channel:+d} {src_info}"))

        elif op_name == "green_channel" and args.green_channel is not None:
            fg = _clamp255(fg + args.green_channel)
            mods.append(("green-channel", f"{args.green_channel:+d} {src_info}"))

        elif op_name == "blue_channel" and args.blue_channel is not None:
            fb = _clamp255(fb + args.blue_channel)
            mods.append(("blue-channel", f"{args.blue_channel:+d} {src_info}"))

        elif op_name == "opacity" and args.opacity is not None:
            fr, fg, fb = op.apply_opacity_on_black(fr, fg, fb, args.opacity)
            mods.append(("opacity-on-black", f"{args.opacity:.2f}% {src_info}"))

        elif op_name == "lock_luminance" and getattr(args, "lock_luminance", False):
            _, a_ok, b_ok = conv.rgb_to_oklab(fr, fg, fb)
            fr, fg, fb = op.gamut_map_oklab_to_srgb(base_l_oklab, a_ok, b_ok)
            mods.append(("lock-oklab-lightness", f"{src_info}"))

        elif op_name == "lock_rel_luminance" and getattr(args, "lock_rel_luminance", False):
            fr, fg, fb = op.lock_relative_luminance(fr, fg, fb, base_rel_lum)
            mods.append(("lock-relative-luminance", f"{src_info}"))

        elif op_name == "target_rel_lum" and getattr(args, "target_rel_lum", None) is not None:
            target_Y = _clamp01(float(args.target_rel_lum))
            fr, fg, fb = op.lock_relative_luminance(fr, fg, fb, target_Y)
            mods.append(("target-rel-luminance", f"{target_Y:.4f} {src_info}"))

        elif op_name == "min_contrast" and mc_with and mc_val is not None:
            fr, fg, fb, changed = op.ensure_min_contrast_with(fr, fg, fb, args.min_contrast_with, args.min_contrast)
            if changed: mods.append(("min-contrast", f">={float(args.min_contrast):.2f} vs #{args.min_contrast_with.upper()} {src_info}"))

    ri, gi, bi = op.finalize_rgb(fr, fg, fb)
    res_hex = conv.rgb_to_hex(ri, gi, bi)
    
    render_adjust_info(base_hex, res_hex, title, mods, args)