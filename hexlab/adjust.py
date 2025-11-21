#!/usr/bin/env python3

import argparse
import sys
import random
import math
import re
import os
import textwrap
import shutil
from typing import Tuple

from input_utils import INPUT_HANDLERS, log, HexlabArgumentParser
from constants import COLOR_NAMES, MAX_DEC, EPS, SRGB_TO_LINEAR_TH, LINEAR_TO_SRGB_TH


def _norm_name_key(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", str(s).lower())


HEX_TO_NAME = {}
for name, hexv in COLOR_NAMES.items():
    HEX_TO_NAME[hexv.upper()] = name

_norm_map = {}
for k, v in COLOR_NAMES.items():
    key = _norm_name_key(k)
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


def rgb_to_oklch(r: float, g: float, b: float) -> Tuple[float, float, float]:
    l, a, bk = rgb_to_oklab(r, g, b)
    return l, math.hypot(a, bk), math.degrees(math.atan2(bk, a)) % 360


def oklch_to_rgb(l: float, c: float, h: float) -> Tuple[float, float, float]:
    hrad = math.radians(h)
    return oklab_to_rgb(l, c * math.cos(hrad), c * math.sin(hrad))


def rgb_to_hsv(r: float, g: float, b: float) -> Tuple[float, float, float]:
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    cmax, cmin = max(rf, gf, bf), min(rf, gf, bf)
    v = cmax
    s = (cmax - cmin) / v if v != 0 else 0
    if cmax == cmin:
        h = 0
    elif cmax == rf:
        h = (60 * ((gf - bf) / (cmax - cmin)) + 360) % 360
    elif cmax == gf:
        h = (60 * ((bf - rf) / (cmax - cmin)) + 120) % 360
    else:
        h = (60 * ((rf - gf) / (cmax - cmin)) + 240) % 360
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


def xyz_to_rgb(x: float, y: float, z: float) -> Tuple[float, float, float]:
    x_n, y_n, z_n = x / 100.0, y / 100.0, z / 100.0
    r_lin = x_n * 3.2404542 + y_n * -1.5371385 + z_n * -0.4985314
    g_lin = x_n * -0.9692660 + y_n * 1.8760108 + z_n * 0.0415560
    b_lin = x_n * 0.0556434 + y_n * -0.2040259 + z_n * 1.0572252
    r = _linear_to_srgb(r_lin)
    g = _linear_to_srgb(g_lin)
    b = _linear_to_srgb(b_lin)
    return _clamp01(r) * 255, _clamp01(g) * 255, _clamp01(b) * 255


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


def _finalize_rgb(r: float, g: float, b: float) -> Tuple[int, int, int]:
    return (
        max(0, min(255, int(round(r)))),
        max(0, min(255, int(round(g)))),
        max(0, min(255, int(round(b)))),
    )

def print_block(hex_code: str, label: str) -> None:
    r, g, b = hex_to_rgb(hex_code)
    print(f"{label:<17} : \033[48;2;{r};{g};{b}m        \033[0m #{hex_code}")


def _apply_linear_gain_rgb(fr: float, fg: float, fb: float, factor: float) -> Tuple[float, float, float]:
    rl, gl, bl = _srgb_to_linear(fr), _srgb_to_linear(fg), _srgb_to_linear(fb)
    rl = _clamp01(rl * factor)
    gl = _clamp01(gl * factor)
    bl = _clamp01(bl * factor)
    return _linear_to_srgb(rl) * 255.0, _linear_to_srgb(gl) * 255.0, _linear_to_srgb(bl) * 255.0


def _apply_linear_contrast_rgb(fr: float, fg: float, fb: float, contrast_amount: float) -> Tuple[float, float, float]:
    f = (259 * (contrast_amount + 255)) / (255 * (259 - contrast_amount))
    rl, gl, bl = _srgb_to_linear(fr), _srgb_to_linear(fg), _srgb_to_linear(fb)
    rl = _clamp01((f * (rl - 0.5)) + 0.5)
    gl = _clamp01((f * (gl - 0.5)) + 0.5)
    bl = _clamp01((f * (bl - 0.5)) + 0.5)
    return _linear_to_srgb(rl) * 255.0, _linear_to_srgb(gl) * 255.0, _linear_to_srgb(bl) * 255.0


def _apply_opacity_on_black(fr: float, fg: float, fb: float, opacity_percent: float) -> Tuple[float, float, float]:
    alpha = _clamp01(opacity_percent / 100.0)
    rl, gl, bl = _srgb_to_linear(fr), _srgb_to_linear(fg), _srgb_to_linear(fb)
    rl *= alpha
    gl *= alpha
    bl *= alpha
    return _linear_to_srgb(rl) * 255.0, _linear_to_srgb(gl) * 255.0, _linear_to_srgb(bl) * 255.0


def _mix_rgb_linear(fr: float, fg: float, fb: float, tr: float, tg: float, tb: float, t: float) -> Tuple[float, float, float]:
    rl1, gl1, bl1 = _srgb_to_linear(fr), _srgb_to_linear(fg), _srgb_to_linear(fb)
    rl2, gl2, bl2 = _srgb_to_linear(tr), _srgb_to_linear(tg), _srgb_to_linear(tb)
    rl = rl1 * (1.0 - t) + rl2 * t
    gl = gl1 * (1.0 - t) + gl2 * t
    bl = bl1 * (1.0 - t) + bl2 * t
    return _linear_to_srgb(rl) * 255.0, _linear_to_srgb(gl) * 255.0, _linear_to_srgb(bl) * 255.0

def _print_steps(mods):
    if not mods:
        return
    print()
    print("steps:")
    term_width = shutil.get_terminal_size(fallback=(80, 20)).columns
    indent = "  "
    for i, (label, val) in enumerate(mods, 1):
        if val:
            line = f"{i:2d}. {label}: {val}"
        else:
            line = f"{i:2d}. {label}"
        wrapped = textwrap.fill(line, width=term_width, subsequent_indent=indent)
        print(indent + wrapped.lstrip())


def handle_adjust_command(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)
    base_hex, title = None, "original"

    if args.random_adjust:
        base_hex, title = f"{random.randint(0, MAX_DEC):06X}", "Random"
    elif args.color_name:
        base_hex = _get_color_name_hex(args.color_name)
        if not base_hex:
            log("error", f"unknown name '{args.color_name}'")
            sys.exit(2)
        title = args.color_name
    elif args.hex:
        base_hex, title = args.hex, f"#{args.hex}"
    elif getattr(args, "decimal_index", None):
        base_hex, title = args.decimal_index, f"Idx {int(args.decimal_index, 16)}"

    if not base_hex:
        log("error", "no input color")
        sys.exit(2)

    r, g, b = hex_to_rgb(base_hex)
    fr, fg, fb = float(r), float(g), float(b)
    mods = []

    mix_target_hex = None
    if args.mix_color:
        target = args.mix_color_hex
        if not target and args.mix_color_name:
            target = _get_color_name_hex(args.mix_color_name)

        if target:
            mix_target_hex = target
            tr, tg, tb = hex_to_rgb(target)
            t = args.mix_amount / 100.0
            target_up = target.upper()
            target_name = HEX_TO_NAME.get(target_up)
            target_desc = f"with #{target_up}"
            if target_name:
                target_desc += f' ("{target_name}")'

            mix_mode = args.mix_mode

            if mix_mode in ("rgb", "srgb-linear"):
                fr, fg, fb = _mix_rgb_linear(fr, fg, fb, float(tr), float(tg), float(tb), t)
                mods.append(
                    (
                        "mix SRGB Linear",
                        f"{args.mix_amount:.4f}% {target_desc}",
                    )
                )
            elif mix_mode == "srgb":
                fr = fr * (1.0 - t) + float(tr) * t
                fg = fg * (1.0 - t) + float(tg) * t
                fb = fb * (1.0 - t) + float(tb) * t
                fr, fg, fb = _clamp255(fr), _clamp255(fg), _clamp255(fb)
                mods.append(
                    (
                        "mix SRGB",
                        f"{args.mix_amount:.4f}% {target_desc}",
                    )
                )
            elif mix_mode == "lab":
                r1_i, g1_i, b1_i = int(round(fr)), int(round(fg)), int(round(fb))
                r2_i, g2_i, b2_i = tr, tg, tb
                x1, y1, z1 = rgb_to_xyz(r1_i, g1_i, b1_i)
                x2, y2, z2 = rgb_to_xyz(r2_i, g2_i, b2_i)
                l1, a1, b1_lab = xyz_to_lab(x1, y1, z1)
                l2, a2, b2_lab = xyz_to_lab(x2, y2, z2)
                l_new = l1 * (1.0 - t) + l2 * t
                a_new = a1 * (1.0 - t) + a2 * t
                b_new = b1_lab * (1.0 - t) + b2_lab * t
                x_new, y_new, z_new = lab_to_xyz(l_new, a_new, b_new)
                fr, fg, fb = xyz_to_rgb(x_new, y_new, z_new)
                fr, fg, fb = _clamp255(fr), _clamp255(fg), _clamp255(fb)
                mods.append(
                    (
                        "mix CIELAB",
                        f"{args.mix_amount:.4f}% {target_desc}",
                    )
                )
            elif mix_mode == "luv":
                r1_i, g1_i, b1_i = int(round(fr)), int(round(fg)), int(round(fb))
                r2_i, g2_i, b2_i = tr, tg, tb
                l1, u1, v1 = rgb_to_luv(r1_i, g1_i, b1_i)
                l2, u2, v2 = rgb_to_luv(r2_i, g2_i, b2_i)
                l_new = l1 * (1.0 - t) + l2 * t
                u_new = u1 * (1.0 - t) + u2 * t
                v_new = v1 * (1.0 - t) + v2 * t
                fr, fg, fb = luv_to_rgb(l_new, u_new, v_new)
                fr, fg, fb = _clamp255(fr), _clamp255(fg), _clamp255(fb)
                mods.append(
                    (
                        "mix CIELUV",
                        f"{args.mix_amount:.4f}% {target_desc}",
                    )
                )
            else:
                # default / explicit oklab
                l1, a1, b1_ok = rgb_to_oklab(fr, fg, fb)
                l2, a2, b2_ok = rgb_to_oklab(tr, tg, tb)
                fr, fg, fb = oklab_to_rgb(
                    l1 * (1.0 - t) + l2 * t,
                    a1 * (1.0 - t) + a2 * t,
                    b1_ok * (1.0 - t) + b2_ok * t,
                )
                fr, fg, fb = _clamp255(fr), _clamp255(fg), _clamp255(fb)
                mods.append(
                    (
                        "mix OKLAB",
                        f"{args.mix_amount:.4f}% {target_desc}",
                    )
                )

    if args.invert:
        fr, fg, fb = 255.0 - fr, 255.0 - fg, 255.0 - fb
        mods.append(("invert", None))
    if args.grayscale:
        l, a, b_ok = rgb_to_oklab(fr, fg, fb)
        fr, fg, fb = oklab_to_rgb(l, 0.0, 0.0)
        fr, fg, fb = _clamp255(fr), _clamp255(fg), _clamp255(fb)
        mods.append(("grayscale OKLAB", None))
    if args.sepia:
        tr = fr * 0.393 + fg * 0.769 + fb * 0.189
        tg = fr * 0.349 + fg * 0.686 + fb * 0.168
        tb = fr * 0.272 + fg * 0.534 + fb * 0.131
        fr, fg, fb = min(255.0, tr), min(255.0, tg), min(255.0, tb)
        fr, fg, fb = _clamp255(fr), _clamp255(fg), _clamp255(fb)
        mods.append(("sepia RGB", None))

    if args.rotate is not None:
        h, s, l = rgb_to_hsl(fr, fg, fb)
        fr, fg, fb = hsl_to_rgb(h + args.rotate, s, l)
        mods.append(("hue rotate HSL)", f"{args.rotate:+.4f}°"))
    if args.brightness is not None:
        factor = 1.0 + (args.brightness / 100.0)
        fr, fg, fb = _apply_linear_gain_rgb(fr, fg, fb, factor)
        mods.append(("brightness linear RGB", f"{args.brightness:+.4f}%"))
    if args.contrast is not None:
        fr, fg, fb = _apply_linear_contrast_rgb(fr, fg, fb, args.contrast)
        mods.append(("contrast linear RGB)", f"{args.contrast:+.4f}%"))

    if any(x is not None for x in [args.lighten, args.darken, args.saturate, args.desaturate]):
        h, s, l = rgb_to_hsl(fr, fg, fb)
        if args.lighten is not None:
            amount = args.lighten / 100.0
            l = _clamp01(l + (1.0 - l) * amount)
            mods.append(("lighten HSL", f"+{args.lighten:.4f}%"))

        if args.darken is not None:
            amount = args.darken / 100.0
            l = _clamp01(l * (1.0 - amount))
            mods.append(("darken HSL", f"-{args.darken:.4f}%"))

        if args.saturate is not None:
            amount = args.saturate / 100.0
            s = _clamp01(s + (1.0 - s) * amount)
            mods.append(("saturate HSL", f"+{args.saturate:.4f}%"))

        if args.desaturate is not None:
            amount = args.desaturate / 100.0
            s = _clamp01(s * (1.0 - amount))
            mods.append(("desaturate HSL", f"-{args.desaturate:.4f}%"))

        fr, fg, fb = hsl_to_rgb(h, s, l)

    if args.whiten is not None or args.blacken is not None:
        h, w, b = rgb_to_hwb(fr, fg, fb)
        if args.whiten is not None:
            w = _clamp01(w + args.whiten / 100.0)
            mods.append(("whiten HWB", f"+{args.whiten:.4f}%"))
        if args.blacken is not None:
            b = _clamp01(b + args.blacken / 100.0)
            mods.append(("blacken HWB", f"+{args.blacken:.4f}%"))
        fr, fg, fb = hwb_to_rgb(h, w, b)

    if args.chroma_boost is not None:
        l, c, h = rgb_to_oklch(fr, fg, fb)
        factor = 1.0 + (args.chroma_boost / 100.0)
        c = max(0.0, c * factor)
        fr, fg, fb = oklch_to_rgb(l, c, h)
        fr, fg, fb = _clamp255(fr), _clamp255(fg), _clamp255(fb)
        mods.append(("chroma boost OKLCH", f"{args.chroma_boost:+.4f}%"))

    if args.warm is not None:
        l, a, b_ok = rgb_to_oklab(fr, fg, fb)
        fr, fg, fb = oklab_to_rgb(l, a + args.warm / 2000.0, b_ok + args.warm / 1000.0)
        fr, fg, fb = _clamp255(fr), _clamp255(fg), _clamp255(fb)
        mods.append(("warm OKLAB", f"+{args.warm:.4f}%"))

    if args.cool is not None:
        l, a, b_ok = rgb_to_oklab(fr, fg, fb)
        fr, fg, fb = oklab_to_rgb(l, a - args.cool / 2000.0, b_ok - args.cool / 1000.0)
        fr, fg, fb = _clamp255(fr), _clamp255(fg), _clamp255(fb)
        mods.append(("cool OKLAB", f"+{args.cool:.4f}%"))

    if args.red_channel is not None:
        fr = _clamp01((fr + args.red_channel) / 255.0) * 255.0
        mods.append(("red channel RGB", f"{args.red_channel:+d}"))
    if args.green_channel is not None:
        fg = _clamp01((fg + args.green_channel) / 255.0) * 255.0
        mods.append(("green channel RGB", f"{args.green_channel:+d}"))
    if args.blue_channel is not None:
        fb = _clamp01((fb + args.blue_channel) / 255.0) * 255.0
        mods.append(("blue channel RGB", f"{args.blue_channel:+d}"))
    if args.opacity is not None:
        fr, fg, fb = _apply_opacity_on_black(fr, fg, fb, args.opacity)
        mods.append(("opacity linear RGB", f"{args.opacity:.4f}%"))

    print()

    ri, gi, bi = _finalize_rgb(fr, fg, fb)
    res_hex = rgb_to_hex(ri, gi, bi)

    base_hex_upper = base_hex.upper()
    is_hex_title = isinstance(title, str) and title.startswith("#") and title[1:].upper() == base_hex_upper

    if mix_target_hex:
        base_label = "Base" if is_hex_title else title
        print_block(base_hex, base_label)
        print_block(mix_target_hex, "mix with")
        print("-" * 18)
        print_block(res_hex, "result")
    elif not mods:
        base_label = "original" if is_hex_title else title
        print_block(base_hex, base_label)
        print()
        return
    else:
        base_label = "original" if is_hex_title else title
        print_block(base_hex, base_label)
        print("-" * 18)
        print_block(res_hex, "adjusted")

    print()
    _print_steps(mods)
    print()


def get_adjust_parser() -> argparse.ArgumentParser:

    p = HexlabArgumentParser(
        prog="hexlab adjust",
        description=(
            "hexlab adjust: advanced color manipulation\n\n"
            "all operations are deterministic and applied in a fixed pipeline. "
            "see the 'steps' section in the output for the exact order"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-H",
        "--hex",
        type=INPUT_HANDLERS["hex"],
        help="base hex code",
    )
    input_group.add_argument(
        "-ra",
        "--random-adjust",
        action="store_true",
        help="use a random base color",
    )
    input_group.add_argument(
        "-cn",
        "--color-name",
        type=INPUT_HANDLERS["color_name"],
        help="base color name from --list-color-names",
    )
    input_group.add_argument(
        "-di",
        "--decimal-index",
        type=INPUT_HANDLERS["decimal_index"],
        help="base color decimal index: 0 to 16777215",
    )

    p.add_argument(
        "-s",
        "--seed",
        type=INPUT_HANDLERS["seed"],
        help="random seed for reproducibility",
    )

    ga = p.add_argument_group("basic HSL tone & saturation from 0 to 100, positive only)")
    ga.add_argument(
        "-l",
        "--lighten",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="increase HSL lightness by N%% relative to current lightness",
    )
    ga.add_argument(
        "-d",
        "--darken",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="decrease HSL lightness by N%% relative multiply towards 0",
    )
    ga.add_argument(
        "-sat",
        "--saturate",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="increase HSL saturation by N%% relative to remaining headroom",
    )
    ga.add_argument(
        "-des",
        "--desaturate",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="decrease HSL saturation by N%% relative multiply towards 0",
    )
    ga.add_argument(
        "-rot",
        "--rotate",
        type=INPUT_HANDLERS["float"],
        metavar="N",
        help="rotate HSL hue by N degrees can be positive or negative",
    )

    adv_group = p.add_argument_group("advanced tonal & colorfulness corrections")
    adv_group.add_argument(
        "-br",
        "--brightness",
        type=INPUT_HANDLERS["float_signed_100"],
        metavar="N",
        help="adjust brightness by N%% in linear RGB from -100 to 100",
    )
    adv_group.add_argument(
        "-ct",
        "--contrast",
        type=INPUT_HANDLERS["float_signed_100"],
        metavar="N",
        help="adjust contrast by N%% in linear RGB from -100 to 100, mid-grey anchored)",
    )
    adv_group.add_argument(
        "-cb",
        "--chroma-boost",
        type=INPUT_HANDLERS["float"],
        metavar="N",
        help="scale OKLCH chroma by 1 + N/100; N>0 boosts, N<0 reduces",
    )
    adv_group.add_argument(
        "-w",
        "--whiten",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="increase HWB whiteness by N%% mix towards white",
    )
    adv_group.add_argument(
        "-b",
        "--blacken",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="increase HWB blackness by N%% mix towards black",
    )
    adv_group.add_argument(
        "-warm",
        "--warm",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="heuristic OKLAB warmth from 0 to 100",
    )
    adv_group.add_argument(
        "-cool",
        "--cool",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help="heuristic OKLAB coolness from 0 to 100",
    )

    filter_group = p.add_argument_group("filters & RGB channels")
    filter_group.add_argument(
        "-g",
        "--grayscale",
        action="store_true",
        help="OKLAB grayscale preserves lightness and removes chroma",
    )
    filter_group.add_argument(
        "-inv",
        "--invert",
        action="store_true",
        help="invert the color)",
    )
    filter_group.add_argument(
        "-sep",
        "--sepia",
        action="store_true",
        help="apply classic RGB sepia matrix",
    )
    filter_group.add_argument(
        "-rc",
        "--red-channel",
        type=INPUT_HANDLERS["int_channel"],
        metavar="N",
        help="add/sub red channel in RGB from -255 to 255",
    )
    filter_group.add_argument(
        "-gc",
        "--green-channel",
        type=INPUT_HANDLERS["int_channel"],
        metavar="N",
        help="add/sub green channel in RGB from -255 to 255",
    )
    filter_group.add_argument(
        "-bc",
        "--blue-channel",
        type=INPUT_HANDLERS["int_channel"],
        metavar="N",
        help="add/sub blue channel in RGB from -255 to 255",
    )
    filter_group.add_argument(
        "-op",
        "--opacity",
        type=INPUT_HANDLERS["float_0_100"],
        metavar="N",
        help=(
            "simulate opacity N%% over black in linear RGB"
        ),
    )

    gm = p.add_argument_group("mixing with another color")
    gm.add_argument(
        "-mix",
        "--mix-color",
        action="store_true",
        help="enable mixing mode",
    )
    mix_ex = gm.add_mutually_exclusive_group()
    mix_ex.add_argument(
        "-mh",
        "--mix-color-hex",
        type=INPUT_HANDLERS["hex"],
        help="hex code to mix with base color",
    )
    mix_ex.add_argument(
        "-mn",
        "--mix-color-name",
        type=INPUT_HANDLERS["color_name"],
        help="color name to mix with from --list-color-names",
    )
    gm.add_argument(
        "-ma",
        "--mix-amount",
        type=INPUT_HANDLERS["float_0_100"],
        default=50.0,
        metavar="N",
        help="mix percentage N%% of secondary color (0-100, default: 50)",
    )
    gm.add_argument(
        "-mm",
        "--mix-mode",
        type=INPUT_HANDLERS["colorspace"],
        choices=["rgb", "srgb", "srgb-linear", "lab", "oklab", "luv"],
        default="rgb",
        help=(
            "mixing interpolation mode"
        ),
    )

    return p


def main() -> None:
    parser = get_adjust_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    handle_adjust_command(args)


if __name__ == "__main__":
    main()