#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/logic/gradient/engine.py

from typing import Tuple

from hexlab.core import conversions as conv

def get_interpolated_color(c1, c2, t: float, colorspace: str) -> Tuple[float, float, float]:
    """Interpolate between two colors in the specified colorspace."""
    if colorspace == "srgb":
        r1, g1, b1 = c1
        r2, g2, b2 = c2
        return r1 + t * (r2 - r1), g1 + t * (g2 - g1), b1 + t * (b2 - b1)

    if colorspace == "srgblinear":
        r_lin1, g_lin1, b_lin1 = c1
        r_lin2, g_lin2, b_lin2 = c2
        r_lin_new = r_lin1 + t * (r_lin2 - r_lin1)
        g_lin_new = g_lin1 + t * (g_lin2 - g_lin1)
        b_lin_new = b_lin1 + t * (b_lin2 - b_lin1)
        return conv._linear_to_srgb(r_lin_new) * 255, conv._linear_to_srgb(g_lin_new) * 255, conv._linear_to_srgb(b_lin_new) * 255

    if colorspace == "lab":
        l_new = c1[0] + t * (c2[0] - c1[0])
        a_new = c1[1] + t * (c2[1] - c1[1])
        b_new = c1[2] + t * (c2[2] - c1[2])
        return conv.lab_to_rgb(l_new, a_new, b_new)

    if colorspace == "oklab":
        l_new = c1[0] + t * (c2[0] - c1[0])
        a_new = c1[1] + t * (c2[1] - c1[1])
        b_new = c1[2] + t * (c2[2] - c1[2])
        return conv.oklab_to_rgb(l_new, a_new, b_new)

    if colorspace in ["lch", "oklch"]:
        l1, c1_val, h1 = c1
        l2, c2_val, h2 = c2
        h1, h2 = h1 % 360, h2 % 360
        h_diff = h2 - h1
        if h_diff > 180: h2 -= 360
        elif h_diff < -180: h2 += 360
        l_new = l1 + t * (l2 - l1)
        c_new = c1_val + t * (c2_val - c1_val)
        h_new = (h1 + t * (h2 - h1)) % 360
        return conv.lch_to_rgb(l_new, c_new, h_new) if colorspace == "lch" else conv.oklch_to_rgb(l_new, c_new, h_new)

    if colorspace == "luv":
        return conv.luv_to_rgb(c1[0] + t * (c2[0] - c1[0]), c1[1] + t * (c2[1] - c1[1]), c1[2] + t * (c2[2] - c1[2]))

    return 0.0, 0.0, 0.0

def convert_rgb_to_space(r: int, g: int, b: int, colorspace: str) -> Tuple[float, ...]:
    """Convert RGB to components in the specified colorspace."""
    if colorspace == "srgb":
        return (float(r), float(g), float(b))
    if colorspace == "srgblinear":
        return (conv._srgb_to_linear(r), conv._srgb_to_linear(g), conv._srgb_to_linear(b))
    if colorspace == "lab":
        return conv.rgb_to_lab(r, g, b)
    if colorspace == "oklab":
        return conv.rgb_to_oklab(r, g, b)
    if colorspace == "lch":
        return conv.rgb_to_lch(r, g, b)
    if colorspace == "oklch":
        return conv.rgb_to_oklch(r, g, b)
    if colorspace == "luv":
        return conv.rgb_to_luv(r, g, b)

    return (float(r), float(g), float(b))