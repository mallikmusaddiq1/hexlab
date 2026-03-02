#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/logic/color/renderer.py

import argparse
from typing import Optional, Dict, Any

from hexlab.core import config as c
from hexlab.shared.naming import get_title_for_hex
from hexlab.shared.formatting import format_colorspace
from hexlab.shared.preview import print_color_block


def _zero_small(v: float, threshold: float = 1e-4) -> float:
    """Zero out small floating-point values below a threshold."""
    return 0.0 if abs(v) <= threshold else v


def _draw_bar(val: float, max_val: float, r_c: int, g_c: int, b_c: int) -> str:
    """Draw a ANSI-colored bar representation of a value."""
    total_len = 16
    abs_val = min(abs(val), max_val)
    percent = abs_val / max_val
    filled = max(0, min(total_len, int(total_len * percent)))
    empty = total_len - filled

    color_ansi = f"\033[38;2;{r_c};{g_c};{b_c}m"
    reset_ansi = "\033[0m"
    empty_ansi = "\033[90m"

    block_char = "█"
    empty_char = "░"

    if val < 0:
        bar_str = (
            f"{empty_ansi}{empty_char * empty}{reset_ansi}"
            f"{color_ansi}{block_char * filled}{reset_ansi}"
        )
    else:
        bar_str = (
            f"{color_ansi}{block_char * filled}{reset_ansi}"
            f"{empty_ansi}{empty_char * empty}{reset_ansi}"
        )

    return bar_str


def render_color_info(
    hex_code: str,
    title: str,
    args: argparse.Namespace,
    rgb: tuple,
    luminance: Optional[float] = None,
    neighbors: Optional[Dict[str, str]] = None,
    tech_data: Optional[Dict[str, Any]] = None,
    wcag_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Strictly prints color information. Data must be pre-calculated by the engine."""
    print()
    # 1. Main color block
    print_color_block(hex_code, f"{c.BOLD_WHITE}{title}{c.RESET}")

    # 2. Neighbors (Pre-calculated by Engine)
    if neighbors:
        print()
        for key in ["next", "previous", "negative"]:
            val = neighbors.get(key)
            if val is not None:
                colored_title = f"{c.MSG_BOLD_COLORS['info']}{key}{c.RESET}"
                print_color_block(val, colored_title)

    hide_bars = getattr(args, "hide_bars", False)
    r, g, b = rgb
    data = tech_data or {}

    # 3. Basic Infos
    if getattr(args, "index", False):
        print(f"\n{c.MSG_BOLD_COLORS['info']}index{c.RESET}             {c.BOLD_WHITE}: {int(hex_code, 16)} / {c.MAX_DEC}{c.RESET}")

    if getattr(args, "name", False):
        name_or_hex = get_title_for_hex(hex_code)
        if not name_or_hex.startswith("#") and name_or_hex.lower() != "unknown":
            print(f"\n{c.MSG_BOLD_COLORS['info']}name{c.RESET}              {c.BOLD_WHITE}: {name_or_hex}{c.RESET}")

    # 4. Luminance
    if luminance is not None and getattr(args, "luminance", False):
        print(f"\n{c.MSG_BOLD_COLORS['info']}luminance{c.RESET}         {c.BOLD_WHITE}: {luminance:.6f}{c.RESET}")
        if not hide_bars:
            print(f"                    {c.BOLD_WHITE}L{c.RESET} {_draw_bar(luminance, 1.0, 200, 200, 200)}")

    # 5. Color Spaces (Using pre-calculated tech_data)
    if getattr(args, "rgb", False):
        print(f"\n{c.MSG_BOLD_COLORS['info']}rgb{c.RESET}               {c.BOLD_WHITE}: {format_colorspace('rgb', r, g, b)}{c.RESET}")
        if not hide_bars:
            print(f"                    {c.BOLD_WHITE}R{c.RESET} {_draw_bar(r, 255, 255, 60, 60)} {c.BOLD_WHITE}{(r / 255) * 100:6.2f}%{c.RESET}")
            print(f"                    {c.BOLD_WHITE}G{c.RESET} {_draw_bar(g, 255, 60, 255, 60)} {c.BOLD_WHITE}{(g / 255) * 100:6.2f}%{c.RESET}")
            print(f"                    {c.BOLD_WHITE}B{c.RESET} {_draw_bar(b, 255, 60, 80, 255)} {c.BOLD_WHITE}{(b / 255) * 100:6.2f}%{c.RESET}")

    if "hsl" in data:
        h, s, l_hsl = data["hsl"]
        print(f"\n{c.MSG_BOLD_COLORS['info']}hsl{c.RESET}               {c.BOLD_WHITE}: {format_colorspace('hsl', h, s, l_hsl)}{c.RESET}")
        if not hide_bars:
            print(f"                    {c.BOLD_WHITE}H{c.RESET} {_draw_bar(h, 360, 255, 200, 0)}")
            print(f"                    {c.BOLD_WHITE}S{c.RESET} {_draw_bar(s, 1.0, 0, 200, 255)}")
            print(f"                    {c.BOLD_WHITE}L{c.RESET} {_draw_bar(l_hsl, 1.0, 200, 200, 200)}")

    if "hsv" in data:
        h, s, v = data["hsv"]
        print(f"\n{c.MSG_BOLD_COLORS['info']}hsv{c.RESET}               {c.BOLD_WHITE}: {format_colorspace('hsv', h, s, v)}{c.RESET}")
        if not hide_bars:
            print(f"                    {c.BOLD_WHITE}H{c.RESET} {_draw_bar(h, 360, 255, 200, 0)}")
            print(f"                    {c.BOLD_WHITE}S{c.RESET} {_draw_bar(s, 1.0, 0, 200, 255)}")
            print(f"                    {c.BOLD_WHITE}V{c.RESET} {_draw_bar(v, 1.0, 200, 200, 200)}")

    if "hwb" in data:
        h, w, b_hwb = data["hwb"]
        print(f"\n{c.MSG_BOLD_COLORS['info']}hwb{c.RESET}               {c.BOLD_WHITE}: {format_colorspace('hwb', h, w, b_hwb)}{c.RESET}")
        if not hide_bars:
            print(f"                    {c.BOLD_WHITE}H{c.RESET} {_draw_bar(h, 360, 255, 200, 0)}")
            print(f"                    {c.BOLD_WHITE}W{c.RESET} {_draw_bar(w, 1.0, 200, 200, 200)}")
            print(f"                    {c.BOLD_WHITE}B{c.RESET} {_draw_bar(b_hwb, 1.0, 100, 100, 100)}")

    if "cmyk" in data:
        cy, m, y_cmyk, k = data["cmyk"]
        print(f"\n{c.MSG_BOLD_COLORS['info']}cmyk{c.RESET}              {c.BOLD_WHITE}: {format_colorspace('cmyk', cy, m, y_cmyk, k)}{c.RESET}")
        if not hide_bars:
            print(f"                    {c.BOLD_WHITE}C{c.RESET} {_draw_bar(cy, 1.0, 0, 255, 255)}")
            print(f"                    {c.BOLD_WHITE}M{c.RESET} {_draw_bar(m, 1.0, 255, 0, 255)}")
            print(f"                    {c.BOLD_WHITE}Y{c.RESET} {_draw_bar(y_cmyk, 1.0, 255, 255, 0)}")
            print(f"                    {c.BOLD_WHITE}K{c.RESET} {_draw_bar(k, 1.0, 100, 100, 100)}")

    if "xyz" in data:
        x, y, z = data["xyz"]
        print(f"\n{c.MSG_BOLD_COLORS['info']}xyz{c.RESET}               {c.BOLD_WHITE}: {format_colorspace('xyz', x, y, z)}{c.RESET}")
        if not hide_bars:
            print(f"                    {c.BOLD_WHITE}X{c.RESET} {_draw_bar(x / 100.0, 1.0, 255, 60, 60)}")
            print(f"                    {c.BOLD_WHITE}Y{c.RESET} {_draw_bar(y / 100.0, 1.0, 60, 255, 60)}")
            print(f"                    {c.BOLD_WHITE}Z{c.RESET} {_draw_bar(z / 100.0, 1.0, 60, 80, 255)}")

    if "lab" in data:
        l_lab, a_lab, b_lab = data["lab"]
        a_comp, b_comp = _zero_small(a_lab), _zero_small(b_lab)
        print(f"\n{c.MSG_BOLD_COLORS['info']}lab{c.RESET}               {c.BOLD_WHITE}: {format_colorspace('lab', l_lab, a_comp, b_comp)}{c.RESET}")
        if not hide_bars:
            print(f"                    {c.BOLD_WHITE}L{c.RESET} {_draw_bar(l_lab / 100.0, 1.0, 200, 200, 200)}")
            print(f"                    {c.BOLD_WHITE}A{c.RESET} {_draw_bar(a_comp, 128.0, 60, 255, 60)}")
            print(f"                    {c.BOLD_WHITE}B{c.RESET} {_draw_bar(b_comp, 128.0, 60, 60, 255)}")

    if "lch" in data:
        l_lch, c_lch, h_lch = data["lch"]
        print(f"\n{c.MSG_BOLD_COLORS['info']}lch{c.RESET}               {c.BOLD_WHITE}: {format_colorspace('lch', l_lch, c_lch, h_lch)}{c.RESET}")
        if not hide_bars:
            print(f"                    {c.BOLD_WHITE}L{c.RESET} {_draw_bar(l_lch / 100.0, 1.0, 200, 200, 200)}")
            print(f"                    {c.BOLD_WHITE}C{c.RESET} {_draw_bar(c_lch / 150.0, 1.0, 255, 60, 255)}")
            print(f"                    {c.BOLD_WHITE}H{c.RESET} {_draw_bar(h_lch, 360, 255, 200, 0)}")

    if "luv" in data:
        l_uv, u_uv, v_uv = data["luv"]
        u_comp, v_comp = _zero_small(u_uv), _zero_small(v_uv)
        print(f"\n{c.MSG_BOLD_COLORS['info']}luv{c.RESET}               {c.BOLD_WHITE}: {format_colorspace('luv', l_uv, u_comp, v_comp)}{c.RESET}")
        if not hide_bars:
            print(f"                    {c.BOLD_WHITE}L{c.RESET} {_draw_bar(l_uv / 100.0, 1.0, 200, 200, 200)}")
            print(f"                    {c.BOLD_WHITE}U{c.RESET} {_draw_bar(u_comp, 100.0, 60, 255, 60)}")
            print(f"                    {c.BOLD_WHITE}V{c.RESET} {_draw_bar(v_comp, 100.0, 60, 60, 255)}")

    if "oklab" in data:
        l_ok, a_ok, b_ok = data["oklab"]
        a_comp, b_comp = _zero_small(a_ok), _zero_small(b_ok)
        print(f"\n{c.MSG_BOLD_COLORS['info']}oklab{c.RESET}             {c.BOLD_WHITE}: {format_colorspace('oklab', l_ok, a_comp, b_comp)}{c.RESET}")
        if not hide_bars:
            print(f"                    {c.BOLD_WHITE}L{c.RESET} {_draw_bar(l_ok, 1.0, 200, 200, 200)}")
            print(f"                    {c.BOLD_WHITE}A{c.RESET} {_draw_bar(a_comp, 0.4, 60, 255, 60)}")
            print(f"                    {c.BOLD_WHITE}B{c.RESET} {_draw_bar(b_comp, 0.4, 60, 60, 255)}")

    if "oklch" in data:
        l_oklch, c_oklch, h_oklch = data["oklch"]
        print(f"\n{c.MSG_BOLD_COLORS['info']}oklch{c.RESET}             {c.BOLD_WHITE}: {format_colorspace('oklch', l_oklch, c_oklch, h_oklch)}{c.RESET}")
        if not hide_bars:
            print(f"                    {c.BOLD_WHITE}L{c.RESET} {_draw_bar(l_oklch, 1.0, 200, 200, 200)}")
            print(f"                    {c.BOLD_WHITE}C{c.RESET} {_draw_bar(c_oklch / 0.4, 1.0, 255, 60, 255)}")
            print(f"                    {c.BOLD_WHITE}H{c.RESET} {_draw_bar(h_oklch, 360, 255, 200, 0)}")

    # 6. Contrast (Using pre-calculated wcag_data)
    if wcag_data and getattr(args, "contrast", False):
        bg_ansi = f"\033[48;2;{r};{g};{b}m"
        reset, info_c = "\033[0m", c.MSG_BOLD_COLORS["info"]
        succ_c, err_c = c.MSG_BOLD_COLORS["success"], c.MSG_BOLD_COLORS["error"]

        line_1_block = f"{bg_ansi}\033[1;38;2;255;255;255m{f'white':^16}{reset}"
        line_2_block = f"{bg_ansi}{'ㅤ' * 8}{reset}"
        line_3_block = f"{bg_ansi}\033[1;38;2;0;0;0m{f'black':^16}{reset}"

        def fmt_status(status: str) -> str:
            return f"{succ_c}Pass{info_c}" if status == "Pass" else f"{err_c}Fail{info_c}"

        w_aa, w_aaa = fmt_status(wcag_data["white"]["levels"]["AA"]), fmt_status(wcag_data["white"]["levels"]["AAA"])
        b_aa, b_aaa = fmt_status(wcag_data["black"]["levels"]["AA"]), fmt_status(wcag_data["black"]["levels"]["AAA"])

        s_white = f"{wcag_data['white']['ratio']:5.2f}:1 {info_c}(AA:{w_aa}, AAA:{w_aaa}){c.RESET}"
        s_black = f"{wcag_data['black']['ratio']:5.2f}:1 {info_c}(AA:{b_aa}, AAA:{b_aaa}){c.RESET}"

        print(f"\n                      {line_1_block}  {c.BOLD_WHITE}{s_white}{c.RESET}")
        print(f"{c.MSG_BOLD_COLORS['info']}contrast{c.RESET}          {c.BOLD_WHITE}:{c.RESET}   {line_2_block}")
        print(f"                      {line_3_block}  {c.BOLD_WHITE}{s_black}{c.RESET}")

    print()
