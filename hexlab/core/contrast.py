#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/core/contrast.py

from . import config as c
from .luminance import get_luminance

def get_wcag_contrast(lum: float) -> dict:
    """
    Calculate WCAG contrast ratios against pure white and pure black.
    
    Source: Web Content Accessibility Guidelines (WCAG) 2.1
    Formula: (L1 + 0.05) / (L2 + 0.05), where L is the relative luminance.
    """
    
    contrast_white = (c.UNIT + c.WCAG_LUMINANCE_OFFSET) / (lum + c.WCAG_LUMINANCE_OFFSET)
    contrast_black = (lum + c.WCAG_LUMINANCE_OFFSET) / (c.GAMMA_MIN + c.WCAG_LUMINANCE_OFFSET)

    def get_pass_fail(ratio: float) -> dict:
        return {
            "AA-Large": "Pass" if ratio >= c.WCAG_AA_LARGE else "Fail",
            "AA": "Pass" if ratio >= c.WCAG_AA_NORMAL else "Fail",
            "AAA-Large": "Pass" if ratio >= c.WCAG_AAA_LARGE else "Fail",
            "AAA": "Pass" if ratio >= c.WCAG_AAA_NORMAL else "Fail",
        }

    return {
        "white": {
            "ratio": round(contrast_white, c.EXP_2), 
            "levels": get_pass_fail(contrast_white)
        },
        "black": {
            "ratio": round(contrast_black, c.EXP_2), 
            "levels": get_pass_fail(contrast_black)
        },
    }

def get_contrast_ratio_rgb(c1: tuple, c2: tuple) -> float:
    """
    Calculate the WCAG 2.1 contrast ratio between two specific RGB colors.
    
    Source: https://www.w3.org/TR/WCAG21/#dfn-contrast-ratio
    """
    y1 = get_luminance(*c1)
    y2 = get_luminance(*c2)
    
    l1, l2 = (y1, y2) if y1 > y2 else (y2, y1)
    
    return (l1 + c.WCAG_LUMINANCE_OFFSET) / (l2 + c.WCAG_LUMINANCE_OFFSET)