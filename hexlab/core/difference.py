#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/core/difference.py

import math
from typing import Tuple

from . import config as c


def delta_e_ciede2000(
    lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]
) -> float:
    """
    Calculate the CIEDE2000 color difference (ΔE_00) between two CIE LAB colors.
    This formula is the CIE recommendation for perceptual color difference.
    
    Source: Sharma, G., Wu, W., & Dalal, E. N. (2005).
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    C_bar = (C1 + C2) / c.DIV_2
    
    C_bar_7 = C_bar ** c.EXP_7
    G = c.G_FACTOR * (c.UNIT - math.sqrt(C_bar_7 / (C_bar_7 + c.POW7_25)))

    a1_prime = (c.UNIT + G) * a1
    a2_prime = (c.UNIT + G) * a2
    
    C1_prime = math.hypot(a1_prime, b1)
    C2_prime = math.hypot(a2_prime, b2)

    h1_prime_deg = math.degrees(math.atan2(b1, a1_prime) % (c.DIV_2 * math.pi))
    h2_prime_deg = math.degrees(math.atan2(b2, a2_prime) % (c.DIV_2 * math.pi))

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    C_prime_bar = (C1_prime + C2_prime) / c.DIV_2

    if C1_prime * C2_prime == 0:
        delta_h_prime_deg = 0
    elif abs(h2_prime_deg - h1_prime_deg) <= c.DEG_180:
        delta_h_prime_deg = h2_prime_deg - h1_prime_deg
    elif h2_prime_deg - h1_prime_deg > c.DEG_180:
        delta_h_prime_deg = (h2_prime_deg - h1_prime_deg) - c.DEG_360
    else:
        delta_h_prime_deg = (h2_prime_deg - h1_prime_deg) + c.DEG_360

    delta_H_prime = c.DIV_2 * math.sqrt(max(0.0, C1_prime * C2_prime)) * math.sin(
        math.radians(delta_h_prime_deg) / c.DIV_2
    )

    L_prime_bar = (L1 + L2) / c.DIV_2

    if C1_prime * C2_prime == 0:
        h_prime_bar_deg = h1_prime_deg + h2_prime_deg
    elif abs(h2_prime_deg - h1_prime_deg) <= c.DEG_180:
        h_prime_bar_deg = (h1_prime_deg + h2_prime_deg) / c.DIV_2
    elif (h1_prime_deg + h2_prime_deg) < c.DEG_360:
        h_prime_bar_deg = (h1_prime_deg + h2_prime_deg + c.DEG_360) / c.DIV_2
    else:
        h_prime_bar_deg = (h1_prime_deg + h2_prime_deg - c.DEG_360) / c.DIV_2

    T = (
        c.UNIT
        - c.T_K1 * math.cos(math.radians(h_prime_bar_deg - c.T_OFFSET_1))
        + c.T_K2 * math.cos(math.radians(c.DIV_2 * h_prime_bar_deg))
        + c.T_K3 * math.cos(math.radians(c.T_MUL_3 * h_prime_bar_deg + c.T_OFFSET_2))
        - c.T_K4 * math.cos(math.radians(c.T_MUL_4 * h_prime_bar_deg - c.T_OFFSET_3))
    )

    L_L_50_SQ = (L_prime_bar - c.L_OFFSET) ** c.EXP_2
    S_L = c.UNIT + (c.S_L_K * L_L_50_SQ) / math.sqrt(c.S_L_DIV + L_L_50_SQ + c.EPS)
    S_C = c.UNIT + c.S_C_K * C_prime_bar
    S_H = c.UNIT + c.S_L_K * C_prime_bar * T

    delta_theta_deg = c.RT_D30 * math.exp(-(((h_prime_bar_deg - c.RT_H_OFFSET) / c.RT_DIV) ** c.EXP_2))
    C_prime_bar_7 = C_prime_bar ** c.EXP_7

    R_C = c.DIV_2 * math.sqrt(C_prime_bar_7 / (C_prime_bar_7 + c.POW7_25))
    R_T = -R_C * math.sin(math.radians(c.DIV_2 * delta_theta_deg))

    k_L, k_C, k_H = c.K_FACTORS

    delta_E = math.sqrt(
        (delta_L_prime / (k_L * S_L)) ** c.EXP_2 +
        (delta_C_prime / (k_C * S_C)) ** c.EXP_2 +
        (delta_H_prime / (k_H * S_H)) ** c.EXP_2 +
        R_T * (delta_C_prime / (k_C * S_C)) * (delta_H_prime / (k_H * S_H))
    )

    return delta_E

def delta_e_euclidean_rgb(
    rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]
) -> float:
    """
    Calculate the standard 3D Euclidean distance between two RGB colors.
    Note: This is a mathematical distance, not a perceptual one.
    """
    r1, g1, b1 = rgb1
    r2, g2, b2 = rgb2
    return math.sqrt((r1 - r2) ** c.EXP_2 + (g1 - g2) ** c.EXP_2 + (b1 - b2) ** c.EXP_2)

def delta_e_euclidean_oklab(
    oklab1: Tuple[float, float, float], oklab2: Tuple[float, float, float]
) -> float:
    """
    Calculate the Euclidean distance between two OKLab colors.
    OKLab Euclidean distance provides a fast and accurate perceptual metric.
    """
    l1, a1, b1 = oklab1
    l2, a2, b2 = oklab2
    return math.sqrt((l1 - l2) ** c.EXP_2 + (a1 - a2) ** c.EXP_2 + (b1 - b2) ** c.EXP_2)