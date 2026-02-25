import math
from typing import Tuple

from ..constants.constants import EPS, POW7_25


def delta_e_ciede2000(
    lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]
) -> float:
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    C_bar = (C1 + C2) / 2
    
    C_bar_7 = C_bar ** 7

    G = 0.5 * (1 - math.sqrt(C_bar_7 / (C_bar_7 + POW7_25)))

    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2
    C1_prime = math.hypot(a1_prime, b1)
    C2_prime = math.hypot(a2_prime, b2)

    h1_prime_rad = math.atan2(b1, a1_prime) % (2 * math.pi)
    h1_prime_deg = math.degrees(h1_prime_rad)

    h2_prime_rad = math.atan2(b2, a2_prime) % (2 * math.pi)
    h2_prime_deg = math.degrees(h2_prime_rad)

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    C_prime_bar = (C1_prime + C2_prime) / 2

    if C1_prime * C2_prime == 0:
        delta_h_prime_deg = 0
    elif abs(h2_prime_deg - h1_prime_deg) <= 180:
        delta_h_prime_deg = h2_prime_deg - h1_prime_deg
    elif h2_prime_deg - h1_prime_deg > 180:
        delta_h_prime_deg = (h2_prime_deg - h1_prime_deg) - 360
    else:
        delta_h_prime_deg = (h2_prime_deg - h1_prime_deg) + 360

    delta_H_prime = 2 * math.sqrt(max(0.0, C1_prime * C2_prime)) * math.sin(
        math.radians(delta_h_prime_deg) / 2
    )

    L_prime_bar = (L1 + L2) / 2

    if C1_prime * C2_prime == 0:
        h_prime_bar_deg = h1_prime_deg + h2_prime_deg
    elif abs(h2_prime_deg - h1_prime_deg) <= 180:
        h_prime_bar_deg = (h1_prime_deg + h2_prime_deg) / 2
    elif (h1_prime_deg + h2_prime_deg) < 360:
        h_prime_bar_deg = (h1_prime_deg + h2_prime_deg + 360) / 2
    else:
        h_prime_bar_deg = (h1_prime_deg + h2_prime_deg - 360) / 2

    T = (
        1
        - 0.17 * math.cos(math.radians(h_prime_bar_deg - 30))
        + 0.24 * math.cos(math.radians(2 * h_prime_bar_deg))
        + 0.32 * math.cos(math.radians(3 * h_prime_bar_deg + 6))
        - 0.20 * math.cos(math.radians(4 * h_prime_bar_deg - 63))
    )

    S_L = 1 + (0.015 * (L_prime_bar - 50) ** 2) / math.sqrt(
        20 + (L_prime_bar - 50) ** 2 + EPS
    )
    S_C = 1 + 0.045 * C_prime_bar
    S_H = 1 + 0.015 * C_prime_bar * T

    delta_theta_deg = 30 * math.exp(-(((h_prime_bar_deg - 275) / 25) ** 2))
    C_prime_bar_7 = C_prime_bar ** 7

    R_C = 2 * math.sqrt(C_prime_bar_7 / (C_prime_bar_7 + POW7_25))
    R_T = -R_C * math.sin(math.radians(2 * delta_theta_deg))

    k_L, k_C, k_H = 1, 1, 1

    delta_E = math.sqrt(
        (delta_L_prime / (k_L * S_L)) ** 2 +
        (delta_C_prime / (k_C * S_C)) ** 2 +
        (delta_H_prime / (k_H * S_H)) ** 2 +
        R_T * (delta_C_prime / (k_C * S_C)) * (delta_H_prime / (k_H * S_H))
    )

    return delta_E


def delta_e_euclidean_rgb(
    rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]
) -> float:

    r1, g1, b1 = rgb1
    r2, g2, b2 = rgb2

    return math.sqrt((r1 - r2) * (r1 - r2) + (g1 - g2) * (g1 - g2) + (b1 - b2) * (b1 - b2))


def delta_e_euclidean_oklab(
    oklab1: Tuple[float, float, float], oklab2: Tuple[float, float, float]
) -> float:
    l1, a1, b1 = oklab1
    l2, a2, b2 = oklab2

    return math.sqrt((l1 - l2) * (l1 - l2) + (a1 - a2) * (a1 - a2) + (b1 - b2) * (b1 - b2))

