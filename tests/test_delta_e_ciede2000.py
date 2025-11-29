import math
from hexlab.color_math.distance.py import delta_e_ciede2000

cases = [
    ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
    ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
    ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0),
    ((100.0, 0.0, 0.0), (0.0, 0.0, 0.0), None),
]

def approx(a, b, tol=1e-3):
    return abs(a - b) <= tol

def test_sharma_vectors():
    for lab1, lab2, expected in cases[:3]:
        val = delta_e_ciede2000(lab1, lab2)
        assert approx(val, expected, tol=1e-3), f"{val} != {expected}"

def test_no_crash_extremes():
    delta_e_ciede2000(cases[3][0], cases[3][1])
    delta_e_ciede2000(cases[4][0], cases[4][1])
