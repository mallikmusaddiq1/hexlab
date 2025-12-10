# File: constants.py
import re

MAX_DEC = 16777215
MAX_STEPS = 1000
MAX_COUNT = 100

DEDUP_DELTA_E_LAB = 7.7
DEDUP_DELTA_E_OKLAB = 0.077
DEDUP_DELTA_E_RGB = 27

SRGB_TO_LINEAR_TH = 0.04045
LINEAR_TO_SRGB_TH = 0.0031308
EPS = 1e-12

__version__ = "0.0.5"

# HEX_REGEX = re.compile(r"^(?:[0-9A-Fa-f]{6}|[0-9A-Fa-f]{3})$")

TECH_INFO_KEYS = [
    'index',
    'rgb',
    'luminance',
    'hsl',
    'hsv',
    'cmyk',
    'contrast',
    'xyz',
    'lab',
    'hwb',
    'oklab',
    'oklch',
    'cieluv',
    'name',
    'lch',
]

SCHEME_KEYS = [
    'complementary',
    'split_complementary',
    'analogous',
    'triadic',
    'tetradic_square',
    'tetradic_rectangular',
    'monochromatic',
]

SIMULATE_KEYS = [
    'protanopia',
    'deuteranopia',
    'tritanopia',
    'achromatopsia',
]

CB_MATRICES = {
    "Protanopia": [
        [0.56667, 0.43333, 0],
        [0.55833, 0.44167, 0],
        [0, 0.24167, 0.75833],
    ],
    "Deuteranopia": [
        [0.625, 0.375, 0],
        [0.70, 0.30, 0],
        [0, 0.30, 0.70],
    ],
    "Tritanopia": [
        [0.95, 0.05, 0],
        [0, 0.43333, 0.56667],
        [0, 0.475, 0.525],
    ],
}

FORMAT_ALIASES = {
    'hex': 'hex',
    'rgb': 'rgb',
    'hsl': 'hsl',
    'hsv': 'hsv',
    'hwb': 'hwb',
    'cmyk': 'cmyk',
    'xyz': 'xyz',
    'lab': 'lab',
    'lch': 'lch',
    'luv': 'luv',
    'oklab': 'oklab',
    'oklch': 'oklch',
    'index': 'index',
    'name': 'name',
}
