# File: constants.py
import re

MAX_DEC = 16777215
MAX_STEPS = 1000
MAX_COUNT = 100

POW7_25 = 6103515625.0

DEDUP_DELTA_E_LAB = 7.7
DEDUP_DELTA_E_OKLAB = 0.077
DEDUP_DELTA_E_RGB = 27

SRGB_TO_LINEAR_TH = 0.04045
LINEAR_TO_SRGB_TH = 0.0031308
EPS = 1e-12

__version__ = "v0.0.6"

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

PIPELINE = [
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

# Standard ANSI Color Codes
MSG_BOLD_COLORS = {
    "error": "\033[1;31m",  # Bold Red
    "warning": "\033[1;33m", # Bold Yellow
    "info": "\033[1;36m",    # Bold Cyan
    "success": "\033[1;32m", # Bold Green
    "dim": "\033[1;2;37m",      # Dim Bold White
}

MSG_COLORS = {
    "error": "\033[0;31m",   # Normal Red
    "warning": "\033[0;33m", # Normal Yellow
    "info": "\033[0;36m",    # Normal Cyan
    "success": "\033[0;32m", # Normal Green
}

RESET = "\033[0m"
BOLD_WHITE = "\033[1;37m"
