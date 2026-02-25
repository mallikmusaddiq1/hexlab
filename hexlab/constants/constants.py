import re

# ==========================================
# Core Metadata & Precision
# ==========================================

__version__ = "v0.0.6"

# Floating-point precision and division-by-zero safety
EPS = 1e-12

# ==========================================
# Color Science Constants & Coefficients
# ==========================================

# Standard Scaling Constants
RGB_MAX = 255.0
HUE_MAX = 360.0
HUE_SECTOR = 60.0  # Degrees per HSL/HSV sector

# sRGB Transfer Function Constants (Source: IEC 61966-2-1:1999)
SRGB_SLOPE = 12.92
SRGB_OFFSET = 0.055
SRGB_DIVISOR = 1.055
SRGB_GAMMA = 2.4
SRGB_TO_LINEAR_TH = 0.04045
LINEAR_TO_SRGB_TH = 0.0031308

# XYZ D65 Reference White (Source: ASTM E308-01 / CIE D65)
D65_X = 95.047
D65_Y = 100.0
D65_Z = 108.883

# sRGB to XYZ Matrix (Source: sRGB D65)
M_SRGB_XYZ_X = (0.4124564, 0.3575761, 0.1804375)
M_SRGB_XYZ_Y = (0.2126729, 0.7151522, 0.0721750)
M_SRGB_XYZ_Z = (0.0193339, 0.1191920, 0.9503041)

# XYZ to sRGB Matrix (Source: sRGB D65 inverse)
M_XYZ_SRGB_R = (3.2404542, -1.5371385, -0.4985314)
M_XYZ_SRGB_G = (-0.9692660, 1.8760108, 0.0415560)
M_XYZ_SRGB_B = (0.0556434, -0.2040259, 1.0572252)

# CIELAB Constants (Source: CIE 15:2004)
LAB_E = 0.008856  # Actual: (6/29)^3
LAB_K = 7.787     # Actual: (1/3) * (29/6)^2
LAB_OFFSET = 16.0 / 116.0
LAB_L_MULT = 116.0
LAB_L_SUB = 16.0
LAB_A_MULT = 500.0
LAB_B_MULT = 200.0
LAB_POW = 1.0 / 3.0
LAB_INV_THR = 0.20689655  # 6/29

# OKLab Matrices (Source: Björn Ottosson, 2020)
M1_OKLAB = [
    (0.4122214708, 0.5363325363, 0.0514459929),
    (0.2119034982, 0.6806995451, 0.1073969566),
    (0.0883024619, 0.2817188376, 0.6299787005),
]
M2_OKLAB = [
    (0.2104542553, 0.7936177850, -0.0040720468),
    (1.9779984951, -2.4285922050, 0.4505937099),
    (0.0259040371, 0.7827717662, -0.8086757660),
]
M2_INV_OKLAB = [
    (1.0, 0.3963377774, 0.2158037573),
    (1.0, -0.1055613458, -0.0638541728),
    (1.0, -0.0894841775, -1.2914855480),
]
M1_INV_OKLAB = [
    (4.0767416621, -3.3077115913, 0.2309699292),
    (-1.2684380046, 2.6097574011, -0.3413193965),
    (-0.0041960863, -0.7034186147, 1.7076147010),
]

# CIELUV Constants (Source: CIELUV 1976)
LUV_U_V_MULT = 13.0
LUV_KAPPA = 903.3
LUV_U_NUM = 4.0
LUV_V_NUM = 9.0
LUV_DENOM_Y = 15.0
LUV_DENOM_Z = 3.0
LUV_Z_CONST = 12.0
LUV_Z_U_MULT = 3.0
LUV_Z_V_MULT = 20.0
LUV_L_THR = 8.0

# ==========================================
# Application Logic & Constraints
# ==========================================

# Limits for CLI and processing
MAX_DEC = 16777215  # Equivalent to #FFFFFF
MAX_STEPS = 1000
MAX_COUNT = 100

# CIEDE2000 Optimization Constant (25^7)
POW7_25 = 6103515625.0

# Default deduplication thresholds for distance metrics
DEDUP_DELTA_E_LAB = 7.7
DEDUP_DELTA_E_OKLAB = 0.077
DEDUP_DELTA_E_RGB = 27

# ==========================================
# CLI UI & Data Structures
# ==========================================

TECH_INFO_KEYS = [
    'index', 'rgb', 'luminance', 'hsl', 'hsv', 'cmyk',
    'contrast', 'xyz', 'lab', 'hwb', 'oklab', 'oklch',
    'cieluv', 'name', 'lch'
]

SCHEME_KEYS = [
    'complementary', 'split_complementary', 'analogous',
    'triadic', 'tetradic_square', 'tetradic_rectangular',
    'monochromatic'
]

SIMULATE_KEYS = ['protanopia', 'deuteranopia', 'tritanopia', 'achromatopsia']

# Color Blindness Simulation Matrices (Source: Machado et al.)
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
    'hex': 'hex', 'rgb': 'rgb', 'hsl': 'hsl', 'hsv': 'hsv',
    'hwb': 'hwb', 'cmyk': 'cmyk', 'xyz': 'xyz', 'lab': 'lab',
    'lch': 'lch', 'luv': 'luv', 'oklab': 'oklab', 'oklch': 'oklch',
    'index': 'index', 'name': 'name'
}

# Image processing and color adjustment sequence
PIPELINE = [
    "exposure", "gamma", "brightness", "contrast",
    "rotate", "rotate_oklch", "lighten", "darken",
    "saturate", "desaturate", "chroma_oklch", "vibrance_oklch",
    "whiten_hwb", "blacken_hwb",
    "warm_oklab", "cool_oklab", "tint",
    "red_channel", "green_channel", "blue_channel",
    "posterize", "threshold", "solarize", "sepia", "grayscale", "invert",
    "lock_luminance", "lock_rel_luminance", "target_rel_lum", "min_contrast",
    "opacity"
]

# Standard ANSI Escape Codes for UI
MSG_BOLD_COLORS = {
    "error": "\033[1;31m",
    "warning": "\033[1;33m",
    "info": "\033[1;36m",
    "success": "\033[1;32m",
    "dim": "\033[1;2;37m",
}

MSG_COLORS = {
    "error": "\033[0;31m",
    "warning": "\033[0;33m",
    "info": "\033[0;36m",
    "success": "\033[0;32m",
}

RESET = "\033[0m"
BOLD_WHITE = "\033[1;37m"