#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/core/config.py

# ==========================================
# Color Science Constants & Coefficients
# ==========================================

# Max size for conversion cache
LRU_CACHE_SIZE = 1024

EPS = 1e-12                        # Floating-point precision and division-by-zero safety
LUM_LOCK_EPS = 1e-7                # Precision for luminance locking

# Relative Luminance Coefficients (Source: ITU-R BT.709 / Rec. 709)
LUMA_R = 0.2126                    # Red component contribution to relative luminance
LUMA_G = 0.7152                    # Green component contribution to relative luminance
LUMA_B = 0.0722                    # Blue component contribution to relative luminance

# WCAG Contrast Thresholds (Source: https://www.w3.org/TR/WCAG21/#contrast-minimum)
WCAG_AA_LARGE = 3.0                # Minimum contrast for large text (Level AA)
WCAG_AA_NORMAL = 4.5               # Minimum contrast for normal text (Level AA)
WCAG_AAA_LARGE = 4.5               # Enhanced contrast for large text (Level AAA)
WCAG_AAA_NORMAL = 7.0              # Enhanced contrast for normal text (Level AAA)
WCAG_MIN_RATIO = 1.0               # Lower bound for WCAG calculation
WCAG_MAX_RATIO = 21.0              # Upper bound for WCAG calculation (Black on White)
CONTRAST_MIN_ABS = 0.001           # Minimum threshold for contrast adjustment

# Standard Scaling & Mathematical Constants
UNIT = 1.0                         # Normalized maximum
DIV_2 = 2.0                        # Standard divisor for averages
RGB_MAX = 255.0                    # 8-bit color depth limit
HUE_MAX = 360.0                    # Full circle degrees
HUE_SECTOR = 60.0                  # Degrees per HSL/HSV sector
HSL_HUE_MOD = 6.0                  # Hue sector divisor for HSL/HSV
EXP_2 = 2                          # Square power
EXP_7 = 7                          # Power for CIEDE2000 chroma calculation

# sRGB Transfer Function Constants (Source: IEC 61966-2-1:1999)
SRGB_SLOPE = 12.92                 # Slope of the linear portion of the sRGB curve
SRGB_OFFSET = 0.055                # Constant offset used in the non-linear sRGB segment
SRGB_DIVISOR = 1.055               # Divisor for normalizing the sRGB component
SRGB_GAMMA = 2.4                   # Effective gamma exponent for sRGB transfer
SRGB_TO_LINEAR_TH = 0.04045        # Threshold for switching from linear to non-linear sRGB
LINEAR_TO_SRGB_TH = 0.0031308      # Threshold for switching from linear to sRGB space

# XYZ D65 Reference White (Source: ASTM E308-01 / CIE D65)
D65_X = 95.047                     # X coordinate for D65 illuminant (2-degree observer)
D65_Y = 100.0                      # Y coordinate (Luminance) for D65 illuminant
D65_Z = 108.883                    # Z coordinate for D65 illuminant

# sRGB to XYZ Matrix (Source: sRGB D65)
M_SRGB_XYZ_X = (0.4124564, 0.3575761, 0.1804375)  # Coefficients for X coordinate calculation
M_SRGB_XYZ_Y = (0.2126729, 0.7151522, 0.0721750)  # Coefficients for Y (Luminance) calculation
M_SRGB_XYZ_Z = (0.0193339, 0.1191920, 0.9503041)  # Coefficients for Z coordinate calculation

# XYZ to sRGB Matrix (Source: sRGB D65 inverse)
M_XYZ_SRGB_R = (3.2404542, -1.5371385, -0.4985314)  # Coefficients for linear Red component calculation
M_XYZ_SRGB_G = (-0.9692660, 1.8760108, 0.0415560)   # Coefficients for linear Green component calculation
M_XYZ_SRGB_B = (0.0556434, -0.2040259, 1.0572252)   # Coefficients for linear Blue component calculation

# CIELAB Constants (Source: CIE 15:2004)
LAB_E = 0.008856                   # Threshold for switching between linear and power functions
LAB_K = 7.787                      # Slope of the linear segment for low luminance values
LAB_OFFSET = 16.0 / 116.0          # Constant offset for normalization in XYZ to Lab conversion
LAB_L_MULT = 116.0                 # Multiplier for Lightness (L*) calculation
LAB_L_SUB = 16.0                   # Subtraction constant for Lightness (L*) calculation
LAB_A_MULT = 500.0                 # Multiplier for 'a*' (green-red) channel calculation
LAB_B_MULT = 200.0                 # Multiplier for 'b*' (blue-yellow) channel calculation
LAB_INV_THR = 0.20689655           # Threshold for inverse conversion (Lab to XYZ)

# CIEDE2000 Constants (Source: Sharma, G., Wu, W., & Dalal, E. N. (2005))
POW7_25 = 6103515625.0             # Constant for chroma normalization (25^7)
G_FACTOR = 0.5                     # Axial adjustment factor for neutral gray
T_K1 = 0.17                        # First T-factor coefficient for hue weighting
T_K2 = 0.24                        # Second T-factor coefficient for hue weighting
T_K3 = 0.32                        # Third T-factor coefficient for hue weighting
T_K4 = 0.20                        # Fourth T-factor coefficient for hue weighting
T_OFFSET_1 = 30.0                  # Primary phase offset for hue angle T-factor
T_OFFSET_2 = 6.0                   # Secondary phase offset for hue angle T-factor
T_OFFSET_3 = 63.0                  # Tertiary phase offset for hue angle T-factor
T_MUL_3 = 3.0                      # Multiplier for tertiary hue angle calculation
T_MUL_4 = 4.0                      # Multiplier for quaternary hue angle calculation
L_OFFSET = 50.0                    # Lightness midpoint for S_L weighting function
S_L_K = 0.015                      # Lightness weighting coefficient for S_L
S_C_K = 0.045                      # Chroma weighting coefficient for S_C
S_L_DIV = 20.0                     # Divisor term for S_L weighting calculation
RT_D30 = 30.0                      # Degree factor for rotation term (R_T) calculation
RT_H_OFFSET = 275.0                # Hue offset for blue region in R_T calculation
RT_DIV = 25.0                      # Hue divisor for blue region in R_T calculation
K_FACTORS = (1.0, 1.0, 1.0)        # Parametric weighting factors (k_L, k_C, k_H)

# Color Blindness Simulation Matrices (Source: Machado et al., 2009)
CB_MATRICES = {
    "Protanopia": [
        [0.56667, 0.43333, 0],      # Transformation for Red-blindness (L-cone deficiency)
        [0.55833, 0.44167, 0],      # Mapping spectral sensitivity to remaining cones
        [0, 0.24167, 0.75833],      # Final Z-axis adjustment for Protan simulation
    ],
    "Deuteranopia": [
        [0.625, 0.375, 0],          # Transformation for Green-blindness (M-cone deficiency)
        [0.70, 0.30, 0],            # Mapping spectral sensitivity to remaining cones
        [0, 0.30, 0.70],            # Final Z-axis adjustment for Deutan simulation
    ],
    "Tritanopia": [
        [0.95, 0.05, 0],            # Transformation for Blue-blindness (S-cone deficiency)
        [0, 0.43333, 0.56667],      # Mapping spectral sensitivity to remaining cones
        [0, 0.475, 0.525],          # Final Z-axis adjustment for Tritan simulation
    ],
}

# CIELUV Constants (Source: CIELUV 1976 / CIE 15:2004)
LUV_U_V_MULT = 13.0                 # Multiplier for 'u' and 'v' chromaticity coordinates
LUV_KAPPA = 903.3                   # Constant used for Lightness (L*) calculation in the linear segment
LUV_U_NUM = 4.0                     # Numerator coefficient for u' chromaticity calculation
LUV_V_NUM = 9.0                     # Numerator coefficient for v' chromaticity calculation
LUV_DENOM_Y = 15.0                  # Y-coefficient for the denominator in chromaticity formulas
LUV_DENOM_Z = 3.0                   # Z-coefficient for the denominator in chromaticity formulas
LUV_Z_CONST = 12.0                  # Constant used in the Z coordinate derivation from LUV
LUV_L_THR = 8.0                     # Lightness threshold for switching between L* calculation methods
LUV_Z_U_MULT = 3.0                  # Z-axis multiplier for LUV to XYZ conversion
LUV_Z_V_MULT = 20.0                 # Z-axis offset multiplier for LUV to XYZ conversion

# Constants for OKLab color space conversions (Source: https://bottosson.github.io/posts/oklab/)
OKLAB_D65_MID_GRAY = 0.18           # sRGB reflectance value for standard 18% mid-gray
OKLAB_CUBE_ROOT_EXP = 1.0 / 3.0     # Power exponent for perceptual LMS non-linearity
XYZ_SCALING = 100.0                 # Factor for normalizing/scaling XYZ coordinates

# XYZ to LMS matrix coefficients (Source: Björn Ottosson, 2020)
OKLAB_XYZ_TO_LMS_RR = 0.4122214708  # Contribution of X to Long-wavelength (L) response
OKLAB_XYZ_TO_LMS_RG = 0.5363325363  # Contribution of Y to Long-wavelength (L) response
OKLAB_XYZ_TO_LMS_RB = 0.0514459929  # Contribution of Z to Long-wavelength (L) response
OKLAB_XYZ_TO_LMS_GR = 0.2119034982  # Contribution of X to Medium-wavelength (M) response
OKLAB_XYZ_TO_LMS_GG = 0.6806995451  # Contribution of Y to Medium-wavelength (M) response
OKLAB_XYZ_TO_LMS_GB = 0.1073969566  # Contribution of Z to Medium-wavelength (M) response
OKLAB_XYZ_TO_LMS_BR = 0.0883024619  # Contribution of X to Short-wavelength (S) response
OKLAB_XYZ_TO_LMS_BG = 0.2817188376  # Contribution of Y to Short-wavelength (S) response
OKLAB_XYZ_TO_LMS_BB = 0.6299787005  # Contribution of Z to Short-wavelength (S) response

# LMS to Lab matrix coefficients (Perceptual lightness and opponency)
OKLAB_LMS_TO_LAB_LL = 0.2104542553         # Weight for Lightness (L) component
OKLAB_LMS_TO_LAB_LM = 0.7936177850         # Weight for L-M difference in Lightness
OKLAB_LMS_TO_LAB_LS = -0.0040720468        # Weight for S component in Lightness
OKLAB_LMS_TO_LAB_AL = 1.9779984951         # Weight for Lightness (L) in 'a' (green-red) component
OKLAB_LMS_TO_LAB_AM = -2.4285922050        # Weight for M in 'a' (green-red) component
OKLAB_LMS_TO_LAB_AS = 0.4505937099         # Weight for S in 'a' (green-red) component
OKLAB_LMS_TO_LAB_BL = 0.0259040371         # Weight for Lightness (L) in 'b' (blue-yellow) component
OKLAB_LMS_TO_LAB_BM = 0.7827717662         # Weight for M in 'b' (blue-yellow) component
OKLAB_LMS_TO_LAB_BS = -0.8086757660        # Weight for S in 'b' (blue-yellow)

# OKLab to LMS' matrix coefficients (Inverse stage part 1)
OKLAB_TO_LMS_PRIME_LA = 0.3963377774       # Contribution of 'a' to L' channel
OKLAB_TO_LMS_PRIME_LB = 0.2158037573       # Contribution of 'b' to L' channel
OKLAB_TO_LMS_PRIME_MA = -0.1055613458      # Contribution of 'a' to M' channel
OKLAB_TO_LMS_PRIME_MB = -0.0638541728      # Contribution of 'b' to M' channel
OKLAB_TO_LMS_PRIME_SA = -0.0894841775      # Contribution of 'a' to S' channel
OKLAB_TO_LMS_PRIME_SB = -1.2914855480      # Contribution of 'b' to S' channel

# LMS' to XYZ matrix coefficients (Inverse stage part 2)
OKLAB_LMS_PRIME_TO_XYZ_RL = 4.0767416621   # Weight of L' for linear Red
OKLAB_LMS_PRIME_TO_XYZ_RM = -3.3077115913  # Weight of M' for linear Red
OKLAB_LMS_PRIME_TO_XYZ_RS = 0.2309699292   # Weight of S' for linear Red
OKLAB_LMS_PRIME_TO_XYZ_GL = -1.2684380046  # Weight of L' for linear Green
OKLAB_LMS_PRIME_TO_XYZ_GM = 2.6097574011   # Weight of M' for linear Green
OKLAB_LMS_PRIME_TO_XYZ_GS = -0.3413193965  # Weight of S' for linear Green
OKLAB_LMS_PRIME_TO_XYZ_BL = -0.0041960863  # Weight of L' for linear Blue
OKLAB_LMS_PRIME_TO_XYZ_BM = -0.7034186147  # Weight of M' for linear Blue
OKLAB_LMS_PRIME_TO_XYZ_BS = 1.7076147010   # Weight of S' for linear Blue

# ==========================================
# Application Logic & Constraints
# ==========================================

MAX_DEC = 16777215                 # Max integer value for 24-bit Hex (0xFFFFFF)
MAX_STEPS = 1000                   # Iteration safety limit to prevent infinite loops
MAX_COUNT = 100                    # Maximum number of colors allowed in batch processing

# Default Deduplication Thresholds (Empirical values for "similar" colors)
DEDUP_DELTA_E_LAB = 7.7            # Delta-E threshold in CIELAB space
DEDUP_DELTA_E_OKLAB = 0.077        # Delta-E threshold in OKLab space
DEDUP_DELTA_E_RGB = 27             # Euclidean distance threshold in RGB space


# ==========================================
# Adjustment & Pipeline Constants
# ==========================================

# Clamping and Iteration limits
RGB_CLAMP_TOLERANCE_LOWER = -0.5         # Lower bound tolerance for gamut mapping and rounding
RGB_CLAMP_TOLERANCE_UPPER = 255.5        # Upper bound tolerance for gamut mapping and rounding
GAMUT_MAP_BINARY_SEARCH_ITERATIONS = 20  # Binary search steps for chroma-based gamut mapping
CONTRAST_BINARY_SEARCH_ITERATIONS = 30   # Binary search steps for target contrast matching

# Filter Coefficients (Source: W3C / Standard Image Processing)
SEPIA_RR = 0.393                         # Red contribution to the output Red channel
SEPIA_RG = 0.769                         # Green contribution to the output Red channel
SEPIA_RB = 0.189                         # Blue contribution to the output Red channel
SEPIA_GR = 0.349                         # Red contribution to the output Green channel
SEPIA_GG = 0.686                         # Green contribution to the output Green channel
SEPIA_GB = 0.168                         # Blue contribution to the output Green channel
SEPIA_BR = 0.272                         # Red contribution to the output Blue channel
SEPIA_BG = 0.534                         # Green contribution to the output Blue channel
SEPIA_BB = 0.131                         # Blue contribution to the output Blue channel

# OKLab Adjustment Scales
WARM_OKLAB_A_SCALE = 2000.0              # Denominator for 'a' channel shift in temperature adjustments
WARM_OKLAB_B_SCALE = 1000.0              # Denominator for 'b' channel shift in temperature adjustments
VIBRANCE_NORMALIZATION_MAX_CHROMA = 0.4  # Max chroma reference for scaling vibrance non-linearly

# Logic & Math Helpers
PERCENT_TO_FACTOR = 100.0                # Divisor to convert percentage values to decimal factors
EXPOSURE_STOPS_SCALE = 10.0              # Scaling factor for exposure stops (10% change per stop)
POSTERIZE_MIN_LEVELS = 2                 # Minimum number of levels allowed for posterization
POSTERIZE_MAX_LEVELS = 256               # Maximum number of levels allowed for posterization
CHANNEL_MIN = -255                       # Lower bound for manual color channel adjustments
CHANNEL_MAX = 255                        # Upper bound for manual color channel adjustments
AVG_DIVISOR = 3.0                        # Divisor used for calculating simple R+G+B averages
GAMMA_MIN = 0.0                          # Minimum valid value for gamma correction
WCAG_LUMINANCE_OFFSET = 0.05             # Standard offset constant in the (L + 0.05) contrast formula

# ==========================================
# CLI UI & Data Structures
# ==========================================

# Keys used to extract and format technical color data
TECH_INFO_KEYS = [
    'index',
    'rgb',
    'luminance',
    'hsl',
    'hsv',
    'cmyk',
    'contrast',
    'xyz', 'lab',
    'hwb',
    'oklab',
    'oklch',
    'cieluv',
    'name',
    'lch'
]

# Supported color harmony schemes
SCHEME_KEYS = [
    'complementary',
    'split_complementary',
    'analogous',
    'triadic',
    'tetradic_square',
    'tetradic_rectangular',
    'monochromatic'
]

# Image processing pipeline order
PIPELINE = [
    "exposure",
    "gamma",
    "brightness",
    "contrast",
    "rotate",
    "rotate_oklch",
    "lighten",
    "darken",
    "saturate",
    "desaturate",
    "chroma_oklch", 
    "vibrance_oklch",
    "whiten_hwb",
    "blacken_hwb",
    "warm_oklab",
    "cool_oklab", 
    "tint",
    "red_channel",
    "green_channel", 
    "blue_channel",
    "posterize",
    "threshold",
    "solarize",
    "sepia", "grayscale", 
    "invert",
    "lock_luminance", 
    "lock_rel_luminance", 
    "target_rel_lum",
    "min_contrast",
    "opacity"
]

# Format aliases for 'convert' command
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

# ANSI Terminal Styling
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