# File: luminance.py
from .conversions import _srgb_to_linear

# ==========================================
# Relative Luminance Coefficients
# ==========================================
# Source: ITU-R BT.709 (Rec. 709) and Web Content Accessibility Guidelines (WCAG) 2.1
# These coefficients represent the human eye's varying sensitivity to different wavelengths of light.
from ..constants.constants import (
    LUMA_R,
    LUMA_G,
    LUMA_B
)


def get_luminance(r: int, g: int, b: int) -> float:
    """
    Calculate the relative luminance (Y) of an sRGB color.
    
    Relative luminance is the relative brightness of any point in a colorspace,
    normalized to 0.0 for darkest black and 1.0 for lightest white.
    
    Args:
        r (int): Red channel value (0-255)
        g (int): Green channel value (0-255)
        b (int): Blue channel value (0-255)
        
    Returns:
        float: The relative luminance value between 0.0 and 1.0
    """
    return (
        LUMA_R * _srgb_to_linear(r) +
        LUMA_G * _srgb_to_linear(g) +
        LUMA_B * _srgb_to_linear(b)
    )