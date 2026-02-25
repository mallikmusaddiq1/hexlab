# File: wcag_contrast.py
from .luminance import get_luminance


def get_wcag_contrast(lum: float) -> dict:
    """
    Calculate WCAG 2.1 contrast ratios against pure white and pure black.
    
    Source: Web Content Accessibility Guidelines (WCAG) 2.1
    Formula: (L1 + 0.05) / (L2 + 0.05), where L is the relative luminance.
    
    Thresholds:
    - AA Large / UI Components: >= 3.0:1
    - AA Normal / AAA Large: >= 4.5:1
    - AAA Normal: >= 7.0:1
    """
    # Luminance of pure white is 1.0, pure black is 0.0
    contrast_white = (1.0 + 0.05) / (lum + 0.05)
    contrast_black = (lum + 0.05) / (0.0 + 0.05)

    def get_pass_fail(ratio: float) -> dict:
        return {
            "AA-Large": "Pass" if ratio >= 3 else "Fail",
            "AA": "Pass" if ratio >= 4.5 else "Fail",
            "AAA-Large": "Pass" if ratio >= 4.5 else "Fail",
            "AAA": "Pass" if ratio >= 7 else "Fail",
        }

    return {
        "white": {"ratio": contrast_white, "levels": get_pass_fail(contrast_white)},
        "black": {"ratio": contrast_black, "levels": get_pass_fail(contrast_black)},
    }


def _wcag_contrast_ratio_from_rgb(
    fr: float, fg: float, fb: float, br: float, bg: float, bb: float
) -> float:
    """
    Calculate the WCAG 2.1 contrast ratio between two specific RGB colors.
    
    Source: Web Content Accessibility Guidelines (WCAG) 2.1
    """
    # Calculate relative luminance for both colors
    y1 = get_luminance(int(round(fr)), int(round(fg)), int(round(fb)))
    y2 = get_luminance(int(round(br)), int(round(bg)), int(round(bb)))
    
    # Ensure y1 is always the lighter color (higher luminance) for the formula
    if y1 < y2:
        y1, y2 = y2, y1
        
    return (y1 + 0.05) / (y2 + 0.05)
