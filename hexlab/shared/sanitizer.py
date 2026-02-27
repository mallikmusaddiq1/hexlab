#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/shared/sanitizer.py

import argparse
import re
import sys

from hexlab.core import config as c
from .logger import log


def _sanitize_for_log(value) -> str:
    """
    Cleans up the input value for safe terminal logging by removing 
    excessive whitespace and newlines.
    """
    if value is None:
        return ""
    return " ".join(str(value).split())


def normalize_hex(value: str) -> str:
    """
    Normalizes various formats of hex strings into a standard 6-character uppercase hex.
    Handles shorthand formats (e.g., 'F', 'FF', 'FFF') by repeating characters appropriately.
    """
    if value is None:
        return ""
    # Remove hash symbol and spaces, convert to uppercase
    s = str(value).replace("#", "").replace(" ", "").upper()
    
    # Regex [0-9A-F] extracts only valid hexadecimal characters, ignoring any garbage input
    extracted = "".join(re.findall(r"[0-9A-F]", s))

    if not extracted:
        return ""

    L = len(extracted)
    if L == 6:
        return extracted
    if L == 3:
        # e.g., 'ABC' becomes 'AABBCC'
        return "".join([c * 2 for c in extracted])
    if L == 1:
        # e.g., 'A' becomes 'AAAAAA'
        return extracted * 6
    if L == 2:
        # e.g., 'AB' becomes 'ABABAB'
        return extracted * 3
    if L == 4:
        # e.g., 'ABCD' becomes 'ABCD00' (Appends zeros)
        return extracted + "00"
    if L == 5:
        # e.g., 'ABCDE' becomes 'ABCDE0' (Appends a zero)
        return extracted + "0"

    # If it's longer than 6, just truncate it to the first 6 characters
    return extracted[:6]


def _extract_positive_only_int(value: str) -> int:
    """
    Extracts a strictly positive integer from a string by stripping out 
    all non-numeric characters (including minus signs).
    """
    if value is None:
        return None
    s = str(value)

    # Regex [^0-9] matches anything that is NOT a digit (0-9) and removes it
    digits_only = re.sub(r"[^0-9]", "", s)

    if not digits_only:
        return None

    try:
        return int(digits_only)
    except ValueError:
        return None


def _extract_signed_int(value: str) -> int:
    """
    Extracts an integer from a string while preserving its mathematical sign (+ or -).
    Ignores alphabetical characters mixed in the string.
    """
    if value is None:
        return None

    s = str(value)
    
    # Check if the original string explicitly starts with a negative sign
    is_negative = s.strip().startswith("-")

    # Regex [0-9] extracts only the numeric digits
    digits_only = "".join(re.findall(r"[0-9]", s))

    if not digits_only:
        return None

    try:
        val = int(digits_only)
        # Re-apply the negative sign if it was present at the start
        if is_negative:
            val = -val
        return val
    except ValueError:
        return None


def _extract_signed_float(value: str) -> float:
    """
    Extracts a floating-point number from a string, preserving the sign and 
    handling multiple decimal points by keeping only the first one encountered.
    """
    if value is None:
        return None

    s = str(value)
    
    is_negative = s.strip().startswith("-")

    # Regex [0-9\.] extracts only numeric digits and literal dot (.) characters
    raw_chars = re.findall(r"[0-9\.]", s)
    if not raw_chars:
        return None
    
    clean_str = ""
    dot_seen = False
    
    # Reconstruct the float string ensuring only a single decimal point is kept
    for char in raw_chars:
        if char == '.':
            if not dot_seen:
                clean_str += char
                dot_seen = True
        else:
            clean_str += char
            
    # Return None if string is empty or just a lonely dot
    if not clean_str or clean_str == '.':
        return None

    try:
        val = float(clean_str)
        if is_negative:
            val = -val
        return val
    except ValueError:
        return None


def _extract_alpha_only(value: str) -> str:
    """
    Extracts only alphabetical characters from a string, lowercasing them.
    Useful for cleaning up color names or metric identifiers.
    """
    if value is None:
        return ""
    # Remove spaces and convert to lowercase
    s = str(value).replace(" ", "").lower()
    # Regex [a-z] extracts strictly english alphabet characters
    extracted = "".join(re.findall(r"[a-z]", s))
    return extracted


# ==========================================
# CLI Argument Type Handlers (Validators)
# ==========================================

def handle_hex(v: str) -> str:
    """Validator for hex string CLI arguments."""
    cleaned = normalize_hex(v)
    if not cleaned:
        raw = _sanitize_for_log(v)
        raise argparse.ArgumentTypeError(f"invalid hex value: '{raw}'")
    return cleaned


def handle_decimal_index(v: str) -> str:
    """
    Validator for decimal indices. Clamps the value between 0 and MAX_DEC, 
    and returns it formatted as a 6-character hex string.
    """
    val = _extract_positive_only_int(v)

    if val is None:
        raw = _sanitize_for_log(v)
        raise argparse.ArgumentTypeError(f"invalid decimal index: '{raw}'")

    if val < 0:
        val = 0
    if val > c.MAX_DEC:
        val = c.MAX_DEC

    return f"{val:06X}"


def handle_color_name(v: str) -> str:
    """Validator for alphabetical color names."""
    cleaned = _extract_alpha_only(v)
    if not cleaned:
        raw = _sanitize_for_log(v)
        raise argparse.ArgumentTypeError(f"invalid color name: '{raw}'")
    return cleaned


def handle_string_clean(v: str) -> str:
    """Validator for pure alphabetical string options (e.g., format names)."""
    cleaned = _extract_alpha_only(v)
    if not cleaned:
        raw = _sanitize_for_log(v)
        raise argparse.ArgumentTypeError(f"invalid string value: '{raw}'")
    return cleaned


def handle_float_any(v: str) -> float:
    """Validator for unbounded floating-point CLI arguments."""
    val = _extract_signed_float(v)
    if val is None:
        raw = _sanitize_for_log(v)
        raise argparse.ArgumentTypeError(f"invalid numeric value: '{raw}'")
    return val


def handle_int_range(min_v: int, max_v: int):
    """
    Factory function returning a validator that ensures an integer 
    is clamped within a specific [min_v, max_v] range.
    """
    def validator(v: str) -> int:
        val = _extract_signed_int(v)

        if val is None:
            raw = _sanitize_for_log(v)
            raise argparse.ArgumentTypeError(f"invalid integer value: '{raw}'")

        if val < min_v:
            val = min_v
        elif val > max_v:
            val = max_v
        return val
    return validator


def handle_positive_int(min_v: int, max_v: int):
    """
    Factory function returning a validator that specifically handles 
    positive integers clamped within a given range.
    """
    def validator(v: str) -> int:
        val = _extract_positive_only_int(v)

        if val is None:
            raw = _sanitize_for_log(v)
            raise argparse.ArgumentTypeError(f"invalid numeric value: '{raw}'")

        if val < min_v:
            val = min_v
        elif val > max_v:
            val = max_v
        return val
    return validator


def handle_float_range(min_v: float, max_v: float):
    """
    Factory function returning a validator that ensures a float 
    is clamped within a specific [min_v, max_v] range.
    """
    def validator(v: str) -> float:
        val = _extract_signed_float(v)

        if val is None:
            raw = _sanitize_for_log(v)
            raise argparse.ArgumentTypeError(f"invalid float value: '{raw}'")

        if val < min_v:
            val = min_v
        elif val > max_v:
            val = max_v
        return val
    return validator


# ==========================================
# Central Mapping for Argparse types
# ==========================================

# This dictionary maps custom CLI argument types to their respective parsing functions.
INPUT_HANDLERS = {
    "hex": handle_hex,
    "decimal_index": handle_decimal_index,
    "color_name": handle_color_name,
    "colorspace": handle_string_clean,
    "distance_metric": handle_string_clean,
    "harmony_model": handle_string_clean,
    "from_format": handle_string_clean,
    "to_format": handle_string_clean,
    "dedup_value": handle_float_any,
    "float": handle_float_any,

    "float_0_1": handle_float_range(0.0, 1.0),
    "float_0_100": handle_float_range(0.0, 100.0),
    "float_signed_100": handle_float_range(-100.0, 100.0),
    "float_signed_360": handle_float_range(-360.0, 360.0),

    "count": handle_int_range(2, c.MAX_COUNT),
    "count_similar": handle_int_range(2, 250),
    "count_distinct": handle_int_range(2, 250),
    "seed": handle_int_range(0, 999_999_999_999_999_999),
    "steps": handle_int_range(1, c.MAX_STEPS),
    "int_channel": handle_int_range(-255, 255),
    "custom_scheme": handle_int_range(-360, 360),
    "intensity": handle_positive_int(0, 100)
}