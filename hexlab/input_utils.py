#!/usr/bin/env python3
"""
Shared input handling & sanitization utilities for hexlab.

Rules (summary):

- Hex (-H/--hex):
  - Extract only 0-9 A-F (case-insensitive), remove spaces.
  - Uppercase output.
  - If length == 3 -> expand like CSS (#F0A -> #FF00AA style) by doubling each char.
  - If length < 6 -> left-pad with '0' to 6 chars.
  - If length > 6 -> take first 6 chars.
  - If after sanitization nothing remains -> argparse.ArgumentTypeError.

- Decimal index (-di/--decimal-index):
  - Remove spaces.
  - Extract only digits 0-9 (minus sign ignored, so "-5" -> "5").
  - Convert to int, clamp to [0, MAX_DEC].
  - Return as string (for later decimal_to_hex_str).
  - If after sanitization no digits -> argparse.ArgumentTypeError.

- Color name (-cn/--color-name) and other string flags (colorspace, distance-metric, harmony-model):
  - Remove spaces.
  - Extract only letters a-z.
  - Lowercase.
  - If after sanitization empty -> argparse.ArgumentTypeError.

- Float-ish flags (e.g. dedup-value):
  - Remove spaces.
  - Extract first float pattern: digits with optional decimal point.
  - Parse as float.
  - If no float pattern found -> argparse.ArgumentTypeError.

- Range-based integer flags (number, seed, steps, total_random, etc.):
  - Use only digits 0-9, remove spaces.
  - If no digits -> argparse.ArgumentTypeError.
  - Convert to int, clamp to [min_val, max_val].

- Range-based float flags (vision parameters like intensity, severity):
  - Use digits and decimal point, remove spaces.
  - If no float -> argparse.ArgumentTypeError.
  - Convert to float, clamp to [min_val, max_val].
"""

import argparse
import re
import sys

# --- CONSTANTS ---
MAX_DEC = 16777215
MAX_RANDOM_COLORS = 100
MAX_STEPS = 10000


# --- LOGGING ---

def log(level: str, message: str) -> None:
    level = str(level).lower()
    stream = sys.stdout if level == "info" else sys.stderr
    print(f"[hexlab][{level}] {message}", file=stream)


def _sanitize_for_log(value) -> str:
    """
    Collapse whitespace, limit length, safe for showing in error messages.
    """
    if value is None:
        return ""
    s = str(value)
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 200:
        s = s[:197] + "..."
    return s


# --- 1. CORE EXTRACTION LOGIC (The Cleaners) ---


def _extract_hex(value: str) -> str:
    """
    Requirements: 0-9, a-f extract. Upper case. Remove spaces.
    Padding: 'F' -> '00000F'.
    Clamping: Takes first 6 valid chars.
    Ignores: Any special chars, unicode.

    NOTE: If no valid hex chars -> returns empty string.
    Caller (argparse type handler) MUST raise ArgumentTypeError on empty.
    """
    if value is None:
        return ""
    s = str(value).replace(" ", "").upper()
    extracted = "".join(re.findall(r"[0-9A-F]", s))

    if not extracted:
        return ""

    # 3-digit expand (CSS-style shorthand)
    if len(extracted) == 3:
        extracted = "".join([c * 2 for c in extracted])

    # Left-pad with 0 if < 6
    if len(extracted) < 6:
        extracted = extracted.zfill(6)

    # Truncate to 6 chars
    return extracted[:6]


def _extract_int(value: str, min_val: int, max_val: int):
    """
    Requirements: Extract integer pattern (including negative signs).
    Remove spaces.
    Clamping: min_val <= value <= max_val.

    Returns:
        int value in [min_val, max_val] if digits found.
        None if no digits after sanitization.
    """
    if value is None:
        return None
    s = str(value).replace(" ", "")
    # Regex to capture optional +/- and digits
    match = re.search(r"[-+]?\d+", s)

    if not match:
        return None

    val = int(match.group())
    if val < min_val:
        val = min_val
    elif val > max_val:
        val = max_val
    return val


def _extract_alpha(value: str) -> str:
    """
    Requirements: a-z extract only. Lower case. Remove spaces.

    Returns:
        sanitized lower-case alpha string (may be empty).
    """
    if value is None:
        return ""
    s = str(value).replace(" ", "").lower()
    extracted = "".join(re.findall(r"[a-z]", s))
    return extracted


def _extract_float(value: str):
    """
    Requirements: Digits, dot, and sign extract.

    Returns:
        float parsed from first "[-+]digits[.digits]" pattern.
        None if no such pattern found.
    """
    if value is None:
        return None
    s = str(value).replace(" ", "")
    # Regex to capture optional +/- , digits, optional dot, digits
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if match:
        try:
            return float(match.group())
        except Exception:
            return None
    return None


# --- 2. ARGPARSE TYPE HANDLERS (Wrappers) ---
# These are meant to be used as `type=` in argparse.add_argument()


def handle_hex(v: str) -> str:
    """
    For -H/--hex flags.

    On success:
        returns sanitized 6-digit uppercase hex string.

    On failure (no valid hex chars after sanitization):
        raises argparse.ArgumentTypeError.
    """
    cleaned = _extract_hex(v)
    if not cleaned:
        raw = _sanitize_for_log(v).replace(" ", "")
        raise argparse.ArgumentTypeError(
            f"invalid hex value: '{raw}'"
        )
    return cleaned


def handle_decimal_index(v: str) -> str:
    """
    For -di/--decimal-index flags.

    Behaviour:
        - Extract digits only.
        - Clamp to [0, MAX_DEC].
        - Return as string, for later decimal_to_hex_str() usage.

    If no digits remain after sanitization:
        raises argparse.ArgumentTypeError.
    """
    # Use 0, MAX_DEC clamp logic from updated _extract_int
    val = _extract_int(v, 0, MAX_DEC)
    if val is None:
        raw = _sanitize_for_log(v).replace(" ", "")
        raise argparse.ArgumentTypeError(
            f"invalid decimal index: '{raw}'"
        )
    return f"{val:06X}"


def handle_color_name(v: str) -> str:
    """
    For -cn/--color-name flags.

    Behaviour:
        - Remove spaces.
        - Extract only letters a-z.
        - Lowercase.
        - Empty after sanitization -> error.
    """
    cleaned = _extract_alpha(v)
    if not cleaned:
        raw = _sanitize_for_log(v).replace(" ", "")
        raise argparse.ArgumentTypeError(
            f"invalid color name: '{raw}'"
        )
    return cleaned


def handle_string_clean(v: str) -> str:
    """
    Generic string-normalizer for flags like:
      --colorspace, --distance-metric, --harmony-model

    Behaviour:
        - Remove spaces.
        - Extract only letters a-z.
        - Lowercase.
        - Empty after sanitization -> error.
    """
    cleaned = _extract_alpha(v)
    if not cleaned:
        raw = _sanitize_for_log(v).replace(" ", "")
        raise argparse.ArgumentTypeError(
            f"invalid string value: '{raw}'"
        )
    return cleaned


def handle_float_any(v: str) -> float:
    """
    For flags that expect a float-like number (e.g. --dedup-value, rotation).

    Behaviour:
        - Remove spaces.
        - Parse first float-like token.
        - Empty / unparsable -> error.
    """
    f = _extract_float(v)
    if f is None:
        raw = _sanitize_for_log(v).replace(" ", "")
        raise argparse.ArgumentTypeError(
            f"invalid numeric value: '{raw}'"
        )
    return f


def handle_int_range(min_v: int, max_v: int):
    """
    Factory for range-based integer flags.

    Example:
        type=handle_int_range(2, 1000)  # for -n/--number

    Behaviour:
        - Remove spaces.
        - Extract digits with sign.
        - No digits => error.
        - Clamp to [min_v, max_v].
    """
    def validator(v: str) -> int:
        val = _extract_int(v, min_v, max_v)
        if val is None:
            raw = _sanitize_for_log(v).replace(" ", "")
            raise argparse.ArgumentTypeError(
                f"invalid integer value: '{raw}'"
            )
        return val

    return validator


def handle_float_range(min_v: float, max_v: float):
    """
    Factory for range-based float flags (NEW for vision.py).

    Example:
        type=handle_float_range(0.0, 1.0)  # for -i/--intensity

    Behaviour:
        - Remove spaces.
        - Extract float pattern with sign.
        - No float found => error.
        - Clamp to [min_v, max_v].
    """
    def validator(v: str) -> float:
        val = _extract_float(v)
        if val is None:
            raw = _sanitize_for_log(v).replace(" ", "")
            raise argparse.ArgumentTypeError(
                f"invalid float value: '{raw}'"
            )
        if val < min_v:
            val = min_v
        elif val > max_v:
            val = max_v
        return val

    return validator


# --- 3. THE INPUT HANDLING DICTIONARY ---

INPUT_HANDLERS = {
    # Core color inputs
    "hex": handle_hex,
    "decimal_index": handle_decimal_index,
    "color_name": handle_color_name,

    # String flags
    "colorspace": handle_string_clean,
    "distance_metric": handle_string_clean,
    "harmony_model": handle_string_clean,
    "from_format": handle_string_clean,
    "to_format": handle_string_clean,

    # Float-ish flags
    "dedup_value": handle_float_any,
    "float": handle_float_any,
    "float_0_1": handle_float_range(0.0, 1.0),       # For vision/severity (0.0 - 1.0)
    "float_0_100": handle_float_range(0.0, 100.0),   # For adjust % (0.0 - 100.0)
    "float_signed_100": handle_float_range(-100.0, 100.0), # For brightness/contrast (-100 to 100)

    # Range-based integer flags
    "number": handle_int_range(2, 1000),             # similar -n
    "seed": handle_int_range(0, 999_999_999),        # -s
    "steps": handle_int_range(1, MAX_STEPS),         # gradient -S
    "total_random": handle_int_range(2, MAX_RANDOM_COLORS),  # mix/gradient
    "int_channel": handle_int_range(-255, 255),      # RGB channel modifications
}


# --- 4. COMMON UTILS ---

def decimal_to_hex_str(dec_str: str) -> str:
    """
    Helper to convert sanitized decimal index string to 6-digit hex.

    If dec_str is invalid/int conversion fails, returns "000000".
    (Realistically, dec_str should always be valid if it came from handle_decimal_index.)
    """
    try:
        val = int(dec_str)
        if val < 0:
            val = 0
        elif val > MAX_DEC:
            val = MAX_DEC
        return f"{val:06X}"
    except Exception:
        return "000000"

class HexlabArgumentParser(argparse.ArgumentParser):
    """
    Custom parser to route argparse errors through the centralized log() function
    for consistent [hexlab][error] formatting.
    """
    def error(self, message):
        log('error', message)
        sys.exit(2)