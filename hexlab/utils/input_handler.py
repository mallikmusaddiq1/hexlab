# File: input_handler.py
#!/usr/bin/env python3

import argparse
import re
import sys

from ..constants.constants import MAX_DEC, MAX_RANDOM_COLORS, MAX_STEPS
from .hexlab_logger import log


def _sanitize_for_log(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def normalize_hex(value: str) -> str:
    if value is None:
        return ""
    s = str(value).replace("#", "").replace(" ", "").upper()
    extracted = "".join(re.findall(r"[0-9A-F]", s))

    if not extracted:
        return ""

    L = len(extracted)
    if L == 6:
        return extracted
    if L == 3:
        return "".join([c * 2 for c in extracted])
    if L == 1:
        return extracted * 6
    if L == 2:
        return extracted * 3
    if L == 4:
        return extracted + "00"
    if L == 5:
        return extracted + "0"

    return extracted[:6]


def _extract_positive_only_int(value: str) -> int:
    if value is None:
        return None
    s = str(value)

    digits_only = re.sub(r"[^0-9]", "", s)

    if not digits_only:
        return None

    try:
        return int(digits_only)
    except ValueError:
        return None


def _extract_signed_int(value: str) -> int:
    if value is None:
        return None

    s = str(value).strip()
    if not s:
        return None

    match = re.search(r"\d+", s)

    if not match:
        return None

    prefix_garbage = s[:match.start()]

    is_negative = "-" in prefix_garbage

    try:
        val = int(match.group())
        if is_negative:
            val = -val
        return val
    except ValueError:
        return None


def _extract_signed_float(value: str) -> float:
    if value is None:
        return None

    s = str(value).strip()
    if not s:
        return None

    match = re.search(r"(\d+\.\d+|\.\d+|\d+)", s)

    if not match:
        return None

    prefix_garbage = s[:match.start()]
    is_negative = "-" in prefix_garbage

    try:
        val = float(match.group())
        if is_negative:
            val = -val
        return val
    except ValueError:
        return None


def _extract_alpha_only(value: str) -> str:
    if value is None:
        return ""
    s = str(value).replace(" ", "").lower()
    extracted = "".join(re.findall(r"[a-z]", s))
    return extracted


def handle_hex(v: str) -> str:
    cleaned = normalize_hex(v)
    if not cleaned:
        raw = _sanitize_for_log(v)
        raise argparse.ArgumentTypeError(f"invalid hex value: '{raw}'")
    return cleaned


def handle_decimal_index(v: str) -> str:
    val = _extract_positive_only_int(v)

    if val is None:
        raw = _sanitize_for_log(v)
        raise argparse.ArgumentTypeError(f"invalid decimal index: '{raw}'")

    if val < 0:
        val = 0
    if val > MAX_DEC:
        val = MAX_DEC

    return f"{val:06X}"


def handle_color_name(v: str) -> str:
    cleaned = _extract_alpha_only(v)
    if not cleaned:
        raw = _sanitize_for_log(v)
        raise argparse.ArgumentTypeError(f"invalid color name: '{raw}'")
    return cleaned


def handle_string_clean(v: str) -> str:
    cleaned = _extract_alpha_only(v)
    if not cleaned:
        raw = _sanitize_for_log(v)
        raise argparse.ArgumentTypeError(f"invalid string value: '{raw}'")
    return cleaned


def handle_float_any(v: str) -> float:
    val = _extract_signed_float(v)
    if val is None:
        raw = _sanitize_for_log(v)
        raise argparse.ArgumentTypeError(f"invalid numeric value: '{raw}'")
    return val


def handle_int_range(min_v: int, max_v: int):
    def validator(v: str) -> int:
        if min_v < 0:
            val = _extract_signed_int(v)
        else:
            val = _extract_positive_only_int(v)

        if val is None:
            raw = _sanitize_for_log(v)
            raise argparse.ArgumentTypeError(f"invalid integer value: '{raw}'")

        if val < min_v:
            val = min_v
        elif val > max_v:
            val = max_v
        return val
    return validator


def handle_float_range(min_v: float, max_v: float):
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

    "number": handle_int_range(2, 1000),
    "seed": handle_int_range(0, 999_999_999_999_999_999),
    "steps": handle_int_range(1, MAX_STEPS),
    "total_random": handle_int_range(2, MAX_RANDOM_COLORS),
    "int_channel": handle_int_range(-255, 255),
    "custom_scheme": handle_int_range(-360, 360)
}


class HexlabArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        log('error', message)
        sys.exit(2)