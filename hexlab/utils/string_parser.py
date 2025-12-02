# File: string_parser.py
import math
import re
import sys
from typing import List, Tuple

from ..constants.constants import EPS
from .clamping import _clamp01
from .hexlab_logger import log


def _normalize_value_string(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'`":
        s = s[1:-1].strip()
    s = s.replace('°', ' ')
    s = s.replace('–', '-')
    s = re.sub(r'deg', ' ', s, flags=re.IGNORECASE)

    s = re.sub(r'^[a-zA-Z]+\s*\(', '', s, flags=re.IGNORECASE)
    s = s.rstrip(')')

    s = s.replace(',', ' ')
    s = s.replace('/', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def _safe_float(s: str) -> float:
    try:
        v = float(s)
    except Exception:
        log('error', f"invalid numeric value '{s}'")
        sys.exit(2)
    if not math.isfinite(v):
        log('error', f"non-finite numeric value '{s}'")
        sys.exit(2)
    return v


def _parse_numerical_string(s: str) -> List[float]:
    s = _normalize_value_string(s)
    pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    try:
        matches = list(re.finditer(pattern, s))
        if not matches:
            raise ValueError

        tokens = []
        i = 0
        while i < len(matches):
            m = matches[i]
            token = m.group()

            if i + 1 < len(matches) and '.' not in token:
                m_next = matches[i + 1]
                next_token = m_next.group()
                if next_token.startswith('.'):
                    between = s[m.end():m_next.start()]
                    if between and re.fullmatch(r'[A-Za-z]+', between):
                        token = token + next_token
                        i += 1

            val = float(token)
            if not math.isfinite(val):
                raise ValueError
            tokens.append(val)
            i += 1

        return tokens
    except Exception:
        raise ValueError(f"could not parse numerical values from '{s}'")


def _parse_h_ss_string(s: str, model_name: str) -> Tuple[float, float, float]:
    s_norm = _normalize_value_string(s)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?%?", s_norm)
    if len(nums) < 3:
        log('error', f"invalid {model_name} string: {s}")
        sys.exit(2)

    h = _safe_float(re.sub(r'%', '', nums[0])) % 360

    val1 = nums[1]
    val2 = nums[2]

    v1_f = _safe_float(re.sub(r'%', '', val1))
    v2_f = _safe_float(re.sub(r'%', '', val2))

    v1_f = v1_f / 100.0 if '%' in val1 or v1_f > 1.0 else v1_f
    v2_f = v2_f / 100.0 if '%' in val2 or v2_f > 1.0 else v2_f

    return h, _clamp01(v1_f), _clamp01(v2_f)


def _parse_3_floats(s: str, model_name: str) -> Tuple[float, float, float]:
    try:
        nums = _parse_numerical_string(s)
    except ValueError:
        log('error', f"invalid {model_name} string: {s}")
        sys.exit(2)
    if len(nums) < 3:
        log('error', f"invalid {model_name} string: {s}")
        sys.exit(2)
    return float(nums[0]), float(nums[1]), float(nums[2])


def parse_rgb_string(s: str) -> Tuple[int, int, int]:
    try:
        nums = _parse_numerical_string(s)
    except ValueError:
        log('error', f"invalid rgb string: {s}")
        sys.exit(2)
    if len(nums) < 3:
        log('error', f"invalid rgb string: {s}")
        sys.exit(2)

    def _to_8bit(val: float) -> int:
        if 0.0 < val < 1.0:
            v = val * 255.0
        else:
            v = val
        return max(0, min(255, int(round(v))))

    r = _to_8bit(nums[0])
    g = _to_8bit(nums[1])
    b = _to_8bit(nums[2])
    return r, g, b


def parse_hsl_string(s: str) -> Tuple[float, float, float]:
    return _parse_h_ss_string(s, "hsl")


def parse_hsv_string(s: str) -> Tuple[float, float, float]:
    return _parse_h_ss_string(s, "hsv")


def parse_hwb_string(s: str) -> Tuple[float, float, float]:
    return _parse_h_ss_string(s, "hwb")


def parse_cmyk_string(s: str) -> Tuple[float, float, float, float]:
    s_norm = _normalize_value_string(s)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?%?", s_norm)
    if len(nums) < 4:
        log('error', f"invalid cmyk string: {s}")
        sys.exit(2)
    vals = []
    for n in nums[:4]:
        v = _safe_float(re.sub(r'%', '', n))
        v = v / 100.0 if '%' in n or v > 1.0 else v
        vals.append(_clamp01(v))
    return tuple(vals)


def parse_xyz_string(s: str) -> Tuple[float, float, float]:
    return _parse_3_floats(s, "xyz")


def parse_lab_string(s: str) -> Tuple[float, float, float]:
    return _parse_3_floats(s, "lab")


def parse_lch_string(s: str) -> Tuple[float, float, float]:
    return _parse_3_floats(s, "lch")


def parse_oklab_string(s: str) -> Tuple[float, float, float]:
    return _parse_3_floats(s, "oklab")


def parse_oklch_string(s: str) -> Tuple[float, float, float]:
    return _parse_3_floats(s, "oklch")


def parse_luv_string(s: str) -> Tuple[float, float, float]:
    return _parse_3_floats(s, "luv")


STRING_PARSERS = {
    'rgb': parse_rgb_string,
    'hsl': parse_hsl_string,
    'hsv': parse_hsv_string,
    'hwb': parse_hwb_string,
    'cmyk': parse_cmyk_string,
    'xyz': parse_xyz_string,
    'lab': parse_lab_string,
    'lch': parse_lch_string,
    'oklab': parse_oklab_string,
    'oklch': parse_oklch_string,
    'luv': parse_luv_string,
}
