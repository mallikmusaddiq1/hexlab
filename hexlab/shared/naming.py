#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/shared/naming.py

import json
import re
import sys
from typing import Optional

from .logger import log
from .sanitizer import normalize_hex
from hexlab.core.color_names import COLOR_NAMES 


def _norm_name_key(s: str) -> str:
    """Normalize color name to alphanumeric lower case."""
    return re.sub(r'[^0-9a-z]', '', str(s).lower())

COLOR_NAMES = {k: normalize_hex(v) for k, v in COLOR_NAMES.items()}

HEX_TO_NAME = {v.upper(): k for k, v in COLOR_NAMES.items()}

_norm_map = {}
for k, v in COLOR_NAMES.items():
    key = _norm_name_key(k)

    if key in _norm_map and _norm_map.get(key) != v:
        original_hex = _norm_map.get(key)
        original_name = HEX_TO_NAME.get(original_hex.upper(), '???')
        log(
            'warn',
            f"color name collision on key '{key}': '{original_name}' and "
            f"'{k}' both normalize to the same key. '{k}' will be used."
        )
    _norm_map[key] = v

COLOR_NAMES_LOOKUP = _norm_map


def get_hex_from_name(sanitized_name: str) -> Optional[str]:
    if not sanitized_name:
        return None
    return COLOR_NAMES_LOOKUP.get(sanitized_name)


def resolve_color_name_or_exit(name_arg: str) -> str:
    hex_val = get_hex_from_name(name_arg)

    if not hex_val:
        log('error', f"unknown color name '{name_arg}'")
        log('info', "use 'hexlab --list-color-names' to see all options")
        sys.exit(2)

    return hex_val


def get_title_for_hex(hex_code: str, fallback: Optional[str] = None) -> str:
    if not hex_code:
        return "unknown"
    clean_hex = normalize_hex(hex_code)
    return HEX_TO_NAME.get(clean_hex, fallback or f"#{clean_hex}")


def handle_list_color_names_action(fmt: str) -> None:

    color_keys = sorted(list(COLOR_NAMES.keys()))

    if fmt == 'text':
        for name in color_keys:
            print(name)
    elif fmt == 'json':
        print(json.dumps(color_keys))
    elif fmt == 'prettyjson':
        print(json.dumps(color_keys, indent=4))

    sys.exit(0)
