#!/usr/bin/env python3

import sys
import json
import re
import os
from typing import Optional
from .hexlab_logger import log
from .input_handler import normalize_hex

def _norm_name_key(s: str) -> str:
    return re.sub(r'[^0-9a-z]', '', str(s).lower())

_json_path = os.path.join(os.path.dirname(__file__), '..', 'constants', 'color_names.json')

try:
    with open(_json_path, 'r', encoding='utf-8') as _f:
        _raw_colors = json.load(_f)
except FileNotFoundError:
    log('error', f"color names file not found at: {_json_path}")
    sys.exit(1)
except json.JSONDecodeError:
    log('error', f"invalid JSON in color names file: {_json_path}")
    sys.exit(1)

COLOR_NAMES = {k: normalize_hex(v) for k, v in _raw_colors.items()}

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

def get_title_for_hex(hex_code: str, fallback: str = None) -> str:
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
