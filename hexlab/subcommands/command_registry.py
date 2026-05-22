#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/subcommands/registry.py

from . import (
    gradient,
    mix,
    scheme,
    vision,
    similar,
    distinct,
    unicode,
    convert,
    adjust
)

SUBCOMMANDS = {
    'gradient': gradient,
    'mix': mix,
    'scheme': scheme,
    'vision': vision,
    'similar': similar,
    'distinct': distinct,
    'unicode': unicode,
    'convert': convert,
    'adjust': adjust
}
