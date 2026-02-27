#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/core/luminance.py

from .conversions import _srgb_to_linear
from . import config as c


def get_luminance(r: int, g: int, b: int) -> float:
    return (
        c.LUMA_R * _srgb_to_linear(r) +
        c.LUMA_G * _srgb_to_linear(g) +
        c.LUMA_B * _srgb_to_linear(b)
    )