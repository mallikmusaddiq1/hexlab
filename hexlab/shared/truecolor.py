#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/shared/truecolor.py

import os
import sys


def ensure_truecolor() -> None:
    """Ensure the COLORTERM environment variable is set to truecolor."""
    if sys.platform == "win32":
        return
    if os.environ.get("COLORTERM") != "truecolor":
        os.environ["COLORTERM"] = "truecolor"
