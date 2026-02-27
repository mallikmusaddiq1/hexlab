#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/shared/logger.py

import sys
import argparse

from hexlab.core import config as c


def log(level: str, message: str) -> None:
    level = str(level).lower()
    stream = sys.stdout if level in ["info", "success"] else sys.stderr
    tag_color = c.MSG_BOLD_COLORS.get(level, c.RESET)
    msg_color = c.MSG_COLORS.get(level, c.RESET)
    print(f"{tag_color}[{level}]{c.RESET} {msg_color}{message}{c.RESET}", file=stream)


class HexlabArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        """
        Overrides the default error method to use our color-coded logger,
        then exits the program with the standard CLI error code 2.
        """
        log('error', message)
        sys.exit(2)