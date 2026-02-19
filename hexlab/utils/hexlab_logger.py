# File: hexlab_logger.py
import sys
import argparse

from ..constants.constants import (
    MSG_BOLD_COLORS,
    MSG_COLORS,
    RESET
)

def log(level: str, message: str) -> None:
    level = str(level).lower()
    stream = sys.stdout if level in ["info", "success"] else sys.stderr
    
    tag_color = MSG_BOLD_COLORS.get(level, RESET)
    msg_color = MSG_COLORS.get(level, RESET)

    print(f"{tag_color}[{level}]{RESET} {msg_color}{message}{RESET}", file=stream)


class HexlabArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        log('error', message)
        sys.exit(2)