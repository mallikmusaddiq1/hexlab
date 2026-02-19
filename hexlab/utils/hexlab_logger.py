# File: hexlab_logger.py
import sys
import argparse

# Standard ANSI Color Codes
LEVEL_COLORS = {
    "error": "\033[1;31m",   # Bold Red
    "warning": "\033[1;33m", # Bold Yellow
    "info": "\033[1;36m",    # Bold Cyan
    "success": "\033[1;32m", # Bold Green
}

MSG_COLORS = {
    "error": "\033[0;31m",   # Normal Red
    "warning": "\033[0;33m", # Normal Yellow
    "info": "\033[0;36m",    # Normal Cyan
    "success": "\033[0;32m", # Normal Green
}

RESET = "\033[0m"

def log(level: str, message: str) -> None:
    level = str(level).lower()
    stream = sys.stdout if level in ["info", "success"] else sys.stderr
    
    tag_color = LEVEL_COLORS.get(level, RESET)
    msg_color = MSG_COLORS.get(level, RESET)

    print(f"{tag_color}[{level}]{RESET} {msg_color}{message}{RESET}", file=stream)


class HexlabArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        log('error', message)
        sys.exit(2)
