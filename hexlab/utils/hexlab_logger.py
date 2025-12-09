# File: hexlab_logger.py
import sys


def log(level: str, message: str) -> None:
    """Log a message to stdout (info) or stderr (other levels)."""
    level = str(level).lower()
    stream = sys.stdout if level == "info" else sys.stderr
    print(f"[{level}] {message}", file=stream)

