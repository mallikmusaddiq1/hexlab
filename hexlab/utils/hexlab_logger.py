# File: hexlab_logger.py
import sys
import argparse

from ..constants.constants import (
    MSG_BOLD_COLORS,
    MSG_COLORS,
    RESET
)

def log(level: str, message: str) -> None:
    """
    Custom logging function that outputs color-coded messages to the terminal.
    
    Routes standard information and success messages to standard output (stdout), 
    while routing warnings and errors to standard error (stderr). This allows 
    users to pipe or redirect standard output without mixing in error messages.
    """
    level = str(level).lower()
    
    # Route non-critical logs to stdout, critical logs to stderr
    stream = sys.stdout if level in ["info", "success"] else sys.stderr
    
    # Retrieve specific ANSI color codes for the tag and message text
    tag_color = MSG_BOLD_COLORS.get(level, RESET)
    msg_color = MSG_COLORS.get(level, RESET)

    # Print formatted output, e.g., "[error] invalid hex value"
    print(f"{tag_color}[{level}]{RESET} {msg_color}{message}{RESET}", file=stream)


class HexlabArgumentParser(argparse.ArgumentParser):
    """
    Custom ArgumentParser that overrides the default argparse error handling.
    
    Instead of printing standard unformatted argparse errors, this class 
    intercepts them and routes them through the custom `log` function 
    to maintain consistent visual styling across the CLI application.
    """
    def error(self, message):
        """
        Overrides the default error method to use our color-coded logger,
        then exits the program with the standard CLI error code 2.
        """
        log('error', message)
        sys.exit(2)
