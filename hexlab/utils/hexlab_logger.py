import sys

def log(level: str, message: str) -> None:
    level = str(level).lower()
    stream = sys.stdout if level == "info" else sys.stderr
    print(f"[hexlab][{level}] {message}", file=stream)
