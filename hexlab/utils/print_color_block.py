import re
from ..color_math.conversions import hex_to_rgb

def get_visible_len(s: str) -> int:
    """ANSI escape codes ko ignore karke visible characters ki length nikalta hai."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return len(ansi_escape.sub('', s))

def print_color_block(hex_code: str, title: str = "color", end: str = "\n") -> None:
    """Print a colored text block with fixed alignment, supporting ANSI colored titles."""
    r, g, b = hex_to_rgb(hex_code)
    vis_len = get_visible_len(title)
    padding = " " * max(0, 18 - vis_len)
    print(f"{title}{padding}:   \033[48;2;{r};{g};{b}m                \033[0m  #{hex_code}", end=end)

