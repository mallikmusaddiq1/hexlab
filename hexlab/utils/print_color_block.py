from ..color_math.conversions import hex_to_rgb

def print_color_block(hex_code: str, title: str = "color", end: str = "\n") -> None:
    r, g, b = hex_to_rgb(hex_code)
    print(f"{title:<18}:   \033[48;2;{r};{g};{b}m                \033[0m  #{hex_code}", end=end)
