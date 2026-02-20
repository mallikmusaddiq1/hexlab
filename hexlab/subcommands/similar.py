# File: similar.py
#!/usr/bin/env python3

import argparse
import random
import sys
from typing import Generator, Tuple, Set

from ..color_math.conversions import (
    hex_to_rgb,
    hsl_to_rgb,
    rgb_to_hex,
    rgb_to_hsl,
    rgb_to_oklab,
    rgb_to_xyz,
    xyz_to_lab,
)
from ..color_math.distance import (
    delta_e_ciede2000,
    delta_e_euclidean_oklab,
    delta_e_euclidean_rgb,
)
from ..constants.constants import (
    DEDUP_DELTA_E_LAB,
    DEDUP_DELTA_E_OKLAB,
    DEDUP_DELTA_E_RGB,
    MAX_DEC,
    MSG_BOLD_COLORS,
    RESET,
    BOLD_WHITE
)
from ..utils.color_names_handler import get_title_for_hex, resolve_color_name_or_exit
from ..utils.hexlab_logger import log, HexlabArgumentParser
from ..utils.input_handler import INPUT_HANDLERS
from ..utils.print_color_block import print_color_block
from ..utils.truecolor import ensure_truecolor


def _to_metric_space(rgb: Tuple[int, int, int], metric: str):
    if metric == 'rgb':
        return rgb
    elif metric == 'oklab':
        return rgb_to_oklab(*rgb)
    else:
        x, y, z = rgb_to_xyz(*rgb)
        return xyz_to_lab(x, y, z)


def generate_similar_colors_streaming(
    base_rgb: Tuple[int, int, int],
    n: int = 5,
    metric: str = 'lab',
    dedup_val: float = 7.7
) -> Generator[Tuple[str, float], None, None]:
    
    base_metric_val = _to_metric_space(base_rgb, metric)
    
    r, g, b = base_rgb
    h_base, s_base, l_base = rgb_to_hsl(r, g, b)

    count_found = 0
    seen_hex: Set[str] = set()
    seen_hex.add(rgb_to_hex(r, g, b))

    attempts = 0
    max_attempts = n * 500

    while count_found < n and attempts < max_attempts:
        attempts += 1
        
        h_delta = random.uniform(-20, 20)
        s_delta = random.uniform(-0.15, 0.15)
        l_delta = random.uniform(-0.15, 0.15)

        new_h = (h_base + h_delta) % 360
        new_s = max(0.0, min(1.0, s_base + s_delta))
        new_l = max(0.0, min(1.0, l_base + l_delta))

        nr, ng, nb = hsl_to_rgb(new_h, new_s, new_l)
        nr_i, ng_i, nb_i = int(round(nr)), int(round(ng)), int(round(nb))
        
        cand_hex = rgb_to_hex(nr_i, ng_i, nb_i)
        
        if cand_hex in seen_hex:
            continue

        cand_rgb = (nr_i, ng_i, nb_i)
        
        diff = 0.0
        if metric == 'lab':
            cand_metric = _to_metric_space(cand_rgb, 'lab')
            diff = delta_e_ciede2000(base_metric_val, cand_metric)
        elif metric == 'oklab':
            cand_metric = _to_metric_space(cand_rgb, 'oklab')
            diff = delta_e_euclidean_oklab(base_metric_val, cand_metric)
        elif metric == 'rgb':
            diff = delta_e_euclidean_rgb(base_rgb[0], base_rgb[1], base_rgb[2], nr_i, ng_i, nb_i)

        if diff >= dedup_val:
            seen_hex.add(cand_hex)
            count_found += 1
            yield (cand_hex, diff)


def handle_similar_command(args: argparse.Namespace) -> None:
    clean_hex = None
    title = "base color"
    if args.seed is not None:
        random.seed(args.seed)

    if args.random:
        current_dec = random.randint(0, MAX_DEC)
        clean_hex = f"{current_dec:06X}"
        title = "random"
    elif args.color_name:
        clean_hex = resolve_color_name_or_exit(args.color_name)
        title = get_title_for_hex(clean_hex)
    elif args.hex:
        clean_hex = args.hex
        title = get_title_for_hex(clean_hex)
    elif getattr(args, "decimal_index", None) is not None:
        clean_hex = args.decimal_index
        idx = int(clean_hex, 16)
        title = get_title_for_hex(clean_hex, f"index {idx}")

    print()
    print_color_block(clean_hex, f"{BOLD_WHITE}{title}{RESET}")
    print()

    base_rgb = hex_to_rgb(clean_hex)
    metric = args.distance_metric

    dedup_val = 0.0
    if args.dedup_value is not None:
        dedup_val = args.dedup_value
    else:
        if metric == 'rgb':
            dedup_val = DEDUP_DELTA_E_RGB
        elif metric == 'oklab':
            dedup_val = DEDUP_DELTA_E_OKLAB
        else:
            dedup_val = DEDUP_DELTA_E_LAB

    num_results = args.count

    similar_gen = generate_similar_colors_streaming(
        base_rgb,
        n=num_results,
        metric=metric,
        dedup_val=dedup_val
    )

    metric_map = {'lab': 'ΔE(2000)', 'oklab': 'ΔE(OKLAB)', 'rgb': 'ΔE(RGB)'}
    metric_label = metric_map.get(metric, 'ΔE')

    found_any = False

    for i, (hex_val, diff) in enumerate(similar_gen):
        found_any = True
   
        label = f"{MSG_BOLD_COLORS['info']}similar{f'{i + 1}':>9}{RESET}"
        
        print_color_block(hex_val, label, end="")
      
        print(f"  {MSG_BOLD_COLORS['info']}({metric_label}: {diff:5.2f}){RESET}")
        sys.stdout.flush()

    if not found_any:
        log('info', "no similar colors found within parameters")

    print()


def get_similar_parser() -> argparse.ArgumentParser:
    parser = HexlabArgumentParser(
        prog="hexlab similar",
        description="hexlab similar: find perceptually similar colors",
        formatter_class=argparse.RawTextHelpFormatter
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-H", "--hex",
        dest="hex",
        type=INPUT_HANDLERS["hex"],
        help="base hex code"
    )
    input_group.add_argument(
        "-r", "--random",
        action="store_true",
        help="use a random base"
    )
    input_group.add_argument(
        "-cn", "--color-name",
        type=INPUT_HANDLERS["color_name"],
        help="base color name"
    )
    input_group.add_argument(
        "-di", "--decimal-index",
        type=INPUT_HANDLERS["decimal_index"],
        help="base decimal index"
    )
    parser.add_argument(
        "-dm", "--distance-metric",
        type=INPUT_HANDLERS["distance_metric"],
        default='lab',
        help="distance metric: lab oklab rgb (default: lab)",
        choices=['lab', 'oklab', 'rgb']
    )
    parser.add_argument(
        "-dv", "--dedup-value",
        type=INPUT_HANDLERS["dedup_value"],
        default=None,
        help=(
            f"custom deduplication threshold (defaults: lab={DEDUP_DELTA_E_LAB}, "
            f"oklab={DEDUP_DELTA_E_OKLAB}, rgb={DEDUP_DELTA_E_RGB})"
        )
    )
    parser.add_argument(
        "-c", "--count",
        type=INPUT_HANDLERS["count_similar"],
        default=10,
        help="number of similar colors to find (min: 2, max: 500, default: 10)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=INPUT_HANDLERS["seed"],
        default=None,
        help="seed for reproducibility of random"
    )
    return parser


def main() -> None:
    parser = get_similar_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    handle_similar_command(args)


if __name__ == "__main__":
    main()