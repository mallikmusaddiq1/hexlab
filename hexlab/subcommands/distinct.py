# File: distinct.py
#!/usr/bin/env python3

import argparse
import random
import sys
import math
from typing import Generator, List, Tuple

from ..color_math.conversions import (
    hex_to_rgb,
    rgb_to_hex,
    rgb_to_oklab,
    rgb_to_xyz,
    xyz_to_lab,
)
from ..color_math.distance import (
    delta_e_ciede2000,
)
from ..constants.constants import (
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

CANDIDATES_PER_STEP = 200


def _generate_random_rgb() -> Tuple[int, int, int]:
    val = random.randint(0, MAX_DEC)
    r = (val >> 16) & 0xFF
    g = (val >> 8) & 0xFF
    b = val & 0xFF
    return r, g, b


def _to_metric_space(rgb: Tuple[int, int, int], metric: str):
    if metric == 'rgb':
        return rgb
    elif metric == 'oklab':
        return rgb_to_oklab(*rgb)
    else:
        x, y, z = rgb_to_xyz(*rgb)
        return xyz_to_lab(x, y, z)


def generate_distinct_colors_greedy(
    base_rgb: Tuple[int, int, int],
    n: int = 5,
    metric: str = 'oklab',
    candidates_per_step: int = CANDIDATES_PER_STEP
) -> Generator[Tuple[str, float], None, None]:
    
    base_metric_val = _to_metric_space(base_rgb, metric)
    
    selected_data = [(base_metric_val, base_rgb)]
    
    is_euclidean = (metric in ['rgb', 'oklab'])

    for _ in range(n):
        best_candidate_rgb = None
        
        max_min_dist_val = -1.0 

        for _ in range(candidates_per_step):
            cand_rgb = _generate_random_rgb()
            cand_metric = _to_metric_space(cand_rgb, metric)
            
            min_dist_for_this_cand = float('inf')
            
            stop_early = False

            if is_euclidean:
                for existing_metric, _ in selected_data:
                    d0 = cand_metric[0] - existing_metric[0]
                    d1 = cand_metric[1] - existing_metric[1]
                    d2 = cand_metric[2] - existing_metric[2]
                    d_sq = d0*d0 + d1*d1 + d2*d2
                    
                    if d_sq < min_dist_for_this_cand:
                        min_dist_for_this_cand = d_sq
                    
                    if min_dist_for_this_cand < max_min_dist_val:
                        stop_early = True
                        break
            else:
                for existing_metric, _ in selected_data:
                    d = delta_e_ciede2000(existing_metric, cand_metric)
                    if d < min_dist_for_this_cand:
                        min_dist_for_this_cand = d
                    if min_dist_for_this_cand < max_min_dist_val:
                        stop_early = True
                        break

            if stop_early:
                continue

            if min_dist_for_this_cand > max_min_dist_val:
                max_min_dist_val = min_dist_for_this_cand
                best_candidate_rgb = cand_rgb
                best_candidate_metric = cand_metric

        if best_candidate_rgb:
            selected_data.append((best_candidate_metric, best_candidate_rgb))
            
            final_dist = max_min_dist_val
            if is_euclidean:
                final_dist = math.sqrt(max_min_dist_val)
                
            yield (rgb_to_hex(*best_candidate_rgb), final_dist)


def handle_distinct_command(args: argparse.Namespace) -> None:
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
    
    distinct_gen = generate_distinct_colors_greedy(
        base_rgb,
        n=args.count,
        metric=metric,
        candidates_per_step=CANDIDATES_PER_STEP
    )

    metric_map = {'lab': 'ΔE2000', 'oklab': 'ΔE(OKLAB)', 'rgb': 'ΔE(RGB)'}
    metric_label = metric_map.get(metric, 'min-dist')

    for i, (hex_val, diff) in enumerate(distinct_gen):
        label = f"{MSG_BOLD_COLORS['info']}distinct {i + 1}{RESET}"
        
        print_color_block(hex_val, label, end="")
        
        print(f"  {MSG_BOLD_COLORS['info']}({metric_label}: {diff:.2f}){RESET}")
        sys.stdout.flush()

    print()


def get_distinct_parser() -> argparse.ArgumentParser:
    parser = HexlabArgumentParser(
        prog="hexlab distinct",
        description="hexlab distinct: generate visually distinct colors",
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
        default='oklab',
        help="distance metric: oklab lab rgb (default: oklab)",
        choices=['lab', 'oklab', 'rgb']
    )
    parser.add_argument(
        "-c", "--count",
        type=INPUT_HANDLERS["count_distinct"],
        default=10,
        help="number of distinct colors to generate (min: 2, max: 250, default: 10)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=INPUT_HANDLERS["seed"],
        default=None,
        help="seed for reproducibility of random"
    )
    
    return parser


def main() -> None:
    parser = get_distinct_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    handle_distinct_command(args)


if __name__ == "__main__":
    main()
