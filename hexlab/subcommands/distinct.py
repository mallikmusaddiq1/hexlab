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
from ..constants.constants import MAX_DEC
from ..utils.color_names_handler import get_title_for_hex, resolve_color_name_or_exit
from ..utils.hexlab_logger import log
from ..utils.input_handler import INPUT_HANDLERS, HexlabArgumentParser
from ..utils.print_color_block import print_color_block
from ..utils.truecolor import ensure_truecolor

CANDIDATES_PER_STEP = 500


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


def _dist_sq_euclidean(t1: Tuple[float, ...], t2: Tuple[float, ...]) -> float:
    """Calculates Squared Euclidean distance (No Sqrt) for speed."""
    d0 = t1[0] - t2[0]
    d1 = t1[1] - t2[1]
    d2 = t1[2] - t2[2]
    return d0*d0 + d1*d1 + d2*d2


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
                    d_sq = _dist_sq_euclidean(cand_metric, existing_metric)
                    
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
        title = "random base"
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
    print_color_block(clean_hex, title)
    print()
    
    base_rgb = hex_to_rgb(clean_hex)
    metric = args.distance_metric
    
    distinct_gen = generate_distinct_colors_greedy(
        base_rgb,
        n=args.number,
        metric=metric,
        candidates_per_step=CANDIDATES_PER_STEP
    )

    metric_map = {'lab': 'ΔE2000', 'oklab': 'ΔE(OKLAB)', 'rgb': 'ΔE(RGB)'}
    metric_label = metric_map.get(metric, 'min-dist')

    for i, (hex_val, diff) in enumerate(distinct_gen):
        label = f"distinct {i + 1}"
        print_color_block(hex_val, label, end="")
        print(f"  ({metric_label}: {diff:.2f})")
        sys.stdout.flush()

    print()


def get_distinct_parser() -> argparse.ArgumentParser:
    parser = HexlabArgumentParser(
        prog="hexlab distinct",
        description="hexlab distinct: generate a set of visually distinct colors",
        formatter_class=argparse.RawTextHelpFormatter
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-H", "--hex",
        dest="hex",
        type=INPUT_HANDLERS["hex"],
        help="start with this hex code"
    )
    input_group.add_argument(
        "-r", "--random",
        action="store_true",
        help="start with a random color"
    )
    input_group.add_argument(
        "-cn", "--color-name",
        type=INPUT_HANDLERS["color_name"],
        help="start with this color name"
    )
    input_group.add_argument(
        "-di", "--decimal-index",
        type=INPUT_HANDLERS["decimal_index"],
        help="start with this decimal index"
    )
    
    parser.add_argument(
        "-n", "--number",
        type=INPUT_HANDLERS["number_distinct"],
        default=10,
        help="number of distinct colors to generate (min: 2, max: 250, default: 10)"
    )
    parser.add_argument(
        "-dm", "--distance-metric",
        type=INPUT_HANDLERS["distance_metric"],
        default='oklab',
        help="distance metric: oklab lab rgb (default: oklab)",
        choices=['lab', 'oklab', 'rgb']
    )
    parser.add_argument(
        "-s", "--seed",
        type=INPUT_HANDLERS["seed"],
        default=None,
        help="seed for reproducibility"
    )
    
    return parser


def main() -> None:
    parser = get_distinct_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    handle_distinct_command(args)


if __name__ == "__main__":
    main()