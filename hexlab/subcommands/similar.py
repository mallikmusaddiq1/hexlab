# File: similar.py
#!/usr/bin/env python3

import argparse
import random
import sys
from typing import List, Tuple

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
)
from ..utils.color_names_handler import get_title_for_hex, resolve_color_name_or_exit
from ..utils.hexlab_logger import log
from ..utils.input_handler import INPUT_HANDLERS, HexlabArgumentParser
from ..utils.print_color_block import print_color_block
from ..utils.truecolor import ensure_truecolor


def _generate_search_cloud(
    base_rgb: Tuple[int, int, int],
    count: int = 5000
) -> List[Tuple[int, int, int]]:
    r, g, b = base_rgb
    h, s, l = rgb_to_hsl(r, g, b)

    candidates = set()

    for _ in range(count):
        h_delta = random.uniform(-20, 20)
        s_delta = random.uniform(-0.15, 0.15)
        l_delta = random.uniform(-0.15, 0.15)

        new_h = (h + h_delta) % 360
        new_s = max(0.0, min(1.0, s + s_delta))
        new_l = max(0.0, min(1.0, l + l_delta))

        nr, ng, nb = hsl_to_rgb(new_h, new_s, new_l)

        candidates.add((nr, ng, nb))

    return list(candidates)


def find_similar_colors_dynamic(
    base_rgb: Tuple[int, int, int],
    n: int = 5,
    metric: str = 'lab',
    dedup_val: float = 7.7
) -> List[Tuple[str, float]]:
    base_r_i, base_g_i, base_b_i = base_rgb

    base_lab, base_oklab = None, None

    if metric == 'lab':
        x, y, z = rgb_to_xyz(base_r_i, base_g_i, base_b_i)
        base_lab = xyz_to_lab(x, y, z)
    elif metric == 'oklab':
        base_oklab = rgb_to_oklab(base_r_i, base_g_i, base_b_i)

    pool_size = max(5000, n * 10)
    candidate_pool = _generate_search_cloud(base_rgb, count=pool_size)

    valid_similar = []

    for cand_rgb in candidate_pool:
        r, g, b = cand_rgb
        if r == base_r_i and g == base_g_i and b == base_b_i:
            continue

        diff = 0.0

        if metric == 'lab':
            x, y, z = rgb_to_xyz(r, g, b)
            cand_lab = xyz_to_lab(x, y, z)
            diff = delta_e_ciede2000(base_lab, cand_lab)
        elif metric == 'oklab':
            cand_oklab = rgb_to_oklab(r, g, b)
            diff = delta_e_euclidean_oklab(base_oklab, cand_oklab)
        elif metric == 'rgb':
            diff = delta_e_euclidean_rgb(base_r_i, base_g_i, base_b_i, r, g, b)

        if diff >= dedup_val:
            valid_similar.append((rgb_to_hex(r, g, b), diff))

    valid_similar.sort(key=lambda x: x[1])

    return valid_similar[:n]


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
    elif args.hexcode:
        clean_hex = args.hexcode
        title = get_title_for_hex(clean_hex)
    elif getattr(args, "decimal_index", None) is not None:
        clean_hex = args.decimal_index
        idx = int(clean_hex, 16)
        title = get_title_for_hex(clean_hex, f"index {idx}")

    print()
    print_color_block(clean_hex, title)
    print()
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

    num_results = args.number

    similar_list = find_similar_colors_dynamic(
        base_rgb,
        n=num_results,
        metric=metric,
        dedup_val=dedup_val
    )

    metric_map = {'lab': 'ΔE2000', 'oklab': 'ΔE(OKLAB)', 'rgb': 'ΔE(RGB)'}
    metric_label = metric_map.get(metric, 'ΔE')

    if not similar_list:
        log('info', "no similar colors found within parameters")
    else:
        for i, (hex_val, diff) in enumerate(similar_list):
            label = f"similar {i + 1}"
            print_color_block(hex_val, label, end="")
            print(f"  ({metric_label}: {diff:.2f})")

    print()


def get_similar_parser() -> argparse.ArgumentParser:
    parser = HexlabArgumentParser(
        prog="hexlab similar",
        description="hexlab similar: find perceptually similar colors from the full 24-bit spectrum",
        formatter_class=argparse.RawTextHelpFormatter
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-H", "--hex",
        dest="hexcode",
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
        help="base color name from --list-color-names"
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
        "-n", "--number",
        type=INPUT_HANDLERS["number"],
        default=5,
        help="number of similar colors to find (min: 2, max: 1000, default: 5)"
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
