#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/subcommands/similar.py

import argparse
import random
import sys
from typing import Generator, Tuple, Set

from hexlab.core.conversions import (
    hex_to_rgb,
    hsl_to_rgb,
    rgb_to_hex,
    rgb_to_hsl,
    rgb_to_oklab,
    rgb_to_xyz,
    xyz_to_lab,
)
from hexlab.core.difference import (
    delta_e_ciede2000,
    delta_e_euclidean_oklab,
    delta_e_euclidean_rgb,
)
from hexlab.core import config as c
from hexlab.shared.naming import get_title_for_hex, resolve_color_name_or_exit
from hexlab.shared.logger import log, HexlabArgumentParser
from hexlab.shared.sanitizer import INPUT_HANDLERS
from hexlab.shared.preview import print_color_block
from hexlab.shared.truecolor import ensure_truecolor

CANDIDATES_PER_STEP = 500


def _generate_random_rgb() -> Tuple[int, int, int]:
    """Generate a random RGB color tuple.

    Returns:
        Tuple[int, int, int]: Random RGB values (0-255).
    """
    val = random.randint(0, c.MAX_DEC)
    r = (val >> 16) & 0xFF
    g = (val >> 8) & 0xFF
    b = val & 0xFF
    return r, g, b


def _to_metric_space(rgb: Tuple[int, int, int], metric: str):
    """Convert RGB to the specified metric space.

    Args:
        rgb (Tuple[int, int, int]): RGB values.
        metric (str): Metric type ('rgb', 'oklab', or 'lab').

    Returns:
        Tuple: Converted values in the metric space.
    """
    if metric == "rgb":
        return rgb
    elif metric == "oklab":
        return rgb_to_oklab(*rgb)
    else:
        x, y, z = rgb_to_xyz(*rgb)
        return xyz_to_lab(x, y, z)


def generate_similar_colors_streaming(
    base_rgb: Tuple[int, int, int],
    n: int = 5,
    metric: str = "lab",
    dedup_val: float = 7.7,
) -> Generator[Tuple[str, float], None, None]:
    """Generate similar colors by perturbing HSL values and checking distance.

    Streams hex codes and distances for colors similar to the base.

    Args:
        base_rgb (Tuple[int, int, int]): Base RGB color.
        n (int, optional): Number of similar colors to generate. Defaults to 5.
        metric (str, optional): Distance metric ('lab', 'oklab', 'rgb'). Defaults to 'lab'.
        dedup_val (float, optional): Minimum distance for deduplication. Defaults to 7.7.

    Yields:
        Generator[Tuple[str, float], None, None]: Hex code and distance for each similar color.
    """
    base_metric_val = _to_metric_space(base_rgb, metric)

    r, g, b = base_rgb
    h_base, s_base, l_base = rgb_to_hsl(r, g, b)

    count_found = 0
    seen_hex: Set[str] = set()
    seen_hex.add(rgb_to_hex(r, g, b))

    attempts = 0
    max_attempts = n * CANDIDATES_PER_STEP

    while count_found < n and attempts < max_attempts:
        attempts += 1

        # Perturb HSL values
        h_delta = random.uniform(-20, 20)
        s_delta = random.uniform(-0.15, 0.15)
        l_delta = random.uniform(-0.15, 0.15)

        new_h = (h_base + h_delta) % 360
        new_s = max(0.0, min(1.0, s_base + s_delta))
        new_l = max(0.0, min(1.0, l_base + l_delta))

        nr, ng, nb = hsl_to_rgb(new_h, new_s, new_l)
        nr_i, ng_i, nb_i = int(round(nr)), int(round(ng)), int(round(nb))

        cand_hex = rgb_to_hex(nr_i, ng_i, nb_i)

        # Skip if already seen
        if cand_hex in seen_hex:
            continue

        cand_rgb = (nr_i, ng_i, nb_i)

        diff = 0.0
        if metric == "lab":
            cand_metric = _to_metric_space(cand_rgb, "lab")
            diff = delta_e_ciede2000(base_metric_val, cand_metric)
        elif metric == "oklab":
            cand_metric = _to_metric_space(cand_rgb, "oklab")
            diff = delta_e_euclidean_oklab(base_metric_val, cand_metric)
        elif metric == "rgb":
            diff = delta_e_euclidean_rgb(base_rgb, cand_rgb)

        # Yield if sufficiently distant
        if diff >= dedup_val:
            seen_hex.add(cand_hex)
            count_found += 1
            yield (cand_hex, diff)


def handle_similar_command(args: argparse.Namespace) -> None:
    """Handle the similar command logic.

    Processes input color, generates similar colors, and prints them with distances.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    clean_hex = None
    title = "base color"
    if args.seed is not None:
        random.seed(args.seed)

    if args.random:
        current_dec = random.randint(0, c.MAX_DEC)
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
    print_color_block(clean_hex, f"{c.BOLD_WHITE}{title}{c.RESET}")
    print()

    base_rgb = hex_to_rgb(clean_hex)
    metric = args.distance_metric

    dedup_val = 0.0
    if args.dedup_value is not None:
        dedup_val = args.dedup_value
    else:
        if metric == "rgb":
            dedup_val = c.DEDUP_DELTA_E_RGB
        elif metric == "oklab":
            dedup_val = c.DEDUP_DELTA_E_OKLAB
        else:
            dedup_val = c.DEDUP_DELTA_E_LAB

    num_results = args.count

    similar_gen = generate_similar_colors_streaming(
        base_rgb,
        n=num_results,
        metric=metric,
        dedup_val=dedup_val,
    )

    metric_map = {"lab": "ΔE(2000)", "oklab": "ΔE(OKLAB)", "rgb": "ΔE(RGB)"}
    metric_label = metric_map.get(metric, "ΔE")

    found_any = False

    for i, (hex_val, diff) in enumerate(similar_gen):
        found_any = True

        label = f"{c.MSG_BOLD_COLORS['info']}similar{f'{i + 1}':>9}{c.RESET}"

        print_color_block(hex_val, label, end="")

        print(f"  {c.MSG_BOLD_COLORS['info']}({metric_label}: {diff:5.2f}){c.RESET}")
        sys.stdout.flush()

    if not found_any:
        log("info", "no similar colors found within parameters")

    print()


def get_similar_parser() -> argparse.ArgumentParser:
    """Create argument parser for similar command.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = HexlabArgumentParser(
        prog="hexlab similar",
        description="hexlab similar: find perceptually similar colors",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-H",
        "--hex",
        dest="hex",
        type=INPUT_HANDLERS["hex"],
        help="base hex code",
    )
    input_group.add_argument(
        "-r",
        "--random",
        action="store_true",
        help="use a random base",
    )
    input_group.add_argument(
        "-cn",
        "--color-name",
        type=INPUT_HANDLERS["color_name"],
        help="base color name",
    )
    input_group.add_argument(
        "-di",
        "--decimal-index",
        type=INPUT_HANDLERS["decimal_index"],
        help="base decimal index",
    )
    parser.add_argument(
        "-dm",
        "--distance-metric",
        type=INPUT_HANDLERS["distance_metric"],
        default="lab",
        help="distance metric: lab oklab rgb (default: lab)",
        choices=["lab", "oklab", "rgb"],
    )
    parser.add_argument(
        "-dv",
        "--dedup-value",
        type=INPUT_HANDLERS["dedup_value"],
        default=None,
        help=(
            f"custom deduplication threshold (defaults: lab={c.DEDUP_DELTA_E_LAB}, "
            f"oklab={c.DEDUP_DELTA_E_OKLAB}, rgb={c.DEDUP_DELTA_E_RGB})"
        ),
    )
    parser.add_argument(
        "-c",
        "--count",
        type=INPUT_HANDLERS["count_similar"],
        default=10,
        help="number of similar colors to find (min: 2, max: 250, default: 10)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=INPUT_HANDLERS["seed"],
        default=None,
        help="seed for reproducibility of random",
    )
    return parser


def main() -> None:
    """Main entry point for similar command."""
    parser = get_similar_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    handle_similar_command(args)


if __name__ == "__main__":
    main()
