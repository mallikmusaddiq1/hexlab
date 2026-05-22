#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/subcommands/unicode.py

"""
Unicode Subcommand Module for Hexlab.

Provides advanced semantic color matching across the 1.1 Million Unicode space.
Uses a hybrid architecture of K-Means Clustering, Delta-E Gatekeeping, and
Earth Mover's Distance (EMD) to find visually identical emojis while ignoring
utility colors like borders, shadows, and transparency anti-aliasing.
"""

import argparse
import concurrent.futures
import io
import os
import random
import signal
import sys
import urllib.request
import unicodedata
from typing import Any, Dict, List, Tuple, Generator

from PIL import Image, ImageDraw, ImageFont

from hexlab.core import config as c
from hexlab.core.conversions import (
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
from hexlab.shared.logger import HexlabArgumentParser, log
from hexlab.shared.preview import print_color_block
from hexlab.shared.sanitizer import INPUT_HANDLERS
from hexlab.shared.truecolor import ensure_truecolor

def sigint_handler(sig: int, frame: Any) -> None:
    """Instantly kills all threads and stops the process on Ctrl+C."""
    os._exit(0)

signal.signal(signal.SIGINT, sigint_handler)

GLOBAL_EMOJI_FONT = None
GLOBAL_USE_NETWORK = False


def ensure_emoji_env() -> None:
    """
    Checks if the local OS can render truecolor emojis.
    Fallback to Twemoji network rendering if local rendering fails.
    """
    global GLOBAL_EMOJI_FONT, GLOBAL_USE_NETWORK
    if GLOBAL_EMOJI_FONT is not None or GLOBAL_USE_NETWORK:
        return

    sizes = [109, 136, 64, 128]
    paths = [
        "/system/fonts/NotoColorEmoji.ttf",
        "/system/fonts/AndroidEmoji.ttf",
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/System/Library/Fonts/Apple Color Emoji.ttc",
        "seguiemj.ttf",
    ]

    for path in paths:
        if not os.path.exists(path):
            continue
        for size in sizes:
            try:
                font = ImageFont.truetype(path, size)
                img = Image.new("RGBA", (32, 32), (255, 0, 255, 255))
                draw = ImageDraw.Draw(img)
                draw.text((2, 2), "🐪", font=font, embedded_color=True)
                
                has_graphic = False
                for p in img.getdata():
                    if p[:3] != (255, 0, 255):
                        has_graphic = True
                        break
                if has_graphic:
                    GLOBAL_EMOJI_FONT = font
                    return
            except Exception:
                continue

    GLOBAL_USE_NETWORK = True


def _to_metric_space(r: int, g: int, b: int, metric: str) -> Any:
    """Converts raw RGB into the chosen mathematical metric space."""
    if metric == "lab":
        x, y, z = rgb_to_xyz(r, g, b)
        return xyz_to_lab(x, y, z)
    elif metric == "oklab":
        return rgb_to_oklab(r, g, b)
    else:
        return (float(r), float(g), float(b))


def _get_distance(val1: Any, val2: Any, metric: str) -> float:
    """Routes the distance calculation to the appropriate core formula."""
    if metric == "lab":
        return delta_e_ciede2000(val1, val2)
    elif metric == "oklab":
        return delta_e_euclidean_oklab(val1, val2)
    else:
        return delta_e_euclidean_rgb(val1, val2)


def calculate_emd(sig1: List[Tuple[float, Any]], sig2: List[Tuple[float, Any]], metric: str) -> float:
    """
    Computes Earth Mover's Distance (EMD) between two color palettes.
    Internal 'work cost' is dynamically mapped via the chosen metric distance.
    """
    if not sig1 or not sig2:
        return float('inf')

    w1 = [s[0] for s in sig1]
    w2 = [s[0] for s in sig2]

    distances = []
    for i, (_, m_val1) in enumerate(sig1):
        for j, (_, m_val2) in enumerate(sig2):
            dist = _get_distance(m_val1, m_val2, metric)
            distances.append((dist, i, j))

    distances.sort(key=lambda x: x[0])

    total_cost = 0.0
    for dist, i, j in distances:
        if w1[i] > 0 and w2[j] > 0:
            amount = min(w1[i], w2[j])
            total_cost += amount * dist
            w1[i] -= amount
            w2[j] -= amount

    return total_cost


def _fast_kmeans_palette(
    color_counts: Dict[Tuple[int, int, int], int], 
    metric: str, 
    k: int = 4, 
    max_iter: int = 8
) -> List[Tuple[float, Any]]:
    """Extracts dominant cluster weights using standard fast RGB math."""
    unique_colors = list(color_counts.keys())
    if not unique_colors:
        return []
    
    k = min(k, len(unique_colors))
    centers = random.sample(unique_colors, k)
    
    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for color in unique_colors:
            best_i, best_dist = 0, float('inf')
            for i, c_cent in enumerate(centers):
                dist = (color[0]-c_cent[0])**2 + (color[1]-c_cent[1])**2 + (color[2]-c_cent[2])**2
                if dist < best_dist:
                    best_dist = dist
                    best_i = i
            clusters[best_i].append(color)
        
        new_centers = []
        for i in range(k):
            if not clusters[i]:
                new_centers.append(centers[i])
                continue
            sr, sg, sb, tw = 0, 0, 0, 0
            for color in clusters[i]:
                w = color_counts[color]
                sr += color[0]*w; sg += color[1]*w; sb += color[2]*w; tw += w
            new_centers.append((int(sr/tw), int(sg/tw), int(sb/tw)))
        
        if centers == new_centers:
            break
        centers = new_centers
        
    total_w = sum(color_counts.values())
    sig = []
    for i in range(k):
        if not clusters[i]:
            continue
        weight = sum(color_counts[c] for c in clusters[i]) / total_w
        m_val = _to_metric_space(*centers[i], metric)
        sig.append((weight, m_val))
        
    sig.sort(key=lambda x: x[0], reverse=True)
    return sig


def _extract_signature_and_rgb(
    img: Image.Image, 
    is_local: bool, 
    metric: str, 
    k: int = 4
) -> Tuple[Tuple[int, int, int], List[Tuple[float, Any]]]:
    """Applies semantic utility stripping and extracts K-Means visual signature."""
    img = img.convert("RGBA")
    img.thumbnail((32, 32), Image.Resampling.NEAREST)
    
    color_counts = {}
    valid_color_counts = {}
    
    for p in img.getdata():
        r, g, b, a = p[0], p[1], p[2], p[3]
        if a < 50:
            continue
        if is_local and r == 255 and g == 0 and b == 255:
            continue
            
        q_rgb = ((r//16)*16, (g//16)*16, (b//16)*16)
        color_counts[q_rgb] = color_counts.get(q_rgb, 0) + 1
        
        # Semantic Utility Stripping (Ignores pure white/black boundaries)
        h, s, l_val = rgb_to_hsl(r, g, b)
        if l_val < 0.15 or l_val > 0.85 or (s < 0.15 and 0.15 <= l_val <= 0.85):
            continue
            
        valid_color_counts[q_rgb] = valid_color_counts.get(q_rgb, 0) + 1

    if not valid_color_counts:
        valid_color_counts = color_counts
    if not valid_color_counts:
        return (0, 0, 0), None

    cluster_signature = _fast_kmeans_palette(valid_color_counts, metric, k=k)
    if not cluster_signature:
        return (0, 0, 0), None

    # Link visual dominant RGB to the highest weighted color space cluster
    dom_val = cluster_signature[0][1]
    best_rgb, best_dist = (0, 0, 0), float('inf')
    
    for q_rgb in valid_color_counts.keys():
        dist = _get_distance(dom_val, _to_metric_space(*q_rgb, metric), metric)
        if dist < best_dist:
            best_dist = dist
            best_rgb = q_rgb
            
    return best_rgb, cluster_signature


def calculate_hybrid_relevance_score(
    base_sig: List[Tuple[float, Any]], 
    cand_sig: List[Tuple[float, Any]], 
    metric: str
) -> Tuple[float, float, float]:
    """
    THE MASTER FORMULA:
    Calculates the exact visual distance using Dominant Delta-E and Palette EMD.
    
    Returns:
        Tuple[float, float, float]: (hybrid_score, dominant_difference, emd_difference)
    """
    base_dom = base_sig[0][1]
    cand_dom = cand_sig[0][1]

    # Gatekeeper 1: Exact Dominant Color Match
    dom_diff = _get_distance(base_dom, cand_dom, metric)

    # Gatekeeper 2: Entire Palette Distribution Distance
    emd_diff = calculate_emd(base_sig, cand_sig, metric)

    # Final Blend: 60% importance on Dominant Color, 40% on Palette Vibe
    final_score = (dom_diff * 0.6) + (emd_diff * 0.4)
    return final_score, dom_diff, emd_diff


def fetch_twemoji_signature(
    char: str, 
    metric: str
) -> Tuple[Tuple[int, int, int], List[Tuple[float, Any]]]:
    """Bypasses local limitations and downloads graphical signatures from Twemoji."""
    codepoints = [hex(ord(c))[2:].lower() for c in char]
    if len(codepoints) > 1 and "fe0f" in codepoints:
        codepoints.remove("fe0f")
    cp_str = "-".join(codepoints)

    url = f"https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/72x72/{cp_str}.png"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as response:
            img_data = response.read()
        img = Image.open(io.BytesIO(img_data))
        return _extract_signature_and_rgb(img, is_local=False, metric=metric)
    except Exception:
        return (0, 0, 0), None


def extract_single_char_signature(
    char: str, 
    metric: str
) -> Tuple[Tuple[int, int, int], List[Tuple[float, Any]]]:
    """Primary handler to determine graphical validity of a unicode character."""
    ensure_emoji_env()
    if GLOBAL_USE_NETWORK:
        return fetch_twemoji_signature(char, metric)

    img = Image.new("RGBA", (32, 32), (255, 0, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        draw.text((2, 2), char, font=GLOBAL_EMOJI_FONT, embedded_color=True)
    except Exception:
        pass
        
    ex = img.getextrema()
    if ex[0][0] == 255 and ex[0][1] == 255 and ex[1][0] == 0 and ex[1][1] == 0 and ex[2][0] == 255 and ex[2][1] == 255:
        return (0, 0, 0), None
        
    return _extract_signature_and_rgb(img, is_local=True, metric=metric)


def get_aligned_base_title(base_str: str, is_sequence: bool = False) -> str:
    """Dynamically aligns base symbol to perfectly match standard label width."""
    target_width = 16
    if not is_sequence:
        pad = max(0, target_width - len(base_str))
        return f"{base_str}{' ' * pad}"
        
    vw = 0
    for ch in base_str:
        if ch in ('\uFE0F', '\u200D', '\u200B'):
            continue
        try:
            vw += 2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1
        except Exception:
            vw += 1
            
    pad = max(0, target_width - vw)
    try:
        if any(unicodedata.east_asian_width(c) in ('W', 'F') for c in base_str):
            return f"{base_str}\u200B{' ' * pad}"
    except Exception:
        pass
    return f"{base_str}{' ' * pad}"


def generate_unicode_chunks(chunk_size: int) -> Generator[List[int], None, None]:
    """Generates grouped integers representing valid Unicode 1.1M bounds."""
    priority_blocks = [
        (0x1F300, 0x1FAFF), # Emoticons, Misc Symbols
        (0x2600, 0x27BF),   # Dingbats, Weather
        (0x2000, 0x25FF),   # Punctuation, Math, Box Drawing
        (0x2B00, 0x2BFF),   # Arrows
    ]
    
    seen = set()
    current_chunk = []

    for start, end in priority_blocks:
        for cp in range(start, end + 1):
            if unicodedata.name(chr(cp), None):
                current_chunk.append(cp)
                seen.add(cp)
                if len(current_chunk) >= chunk_size:
                    yield current_chunk
                    current_chunk = []
                    
    if current_chunk:
        yield current_chunk
        current_chunk = []

    if not GLOBAL_USE_NETWORK:
        for cp in range(0x0020, 0x110000):
            if cp in seen:
                continue
            if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF:
                continue
            
            char = chr(cp)
            if unicodedata.name(char, None) is not None:
                current_chunk.append(cp)
                if len(current_chunk) >= chunk_size:
                    yield current_chunk
                    current_chunk = []
                    
        if current_chunk:
            yield current_chunk


def handle_unicode_command(args: argparse.Namespace) -> None:
    """Core execution logic mapped to the unicode subcommand flags."""
    ensure_emoji_env()
    
    metric = args.distance_metric
    clean_hex = None
    base_rgb = None
    base_sig = None
    title = "base symbol"

    if args.symbol:
        target_char = args.symbol
        base_rgb, base_sig = extract_single_char_signature(target_char, metric)
        if base_rgb and base_rgb != (0, 0, 0): 
            clean_hex = rgb_to_hex(*base_rgb)
        title = get_aligned_base_title(target_char, is_sequence=True)
    elif args.unicode:
        target_char = args.unicode 
        base_rgb, base_sig = extract_single_char_signature(target_char, metric)
        if base_rgb and base_rgb != (0, 0, 0): 
            clean_hex = rgb_to_hex(*base_rgb)
        title = get_aligned_base_title(target_char, is_sequence=True)
    elif getattr(args, 'decimal_index', None) is not None:
        target_char = args.decimal_index 
        base_rgb, base_sig = extract_single_char_signature(target_char, metric)
        if base_rgb and base_rgb != (0, 0, 0): 
            clean_hex = rgb_to_hex(*base_rgb)
        title = get_aligned_base_title(target_char, is_sequence=True)

    if not clean_hex or not base_sig:
        log("error", "could not extract a valid visual signature for the input.")
        sys.exit(1)

    print()
    print_color_block(clean_hex, f"{c.BOLD_WHITE}{title}{c.RESET}")
    print()
    
    f_out = None
    if args.output_txt:
        try:
            f_out = open(args.output_txt, "w", encoding="utf-8")
        except Exception as e:
            log("error", f"could not open output file: {e}")
            sys.exit(1)

    # Determine Dynamic Max Threshold limits from Core Configuration
    if args.dedup_value is not None:
        max_threshold = args.dedup_value
    else:
        if metric == "lab":
            max_threshold = getattr(c, "DEDUP_DELTA_E_LAB_UNICODE", 18.0)
        elif metric == "oklab":
            max_threshold = getattr(c, "DEDUP_DELTA_E_OKLAB_UNICODE", 0.18)
        else:
            max_threshold = getattr(c, "DEDUP_DELTA_E_RGB_UNICODE", 60.0)

    metric_map = {"lab": "ΔE(2000)", "oklab": "ΔE(OKLAB)", "rgb": "ΔE(RGB)"}
    metric_label = metric_map.get(metric, "ΔE")

    found_count = 0
    dynamic_chunk_size = 50 if GLOBAL_USE_NETWORK else 250
    
    if GLOBAL_USE_NETWORK:
        log("info", "twemoji cloud bypass active and network rendering engaged\n")

    def process_candidate(cp: int) -> Any:
        char = chr(cp)
        if GLOBAL_USE_NETWORK:
            cand_rgb, cand_sig = fetch_twemoji_signature(char, metric)
        else:
            img = Image.new("RGBA", (32, 32), (255, 0, 255, 255))
            draw = ImageDraw.Draw(img)
            try:
                draw.text((2, 2), char, font=GLOBAL_EMOJI_FONT, embedded_color=True)
            except Exception:
                pass
                
            ex = img.getextrema()
            if ex[0][0] == 255 and ex[0][1] == 255 and ex[1][0] == 0 and ex[1][1] == 0 and ex[2][0] == 255 and ex[2][1] == 255:
                return None
                
            cand_rgb, cand_sig = _extract_signature_and_rgb(img, is_local=True, metric=metric)

        if cand_rgb != (0, 0, 0) and cand_sig is not None:
            # Breakdown the hybrid score into its individual reporting components
            hybrid_score, dom_diff, emd_diff = calculate_hybrid_relevance_score(base_sig, cand_sig, metric)
            
            if hybrid_score <= max_threshold:
                return (hybrid_score, dom_diff, emd_diff, cand_rgb, char, cp)
        return None

    # Core Execution Loop
    for chunk in generate_unicode_chunks(chunk_size=dynamic_chunk_size):
        if found_count >= args.count:
            break
            
        results = []
        if GLOBAL_USE_NETWORK:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(process_candidate, cp): cp for cp in chunk}
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res: 
                        results.append(res)
        else:
            for cp in chunk:
                res = process_candidate(cp)
                if res: 
                    results.append(res)
        
        if results:
            results.sort(key=lambda x: x[0])
            
            for score, dom_diff, emd_diff, rgb, char_val, cp in results:
                if found_count >= args.count:
                    break
                    
                found_count += 1
                hex_key = rgb_to_hex(*rgb)
                
                if f_out:
                    out_str = char_val if args.unicode_form == "symbol" else f"U+{hex(ord(char_val))[2:].upper().zfill(4)}"
                    f_out.write(f"{out_str}\n")
                    f_out.flush()
                
                label = f"{c.MSG_BOLD_COLORS['info']}unicode{f'{found_count}':>9}{c.RESET}"
                print_color_block(hex_key, label, end="")
                
                u_code = f"{c.BOLD_WHITE}U+{hex(ord(char_val))[2:].upper().zfill(4)}{c.RESET}"
                
                try:
                    eaw = unicodedata.east_asian_width(char_val)
                except Exception:
                    eaw = 'N'
                visual_pad = " " if eaw not in ('W', 'F') else ""
                
                # Dynamically aligned to 6 characters (>6.2f) to prevent shifting on 3-digit numbers, gaps reduced to prevent word-wrap
                print(f" {c.BOLD_WHITE}[{char_val}]{visual_pad}{c.RESET} {u_code:<7} {c.MSG_BOLD_COLORS['info']}({metric_label}: {dom_diff:>6.2f}) (EMD: {emd_diff:>6.2f}){c.RESET}")
                sys.stdout.flush()

    if f_out:
        f_out.close()

    if found_count == 0:
        log("info", f"no exact visual matches found within the distance threshold of {max_threshold}")
    else:
        if args.output_txt:
            print(f"\n{c.MSG_COLORS['success']}[success]{c.RESET} {found_count} elements written cleanly to {args.output_txt}")
    print()


def get_unicode_parser() -> argparse.ArgumentParser:
    """Instantiates the subcommand argument parser context."""
    parser = HexlabArgumentParser(
        prog="hexlab unicode",
        description="hexlab unicode: search 1.1M space using Hybrid K-Means + EMD + Distance Filtering",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    uni_input = parser.add_mutually_exclusive_group(required=True)
    uni_input.add_argument(
        "-s",
        "--symbol",
        type=str,
        help="direct emoji asset to extract reference"
    )
    uni_input.add_argument(
         "-u",
         "--unicode",
         type=INPUT_HANDLERS.get("unicode_seq", str),
         help="U+XXXX string sequence to extract reference"
    )
    uni_input.add_argument(
        "-di",
        "--decimal-index",
        type=INPUT_HANDLERS.get("decimal_seq", str),
        help="decimal sequence index of the unicode symbol"
    )

    parser.add_argument(
        "-dm", "--distance-metric", 
        type=INPUT_HANDLERS.get("distance_metric", str), 
        choices=["lab", "oklab", "rgb"], 
        default="lab", 
        help="distance metric: lab oklab rgb (default: lab)"
    )
    parser.add_argument(
        "-dv", "--dedup-value", 
        type=INPUT_HANDLERS.get("dedup_value", float), 
        default=None, 
        help=(
            "custom relevance threshold cutoff for unicode matches. Defines visual strictness.\n"
            f"ideal defaults: lab={getattr(c, 'DEDUP_DELTA_E_LAB_UNICODE', 18.0)}, "
            f"oklab={getattr(c, 'DEDUP_DELTA_E_OKLAB_UNICODE', 0.18)}, "
            f"rgb={getattr(c, 'DEDUP_DELTA_E_RGB_UNICODE', 60.0)}"
        )
    )
    parser.add_argument(
        "-c",
        "--count",
        type=INPUT_HANDLERS.get("count_unicode", int),
        default=1000,
        help="max number of similar emojis to find (default: 1000)"
    )
    parser.add_argument(
        "-ot",
        "--output-txt",
        type=str,
        default=None, help="target dump path for matched output files"
    )
    parser.add_argument(
        "-uf",
        "--unicode-form",
        choices=["symbol", "unicode"],
        default="symbol",
        help="styling layout mode inside target text files"
    )

    return parser


def main() -> None:
    """Entry point execution binding for CLI invocation."""
    parser = get_unicode_parser()
    args = parser.parse_args(sys.argv[1:])
    ensure_truecolor()
    handle_unicode_command(args)


if __name__ == "__main__":
    main()