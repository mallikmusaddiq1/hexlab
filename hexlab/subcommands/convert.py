#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: hexlab/subcommands/convert.py

import argparse
import sys
from hexlab.core import config as c
from hexlab.shared.logger import HexlabArgumentParser, log
from hexlab.shared.sanitizer import INPUT_HANDLERS
from hexlab.logic.convert.engine import run

def handle_convert_command(args: argparse.Namespace) -> None:
    try:
        f_fmt = c.FORMAT_ALIASES[args.from_format]
        t_fmt = c.FORMAT_ALIASES[args.to_format]
    except KeyError as e:
        log("error", f"invalid format: {e}")
        sys.exit(2)

    print(run(args.value, f_fmt, t_fmt, args.random, args.seed, args.verbose))

def get_convert_parser() -> argparse.ArgumentParser:

    parser = HexlabArgumentParser(prog="hexlab convert", add_help=False)
    parser.add_argument(
        "-h",
        "--help",
        action="help"
    )

    parser.add_argument(
        "-f",
        "--from-format",
        required=True,
        type=INPUT_HANDLERS["from_format"]
    )
    
    parser.add_argument(
        "-t",
        "--to-format",
        required=True,
        type=INPUT_HANDLERS["to_format"]
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
         "-v",
         "--value",
         type=str
    )
    
    group.add_argument(
         "-r",
         "--random",
         action="store_true"
    )
    
    parser.add_argument(
         "-s",
         "--seed",
         type=INPUT_HANDLERS["seed"]
    )

    parser.add_argument(
         "-V",
         "--verbose",
         action="store_true")
    
    return parser

def main():
    args = get_convert_parser().parse_args(sys.argv[1:])
    handle_convert_command(args)

if __name__ == "__main__":
    main()