# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2023-2026 the iorec authors
"""Command line conversion between mesh formats: `iorec-convert SOURCE TARGET`.

Also available as `python -m iorec SOURCE TARGET`. Examples::

    iorec-convert cells.rec cells.fe --fix-volumes --tension-file tensions.txt
    iorec-convert cells.dmp cells.rec --binary
    iorec-convert cells.rec cells.obj --group-by cell
    iorec-convert cells.rec cells.3mf --unit micron --scale 500 --extruder 3=1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from iorec.mesh_io import _accepted_options, convert


def _program_name() -> str:
    name = Path(sys.argv[0]).name
    return "python -m iorec" if name == "__main__.py" else name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_program_name(),
        description=(
            "Convert a labelled mesh between .rec/.arec, .fe/.dmp (Surface Evolver), "
            ".ply, .obj and .3mf, chosen by the file suffixes."
        ),
    )
    parser.add_argument("source", type=Path, help="mesh file to read")
    parser.add_argument("target", type=Path, help="mesh file to write")
    parser.add_argument("--validate", action="store_true", help="check the source")
    parser.add_argument(
        "--exterior-label",
        type=int,
        default=None,
        metavar="N",
        help="label of the exterior; default: the one stored in .obj/.3mf, else 0",
    )
    encoding = parser.add_mutually_exclusive_group()
    encoding.add_argument("--binary", action="store_true", help=".rec/.ply: binary")
    encoding.add_argument("--text", action="store_true", help=".rec/.ply: text")
    parser.add_argument("--colors", action="store_true", help=".ply: face colours")
    parser.add_argument("--group-by", choices=("interface", "cell"), default=None)
    parser.add_argument("--unit", default=None, help=".3mf: length unit")
    parser.add_argument("--scale", type=float, default=None, help=".3mf: size factor")
    parser.add_argument(
        "--extruder",
        action="append",
        default=[],
        metavar="LABEL=E",
        help=".3mf: print LABEL with Cura extruder E (0 = first); repeatable",
    )
    parser.add_argument(
        "--fix-volumes",
        action="store_true",
        help=".fe: hold each body at the volume it has on loading",
    )
    parser.add_argument(
        "--tension-file",
        type=Path,
        default=None,
        help=".fe: text file of 'label_a label_b tension' rows",
    )
    return parser


def _read_tensions(path: Path) -> dict[tuple[int, int], float]:
    rows = np.loadtxt(path, ndmin=2)
    return {(int(a), int(b)): float(t) for a, b, t in rows[:, :3].tolist()}


def _save_options(args: argparse.Namespace) -> dict[str, Any]:
    """Keyword arguments of the writer of the target format."""
    options: dict[str, Any] = {}
    if args.binary or args.text:
        options["binary_mode"] = args.binary
    if args.colors:
        options["colors"] = True
    if args.group_by:
        options["group_by"] = args.group_by
    options.update(_print_options(args), **_evolver_options(args))
    accepted = _accepted_options(args.target, "write")
    return {key: value for key, value in options.items() if key in accepted}


def _print_options(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if args.unit:
        options["unit"] = args.unit
    if args.scale is not None:
        options["scale"] = args.scale
    pairs = [item.split("=") for item in args.extruder]
    options["extruders"] = {int(label): int(e) for label, e in pairs}
    return options


def _evolver_options(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if args.fix_volumes:
        options["volumes"] = "current"
    if args.tension_file:
        options["tensions"] = _read_tensions(args.tension_file)
    return options


def main(argv: list[str] | None = None) -> int:
    """Run the conversion; return the process exit status."""
    args = _parser().parse_args(argv)
    load_options: dict[str, Any] = {"validate": args.validate}
    if args.exterior_label is not None:
        load_options["exterior_label"] = args.exterior_label
    try:
        convert(
            args.source,
            args.target,
            load_options=load_options,
            save_options=_save_options(args),
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
