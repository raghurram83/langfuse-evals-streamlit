from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from .core import IndentResult, analyze_paths


def format_result(result: IndentResult) -> str:
    style = result.dominant_style()
    width = result.most_common_space_width()
    extras: list[str] = []
    if width:
        extras.append(f"common width {width}")
    extras.append(f"tabbed {result.tabbed_lines}")
    extras.append(f"spaced {result.spaced_lines}")
    if result.scanned_lines:
        extras.append(f"scanned {result.scanned_lines} lines")
    extra_str = ", ".join(extras)
    return f"{result.path}: {style} ({extra_str})"


def run(paths: Iterable[str]) -> list[str]:
    results = analyze_paths(paths)
    return [format_result(result) for result in results]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Infer indentation style and size for files or directories."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to inspect. Directories are walked recursively.",
    )
    args = parser.parse_args(argv)

    outputs = run(args.paths)
    if not outputs:
        print("No readable text files found.")
        return

    for line in outputs:
        print(line)


if __name__ == "__main__":
    main()
