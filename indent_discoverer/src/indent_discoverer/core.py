from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

_WHITESPACE_RE = re.compile(r"^([ \t]+)")


@dataclass
class IndentResult:
    path: Path
    tabbed_lines: int
    spaced_lines: int
    space_widths: Counter
    scanned_lines: int

    def dominant_style(self) -> str:
        """Return a human-friendly label for the likely indent style."""
        if self.tabbed_lines == 0 and self.spaced_lines == 0:
            return "none detected"
        if self.tabbed_lines > self.spaced_lines:
            return "tabs"
        if self.spaced_lines > self.tabbed_lines:
            width = self.most_common_space_width()
            return f"spaces ({width})" if width else "spaces"
        return "mixed"

    def most_common_space_width(self) -> int | None:
        if not self.space_widths:
            return None
        return self.space_widths.most_common(1)[0][0]


def is_probably_text(path: Path, sample_size: int = 2048) -> bool:
    """Heuristic to skip binary files quickly."""
    try:
        data = path.read_bytes()[:sample_size]
    except (OSError, PermissionError):
        return False
    return b"\0" not in data


def discover_files(paths: Iterable[str]) -> List[Path]:
    """Expand file and directory inputs into a flat list of candidate files."""
    files: List[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    files.append(child)
        elif path.is_file():
            files.append(path)
    return files


def analyze_file(path: Path) -> IndentResult | None:
    """Inspect a single file for indentation patterns."""
    if not is_probably_text(path):
        return None

    tabbed_lines = 0
    spaced_lines = 0
    space_widths: Counter = Counter()
    scanned_lines = 0

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                scanned_lines += 1
                match = _WHITESPACE_RE.match(line)
                if not match:
                    continue
                prefix = match.group(1)
                if prefix.startswith("\t"):
                    tabbed_lines += 1
                elif prefix.startswith(" "):
                    spaced_lines += 1
                    space_widths[len(prefix)] += 1
    except (OSError, UnicodeError):
        return None

    return IndentResult(
        path=path,
        tabbed_lines=tabbed_lines,
        spaced_lines=spaced_lines,
        space_widths=space_widths,
        scanned_lines=scanned_lines,
    )


def analyze_paths(paths: Iterable[str]) -> List[IndentResult]:
    """Analyze many paths and collect results."""
    results: List[IndentResult] = []
    for file_path in discover_files(paths):
        result = analyze_file(file_path)
        if result:
            results.append(result)
    return results
