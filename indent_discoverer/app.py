"""
Streamlit UI for Indent Discoverer.

Run from repo root:
  PYTHONPATH=src streamlit run indent_discoverer/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure the package is importable whether run via `streamlit run indent_discoverer/app.py`
# or directly from this directory without installation.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from indent_discoverer.core import analyze_paths


def parse_inputs(raw: str) -> list[str]:
    parts = []
    for line in raw.splitlines():
        for piece in line.split(","):
            cleaned = piece.strip()
            if cleaned:
                parts.append(cleaned)
    return parts


def to_rows(results):
    for res in results:
        yield {
            "path": str(res.path),
            "style": res.dominant_style(),
            "tabbed_lines": res.tabbed_lines,
            "spaced_lines": res.spaced_lines,
            "common_space_width": res.most_common_space_width(),
            "scanned_lines": res.scanned_lines,
        }


def main() -> None:
    st.set_page_config(page_title="Indent Discoverer", layout="wide")
    st.title("Indent Discoverer")
    st.write("Inspect files or directories to infer indentation style and size.")

    default_paths = "."
    raw = st.text_area(
        "Paths (comma or newline separated)",
        value=default_paths,
        placeholder="e.g. src/, README.md",
        height=120,
    )

    if st.button("Analyze", type="primary"):
        paths = parse_inputs(raw)
        if not paths:
            st.warning("Enter at least one file or directory path.")
            return

        with st.spinner("Scanning..."):
            results = analyze_paths(paths)

        if not results:
            st.info("No readable text files found in the provided paths.")
            return

        df = pd.DataFrame(to_rows(results))
        st.success(f"Analyzed {len(results)} file(s).")
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
