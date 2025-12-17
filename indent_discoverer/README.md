# Indent Discoverer

Indent Discoverer is a small Python tool that inspects source files to infer indentation style (tabs vs. spaces) and the most likely indent size. It is useful when onboarding to unfamiliar codebases or normalizing formatting settings across editors.

## Quick start

1. Install dependencies (only `Python 3.9+` required).
2. Run the CLI with paths to files or directories:

```bash
python -m indent_discoverer.cli path/to/file.py another_dir/
```

The tool walks provided files, skips binary data, and prints a summary of detected styles.

## Project layout

- `src/indent_discoverer/core.py` — indentation detection logic.
- `src/indent_discoverer/cli.py` — simple command-line interface.
- `tests/` — space for future automated tests.
