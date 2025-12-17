import tempfile
from pathlib import Path

from indent_discoverer.core import analyze_file


def write_sample(lines: list[str], suffix: str = ".txt") -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=suffix)
    tmp.write("\n".join(lines))
    tmp.flush()
    return Path(tmp.name)


def test_spaces_indent_detection():
    path = write_sample(["    def func():", "        return 1", "print('done')"])
    result = analyze_file(path)
    assert result is not None
    assert result.dominant_style().startswith("spaces")
    assert result.most_common_space_width() == 4


def test_tabs_indent_detection():
    path = write_sample(["\tdef func():", "\t\treturn 1"])
    result = analyze_file(path)
    assert result is not None
    assert result.dominant_style() == "tabs"
