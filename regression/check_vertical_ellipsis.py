"""세로쓰기 말줄임표의 점 개수와 외곽선 회귀 검사.

실행: runtime/python.exe regression/check_vertical_ellipsis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.text.drawing_engine import _draw_vertical_ellipsis  # noqa: E402
from core.text.layout_engine import _measure_vertical_unit  # noqa: E402


class RecordingCanvas:
    def __init__(self) -> None:
        self.circles = []

    def drawCircle(self, x, y, radius, paint) -> None:
        self.circles.append((x, y, radius, paint))


def main() -> int:
    for text, expected in (("..", 2), ("...", 3), ("‥", 2), ("…", 3)):
        unit = _measure_vertical_unit(text, "regular", 20, {}, {})
        assert unit is not None
        assert unit["draw_vertical_ellipsis"] is True
        assert unit["ellipsis_dot_count"] == expected, (text, unit)

    canvas = RecordingCanvas()
    _draw_vertical_ellipsis(canvas, 10, 20, 2, 6, 1.5, "fill", "outline")
    assert canvas.circles == [
        (10.0, 17.0, 2.0, "outline"),
        (10.0, 17.0, 1.5, "fill"),
        (10.0, 23.0, 2.0, "outline"),
        (10.0, 23.0, 1.5, "fill"),
    ]
    print("세로쓰기 말줄임표 회귀 검사 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
