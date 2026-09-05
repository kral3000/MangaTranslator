"""가로 텍스트가 박스 전체 폭을 기준으로 정렬되는지 검사한다.

실행: runtime/python.exe regression/check_text_alignment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.text.drawing_engine import _horizontal_line_start_x  # noqa: E402


def main() -> int:
    center_x = 100.0
    box_width = 160.0
    line_width = 80.0
    assert _horizontal_line_start_x(center_x, box_width, line_width, "left") == 20.0
    assert _horizontal_line_start_x(center_x, box_width, line_width, "center") == 60.0
    assert _horizontal_line_start_x(center_x, box_width, line_width, "right") == 100.0
    print("텍스트 박스 정렬 회귀 검사 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
