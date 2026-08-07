"""세로/가로 방향 판정의 회귀 검사.

실행:  runtime\\python.exe regression\\check_vertical_orientation.py

로직을 여기에 복제하지 않고 core.text.text_renderer 의 실제 판정 함수를 호출한다.
그래야 정책을 고쳤을 때 이 스크립트가 같이 따라오고, 거짓 통과가 생기지 않는다.

검사 대상은 "세로쓰기를 시도할지" 결정하는 관문이다.  관문이 막으면 결과는 반드시
가로다.  관문을 통과하면 그 뒤에 가로/세로 레이아웃을 실제로 계산해 더 나은 쪽을
고르므로, 세로가 확정되는 것은 아니다.  현재 조정 대상이 이 관문이라 여기서 끊는다.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: E402

from core.config import RenderingConfig  # noqa: E402
from core.pipeline import _apply_tag_style_overrides  # noqa: E402
from core.text.text_renderer import _should_try_auto_vertical_text  # noqa: E402
from ui.settings_manager import get_saved_settings  # noqa: E402
from ui.ui_models import UIConfigState, map_ui_to_backend_config  # noqa: E402

CASES_PATH = Path(__file__).with_name("vertical_orientation_cases.json")


def load_backend_config():
    """UI가 실제로 쓰는 설정을 실제 매핑 경로로 백엔드 config 까지 만든다.

    여기서 지름길로 JSON 을 직접 읽으면 안 된다.  한 번 그렇게 했다가
    ui_models 의 태그 스타일 매핑에 값이 빠진 버그를 이 검사가 놓쳤다.
    실제 파이프라인이 쓰는 경로를 그대로 타야 거짓 통과가 안 생긴다.
    """
    state = UIConfigState.from_dict(get_saved_settings())
    return map_ui_to_backend_config(
        ui_state=state,
        fonts_base_dir=REPO_ROOT / "fonts",
        target_device=torch.device("cpu"),
        is_batch=False,
    )


def config_for_tag(tag: str, backend_config) -> RenderingConfig:
    """파이프라인과 같은 방식으로 해당 태그의 렌더 설정을 만든다."""
    base = backend_config.rendering
    (
        _font_dir,
        _min_font,
        _max_font,
        _line_spacing,
        _vertical_line_spacing,
        tag_vertical_max_chars,
        tag_vertical_min_aspect,
    ) = _apply_tag_style_overrides(
        tag,
        backend_config,
        base.font_dir,
        base.min_font_size,
        base.max_font_size,
        base.line_spacing_mult,
        base.vertical_line_spacing_mult,
    )

    cfg = RenderingConfig()
    cfg.auto_vertical_text = True
    cfg.vertical_max_chars = (
        tag_vertical_max_chars
        if tag_vertical_max_chars is not None
        else base.vertical_max_chars
    )
    cfg.vertical_min_aspect_ratio = (
        tag_vertical_min_aspect
        if tag_vertical_min_aspect is not None
        else base.vertical_min_aspect_ratio
    )
    return cfg


def main() -> int:
    with open(CASES_PATH, encoding="utf-8") as fh:
        cases = json.load(fh)["cases"]
    backend_config = load_backend_config()

    header = f"{'텍스트':24}{'태그':>6}{'비율':>7}{'기대':>10}{'판정':>10}   결과"
    print(header)
    print("-" * len(header))

    failed = unknown = 0
    for case in cases:
        text, tag = case["text"], case["tag"]
        width, height = float(case["width"]), float(case["height"])
        expect = case.get("expect")

        tries_vertical = _should_try_auto_vertical_text(
            text, width, height, config_for_tag(tag, backend_config)
        )
        verdict = "vertical" if tries_vertical else "horizontal"

        if expect is None:
            unknown += 1
            mark = "- 미확인"
        elif expect == "vertical" and not tries_vertical:
            # 관문이 막았으므로 세로가 될 수 없다. 확실한 실패.
            failed += 1
            mark = "X 실패"
        elif expect == "horizontal" and tries_vertical:
            # 관문은 통과했다. 최종 결과는 레이아웃 비교가 정하므로 단정하지 않는다.
            failed += 1
            mark = "X 실패(세로 시도됨)"
        else:
            mark = "OK"

        ratio = height / width if width else 0.0
        print(f"{text[:22]:24}{tag:>6}{ratio:7.2f}{str(expect):>10}{verdict:>10}   {mark}")

    total = len(cases)
    print()
    print(f"전체 {total}건 · 실패 {failed}건 · 미확인 {unknown}건")
    if failed:
        print("\n실패가 있습니다. 정책을 바꿨다면 의도한 변화인지 확인하세요.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
