---
version: 1
slug: "advanced-editor-launcher-py"
primary_target: "advanced_editor/launcher.py"
related_targets: ["advanced_editor/test_undo_preview_sync.py"]
---

# 고급 편집기 화면 브리프

- 범위/모드: `advanced_editor/launcher.py`의 Windows PyQt6 편집 화면, Operate.
- 사용자 작업: 번역 영역을 선택하고 텍스트·글꼴·OCR·마스크·인페인팅을 반복 조정하며 결과를 즉시 확인한다.
- 성공 조건: 실행 취소·다시 실행 후 별도 클릭 없이 최신 상태가 보이고, 편집 도구와 설정을 왼쪽 한 방향에서 찾을 수 있다.
- 보존 조건: 기존 기능, 단축키, 세션 형식, 출력 호환성을 유지하고 번역 파이프라인과 공용 코어는 건드리지 않는다.
- 선택한 방향: Tubik/Ribbet식 얇은 왼쪽 도구 레일 + 왼쪽 설정 패널 + 넓은 중앙 캔버스 + 최소 상단 작업 표시줄.
- 기억할 장면: 도구를 왼쪽에서 고르고 바로 옆 설정을 바꾸면 중앙 만화 페이지가 즉시 갱신된다.
- 다음 결정: 도구별 상황별 설정 묶음과 영역 목록의 접이식 표시 방식을 실제 사용 흐름으로 검증한다.
