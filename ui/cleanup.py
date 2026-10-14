from pathlib import Path

import gradio as gr

from core.workspace_cleanup import (
    execute_cleanup,
    filtered_choices,
    output_records,
    prepare_cleanup,
)


def _size_label(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:,.1f} {unit}"
        size /= 1024


def build_cleanup_panel(kind, root_input=None):
    """Same selection and confirmation flow for editor jobs and output runs."""
    from ui import callbacks

    repo = Path(__file__).resolve().parents[1]
    title = "저장된 편집 작업 정리" if kind == "sessions" else "번역 결과물 정리"
    scope = (
        "편집 정보와 전용 복구 자산을 정리합니다. 원본과 번역 결과 이미지는 유지됩니다."
        if kind == "sessions"
        else "위 저장 폴더의 번역 결과를 작업별로 정리합니다. 선택한 결과 폴더 전체가 휴지통으로 이동합니다."
    )
    with gr.Accordion(title, open=False, elem_classes=["cleanup-panel"]) as panel:
        records = gr.State([])
        plan = gr.State(None)
        root = gr.State(str(repo / "output"))
        root_value = root_input or root
        gr.Markdown(scope, elem_classes=["cleanup-description"])
        period = gr.Radio(
            choices=[
                ("전체", 0),
                ("7일 이전", 7),
                ("30일 이전", 30),
                ("90일 이전", 90),
                ("1년 이전", 365),
                ("직접 선택", -1),
            ],
            value=30,
            label="수정일 기준",
            show_label=False,
            container=False,
            elem_classes=["cleanup-period"],
        )
        with gr.Row(visible=False, elem_classes=["cleanup-dates"]) as date_range:
            start_date = gr.DateTime(
                include_time=False,
                type="string",
                timezone="Asia/Seoul",
                label="시작일",
                scale=1,
                min_width=180,
            )
            end_date = gr.DateTime(
                include_time=False,
                type="string",
                timezone="Asia/Seoul",
                label="종료일 (당일 포함)",
                scale=1,
                min_width=180,
            )
        with gr.Row(elem_classes=["cleanup-filters"]):
            search = gr.Textbox(
                label="작업 검색",
                show_label=False,
                container=False,
                placeholder="작품·파일·날짜 검색",
                scale=3,
                min_width=160,
            )
            refresh = gr.Button("새로고침", scale=0, min_width=88)
        summary = gr.Markdown(
            "정리할 작업을 불러오세요.", elem_classes=["cleanup-summary"]
        )
        picker = gr.CheckboxGroup(
            choices=[],
            value=[],
            label="정리할 작업",
            interactive=True,
            elem_classes=["cleanup-picker"],
            container=False,
        )
        gr.Markdown(
            "체크박스 드래그로 선택·해제 · Shift로 범위 선택 · 드래그 중 휠 또는 가장자리로 스크롤",
            elem_classes=["cleanup-hint"],
        )
        with gr.Row(elem_classes=["cleanup-actions"]):
            select_all = gr.Button("목록 전체 선택", scale=0)
            deselect = gr.Button("선택 해제", scale=0)
            prepare = gr.Button("선택 정리", variant="primary", scale=0)
        with gr.Group(visible=False, elem_classes=["cleanup-confirm"]) as confirmation:
            detail = gr.Markdown()
            with gr.Row(elem_classes=["cleanup-actions"]):
                cancel = gr.Button("취소", scale=0)
                remove = gr.Button("휴지통으로 보내기", variant="stop", scale=0)

    def choices_for(items, days, query, start, end):
        return filtered_choices(items, days, query, start_date=start, end_date=end)

    filter_inputs = [period, search, start_date, end_date]

    def load(output_root, days, query, start, end):
        output_root = str((repo / str(output_root or "output")).resolve())
        try:
            if kind == "sessions":
                _choices, invalid, items = callbacks._discover_existing_editor_sessions(
                    repo, include_members=True
                )
                notice = f" · 읽을 수 없는 세션 {invalid}개 제외" if invalid else ""
            else:
                items, notice = output_records(Path(output_root)), ""
            visible = choices_for(items, days, query, start, end)
            return (
                items,
                output_root,
                gr.update(choices=visible, value=[]),
                None,
                gr.update(visible=False),
                f"작업 {len(visible)}개 표시 · 선택 0개{notice}",
            )
        except (OSError, ValueError) as exc:
            return (
                [],
                output_root,
                gr.update(choices=[], value=[]),
                None,
                gr.update(visible=False),
                f"목록을 불러오지 못했습니다: {exc}",
            )

    listing_outputs = [records, root, picker, plan, confirmation, summary]
    for trigger in (panel.expand, refresh.click):
        trigger(
            load,
            [root_value, *filter_inputs],
            listing_outputs,
            queue=False,
            show_progress="minimal",
        )

    def filter_list(items, days, query, start, end):
        try:
            choices = choices_for(items, days, query, start, end)
            message = f"작업 {len(choices)}개 표시 · 선택 0개"
        except ValueError as exc:
            choices, message = [], str(exc)
        return (
            gr.update(choices=choices, value=[]),
            None,
            gr.update(visible=False),
            message,
            gr.update(visible=days == -1),
        )

    for control in filter_inputs:
        control.change(
            filter_list,
            [records, *filter_inputs],
            [picker, plan, confirmation, summary, date_range],
            queue=False,
            show_progress="hidden",
        )

    def set_selection(values):
        return values, None, gr.update(visible=False), f"선택 {len(values)}개"

    def select_visible(items, days, query, start, end):
        try:
            return set_selection(
                [value for _label, value in choices_for(items, days, query, start, end)]
            )
        except ValueError as exc:
            return [], None, gr.update(visible=False), str(exc)

    select_all.click(
        select_visible,
        [records, *filter_inputs],
        [picker, plan, confirmation, summary],
        queue=False,
    )
    deselect.click(
        lambda: set_selection([]),
        outputs=[picker, plan, confirmation, summary],
        queue=False,
    )
    picker.input(
        lambda values: (None, gr.update(visible=False), f"선택 {len(values or [])}개"),
        picker,
        [plan, confirmation, summary],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    def review(items, selected, output_root):
        try:
            pending = prepare_cleanup(kind, items, selected, repo, output_root)
            names = "\n".join(f"- {item['label']}" for item in pending["records"][:5])
            more = (
                f"\n\n외 {pending['count'] - 5}개 작업" if pending["count"] > 5 else ""
            )
            text = (
                f"**작업 {pending['count']}개 · {_size_label(pending['bytes'])}**를 휴지통으로 보냅니다. "
                "필요하면 Windows 휴지통에서 복원할 수 있습니다.\n\n" + names + more
            )
            return (
                pending,
                gr.update(visible=True),
                text,
                "대상을 확인한 뒤 휴지통으로 보내 주세요.",
            )
        except (OSError, ValueError) as exc:
            return None, gr.update(visible=False), "", str(exc)

    prepare.click(
        review,
        [records, picker, root],
        [plan, confirmation, detail, summary],
        queue=False,
        show_progress="minimal",
    )
    cancel.click(
        lambda: (None, gr.update(visible=False)),
        outputs=[plan, confirmation],
        queue=False,
    )

    def remove_selected(pending, selected, output_root, days, query, start, end):
        try:
            if not pending or set(selected or []) != set(pending["selected"]):
                raise ValueError("선택이 바뀌었습니다. 선택 정리를 다시 눌러 주세요.")
            completed, errors = execute_cleanup(pending)
            result = list(load(output_root, days, query, start, end))
            result[-1] = (
                f"휴지통으로 {len(completed)}개 파일·폴더를 보냈습니다. "
                + result[-1]
                + ("\n\n" + "\n".join(errors) if errors else "")
            )
            return tuple(result)
        except (OSError, ValueError) as exc:
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                None,
                gr.update(visible=False),
                str(exc),
            )

    event = remove.click(
        remove_selected,
        [plan, picker, root, *filter_inputs],
        listing_outputs,
        queue=False,
        show_progress="minimal",
    )
    if root_input is not None:
        root_input.change(
            lambda: (
                [],
                gr.update(choices=[], value=[]),
                None,
                gr.update(visible=False),
                "저장 경로가 바뀌었습니다. 목록을 새로고침해 주세요.",
            ),
            outputs=[records, picker, plan, confirmation, summary],
            queue=False,
        )
    return event
