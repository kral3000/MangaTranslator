"""Compact, non-modal before/after review of one translation suggestion."""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from advanced_editor.translation_reroll import REROLL_STYLES


class TranslationRerollPreview(QDialog):
    def __init__(self, draft, apply_callback, reroll_callback, parent=None):
        super().__init__(parent)
        self.setWindowTitle("재번역 · 번역안 비교")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(620, 300)
        self.setMinimumSize(440, 250)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        label = REROLL_STYLES[draft["style"]][0]
        note = QLabel(label + " · 적용하기 전까지 기존 문장은 유지됩니다.")
        note.setWordWrap(True)
        layout.addWidget(note)
        if draft.get("request"):
            request = QLabel(draft["request"])
            request.setWordWrap(True)
            layout.addWidget(request)
        row = QHBoxLayout()
        row.setSpacing(12)
        for title, value, read_only in (
            ("현재 번역", draft["current"], True),
            ("새 번역안", draft["text"], False),
        ):
            column = QWidget()
            content = QVBoxLayout(column)
            content.setContentsMargins(0, 0, 0, 0)
            content.setSpacing(6)
            content.addWidget(QLabel(title))
            edit = QTextEdit()
            edit.setAcceptRichText(False)
            edit.setPlainText(value)
            edit.setReadOnly(read_only)
            edit.setAccessibleName(title)
            content.addWidget(edit, 1)
            if not read_only:
                self.candidate_edit = edit
            row.addWidget(column, 1)
        layout.addLayout(row, 1)
        self.feedback = QLabel("새 번역안은 여기서 직접 다듬어도 됩니다.")
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)
        actions = QHBoxLayout()
        again = QPushButton("다시 제안")
        again.clicked.connect(lambda: self.close() if reroll_callback() else None)
        actions.addWidget(again)
        actions.addStretch()
        close = QPushButton("닫기")
        close.clicked.connect(self.close)
        actions.addWidget(close)
        apply = QPushButton("이 문장 적용")
        apply.setDefault(True)

        def apply_candidate():
            value = self.candidate_edit.toPlainText().strip()
            if not value:
                self.feedback.setText("적용할 문장이 비어 있습니다.")
            elif apply_callback(value):
                self.accept()
            else:
                self.feedback.setText("원문이나 현재 번역이 바뀌었습니다. 현재 영역에서 다시 제안받아 주세요.")
        apply.clicked.connect(apply_candidate)
        actions.addWidget(apply)
        layout.addLayout(actions)
