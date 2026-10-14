from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class SfxPreviewDialog(QDialog):
    """Non-blocking comparison; applying is an explicit, separate action."""

    def __init__(self, patch, apply_callback, parent=None):
        super().__init__(parent)
        failed = bool(patch.get("failed"))
        self.setWindowTitle(
            "AI 효과음 식자 · 실패 결과 확인"
            if failed
            else "AI 효과음 식자 · 원본 비교"
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(900, 640)
        self.setMinimumSize(620, 420)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        note = QLabel(
            "적용하지 않은 생성 결과입니다. 비율과 구도를 확인할 수 있으며 페이지는 바뀌지 않습니다."
            if failed
            else "글자와 그림 경계를 비교한 뒤 적용해 주세요. 적용 전에는 현재 페이지가 바뀌지 않습니다."
            if apply_callback is not None
            else "저장된 식자 결과를 원본과 비교합니다."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QHBoxLayout()
        row.setSpacing(16)
        self.images = []
        for title, field in (
            ("원본", "ai_original_path"),
            (
                "생성 원본 · 적용 안 됨"
                if failed
                else "적용할 결과"
                if apply_callback is not None
                else "현재 식자",
                "ai_processed_path",
            ),
        ):
            column = QWidget()
            content = QVBoxLayout(column)
            content.setContentsMargins(0, 0, 0, 0)
            content.setSpacing(8)
            content.addWidget(QLabel(title))
            image = QLabel()
            image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image.setMinimumSize(1, 1)
            image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            image.setStyleSheet("background: #202225; border-radius: 6px;")
            content.addWidget(image, 1)
            pixmap = QPixmap(patch[field])
            self.images.append((image, pixmap))
            row.addWidget(column, 1)
        layout.addLayout(row, 1)
        warning = QLabel(
            patch.get("warning")
            or (
                "적용 후에는 고정된 이미지 영역이 됩니다. ‘식자 표시’를 끄면 원본을 표시합니다."
                if apply_callback is not None
                else "‘식자 표시’를 끄면 원본이 표시됩니다."
            )
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)
        actions = QHBoxLayout()
        actions.addStretch()
        if failed:
            folder = QPushButton("결과 폴더 열기")
            folder.clicked.connect(
                lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(patch["directory"]))
            )
            actions.addWidget(folder)
        close = QPushButton("닫기")
        close.clicked.connect(self.close)
        actions.addWidget(close)
        apply = QPushButton("이 결과 적용")
        if apply_callback is None:
            apply.hide()
        else:
            apply.setDefault(True)
            apply.clicked.connect(lambda: self.accept() if apply_callback() else None)
        actions.addWidget(apply)
        layout.addLayout(actions)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_images()

    def showEvent(self, event):
        super().showEvent(event)
        self._fit_images()

    def _fit_images(self):
        for label, pixmap in self.images:
            label.setPixmap(
                pixmap.scaled(
                    label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
