"""Fixed canvas-toolbar feedback, with a compact disclosure for narrow windows."""

from PyQt6.QtCore import QPoint, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class AiJobFeedback(QWidget):
    stopRequested = pyqtSignal()
    resultRequested = pyqtSignal()
    dismissed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AiJobFeedback")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(80)
        self.setMaximumWidth(370)
        self.setFixedHeight(30)
        self._title = self._detail = ""
        self._compact = False
        self._busy = False
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.compact_button = QPushButton("결과 보기")
        self.compact_button.setProperty("noAutoIcon", True)
        self.compact_button.setFixedSize(80, 30)
        self.compact_button.setAccessibleName("AI 작업 현황과 결과 펼치기")
        self.compact_button.clicked.connect(self._show_popup)
        self._layout.addWidget(self.compact_button)
        self.compact_button.hide()
        self.content = QWidget(self)
        self.content.setObjectName("AiJobContent")
        row = QHBoxLayout(self.content)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.dot = QWidget()
        self.dot.setFixedSize(6, 6)
        row.addWidget(self.dot)
        column = QWidget()
        column.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        labels = QVBoxLayout(column)
        labels.setContentsMargins(0, 0, 0, 0)
        labels.setSpacing(0)
        self.title_label = QLabel()
        self.title_label.setObjectName("AiJobTitle")
        self.detail_label = QLabel()
        self.detail_label.setObjectName("AiJobDetail")
        for label in (self.title_label, self.detail_label):
            label.setTextFormat(Qt.TextFormat.PlainText)
            labels.addWidget(label)
        row.addWidget(column, 1)
        self.stop_button = QPushButton("중단")
        self.stop_button.setProperty("noAutoIcon", True)
        self.stop_button.setFixedSize(48, 30)
        self.stop_button.setAccessibleName("진행 중인 AI 작업 중단")
        self.stop_button.clicked.connect(self.stopRequested)
        row.addWidget(self.stop_button)
        self.result_button = QPushButton("결과 보기")
        self.result_button.setObjectName("AiJobResult")
        self.result_button.setProperty("noAutoIcon", True)
        self.result_button.setFixedSize(84, 30)
        self.result_button.clicked.connect(self._open_result)
        row.addWidget(self.result_button)
        self.dismiss_button = QToolButton()
        self.dismiss_button.setObjectName("AiJobDismiss")
        self.dismiss_button.setFixedSize(24, 30)
        self.dismiss_button.setToolTip("작업 현황 닫기 — 준비된 결과는 유지됩니다.")
        self.dismiss_button.setAccessibleName("작업 현황 닫기")
        self.dismiss_button.clicked.connect(self.dismissed)
        row.addWidget(self.dismiss_button)
        self._layout.addWidget(self.content)
        self.popup = QFrame(self, Qt.WindowType.Popup)
        self.popup.setObjectName("AiJobPopup")
        self.popup.setFixedWidth(360)
        popup_layout = QVBoxLayout(self.popup)
        popup_layout.setContentsMargins(10, 8, 10, 8)
        self.setStyleSheet("""
            QWidget#AiJobContent { background: transparent; }
            QLabel#AiJobTitle { color: #e5e7eb; font-size: 11px; font-weight: 600; }
            QLabel#AiJobDetail { color: #c4cfdd; font-size: 10px; font-weight: 400; }
            QPushButton, QToolButton { color: #f8fafc; background: #364152;
                border: 1px solid #4b5a6f; border-radius: 5px; min-height: 0; padding: 0; }
            QPushButton:hover, QToolButton:hover { background: #475569; border-color: #93c5fd; }
            QPushButton:focus, QToolButton:focus { border-color: #93c5fd; }
            QPushButton:disabled { color: #a8b4c4; background: #303a49; border-color: #4b5a6f; }
            QPushButton#AiJobResult[ready="true"] { background: #2563eb; border-color: #60a5fa; }
            QPushButton#AiJobResult[ready="true"]:hover { background: #1d4ed8; }
            QFrame#AiJobPopup { background: #252d3a; border: 1px solid #4b5a6f; border-radius: 6px; }
        """)

    def set_state(
        self,
        *,
        title="",
        detail="",
        busy=False,
        cancelled=False,
        ready=False,
        result_text="결과 보기",
        tooltip="",
        state="idle",
        compact_text="",
    ):
        self._title, self._detail, self._busy = title, detail, busy
        self.setToolTip(tooltip or "\n".join(filter(None, (title, detail))))
        self.content.setToolTip(self.toolTip())
        self.dot.setVisible(bool(title))
        color = {"busy": "#60a5fa", "ready": "#4ade80", "error": "#fbbf24"}.get(
            state, "#a8b4c4"
        )
        self.dot.setStyleSheet(f"background: {color}; border-radius: 3px;")
        self.stop_button.setVisible(busy)
        self.stop_button.setEnabled(busy and not cancelled)
        self.stop_button.setToolTip(
            "중단한 요청은 결과를 적용하지 않습니다. 이미 보낸 요청은 서버에서 계속 처리될 수 있습니다."
        )
        self.result_button.setText(result_text)
        self.result_button.setEnabled(ready and not busy)
        self.result_button.setProperty("ready", ready and result_text == "비교·적용")
        self.result_button.style().unpolish(self.result_button)
        self.result_button.style().polish(self.result_button)
        self.dismiss_button.setVisible(bool(title) and not busy)
        self.compact_button.setText(compact_text or result_text)
        self.compact_button.setEnabled(bool(title) or ready)
        self.updateGeometry()
        QTimer.singleShot(0, self._elide)

    def sizeHint(self):
        return QSize(370 if self._title else 84, 30)

    def minimumSizeHint(self):
        return QSize(80, 30)

    def _open_result(self):
        self.popup.hide()
        self.resultRequested.emit()

    def _show_popup(self):
        if not self._title and self.result_button.isEnabled():
            self._open_result()
            return
        self.popup.adjustSize()
        screen = self.screen().availableGeometry()
        position = self.mapToGlobal(
            QPoint(self.width() - self.popup.width(), self.height() + 5)
        )
        position.setX(
            max(
                screen.left(),
                min(position.x(), screen.right() - self.popup.width() + 1),
            )
        )
        position.setY(
            max(
                screen.top(),
                min(position.y(), screen.bottom() - self.popup.height() + 1),
            )
        )
        self.popup.move(position)
        self._elide()
        self.popup.show()
        QTimer.singleShot(0, self._elide)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        compact = self.width() < 240
        if compact != self._compact:
            self._compact = compact
            self.popup.hide()
            if compact:
                self._layout.removeWidget(self.content)
                self.popup.layout().addWidget(self.content)
            else:
                self.popup.layout().removeWidget(self.content)
                self._layout.addWidget(self.content)
            self.content.show()
            self.compact_button.setVisible(compact)
        QTimer.singleShot(0, self._elide)

    def _elide(self):
        for label, value in (
            (self.title_label, self._title),
            (self.detail_label, self._detail),
        ):
            label.setText(
                label.fontMetrics().elidedText(
                    value, Qt.TextElideMode.ElideRight, max(0, label.width())
                )
            )
