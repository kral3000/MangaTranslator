"""Compact canvas-toolbar job summary and a scrollable per-job disclosure."""

import time

from PyQt6.QtCore import QPoint, QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class JobRow(QWidget):
    def __init__(self, job_id, stop, show, parent=None):
        super().__init__(parent)
        self.setObjectName("AiJobRow")
        self.setFixedHeight(58)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)
        labels = QVBoxLayout()
        labels.setSpacing(2)
        self.title = QLabel()
        self.detail = QLabel()
        self.title.setObjectName("AiJobTitle")
        self.detail.setObjectName("AiJobDetail")
        for label in (self.title, self.detail):
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
            )
            labels.addWidget(label)
        layout.addLayout(labels, 1)
        self.action = QPushButton()
        self.action.setProperty("noAutoIcon", True)
        self.action.setFixedSize(84, 28)
        self._job_id, self._stop, self._show = job_id, stop, show
        self._pending = False
        self.action.clicked.connect(self._activate)
        layout.addWidget(self.action)

    def _activate(self):
        (self._stop if self._pending else self._show)(self._job_id)

    def update_job(self, job):
        self._job = job
        self._pending = job["state"] in {"queued", "running"}
        self.title.setText(job["target"])
        elapsed = int(
            (job.get("finished") or time.monotonic())
            - (job["started"] or job["submitted"])
        )
        stage = job.get("stage", "")
        detail = f"{stage} · {elapsed}초" if self._pending else stage
        self.detail.setText(
            self.detail.fontMetrics().elidedText(
                detail, Qt.TextElideMode.ElideRight, max(0, self.detail.width())
            )
        )
        self.setToolTip(job["target"] + "\n" + detail + "\n" + job["session_path"])
        colors = {
            "queued": "#c4cfdd",
            "running": "#93c5fd",
            "ready": "#86efac",
            "error": "#fcd34d",
        }
        self.detail.setStyleSheet("color: " + colors.get(job["state"], "#c4cfdd") + ";")
        if self._pending:
            self.action.setText("대기 취소" if job["state"] == "queued" else "중단")
            self.action.setEnabled(not job["cancel"].is_set())
        else:
            self.action.setText("비교·적용" if job["state"] == "ready" else "결과 보기")
            self.action.setEnabled(
                bool(
                    job.get("draft")
                    or (job.get("result") or {}).get("failure_preview")
                    or job["state"] == "error"
                )
            )
        self.action.setAccessibleName(job["target"] + " " + self.action.text())


class AiJobFeedback(QWidget):
    jobStopRequested = pyqtSignal(str)
    jobResultRequested = pyqtSignal(str)
    concurrencyChanged = pyqtSignal(int)
    historyRequested = pyqtSignal()
    clearRequested = pyqtSignal()
    resultRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AiJobFeedback")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(80)
        self.setMaximumWidth(280)
        self.setFixedHeight(30)
        self._summary = "AI 작업"
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.summary_button = QPushButton(self._summary)
        self.summary_button.setProperty("noAutoIcon", True)
        self.summary_button.setMinimumWidth(80)
        self.summary_button.setFixedHeight(30)
        self.summary_button.setAccessibleName("AI 작업 현황과 동시 실행 설정 펼치기")
        self.summary_button.clicked.connect(self._show_popup)
        layout.addWidget(self.summary_button, 1)
        self.result_button = QPushButton("결과 보기")
        self.result_button.setProperty("noAutoIcon", True)
        self.result_button.setFixedSize(84, 30)
        self.result_button.clicked.connect(self.resultRequested)
        layout.addWidget(self.result_button)
        self.popup = QFrame(self, Qt.WindowType.Popup)
        self.popup.setObjectName("AiJobPopup")
        self.popup.setFixedWidth(420)
        content = QVBoxLayout(self.popup)
        content.setContentsMargins(10, 10, 10, 8)
        content.setSpacing(8)
        header = QHBoxLayout()
        title = QLabel("AI 작업")
        title.setObjectName("AiJobTitle")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(QLabel("동시 식자"))
        self.concurrency_spin = QSpinBox()
        self.concurrency_spin.setRange(1, 16)
        self.concurrency_spin.setValue(2)
        self.concurrency_spin.setFixedSize(56, 28)
        self.concurrency_spin.setAccessibleName("AI 식자 동시 실행 수")
        self.concurrency_spin.setToolTip(
            "식자 요청을 동시에 보낼 개수입니다. 재번역은 별도로 실행됩니다.\n개수를 낮춰도 진행 중인 요청은 중단하지 않습니다."
        )
        self.concurrency_spin.valueChanged.connect(self.concurrencyChanged)
        header.addWidget(self.concurrency_spin)
        content.addLayout(header)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setFixedHeight(260)
        self.list = QWidget()
        self.rows_layout = QVBoxLayout(self.list)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(0)
        self.rows_layout.addStretch(1)
        self.scroll.setWidget(self.list)
        content.addWidget(self.scroll)
        self.empty = QLabel("AI 식자·재번역을 요청하면 여기에 표시됩니다.")
        self.empty.setWordWrap(True)
        self.rows_layout.insertWidget(0, self.empty)
        footer = QHBoxLayout()
        self.counts_label = QLabel("진행 중인 작업 없음")
        self.counts_label.setObjectName("AiJobDetail")
        footer.addWidget(self.counts_label, 1)
        self.selection_result_button = QPushButton("선택 영역 결과")
        self.selection_result_button.setProperty("noAutoIcon", True)
        self.selection_result_button.clicked.connect(self._open_selection_result)
        footer.addWidget(self.selection_result_button)
        self.clear_button = QPushButton("완료·오류 정리")
        self.clear_button.setProperty("noAutoIcon", True)
        self.clear_button.setToolTip(
            "적용한 결과·오류·중단 항목을 목록에서 지웁니다. 미적용 결과는 유지됩니다."
        )
        self.clear_button.clicked.connect(self.clearRequested)
        footer.addWidget(self.clear_button)
        content.addLayout(footer)
        history = QPushButton("식자 기록 보기")
        history.clicked.connect(self.historyRequested)
        content.addWidget(history)
        self.rows = {}
        self.setStyleSheet("""
            QPushButton, QSpinBox { color: #f8fafc; background: #364152;
                border: 1px solid #4b5a6f; border-radius: 5px; min-height: 0; padding: 3px 5px; }
            QPushButton:hover { background: #475569; border-color: #93c5fd; }
            QPushButton:focus, QSpinBox:focus { border-color: #93c5fd; }
            QPushButton:disabled { color: #a8b4c4; background: #303a49; }
            QFrame#AiJobPopup { background: #252d3a; border: 1px solid #4b5a6f; }
            QScrollArea, QScrollArea > QWidget > QWidget { background: #252d3a; }
            QLabel { color: #e5e7eb; font-size: 11px; }
            QLabel#AiJobTitle { font-weight: 600; }
            QLabel#AiJobDetail { color: #c4cfdd; }
            QWidget#AiJobRow { border-bottom: 1px solid #3a4657; }
        """)

    def set_jobs(self, jobs, concurrency, *, ready=False):
        self.concurrency_spin.blockSignals(True)
        self.concurrency_spin.setValue(concurrency)
        self.concurrency_spin.blockSignals(False)
        for job_id in list(self.rows):
            if job_id not in jobs:
                row = self.rows.pop(job_id)
                self.rows_layout.removeWidget(row)
                row.deleteLater()
        for job_id, job in jobs.items():
            if job_id not in self.rows:
                row = JobRow(
                    job_id, self.jobStopRequested.emit, self.jobResultRequested.emit
                )
                self.rows[job_id] = row
                self.rows_layout.insertWidget(0, row)
            self.rows[job_id].update_job(job)
        self.empty.setVisible(not jobs)
        running = sum(job["state"] == "running" for job in jobs.values())
        queued = sum(job["state"] == "queued" for job in jobs.values())
        completed = sum(job["state"] in {"ready", "stale"} for job in jobs.values())
        self._summary = f"AI 작업 · 진행 {running} · 대기 {queued} · 결과 {completed}"
        self.counts_label.setText(
            f"진행 {running} · 대기 {queued} · 미적용 {completed}"
        )
        self.clear_button.setEnabled(
            any(
                job["state"] in {"applied", "error", "cancelled"}
                for job in jobs.values()
            )
        )
        self.result_button.setEnabled(ready)
        self.summary_button.setToolTip(self._summary)
        self.selection_result_button.setEnabled(ready)
        self._elide()

    def _open_selection_result(self):
        self.popup.hide()
        self.resultRequested.emit()

    def sizeHint(self):
        return QSize(280, 30)

    def minimumSizeHint(self):
        return QSize(80, 30)

    def _elide(self):
        self.summary_button.setText(
            self.summary_button.fontMetrics().elidedText(
                self._summary,
                Qt.TextElideMode.ElideRight,
                max(20, self.summary_button.width() - 18),
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.result_button.setVisible(self.width() >= 190)
        self._elide()

    def _show_popup(self):
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
        self.popup.show()
        self.popup.layout().activate()
        for row in self.rows.values():
            row.update_job(row._job)
