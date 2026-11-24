"""Compact canvas-toolbar job summary and a scrollable per-job disclosure."""

import time

from PyQt6.QtCore import QPoint, QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)


class JobRow(QWidget):
    HEIGHT = 58

    def __init__(self, job_id, stop, show, parent=None):
        super().__init__(parent)
        self.setObjectName("AiJobRow")
        self.setFixedHeight(self.HEIGHT)
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

    def __init__(self, parent=None, *, chevron_icon=None):
        super().__init__(parent)
        self.setObjectName("AiJobFeedback")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(80)
        self.setMaximumWidth(346)
        self.setFixedHeight(30)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.summary_button = QPushButton()
        self.summary_button.setObjectName("AiJobSummary")
        self.summary_button.setProperty("noAutoIcon", True)
        self.summary_button.setMinimumWidth(80)
        self.summary_button.setFixedHeight(30)
        self.summary_button.clicked.connect(self._show_popup)
        layout.addWidget(self.summary_button, 1)
        summary_layout = QHBoxLayout(self.summary_button)
        summary_layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        summary_layout.setContentsMargins(8, 0, 7, 0)
        summary_layout.setSpacing(5)
        self.summary_title = QLabel("AI 작업")
        self.summary_title.setObjectName("AiJobSummaryTitle")
        summary_layout.addWidget(self.summary_title)
        summary_layout.addStretch()
        self.summary_stats = QWidget()
        stats_layout = QHBoxLayout(self.summary_stats)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(2)
        self.summary_badges = []
        for status, name in (("running", "진행"), ("queued", "대기"), ("ready", "결과")):
            badge = QLabel(f"{name} 0")
            badge.setObjectName("AiJobSummaryBadge")
            badge.setProperty("status", status)
            self.summary_badges.append((name, badge))
            stats_layout.addWidget(badge, 0, Qt.AlignmentFlag.AlignVCenter)
        summary_layout.addWidget(self.summary_stats)
        summary_layout.addStretch(0)
        chevron = QLabel()
        chevron.setFixedWidth(12)
        if chevron_icon is None:
            chevron_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowDown)
        chevron.setPixmap(chevron_icon.pixmap(12, 12))
        summary_layout.addWidget(chevron)
        for child in (self.summary_title, self.summary_stats, chevron):
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
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
            QPushButton#AiJobSummary { background: #273745; border: 1px solid #415363; border-radius: 4px; }
            QPushButton#AiJobSummary:hover { background: #344a5d; }
            QPushButton#AiJobSummary:focus { border-color: #a3cbf6; }
            QFrame#AiJobPopup { background: #252d3a; border: 1px solid #4b5a6f; }
            QScrollArea, QScrollArea > QWidget > QWidget { background: #252d3a; }
            QLabel { color: #e5e7eb; font-size: 12px; }
            QLabel#AiJobTitle { font-weight: 600; }
            QLabel#AiJobDetail { color: #c4cfdd; }
            QLabel#AiJobSummaryTitle { color: #f8fafc; font-size: 14px; font-weight: 600; }
            QLabel#AiJobSummaryBadge {
                background: #2b3544; color: #cbd5e1; border-radius: 4px;
                padding: 1px 3px; font-size: 13px; font-weight: 500;
            }
            QLabel#AiJobSummaryBadge[active="false"] { background: transparent; }
            QLabel#AiJobSummaryBadge[active="true"][status="running"] {
                background: #203e62; color: #bfdbfe;
            }
            QLabel#AiJobSummaryBadge[active="true"][status="queued"] {
                background: #4c3d23; color: #fde68a;
            }
            QLabel#AiJobSummaryBadge[active="true"][status="ready"] {
                background: #23463d; color: #bbf7d0;
            }
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
        for (name, badge), count in zip(self.summary_badges, (running, queued, completed)):
            text = f"{name} {count}"
            if badge.text() != text:
                badge.setText(text)
            active = count > 0
            if badge.property("active") != active:
                badge.setProperty("active", active)
                badge.style().unpolish(badge)
                badge.style().polish(badge)
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
        summary = f"AI 작업 · 진행 {running} · 대기 {queued} · 결과 {completed}"
        self.summary_button.setToolTip(summary)
        self.summary_button.setAccessibleName(f"{summary} · 상세 열기")
        self.selection_result_button.setEnabled(ready)
        self._fit_summary()

    def _open_selection_result(self):
        self.popup.hide()
        self.resultRequested.emit()

    def sizeHint(self):
        return QSize(self.maximumWidth(), 30)

    def minimumSizeHint(self):
        return QSize(80, 30)

    def _fit_summary(self):
        badges_width = self.summary_stats.sizeHint().width()
        summary_layout = self.summary_button.layout()
        margins = summary_layout.contentsMargins()
        summary_insets = margins.left() + margins.right() + 12 + 2 * summary_layout.spacing()
        toolbar_gap = self.layout().spacing()
        result_width = self.result_button.width() + toolbar_gap
        title_width = self.summary_title.sizeHint().width() + summary_layout.spacing()
        summary_width = badges_width + summary_insets + title_width
        show_result = self.width() >= summary_width + result_width
        self.result_button.setVisible(show_result)
        available = self.width() - (result_width if show_result else 0)
        show_stats = available >= badges_width + summary_insets
        show_title = available >= summary_width
        self.summary_stats.setVisible(show_stats)
        self.summary_title.setVisible(show_title or not show_stats)
        summary_layout.setStretch(3, int(show_stats and not show_title))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_summary()

    def _show_popup(self):
        self.scroll.setFixedHeight(min(260, max(1, len(self.rows)) * JobRow.HEIGHT + 2))
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
