"""Native session, lettering and output storage browser."""

import os
import threading
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QDate, QObject, QRect, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QPainter
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDateEdit,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from advanced_editor.storage_inventory import inventory
from advanced_editor.storage_preview import StoragePreview
from core.workspace_cleanup import execute_cleanup, filtered_choices, prepare_cleanup

REPO = Path(__file__).resolve().parents[1]
TITLES = {
    "sessions": "편집 작업",
    "history": "AI 식자 기록",
    "outputs": "출력·다운로드",
    "maintenance": "캐시·로그 정리",
}


def size_text(value):
    if value is None:
        return "…"
    if value >= 1024**3:
        return f"{value / 1024**3:.2f} GiB"
    if value >= 1024**2:
        return f"{value / 1024**2:.1f} MiB"
    return f"{value / 1024:.0f} KiB"


class Signals(QObject):
    event = pyqtSignal(object)


class EditorJobTitleDelegate(QStyledItemDelegate):
    STATUS_ROLE = Qt.ItemDataRole.UserRole + 2

    def paint(self, painter, option, index):
        badge = index.data(self.STATUS_ROLE)
        if not badge:
            super().paint(painter, option, index)
            return
        row = QStyleOptionViewItem(option)
        self.initStyleOption(row, index)
        title = row.text
        row.text = ""
        style = row.widget.style() if row.widget else QApplication.style()
        style.drawControl(
            QStyle.ControlElement.CE_ItemViewItem, row, painter, row.widget
        )
        badge_width = row.fontMetrics.horizontalAdvance(badge) + 14
        pill = QRect(row.rect.left() + 7, row.rect.center().y() - 10, badge_width, 21)
        label = row.rect.adjusted(badge_width + 15, 0, -6, 0)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(row.font)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#dce9fc"))
        painter.drawRoundedRect(pill, 5, 5)
        painter.setPen(QColor("#1e40af"))
        painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, badge)
        painter.setPen(
            QColor(
                "#1e3a8a" if row.state & QStyle.StateFlag.State_Selected else "#172033"
            )
        )
        elided = row.fontMetrics.elidedText(
            title, Qt.TextElideMode.ElideRight, max(0, label.width())
        )
        painter.drawText(
            label, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, elided
        )
        painter.restore()


class CheckTree(QTreeWidget):
    """Paint checkmarks in the first column; use native row range selection elsewhere."""

    def __init__(self):
        super().__init__()
        self.drag_state = None
        self.last_row = None
        self.scroll_timer = QTimer(self, interval=38)
        self.scroll_timer.timeout.connect(self._scroll)

    def mousePressEvent(self, event):
        point = event.position().toPoint()
        row = self.itemAt(point)
        if (
            row
            and row.data(0, Qt.ItemDataRole.UserRole) is not None
            and self.columnAt(point.x()) == 0
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self.setCurrentItem(row)
            self.drag_state = (
                Qt.CheckState.Unchecked
                if row.checkState(0) == Qt.CheckState.Checked
                else Qt.CheckState.Checked
            )
            self.last_row = row
            self.drag_y = point.y()
            row.setCheckState(0, self.drag_state)
            self.scroll_timer.start()
            return
        super().mousePressEvent(event)

    def _extend(self, y):
        row = self.itemAt(10, min(max(0, y), self.viewport().height() - 1))
        if row is None or self.last_row is None or row is self.last_row:
            return
        forward = (
            self.visualItemRect(row).top() > self.visualItemRect(self.last_row).top()
        )
        step = self.itemBelow if forward else self.itemAbove
        current = self.last_row
        self.blockSignals(True)
        try:
            while current is not row:
                current = step(current)
                if current is None:
                    break
                if current.data(0, Qt.ItemDataRole.UserRole) is not None:
                    current.setCheckState(0, self.drag_state)
        finally:
            self.blockSignals(False)
        self.last_row = row
        self.itemChanged.emit(row, 0)

    def _scroll(self):
        bar = self.verticalScrollBar()
        height = self.viewport().height()
        if self.drag_y < 20 or self.drag_y > height - 20:
            bar.setValue(bar.value() + (-1 if self.drag_y < 20 else 1))
            self._extend(self.drag_y)

    def mouseMoveEvent(self, event):
        if self.drag_state is not None:
            self.drag_y = event.position().toPoint().y()
            self._extend(self.drag_y)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drag_state is not None:
            self.scroll_timer.stop()
            self.drag_state = None
        else:
            super().mouseReleaseEvent(event)


class StorageDialog(QDialog):
    def __init__(
        self,
        parent=None,
        *,
        editor=None,
        history=False,
        repo=REPO,
        output=None,
        jobs=None,
        kind="sessions",
        scope=None,
        context_title="",
    ):
        super().__init__(parent)
        self.repo = Path(repo).resolve()
        if output is None:
            import json

            from ui.settings_path import CONFIG_FILE

            try:
                output = json.loads(CONFIG_FILE.read_text(encoding="utf-8")).get(
                    "output_directory"
                )
            except (OSError, ValueError):
                pass
        self.output = (
            self.repo / Path(os.path.expandvars(str(output or "output"))).expanduser()
        ).resolve()
        self.kind = "history" if history else kind
        if scope is None:
            scope = getattr(editor, "related_session_paths", None) or []
            if not scope and getattr(editor, "session_path", None):
                scope = [editor.session_path]
        self.scope = {Path(p) for p in scope}
        self.partial = False
        self.jobs = jobs
        self.storage_changed = False
        self.catalog = None
        self.groups = {}
        self.open_path = self.draft = None
        self.rows = {}
        self.items = {}
        self.request = 0
        self.cancel = threading.Event()
        self.signals = Signals()
        self.signals.event.connect(self._receive)
        self.setWindowTitle(
            "작업 열기" if self.kind == "sessions" else TITLES[self.kind]
        )
        self.resize(1110, 635)
        self.setMinimumSize(760, 440)
        self.setObjectName("WorkLibrary")
        stylesheet = Path(__file__).with_name("work_library.qss")
        self.setStyleSheet(stylesheet.read_text(encoding="utf-8").replace(
            "@chevron", stylesheet.with_name("library_chevron.svg").as_posix()
        ))
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        bar = QWidget()
        bar.setObjectName("LibraryBar")
        toolbar = QHBoxLayout(bar)
        toolbar.setContentsMargins(16, 9, 16, 9)
        title = QLabel(TITLES[self.kind])
        title.setObjectName("LibraryTitle")
        toolbar.addWidget(title)
        if self.kind == "history" and context_title:
            context = self.scope_title = QLabel(context_title)
            context.setTextFormat(Qt.TextFormat.PlainText)
            context.setObjectName("ContextTitle")
            context.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
            )
            context.setToolTip(context_title)
            toolbar.addWidget(context, 1)
        else:
            toolbar.addStretch()
        if self.kind == "sessions":
            self.storage_button = QToolButton()
            self.storage_button.setText("저장 공간 관리")
            self.storage_button.setAccessibleName("저장 공간 관리")
            self.storage_button.setToolTip("출력·다운로드 열기 · 화살표로 정리 메뉴 보기")
            self.storage_button.setPopupMode(
                QToolButton.ToolButtonPopupMode.MenuButtonPopup
            )
            menu = QMenu(self.storage_button)
            for key in ("outputs", "maintenance"):
                menu.addAction(
                    TITLES[key], lambda _=False, k=key: self._open_section(k)
                )
            self.storage_button.setMenu(menu)
            self.storage_button.clicked.connect(lambda: self._open_section("outputs"))
            toolbar.addWidget(self.storage_button)
        outer.addWidget(bar)
        body = QWidget()
        outer.addWidget(body, 1)
        right = QVBoxLayout(body)
        right.setContentsMargins(16, 14, 16, 12)
        right.setSpacing(10)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(1)
        self.splitter.setChildrenCollapsible(False)
        right.addWidget(self.splitter, 1)
        left = QWidget()
        left.setMinimumWidth(310)
        listing = QVBoxLayout(left)
        listing.setContentsMargins(0, 0, 8, 0)
        listing.setSpacing(8)
        self.splitter.addWidget(left)
        filters = QHBoxLayout()
        self.search = QLineEdit(placeholderText="작업·페이지·상태 검색")
        self.search.setClearButtonEnabled(True)
        filters.setSpacing(6)
        self.search.setAccessibleName("목록 검색")
        filters.addWidget(self.search, 1)
        self.scope_box = QComboBox()
        self.scope_box.addItems(["전체 작업", "이 작업"])
        self.scope_box.setCurrentIndex(int(self.kind == "history" and bool(self.scope)))
        self.scope_box.setVisible(self.kind == "history" and bool(self.scope))
        filters.addWidget(self.scope_box)
        self.period = QComboBox()
        self.period.addItems(
            ["전체 기간", "7일 이전", "30일 이전", "90일 이전", "기간 지정"]
        )
        filters.addWidget(self.period)
        refresh = QPushButton("새로고침")
        refresh.clicked.connect(lambda: self._load(force=True))
        filters.addWidget(refresh)
        listing.addLayout(filters)
        self.dates = QWidget()
        dates = QHBoxLayout(self.dates)
        dates.setContentsMargins(0, 0, 0, 0)
        self.start = QDateEdit(QDate.currentDate().addMonths(-1), calendarPopup=True)
        self.end = QDateEdit(QDate.currentDate(), calendarPopup=True)
        for widget in (QLabel("수정일"), self.start, QLabel("~"), self.end):
            dates.addWidget(widget)
        dates.addStretch()
        self.dates.hide()
        listing.addWidget(self.dates)
        self.tree = CheckTree()
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(False)
        self.tree.setAccessibleName(TITLES[self.kind] + " 목록")
        self.tree.setHeaderLabels(["", "작업", "상태", "크기", "수정일"])
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for i, w in ((0, 32), (2, 85), (3, 80), (4, 136)):
            self.tree.setColumnWidth(i, w)
        self.tree.setIndentation(12)
        if self.kind == "sessions":
            self.tree.setItemDelegateForColumn(1, EditorJobTitleDelegate(self.tree))
        self.tree.headerItem().setText(
            2, "페이지" if self.kind == "sessions" else "상태"
        )
        self.tree.setColumnWidth(2, 66 if self.kind == "sessions" else 90)
        self.tree.setColumnWidth(4, 119)
        listing.addWidget(self.tree, 1)
        inspection = QWidget()
        pane = QVBoxLayout(inspection)
        pane.setContentsMargins(8, 0, 0, 0)
        pane.setSpacing(8)
        preview_header = QHBoxLayout()
        self.selection_title = QLabel("페이지 미리보기")
        self.selection_title.setTextFormat(Qt.TextFormat.PlainText)
        self.selection_title.setObjectName("SelectionTitle")
        self.selection_title.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        preview_header.addWidget(self.selection_title, 1)
        self.history_button = QPushButton("AI 식자 기록")
        self.history_button.clicked.connect(self._open_history)
        self.history_button.setVisible(self.kind == "sessions")
        preview_header.addWidget(self.history_button)
        pane.addLayout(preview_header)
        self.preview = StoragePreview()
        pane.addWidget(self.preview, 1)
        self.splitter.addWidget(inspection)
        inspection.setVisible(self.kind != "maintenance")
        self.splitter.setSizes([475, 575] if self.kind == "sessions" else [540, 510])
        footer = QHBoxLayout()
        self.status = QLabel()
        self.status.setObjectName("LibraryStatus")
        self.select_mode = QPushButton("선택 정리")
        self.select_mode.setCheckable(True)
        self.select_mode.setChecked(self.kind != "sessions")
        self.select_mode.setVisible(self.kind == "sessions")
        footer.addWidget(self.select_mode)
        footer.addWidget(self.status, 1)
        self.open = QPushButton("열기")
        self.trash = QPushButton("휴지통으로")
        close = QPushButton("닫기")
        self.open.setObjectName("Primary")
        self.trash.setObjectName("Trash")
        for btn in (self.trash, close, self.open):
            footer.addWidget(btn)
        right.addLayout(footer)
        self.open.clicked.connect(self._open)
        self.trash.clicked.connect(self._review)
        close.clicked.connect(self.reject)
        self.tree.itemActivated.connect(
            lambda item, col: (
                self._open(self.rows[item.data(0, Qt.ItemDataRole.UserRole)])
                if (col != 0 or self.tree.isColumnHidden(0))
                and item.data(0, Qt.ItemDataRole.UserRole) in self.rows
                else None
            )
        )
        self.tree.itemChanged.connect(self._summary)
        self.tree.itemSelectionChanged.connect(self._summary)
        self.tree.currentItemChanged.connect(self._preview_selection)
        self.search.textChanged.connect(self._filter)
        self.period.currentIndexChanged.connect(self._filter)
        self.start.dateChanged.connect(self._filter)
        self.end.dateChanged.connect(self._filter)
        self.select_mode.toggled.connect(self._selection_mode)
        self.search.returnPressed.connect(self._open)
        self.open.setAutoDefault(False)
        self.open.setDefault(False)
        self.scope_box.currentIndexChanged.connect(self._load)
        self._selection_mode(self.select_mode.isChecked())
        self.search.setFocus()
        self._load()

    def _selection_mode(self, enabled):
        if enabled and self.kind == "sessions" and self.catalog is not None:
            self.catalog.request_sizes()
        self.tree.setColumnHidden(0, not enabled)
        self.tree.setColumnHidden(3, not enabled)
        self.trash.setVisible(enabled)
        if not enabled:
            self.tree.blockSignals(True)
            for item in self.items.values():
                item.setCheckState(0, Qt.CheckState.Unchecked)
            self.tree.blockSignals(False)
        self._summary()

    def _open_history(self):
        row = self.preview.row
        if self.partial:
            return
        if row and row["paths"]:
            self._open_section("history", scope=row["paths"], title=row["title"])
        elif row is None:
            self._open_section("history")

    def _open_section(self, kind, *, scope=None, title=""):
        dialog = StorageDialog(
            self,
            kind=kind,
            scope=scope,
            context_title=title,
            repo=self.repo,
            output=self.output,
        )
        dialog.exec()
        if dialog.draft:
            self.draft, self.open_path = dialog.draft, dialog.open_path
            self.accept()
        elif dialog.storage_changed:
            self._load(force=True)
        dialog.deleteLater()

    def done(self, result):
        self.preview.select(None)
        if self.catalog is not None:
            self.catalog.updated.disconnect(self._catalog_update)
            self.catalog = None
        self.cancel.set()
        self.request += 1
        super().done(result)

    def _worker(self, fn):
        token, signals, cancel = self.request, self.signals, self.cancel

        def publish(event, value):
            if not cancel.is_set():
                signals.event.emit((token, event, value))

        def run():
            try:
                fn(publish, cancel)
            except Exception as exc:
                publish("error", str(exc))

        threading.Thread(target=run, daemon=True).start()

    def _load(self, *_, force=False):
        self.cancel.set()
        self.cancel = threading.Event()
        self.request += 1
        kind = self.kind
        scope = (
            tuple(sorted(self.scope))
            if kind == "history" and self.scope_box.currentIndex()
            else ()
        )
        catalog = inventory(self.repo, self.output, kind, scope)
        if hasattr(self, "scope_title"):
            self.scope_title.setVisible(bool(scope))
        if catalog is not self.catalog:
            same_view = self.catalog is not None and scope == self.catalog.scope
            if self.catalog is not None:
                self.catalog.updated.disconnect(self._catalog_update)
            self.catalog = catalog
            self.partial = catalog.partial
            if catalog.rows is not None or not same_view:
                self._populate(catalog.rows or [])
            catalog.updated.connect(self._catalog_update)
        if catalog.rows is None:
            self.status.setText("작업을 찾는 중…")
        if force:
            self.status.setText("목록을 갱신하는 중…")
        catalog.load(
            self.jobs if kind == "sessions" and not force else None, force=force
        )
        if kind == "sessions" and self.select_mode.isChecked():
            catalog.request_sizes()
        self.jobs = None

    def _catalog_update(self, mode, value):
        if self.sender() is not self.catalog:
            return
        if mode in {"rows", "recent"}:
            self.partial = mode == "recent"
            self._populate(value)
        elif mode == "sizes":
            self.tree.blockSignals(True)
            for key, size in value.items():
                if key in self.items:
                    self.items[key].setText(3, size_text(size))
            self.tree.blockSignals(False)
        elif mode == "error":
            self.status.setText(value)
            self.status.setToolTip(value)

    def _populate(self, rows):
        current = self.tree.currentItem()
        selected = current.data(0, Qt.ItemDataRole.UserRole) if current else None
        previous = self.rows.get(selected, {})
        checks = {
            key
            for key, item in self.items.items()
            if item.checkState(0) == Qt.CheckState.Checked
        }
        expanded = {key for key, group in self.groups.items() if group.isExpanded()}
        history = self.kind == "history"
        self.tree.blockSignals(True)
        try:
            self.tree.clear()
            self.tree.last_row = self.tree.drag_state = None
            self.tree.scroll_timer.stop()
            self.tree.setRootIsDecorated(history)
            self.tree.setColumnWidth(0, 50 if history else 32)
            self.groups, self.items, self.rows = {}, {}, {}
            for row in rows:
                key = row["id"]
                row.setdefault("label", row["title"] + " " + row.get("state", ""))
                self.rows[key] = row
                parent = self.tree
                if history:
                    gid = row["group_id"]
                    if gid not in self.groups:
                        group = QTreeWidgetItem(self.tree, ["", row["group_title"]])
                        font = group.font(1)
                        font.setBold(True)
                        group.setFont(1, font)
                        group.setToolTip(1, row["group_title"])
                        self.groups[gid] = group
                    parent = self.groups[gid]
                item = QTreeWidgetItem(
                    parent,
                    [
                        "",
                        row.get("detail", row["title"]),
                        (
                            f"{row['page_count']} / {row['total_pages']}"
                            if row.get("translating")
                            else row["state"]
                        ),
                        size_text(self.catalog.sizes.get(key, row.get("size"))),
                        datetime.fromtimestamp(row["modified"]).strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                    ],
                )
                item.setCheckState(
                    0,
                    Qt.CheckState.Checked if key in checks else Qt.CheckState.Unchecked,
                )
                item.setData(0, Qt.ItemDataRole.UserRole, key)
                item.setToolTip(1, row["label"])
                if row.get("translating"):
                    item.setData(1, EditorJobTitleDelegate.STATUS_ROLE, "번역 중")
                    item.setData(
                        1,
                        Qt.ItemDataRole.AccessibleTextRole,
                        "번역 중 · " + row["title"],
                    )
                self.items[key] = item
            if "" in self.groups:
                unknown = self.groups[""]
                self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(unknown))
                self.tree.addTopLevelItem(unknown)
            for key, group in self.groups.items():
                group.setText(4, group.child(0).text(4))
                group.setText(2, f"{group.childCount()}건")
                group.setExpanded(key in expanded or len(self.groups) == 1)
            self._filter()
            current = self.items.get(selected)
            if current is None and previous.get("group_key"):
                current = next(
                    (
                        self.items[k]
                        for k, r in self.rows.items()
                        if r.get("group_key") == previous["group_key"]
                    ),
                    None,
                )
            if current is None or current.isHidden():
                current = next(
                    (
                        self.tree.topLevelItem(i)
                        for i in range(self.tree.topLevelItemCount())
                        if not self.tree.topLevelItem(i).isHidden()
                    ),
                    None,
                )
            if current:
                self.tree.setCurrentItem(current)
        finally:
            self.tree.blockSignals(False)
        self._summary()
        self._preview_selection()

    def _preview_selection(self, *_):
        item = self.tree.currentItem()
        if item and not item.isHidden() and item.childCount():
            item = next(
                (
                    item.child(i)
                    for i in range(item.childCount())
                    if not item.child(i).isHidden()
                ),
                None,
            )
        key = (
            item.data(0, Qt.ItemDataRole.UserRole)
            if item and not item.isHidden()
            else None
        )
        row = self.rows.get(key)
        self.preview.select(row if self.kind != "maintenance" else None)
        title = row.get("title") if row else ""
        self.selection_title.setText(title or "미리보기")
        self.selection_title.setToolTip(title or "")
        self.history_button.setText("AI 식자 기록" if row else "전체 식자 기록")
        self.history_button.setEnabled(
            not self.partial and (row is None or bool(row["paths"]))
        )

    def _receive(self, event):
        token, mode, value = event
        if token != self.request:
            return
        if mode == "plan":
            self.setEnabled(True)
            total = sum(plan["bytes"] for plan in value)
            count = sum(plan["count"] for plan in value)
            if (
                QMessageBox.question(
                    self,
                    "선택한 항목 정리",
                    f"선택한 {count}개 · {size_text(total)}를 휴지통으로 보낼까요?",
                )
                == QMessageBox.StandardButton.Yes
            ):
                self.setEnabled(False)
                self.status.setText("휴지통으로 보내는 중…")

                def remove(publish, cancel):
                    errors = []
                    for plan in value:
                        _, failed = execute_cleanup(plan)
                        errors.extend(failed)
                    publish("finished", errors)

                self._worker(remove)
            else:
                self._summary()
        elif mode == "finished":
            self.setEnabled(True)
            if value:
                QMessageBox.warning(self, "일부 항목 정리 실패", "\n".join(value))
            self.storage_changed = True
            inventory.cache_clear()
            self._load(force=True)
        elif mode == "error":
            self.setEnabled(True)
            self.status.setText(value)
            self.status.setToolTip(value)

    def _filter(self, *_):
        days = [0, 7, 30, 90, -1][self.period.currentIndex()]
        self.dates.setVisible(days == -1)
        try:
            visible = {
                key
                for _, key in filtered_choices(
                    self.rows.values(),
                    days,
                    self.search.text(),
                    start_date=self.start.date().toString("yyyy-MM-dd"),
                    end_date=self.end.date().toString("yyyy-MM-dd"),
                )
            }
        except ValueError as exc:
            self.status.setText(str(exc))
            self.trash.setEnabled(False)
            return
        for key, item in self.items.items():
            item.setHidden(key not in visible)
        for group in self.groups.values():
            count = sum(
                not group.child(i).isHidden() for i in range(group.childCount())
            )
            group.setHidden(count == 0)
            group.setText(2, f"{count}건")
            if count and self.search.text().strip():
                group.setExpanded(True)
        current = self.tree.currentItem()
        if current is None or current.isHidden():
            first = next(
                (
                    self.tree.topLevelItem(i)
                    for i in range(self.tree.topLevelItemCount())
                    if not self.tree.topLevelItem(i).isHidden()
                ),
                None,
            )
            self.tree.setCurrentItem(first)
        self._summary()
        self._preview_selection()

    def _selected(self):
        checked = [
            k
            for k, i in self.items.items()
            if not i.isHidden() and i.checkState(0) == Qt.CheckState.Checked
        ]
        keys = checked or [
            i.data(0, Qt.ItemDataRole.UserRole)
            for i in self.tree.selectedItems()
            if not i.isHidden() and i.data(0, Qt.ItemDataRole.UserRole) in self.rows
        ]
        return [self.rows[k] for k in keys]

    def _summary(self, *_):
        rows = self._selected()
        total = sum(not i.isHidden() for i in self.items.values())
        if self.partial:
            self.status.setText("최근 작업 표시 중 · 전체 목록을 찾는 중…")
        elif total:
            self.status.setText(
                f"{total}개"
                + (f" · 선택 {len(rows)}개" if self.select_mode.isChecked() else "")
            )
        else:
            self.status.setText(
                "검색 결과 없음" if self.search.text() else "표시할 항목이 없습니다"
            )
        if self.kind == "sessions" and self.catalog and self.catalog.invalid_count and not self.partial:
            self.status.setText(f"{self.status.text()} · 읽기 실패 {self.catalog.invalid_count}개")
        self.trash.setEnabled(
            bool(rows)
            and not self.partial
            and all(
                r["state"] not in {"적용 중", "생성 중", "번역 중", "확인 필요"}
                for r in rows
            )
        )
        self.open.setEnabled(len(rows) == 1 and bool(rows[0]["paths"]))
        self.open.setText(
            {"sessions": "편집기에서 열기", "history": "결과 보기"}.get(
                self.kind, "폴더 열기"
            )
        )

    def _open(self, row=None):
        rows = [row] if isinstance(row, dict) else self._selected()
        if len(rows) != 1:
            return
        row = rows[0]
        if row["kind"] == "history":
            if row.get("draft") and row["state"] == "미적용":
                self.open_path = Path(row["session"])
                self.draft = row["draft"]
                self.accept()
                return
            from advanced_editor.ai_sfx_preview import SfxPreviewDialog

            if not row.get("after") or not row["after"].is_file():
                QMessageBox.information(
                    self,
                    "식자 기록",
                    row.get("error") or "저장된 생성 이미지가 없습니다.",
                )
                return
            SfxPreviewDialog(
                dict(
                    ai_original_path=str(row.get("before") or ""),
                    ai_processed_path=str(row["after"]),
                    failed=row["state"] == "실패",
                    directory=row["id"],
                    warning=row.get("error") or row["state"],
                ),
                None,
                self,
            ).exec()
        elif row["kind"] == "sessions" and row["paths"]:
            selected_page = (
                self.preview.gallery.selected_path()
                if self.preview.row is row
                else None
            )
            self.open_path = Path(selected_page or row["paths"][0])
            self.accept()
        else:
            path = Path(row["id"])
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(path if path.is_dir() else path.parent))
            )

    def _review(self):
        if self.partial or not self.select_mode.isChecked():
            return
        rows = self._selected()
        if not rows:
            return
        self.setEnabled(False)
        self.status.setText("사용 중인 파일과 용량을 확인하는 중…")

        def review(publish, cancel):
            plans = []
            for kind in dict.fromkeys(r["kind"] for r in rows):
                chosen = [r for r in rows if r["kind"] == kind]
                plans.append(
                    prepare_cleanup(
                        kind, chosen, [r["id"] for r in chosen], self.repo, self.output
                    )
                )
            publish("plan", plans)

        self._worker(review)
