from pathlib import Path

from PyQt6.QtCore import QPoint, QPersistentModelIndex, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import QAbstractItemView, QListWidget, QStyle, QStyledItemDelegate

from advanced_editor.storage_preview import request_page_image


_THUMBNAIL_REQUESTED = Qt.ItemDataRole.UserRole + 1


class PageListDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        return QSize(100, 48 if self.parent().thumbnails else 32)

    def paint(self, painter, option, index):
        view = self.parent()
        current = index == view.currentIndex()
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        row = option.rect.adjusted(1, 1, -2, -1)
        painter.save()
        painter.setClipRect(option.rect)
        painter.setPen(Qt.PenStyle.NoPen)
        if current or hovered or selected:
            color = "#e1ecfa" if current else "#edf2f7"
            painter.setBrush(QColor(color))
            painter.drawRoundedRect(row, 4, 4)
        if current:
            painter.fillRect(row.x(), row.y() + 5, 2, row.height() - 10, QColor("#527faf"))

        number_rect = row.adjusted(7, 0, 0, 0)
        number_rect.setWidth(26)
        painter.setFont(option.font)
        painter.setPen(QColor("#4d6077"))
        painter.drawText(number_rect, Qt.AlignmentFlag.AlignCenter, str(index.row() + 1))

        text_left = 40
        if view.thumbnails:
            icon = index.data(Qt.ItemDataRole.DecorationRole)
            if icon is not None and not icon.isNull():
                thumbnail_rect = row.adjusted(35, 0, 0, 0)
                thumbnail_rect.setWidth(26)
                icon.paint(painter, thumbnail_rect, Qt.AlignmentFlag.AlignCenter)
            text_left = 67

        text_rect = row.adjusted(text_left, 0, -24, 0)
        font = option.font
        font.setBold(current)
        painter.setFont(font)
        painter.setPen(QColor("#244e82" if current else "#26374b"))
        file_name = index.data(Qt.ItemDataRole.DisplayRole) or ""
        text = painter.fontMetrics().elidedText(file_name, Qt.TextElideMode.ElideRight, text_rect.width())
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, text)
        if selected:
            check_rect = row.adjusted(row.width() - 19, 0, -5, 0)
            view.check_icon.paint(painter, check_rect, Qt.AlignmentFlag.AlignCenter)
        painter.restore()


class EditorPageList(QListWidget):
    thumbnail_ready = pyqtSignal(object, object)

    def __init__(self, check_icon: QIcon):
        super().__init__()
        self.check_icon = check_icon
        self.thumbnails = False
        self._pending = False
        self.setItemDelegate(PageListDelegate(self))
        self.setUniformItemSizes(True)
        self.setMouseTracking(True)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._load_timer = QTimer(self, singleShot=True, interval=40)
        self._load_timer.timeout.connect(self._load_visible)
        self.verticalScrollBar().valueChanged.connect(self._schedule)
        self.model().rowsInserted.connect(self._schedule)
        self.thumbnail_ready.connect(self._receive, Qt.ConnectionType.QueuedConnection)

    def set_thumbnails(self, enabled: bool) -> None:
        top = self.indexAt(QPoint(4, 5))
        self.thumbnails = enabled
        self.doItemsLayout()
        if top.isValid():
            self.scrollTo(top, QAbstractItemView.ScrollHint.PositionAtTop)
        self.viewport().update()
        self._schedule()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._schedule()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._schedule()

    def _schedule(self, *_args) -> None:
        if self.thumbnails:
            self._load_timer.start()

    def _load_visible(self) -> None:
        if not self.thumbnails or self._pending or not self.isVisible() or not self.count():
            return
        top = max(0, self.indexAt(QPoint(4, 5)).row())
        visible_rows = self.viewport().height() // self.sizeHintForRow(top) + 2
        for row in range(top, min(self.count(), top + visible_rows)):
            item = self.item(row)
            if item.data(_THUMBNAIL_REQUESTED):
                continue
            item.setData(_THUMBNAIL_REQUESTED, True)
            self._pending = True
            index = QPersistentModelIndex(self.indexFromItem(item))
            future = request_page_image(Path(item.data(Qt.ItemDataRole.UserRole)), (52, 72))

            def finish(task):
                image = None if task.cancelled() or task.exception() else task.result()
                try:
                    self.thumbnail_ready.emit(index, image)
                except RuntimeError:
                    pass  # The editor closed during decoding.

            future.add_done_callback(finish)
            break

    def _receive(self, index, image) -> None:
        if index.isValid() and image is not None:
            pixmap = QPixmap.fromImage(image)
            pixmap.setDevicePixelRatio(2)
            self.item(index.row()).setIcon(QIcon(pixmap))
        self._pending = False
        if not self._load_timer.isActive():
            self._load_visible()
