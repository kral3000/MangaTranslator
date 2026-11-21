"""Small previews of saved files. The editor renderer is never started here."""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from advanced_editor.storage_records import session_assets
from core.editor_paths import _resolve_stored_path

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="storage-preview")


def thumbnail(image, bounds):
    image.thumbnail(bounds, Image.Resampling.LANCZOS)
    return ImageQt(image.convert("RGBA")).copy()


@lru_cache(maxsize=24)
def read_thumbnail(path, mtime, size, bounds=(540, 640)):
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as z:
            pages = sorted(
                n
                for n in z.namelist()
                if Path(n).suffix.lower() in Image.registered_extensions()
            )
            if not pages:
                return None
            with z.open(pages[0]) as file, Image.open(file) as image:
                return thumbnail(image, bounds)
    with Image.open(path) as image:
        return thumbnail(image, bounds)


def read_previews(row):
    kind = row["kind"]
    if kind == "history":
        sources = [row.get("before"), row.get("after")]
    else:
        sources = [row.get("preview_path") or row["id"]]
    images = []
    for source in sources:
        try:
            path = Path(source)
            st = path.stat()
            images.append(read_thumbnail(path, st.st_mtime_ns, st.st_size))
        except Exception:
            images.append(None)
    return images


def page_image(page, bounds):
    try:
        data, _, edited = session_assets(page)
        candidates = [edited] + [
            _resolve_stored_path(page, data.get(key))
            for key in (
                "rendered_snapshot_path",
                "translated_image_path",
                "base_image_path",
                "original_snapshot_path",
            )
        ]
    except (OSError, ValueError):
        return None
    for source in candidates:
        if source:
            try:
                stat = source.stat()
                return read_thumbnail(source, stat.st_mtime_ns, stat.st_size, bounds)
            except (OSError, ValueError):
                continue
    return None


class ImagePane(QLabel):
    def set_image(self, image):
        self.original = QPixmap.fromImage(image) if image is not None else None
        self.fit()

    def fit(self):
        if self.original:
            self.setPixmap(
                self.original.scaled(
                    self.contentsRect().size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self.setText("저장된 이미지 없음")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "original"):
            self.fit()


class PageThumbnailDelegate(QStyledItemDelegate):
    height = 98

    def sizeHint(self, option, index):
        return QSize(72, self.height)

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.decorationPosition = QStyleOptionViewItem.Position.Top
        option.decorationAlignment = Qt.AlignmentFlag.AlignCenter
        option.displayAlignment = Qt.AlignmentFlag.AlignCenter


class SessionPages(QWidget):
    loaded = pyqtSignal(int, int, bool, object)

    def __init__(self):
        super().__init__()
        self.paths = []
        self.version = 0
        self.loaded_small = set()
        self.pending = {}
        self.loaded.connect(self._receive)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.position = QLabel()
        self.position.setObjectName("PagePosition")
        layout.addWidget(self.position)
        body = QHBoxLayout()
        body.setSpacing(6)
        layout.addLayout(body, 1)
        self.pages = QListWidget()
        self.pages.setObjectName("PageThumbnails")
        self.pages.setAccessibleName("페이지 썸네일 목록")
        self.pages.setItemDelegate(PageThumbnailDelegate(self.pages))
        self.pages.setUniformItemSizes(True)
        self.pages.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.pages.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.pages.setIconSize(QSize(68, 78))
        self.pages.setFixedWidth(86)
        body.addWidget(self.pages)
        self.large = ImagePane()
        self.large.setObjectName("StoragePreviewImage")
        self.large.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.large.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        body.addWidget(self.large, 1)
        self.timer = QTimer(self, singleShot=True, interval=45)
        self.timer.timeout.connect(self._load_visible)
        self.pages.currentRowChanged.connect(self._change_page)
        self.pages.verticalScrollBar().valueChanged.connect(
            lambda _: self.timer.start()
        )

    def set_pages(self, paths, keep=None):
        self.version += 1
        for future in self.pending.values():
            future.cancel()
        self.pending.clear()
        self.timer.stop()
        self.loaded_small.clear()
        self.paths = list(paths)
        self.pages.blockSignals(True)
        self.pages.clear()
        for number, path in enumerate(self.paths, 1):
            item = QListWidgetItem(str(number), self.pages)
            item.setToolTip(f"{number}페이지 · {Path(path).name}")
            item.setData(Qt.ItemDataRole.AccessibleTextRole, f"{number}페이지")
        self.pages.blockSignals(False)
        if self.paths:
            self.pages.setCurrentRow(
                self.paths.index(keep) if keep in self.paths else 0
            )
        else:
            self._change_page(-1)

    def selected_path(self):
        index = self.pages.currentRow()
        return self.paths[index] if 0 <= index < len(self.paths) else None

    def _change_page(self, index):
        self.large.original = None
        self.large.setText("…" if index >= 0 else "완료된 페이지 없음")
        self.position.setText(
            f"{index + 1} / {len(self.paths)} 페이지" if index >= 0 else "0페이지"
        )
        old = self.pending.pop(True, None)
        if old:
            old.cancel()
        if index >= 0:
            self._request(index, True)
            self.timer.start()

    def _request(self, index, large):
        version = self.version
        future = _pool.submit(
            page_image, Path(self.paths[index]), (1050, 1400) if large else (108, 138)
        )
        self.pending[large] = future

        def finish(task):
            if task.cancelled() or version != self.version:
                return
            try:
                self.loaded.emit(
                    version,
                    index,
                    large,
                    task.result() if not task.exception() else None,
                )
            except RuntimeError:
                pass

        future.add_done_callback(finish)

    def _load_visible(self):
        if not self.paths or False in self.pending:
            return
        height = PageThumbnailDelegate.height
        top = self.pages.verticalScrollBar().value() // height
        end = top + self.pages.viewport().height() // height + 3
        for index in range(max(0, top - 1), min(len(self.paths), end)):
            if index not in self.loaded_small:
                self._request(index, False)
                break

    def _receive(self, version, index, large, image):
        if version != self.version or (large and index != self.pages.currentRow()):
            return
        self.pending.pop(large, None)
        if large:
            self.large.set_image(image)
        else:
            self.loaded_small.add(index)
            if image is not None:
                pixmap = QPixmap.fromImage(image)
                icon = QIcon(pixmap)
                icon.addPixmap(pixmap, QIcon.Mode.Selected)
                self.pages.item(index).setIcon(icon)
            else:
                self.pages.item(index).setText(f"{index + 1} · 없음")
            self._load_visible()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.timer.start()


class StoragePreview(QWidget):
    finished = pyqtSignal(int, object)

    def __init__(self):
        super().__init__()
        self.setObjectName("StoragePreview")
        self.setMinimumWidth(155)
        self.row = self.future = None
        self.serial = 0
        self.timer = QTimer(self, singleShot=True, interval=80)
        self.timer.timeout.connect(self._start)
        self.finished.connect(self._show)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 8, 8)
        self.heading = QLabel("미리보기")
        self.heading.setWordWrap(True)
        layout.addWidget(self.heading)
        self.gallery = SessionPages()
        layout.addWidget(self.gallery, 1)
        self.captions, self.panes = [], []
        for _ in range(2):
            caption, pane = QLabel(), ImagePane()
            pane.setObjectName("StoragePreviewImage")
            pane.setMinimumSize(40, 40)
            pane.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pane.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            layout.addWidget(caption)
            layout.addWidget(pane, 1)
            self.captions.append(caption)
            self.panes.append(pane)
        self.prompt = QLabel("목록에서 항목을 선택하세요")
        self.prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prompt.setWordWrap(True)
        layout.addWidget(self.prompt, 1)
        self.select(None, force=True)

    def select(self, row, *, force=False):
        if not force and row is self.row:
            return
        self.serial += 1
        self.timer.stop()
        if self.future:
            self.future.cancel()
        is_session = bool(row and row["kind"] == "sessions")
        keep = (
            self.gallery.selected_path()
            if self.row
            and row
            and (
                self.row["id"] == row["id"]
                or (
                    row.get("group_key")
                    and self.row.get("group_key") == row["group_key"]
                )
            )
            else None
        )
        if is_session or self.gallery.paths:
            self.gallery.set_pages(row["paths"] if is_session else [], keep)
        self.gallery.setVisible(is_session)
        self.row = row
        self.prompt.setVisible(row is None)
        self.heading.setText(row.get("detail", "미리보기") if row else "미리보기")
        self.heading.setVisible(not is_session)
        labels = (
            ["원본", "식자 결과"]
            if row and row["kind"] == "history"
            else ["저장된 페이지"]
        )
        for index, (caption, pane) in enumerate(zip(self.captions, self.panes)):
            visible = row is not None and not is_session and index < len(labels)
            caption.setVisible(visible)
            pane.setVisible(visible)
            pane.original = None
            pane.setText("…")
            if visible:
                caption.setText(labels[index])
        if row and not is_session:
            self.timer.start()

    def _start(self):
        token = self.serial
        future = self.future = _pool.submit(read_previews, self.row)

        def done(task):
            if task.cancelled() or token != self.serial:
                return
            try:
                self.finished.emit(token, task.result() if not task.exception() else [])
            except RuntimeError:
                pass  # The dialog was destroyed while the image was decoding.

        future.add_done_callback(done)

    def _show(self, token, images):
        if token == self.serial:
            for index, pane in enumerate(self.panes):
                pane.set_image(images[index] if index < len(images) else None)
