from __future__ import annotations

import copy
import io
import json
import os
import re
import sys
import tempfile
import threading
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image
from PyQt6.QtCore import (
    QByteArray,
    QObject,
    QPointF,
    QRect,
    QRectF,
    QSignalBlocker,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QAction,
    QColor,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QImage,
    QIcon,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QLineEdit,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtSvg import QSvgRenderer

from advanced_editor.backend import (
    detect_and_recognize,
    get_translation_defaults,
    run_inpainting,
    translate_texts,
)
from core.config import DEFAULT_FLUX_KLEIN_PROMPT, DEFAULT_FLUX_KONTEXT_PROMPT
from core.editor_package import embed_editor_package, resolve_editor_session_input
from core.editor_session import (
    _region_render_signature,
    _decode_uploaded_relative_path,
    _resolve_repo_relative_path,
    add_region,
    delete_region,
    load_session,
    related_edit_session_paths,
    render_session_image,
    resolve_region_font_size,
    rerender_session,
    save_session,
    set_region_font_size,
)
from core.text.text_processing import extract_translation_tag


TOOLS = {
    "select": "선택/이동",
    "textbox": "텍스트 박스",
    "inpaint_brush": "인페인팅 브러시",
    "inpaint_eraser": "인페인팅 지우개",
    "paint_brush": "그리기 브러시",
    "paint_eraser": "그리기 지우개",
    "restore_brush": "원본 복구 브러시",
    "restore_eraser": "원본 복구 지우개",
}

TOOL_RAIL_LABELS = {
    "select": "선택",
    "textbox": "텍스트 박스",
    "inpaint_brush": "인페인팅 브러시",
    "inpaint_eraser": "인페인팅 마스크 지우개",
    "paint_brush": "그리기 브러시",
    "paint_eraser": "그리기 지우개",
    "restore_brush": "원본 복구 브러시",
    "restore_eraser": "원본 복구 지우개",
}

EDITOR_ICON_PATHS = {
    "select": '<path d="M5 3l11 10-6 1 3 6-3 1-3-6-4 4z"/>',
    "textbox": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M8 8h8M12 8v8M9 16h6"/>',
    "brush": '<path d="M14 4l6 6-9 9H5v-6z"/><path d="M4 20c2 0 3-1 3-3"/>',
    "eraser": '<path d="M13 4l7 7-8 8H7l-4-4z"/><path d="M9 19h11"/>',
    "restore": '<path d="M4 8V3m0 0h5M4 3l4 4"/><path d="M5 13a8 8 0 1 0 2-7"/>',
    "undo": '<path d="M9 7H4V2"/><path d="M4 7l4-4"/><path d="M5 7h8a6 6 0 0 1 0 12H9"/>',
    "redo": '<path d="M15 7h5V2"/><path d="M20 7l-4-4"/><path d="M19 7h-8a6 6 0 0 0 0 12h4"/>',
    "history": '<circle cx="12" cy="12" r="9"/><path d="M12 7v6l4 2"/>',
    "save": '<path d="M4 3h13l3 3v15H4z"/><path d="M8 3v6h8V3M8 21v-7h8v7"/>',
    "export": '<path d="M14 3h7v7M21 3l-10 10"/><path d="M18 14v7H3V6h7"/>',
    "add": '<path d="M12 5v14M5 12h14"/>',
    "trash": '<path d="M4 7h16M9 7V4h6v3M7 7l1 14h8l1-14M10 11v6M14 11v6"/>',
    "refresh": '<path d="M20 7V3l-3 3a8 8 0 1 0 2 9"/>',
    "folder": '<path d="M3 6h7l2 2h9v12H3z"/>',
    "play": '<path d="M8 5l11 7-11 7z"/>',
    "palette": '<path d="M12 3a9 9 0 0 0 0 18h2a2 2 0 0 0 0-4h-1a2 2 0 0 1 0-4h4a4 4 0 0 0 4-4c0-4-4-6-9-6z"/><path d="M7 10h.01M9 6h.01M14 6h.01M18 9h.01"/>',
    "bold": '<path d="M7 4h6a4 4 0 0 1 0 8H7zM7 12h7a4 4 0 0 1 0 8H7z"/>',
    "italic": '<path d="M10 4h8M6 20h8M14 4L10 20"/>',
    "hand": '<path d="M7 12V7a2 2 0 0 1 4 0v3-5a2 2 0 0 1 4 0v5-3a2 2 0 0 1 4 0v7c0 5-3 7-7 7-3 0-5-2-7-5l-2-3a2 2 0 0 1 4-2z"/>',
    "fit": '<path d="M8 3H3v5M16 3h5v5M8 21H3v-5M16 21h5v-5"/>',
    "reset": '<path d="M4 8V3m0 0h5M4 3l4 4"/><path d="M5 13a8 8 0 1 0 2-7"/>',
    "minus": '<path d="M5 12h14"/>',
    "action": '<path d="M5 12h14M14 6l6 6-6 6"/>',
}


def _editor_icon(name: str, color: str = "#e5e7eb") -> QIcon:
    paths = EDITOR_ICON_PATHS.get(name, EDITOR_ICON_PATHS["select"])
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
        'viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round">{paths}</svg>'
    )
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)


class ResizableTextEdit(QTextEdit):
    """Compact text edit with a browser-style vertical resize corner."""

    _GRIP_SIZE = 16

    def __init__(self, height: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._resize_start_y: Optional[int] = None
        self._resize_start_height = height
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setFixedHeight(height)
        self.setToolTip("오른쪽 아래 모서리를 끌어 입력 상자 높이를 조절할 수 있습니다.")

    def _over_resize_grip(self, event: QMouseEvent) -> bool:
        position = event.position()
        return (
            position.x() >= self.viewport().width() - self._GRIP_SIZE
            and position.y() >= self.viewport().height() - self._GRIP_SIZE
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._over_resize_grip(event)
        ):
            self._resize_start_y = int(event.globalPosition().y())
            self._resize_start_height = self.height()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._resize_start_y is not None:
            delta = int(event.globalPosition().y()) - self._resize_start_y
            self.setFixedHeight(max(52, min(360, self._resize_start_height + delta)))
            event.accept()
            return
        cursor = (
            Qt.CursorShape.SizeFDiagCursor
            if self._over_resize_grip(event)
            else Qt.CursorShape.IBeamCursor
        )
        self.viewport().setCursor(cursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._resize_start_y is not None:
            self._resize_start_y = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        painter.setPen(QPen(QColor("#475569"), 1))
        right = self.viewport().width() - 4
        bottom = self.viewport().height() - 4
        for offset in (4, 8, 12):
            painter.drawLine(right - offset, bottom, right, bottom - offset)


class AutoHideStatusLabel(QLabel):
    def setText(self, text: str) -> None:
        super().setText(text)
        self.setVisible(bool(str(text or "").strip()))


class TaskSignals(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

REGION_TYPE_LABELS = {
    "speech_bubble": "말풍선",
    "outside_text": "말풍선 밖 텍스트",
    "added_text": "추가한 텍스트",
    "region": "영역",
}


def _pil_to_qimage(image: Image.Image) -> QImage:
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimage = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888)
    return qimage.copy()


def _load_rgba(path: Any, size: Optional[tuple[int, int]] = None) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    if size and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def _load_mask(path: Any, size: tuple[int, int]) -> np.ndarray:
    if path and Path(path).exists():
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.ndim > 2:
                mask = mask[..., 0]
            if mask.shape[:2] != (size[1], size[0]):
                mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
                if mask.ndim > 2:
                    mask = mask[..., 0]
            return np.where(mask > 0, 255, 0).astype(np.uint8)
    return np.zeros((size[1], size[0]), dtype=np.uint8)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = np.asarray(mask)
    if normalized.ndim > 2:
        normalized = normalized[..., 0]
    cv2.imwrite(str(path), np.where(normalized > 0, 255, 0).astype(np.uint8))


def _overlay_to_array(path: Any, size: tuple[int, int]) -> np.ndarray:
    if path and Path(path).exists():
        image = _load_rgba(path, size)
        return np.asarray(image, dtype=np.uint8).copy()
    return np.zeros((size[1], size[0], 4), dtype=np.uint8)


def _array_to_overlay_image(overlay: np.ndarray) -> Image.Image:
    return Image.fromarray(overlay.astype(np.uint8, copy=False), mode="RGBA")


def _bbox(region: dict[str, Any]) -> list[float]:
    values = region.get("bbox") or [0, 0, 1, 1]
    try:
        x1, y1, x2, y2 = [float(v) for v in values[:4]]
    except Exception:
        return [0.0, 0.0, 1.0, 1.0]
    return [x1, y1, max(x1 + 1.0, x2), max(y1 + 1.0, y2)]


def _rgb_from_hex(value: str, fallback: tuple[int, int, int] = (0, 0, 0)) -> tuple[int, int, int]:
    text = str(value or "").strip()
    if text.startswith("#") and len(text) == 7:
        try:
            return int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)
        except ValueError:
            return fallback
    return fallback


def _hex_from_qcolor(color: QColor) -> str:
    return f"#{color.red():02x}{color.green():02x}{color.blue():02x}"


EDITOR_SETTINGS_FILE = (
    Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")))
    / "MangaTranslator"
    / "advanced_editor_settings.json"
)
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SUPPORTED_FONT_SUFFIXES = {".ttf", ".otf", ".ttc"}
FONT_STYLE_ORDER = {
    "Thin": 100,
    "Hairline": 110,
    "Extra Light": 200,
    "Ultra Light": 210,
    "Light": 300,
    "Book": 350,
    "Regular": 400,
    "Normal": 410,
    "Roman": 420,
    "Medium": 500,
    "Semi Bold": 600,
    "Demi Bold": 610,
    "Bold": 700,
    "Extra Bold": 800,
    "Ultra Bold": 810,
    "Black": 900,
    "Heavy": 910,
}
FONT_STYLE_SUFFIXES = (
    ("Extra Bold Italic", r"(?:extra|ultra)[ _-]*bold[ _-]*(?:italic|oblique)"),
    ("Semi Bold Italic", r"(?:semi|demi)[ _-]*bold[ _-]*(?:italic|oblique)"),
    ("Bold Italic", r"bold[ _-]*(?:italic|oblique)"),
    ("Extra Light Italic", r"(?:extra|ultra)[ _-]*light[ _-]*(?:italic|oblique)"),
    ("Light Italic", r"light[ _-]*(?:italic|oblique)"),
    ("Medium Italic", r"medium[ _-]*(?:italic|oblique)"),
    ("Regular Italic", r"(?:regular|normal|roman)[ _-]*(?:italic|oblique)"),
    ("Italic", r"(?:italic|oblique|slanted|inclined)"),
    ("Extra Bold", r"(?:extra|ultra)[ _-]*bold"),
    ("Semi Bold", r"(?:semi|demi)[ _-]*bold"),
    ("Extra Light", r"(?:extra|ultra)[ _-]*light"),
    ("Thin", r"(?:thin|hairline)"),
    ("Light", r"light"),
    ("Medium", r"medium"),
    ("Bold", r"bold"),
    ("Black", r"(?:black|heavy)"),
    ("Regular", r"(?:regular|normal|roman|book)"),
)


def _font_name_record(font: TTFont, name_ids: tuple[int, ...]) -> str:
    try:
        names = list(font["name"].names)
    except Exception:
        return ""
    for name_id in name_ids:
        records = [record for record in names if record.nameID == name_id]
        records.sort(
            key=lambda record: (
                record.platformID != 3,
                record.langID not in {0x0409, 0},
            )
        )
        for record in records:
            try:
                value = record.toUnicode().strip()
            except Exception:
                continue
            if value:
                return value
    return ""


def _split_font_filename(stem: str) -> tuple[str, str]:
    original = str(stem or "").strip()
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", original)
    spaced = re.sub(r"[_-]+", " ", spaced)
    spaced = re.sub(r"\s+", " ", spaced).strip()
    for style, pattern in FONT_STYLE_SUFFIXES:
        match = re.search(rf"(?:\s|^){pattern}\s*$", spaced, flags=re.IGNORECASE)
        if match:
            family = spaced[: match.start()].strip(" _-")
            return family or original, style
        compact_match = re.search(rf"{pattern}$", original, flags=re.IGNORECASE)
        if compact_match and compact_match.start() > 0:
            family = original[: compact_match.start()].strip(" _-")
            return family or original, style
    short_regular = re.search(r"(?:[-_]|(?<=[a-z]))(?:b?rg|r)$", original)
    if short_regular:
        family = original[: short_regular.start()].strip(" _-")
        if family:
            return family, "Regular"
    return spaced or original, "Regular"


def _normalized_font_style(value: str, filename_style: str) -> str:
    text = re.sub(r"[_-]+", " ", str(value or "")).strip()
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\s+", " ", text)
    lowered = text.casefold()
    italic = any(token in lowered for token in ("italic", "oblique", "slanted"))
    weight = "Regular"
    weight_patterns = (
        ("Extra Bold", ("extra bold", "extrabold", "ultra bold", "ultrabold")),
        ("Semi Bold", ("semi bold", "semibold", "demi bold", "demibold")),
        ("Extra Light", ("extra light", "extralight", "ultra light", "ultralight")),
        ("Thin", ("thin", "hairline")),
        ("Light", ("light",)),
        ("Medium", ("medium",)),
        ("Bold", ("bold",)),
        ("Black", ("black", "heavy")),
        ("Book", ("book",)),
        ("Regular", ("regular", "normal", "roman")),
    )
    for label, tokens in weight_patterns:
        if any(token in lowered for token in tokens):
            weight = label
            break
    if weight == "Regular" and filename_style not in {"", "Regular"}:
        return filename_style
    if italic:
        return "Italic" if weight == "Regular" else f"{weight} Italic"
    if text and weight == "Regular" and lowered not in {
        "regular",
        "normal",
        "roman",
    }:
        return text
    return weight


def _font_descriptor(path: Path, fonts_root: Path) -> dict[str, str]:
    filename_family, filename_style = _split_font_filename(path.stem)
    family = ""
    style = ""
    try:
        font = TTFont(path, fontNumber=0, lazy=True)
        try:
            family = _font_name_record(font, (16, 1))
            style = _font_name_record(font, (17, 2))
        finally:
            font.close()
    except Exception:
        pass
    family = re.sub(r"\s+", " ", family).strip() or filename_family
    style = _normalized_font_style(style, filename_style)
    try:
        relative = str(path.resolve().relative_to(fonts_root.resolve()))
    except Exception:
        relative = path.name
    family_key = "".join(char for char in family.casefold() if char.isalnum())
    return {
        "family": family,
        "family_key": family_key or path.stem.casefold(),
        "style": style or "Regular",
        "path": str(path.resolve()),
        "relative": relative,
    }


def _font_style_sort_key(style: str) -> tuple[int, int, str]:
    text = str(style or "Regular")
    base = re.sub(r"\s+(?:Italic|Oblique)$", "", text, flags=re.IGNORECASE)
    italic = int(bool(re.search(r"(?:Italic|Oblique)$", text, flags=re.IGNORECASE)))
    return FONT_STYLE_ORDER.get(base, 1000), italic, text.casefold()


def _is_temporary_upload_path(path: Any) -> bool:
    try:
        resolved = Path(str(path or "")).resolve()
    except Exception:
        return False
    lowered_parts = {part.lower() for part in resolved.parts}
    if "gradio" in lowered_parts:
        return True
    return resolved.name.startswith("__MT_RELATIVE_PATH__")


def _load_editor_settings() -> dict[str, Any]:
    try:
        if EDITOR_SETTINGS_FILE.is_file():
            with open(EDITOR_SETTINGS_FILE, "r", encoding="utf-8") as file:
                loaded = json.load(file)
            if isinstance(loaded, dict):
                return loaded
    except Exception:
        pass
    return {}


def _save_editor_settings(settings: dict[str, Any]) -> None:
    try:
        EDITOR_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = EDITOR_SETTINGS_FILE.with_suffix(".tmp")
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(settings, file, ensure_ascii=False, indent=2)
        temporary.replace(EDITOR_SETTINGS_FILE)
    except Exception:
        pass


class EditorCanvas(QWidget):
    def __init__(self, window: "AdvancedEditorWindow"):
        super().__init__()
        self.window = window
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(480, 480)
        self.scale = 1.0
        self.offset = QPointF(0.0, 0.0)
        self.pan_offset = QPointF(0.0, 0.0)
        self.pan_drag_start: Optional[QPointF] = None
        self.pan_drag_origin = QPointF(0.0, 0.0)
        self.tool = "select"
        self.drag_mode: Optional[str] = None
        self.drag_start_image = QPointF()
        self.drag_start_view = QPointF()
        self.drag_start_bbox: Optional[list[float]] = None
        self.drag_has_moved = False
        self.drag_history_pushed = False
        self.stroke_points: list[tuple[int, int]] = []
        self.hover_image_pos: Optional[QPointF] = None
        self.textbox_start: Optional[QPointF] = None
        self.textbox_current: Optional[QPointF] = None

    def sizeHint(self) -> QSize:
        return QSize(900, 900)

    def keyPressEvent(self, event) -> None:
        if self.window.handle_editor_key_event(event):
            return
        super().keyPressEvent(event)

    def set_tool(self, tool: str) -> None:
        self.tool = tool
        self.drag_mode = None
        self.pan_drag_start = None
        self.stroke_points = []
        self.textbox_start = None
        self.textbox_current = None
        if tool == "pan":
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()
        self.update()

    def _base_fit_scale(self) -> float:
        image = self.window.display_image
        if image is None:
            return 1.0
        margin = 28.0
        available_w = max(1.0, self.width() - margin * 2)
        available_h = max(1.0, self.height() - margin * 2)
        return max(
            0.0001,
            min(available_w / image.width, available_h / image.height),
        )

    def _constrain_pan(self, base_offset: QPointF) -> None:
        image = self.window.display_image
        if image is None:
            self.pan_offset = QPointF(0.0, 0.0)
            self.offset = QPointF(0.0, 0.0)
            self.window.update_canvas_scrollbars(0, 0, 0, 0)
            return
        margin = 28.0
        draw_w = image.width * self.scale
        draw_h = image.height * self.scale
        desired_x = base_offset.x() + self.pan_offset.x()
        desired_y = base_offset.y() + self.pan_offset.y()
        if draw_w <= self.width() - margin * 2:
            final_x = (self.width() - draw_w) / 2.0
        else:
            final_x = max(
                self.width() - draw_w - margin,
                min(margin, desired_x),
            )
        if draw_h <= self.height() - margin * 2:
            final_y = (self.height() - draw_h) / 2.0
        else:
            final_y = max(
                self.height() - draw_h - margin,
                min(margin, desired_y),
            )
        self.offset = QPointF(final_x, final_y)
        self.pan_offset = QPointF(
            final_x - base_offset.x(),
            final_y - base_offset.y(),
        )
        horizontal_max = max(
            0,
            int(round(draw_w - self.width() + margin * 2.0)),
        )
        vertical_max = max(
            0,
            int(round(draw_h - self.height() + margin * 2.0)),
        )
        horizontal_value = (
            max(0, min(horizontal_max, int(round(margin - final_x))))
            if horizontal_max
            else 0
        )
        vertical_value = (
            max(0, min(vertical_max, int(round(margin - final_y))))
            if vertical_max
            else 0
        )
        self.window.update_canvas_scrollbars(
            horizontal_max,
            horizontal_value,
            vertical_max,
            vertical_value,
        )

    def set_scroll_position(
        self,
        horizontal_value: Optional[int] = None,
        vertical_value: Optional[int] = None,
    ) -> None:
        image = self.window.display_image
        if image is None:
            return
        self._fit_transform()
        margin = 28.0
        draw_w = image.width * self.scale
        draw_h = image.height * self.scale
        base_offset = QPointF(
            (self.width() - draw_w) / 2.0,
            (self.height() - draw_h) / 2.0,
        )
        desired_x = self.offset.x()
        desired_y = self.offset.y()
        if horizontal_value is not None and draw_w > self.width() - margin * 2:
            desired_x = margin - float(horizontal_value)
        if vertical_value is not None and draw_h > self.height() - margin * 2:
            desired_y = margin - float(vertical_value)
        self.pan_offset = QPointF(
            desired_x - base_offset.x(),
            desired_y - base_offset.y(),
        )
        self._constrain_pan(base_offset)
        self.update()

    def _fit_transform(self) -> None:
        image = self.window.display_image
        if image is None:
            self.scale = 1.0
            self.offset = QPointF(0.0, 0.0)
            self.pan_offset = QPointF(0.0, 0.0)
            self.window.update_zoom_display(self.scale)
            return
        fit_scale = self._base_fit_scale()
        self.scale = max(
            0.02,
            min(8.0, fit_scale * self.window.zoom_factor),
        )
        self.window.zoom_factor = self.scale / fit_scale
        draw_w = image.width * self.scale
        draw_h = image.height * self.scale
        base_offset = QPointF(
            (self.width() - draw_w) / 2.0,
            (self.height() - draw_h) / 2.0,
        )
        self._constrain_pan(base_offset)
        self.window.update_zoom_display(self.scale)

    @staticmethod
    def _smooth_image_transform_for_scale(scale: float) -> bool:
        return float(scale) < 1.0

    def _draw_display_image(self, painter: QPainter, image: Image.Image) -> None:
        qimage = self.window.display_qimage
        if qimage is None:
            qimage = _pil_to_qimage(image)
            self.window.display_qimage = qimage
        target = QRectF(
            self.offset.x(),
            self.offset.y(),
            image.width * self.scale,
            image.height * self.scale,
        )
        painter.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform,
            self._smooth_image_transform_for_scale(self.scale),
        )
        painter.drawImage(target, qimage)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

    def reset_view(self) -> None:
        self.window.zoom_factor = 1.0
        self.pan_offset = QPointF(0.0, 0.0)
        self._fit_transform()
        self.update()

    def set_absolute_scale(self, scale: float, anchor: Optional[QPointF] = None) -> None:
        self._fit_transform()
        target_scale = max(0.02, min(8.0, float(scale)))
        current_scale = max(0.0001, self.scale)
        self.zoom_at(
            anchor or QPointF(self.width() / 2.0, self.height() / 2.0),
            target_scale / current_scale,
        )

    def zoom_at(self, view_pos: QPointF, factor: float) -> None:
        image = self.window.display_image
        if image is None or factor <= 0:
            return
        self._fit_transform()
        image_rect = QRectF(
            self.offset.x(),
            self.offset.y(),
            image.width * self.scale,
            image.height * self.scale,
        )
        anchor = QPointF(view_pos)
        if not image_rect.contains(anchor):
            anchor = QPointF(self.width() / 2.0, self.height() / 2.0)
        image_x = (anchor.x() - self.offset.x()) / max(self.scale, 0.0001)
        image_y = (anchor.y() - self.offset.y()) / max(self.scale, 0.0001)
        fit_scale = self._base_fit_scale()
        target_scale = max(0.02, min(8.0, self.scale * factor))
        self.window.zoom_factor = target_scale / fit_scale
        draw_w = image.width * target_scale
        draw_h = image.height * target_scale
        base_offset = QPointF(
            (self.width() - draw_w) / 2.0,
            (self.height() - draw_h) / 2.0,
        )
        self.scale = target_scale
        self.pan_offset = QPointF(
            anchor.x() - image_x * target_scale - base_offset.x(),
            anchor.y() - image_y * target_scale - base_offset.y(),
        )
        self._constrain_pan(base_offset)
        self.window.update_zoom_display(self.scale)
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_transform()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        steps = delta / 120.0
        self.zoom_at(QPointF(event.position()), 1.2 ** steps)
        event.accept()

    def image_to_view(self, point: QPointF) -> QPointF:
        return QPointF(point.x() * self.scale + self.offset.x(), point.y() * self.scale + self.offset.y())

    def view_to_image(self, point: QPointF) -> QPointF:
        image = self.window.display_image
        x = (point.x() - self.offset.x()) / self.scale
        y = (point.y() - self.offset.y()) / self.scale
        if image is not None:
            x = max(0.0, min(float(image.width - 1), x))
            y = max(0.0, min(float(image.height - 1), y))
        return QPointF(x, y)

    def _region_rect_view(self, region: dict[str, Any]) -> QRectF:
        x1, y1, x2, y2 = _bbox(region)
        top_left = self.image_to_view(QPointF(x1, y1))
        bottom_right = self.image_to_view(QPointF(x2, y2))
        return QRectF(top_left, bottom_right).normalized()

    def _handle_rects(self, rect: QRectF) -> dict[str, QRectF]:
        size = 10.0
        half = size / 2.0
        points = {
            "nw": rect.topLeft(),
            "ne": rect.topRight(),
            "sw": rect.bottomLeft(),
            "se": rect.bottomRight(),
        }
        return {key: QRectF(point.x() - half, point.y() - half, size, size) for key, point in points.items()}

    def _hit_test(self, pos: QPointF) -> tuple[Optional[int], Optional[str]]:
        selected_index = self.window.selected_index
        if selected_index is not None:
            region = self.window.region_at(selected_index)
            if region is not None:
                rect = self._region_rect_view(region)
                for key, handle_rect in self._handle_rects(rect).items():
                    if handle_rect.contains(pos):
                        return selected_index, key

        for index in reversed(range(len(self.window.session.get("regions", [])))):
            region = self.window.region_at(index)
            if region is not None and self._region_rect_view(region).contains(pos):
                return index, "move"
        return None, None

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.fillRect(self.rect(), QColor("#202225"))
        self._fit_transform()

        image = self.window.display_image
        if image is None:
            painter.setPen(QColor("#dddddd"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "편집 세션을 불러오지 못했습니다")
            return

        self._draw_display_image(painter, image)

        if self.window.show_inpaint_mask_checkbox.isChecked():
            self._draw_inpaint_mask(painter)
        if (
            hasattr(self.window, "show_restore_mask_checkbox")
            and self.window.show_restore_mask_checkbox.isChecked()
        ):
            self._draw_restore_mask(painter)

        self._draw_regions(painter)
        self._draw_stroke_preview(painter)
        self._draw_brush_cursor(painter)
        self._draw_textbox_preview(painter)

    def _preview_text(self, region: dict[str, Any]) -> str:
        text = str(region.get("translation") or "").strip()
        _tag, text = extract_translation_tag(text)
        text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
        return text.replace("\\n", "\n")

    def _draw_fast_text_preview(self, painter: QPainter) -> None:
        for region in self.window.fast_preview_regions():
            rect = self._region_rect_view(region)
            self.window.draw_region_fast_text(painter, region, rect, self.scale)

    def _draw_inpaint_mask(self, painter: QPainter) -> None:
        self._draw_mask_overlay(
            painter,
            self.window.session.get("inpaint_mask_path"),
            [0, 190, 255, 105],
        )

    def _draw_restore_mask(self, painter: QPainter) -> None:
        self._draw_mask_overlay(
            painter,
            self.window.session.get("restore_mask_path"),
            [255, 145, 0, 115],
        )

    def _draw_mask_overlay(
        self,
        painter: QPainter,
        mask_path: Any,
        color_rgba: list[int],
    ) -> None:
        if not mask_path:
            return
        mask = _load_mask(mask_path, self.window.image_size)
        if not np.any(mask):
            return
        rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba[mask > 0] = color_rgba
        overlay = Image.fromarray(rgba, mode="RGBA")
        target = QRectF(
            self.offset.x(),
            self.offset.y(),
            overlay.width * self.scale,
            overlay.height * self.scale,
        )
        painter.drawImage(target, _pil_to_qimage(overlay))

    @staticmethod
    def _overlap_area(first: QRectF, second: QRectF) -> float:
        overlap = first.intersected(second)
        return max(0.0, overlap.width()) * max(0.0, overlap.height())

    @classmethod
    def _choose_region_badge_rect(
        cls,
        region_rect: QRectF,
        badge_width: float,
        badge_height: float,
        other_regions: list[QRectF],
        placed_badges: list[QRectF],
        viewport_rect: QRectF,
        image_rect: QRectF,
    ) -> QRectF:
        gap = 3.0
        attached_tab = QRectF(
            region_rect.left(),
            region_rect.top() - badge_height,
            badge_width,
            badge_height,
        )
        if viewport_rect.contains(attached_tab):
            blocks_region = any(
                cls._overlap_area(attached_tab, obstacle.adjusted(-1, -1, 1, 1))
                > 0
                for obstacle in other_regions
            )
            blocks_badge = any(
                cls._overlap_area(attached_tab, obstacle.adjusted(-2, -2, 2, 2))
                > 0
                for obstacle in placed_badges
            )
            if not blocks_region and not blocks_badge:
                return attached_tab

        centered_x = region_rect.center().x() - badge_width / 2.0
        centered_y = region_rect.center().y() - badge_height / 2.0
        candidates = (
            QRectF(centered_x, region_rect.top() - badge_height - gap, badge_width, badge_height),
            QRectF(centered_x, region_rect.bottom() + gap, badge_width, badge_height),
            QRectF(region_rect.left(), region_rect.top() - badge_height - gap, badge_width, badge_height),
            QRectF(region_rect.right() - badge_width, region_rect.top() - badge_height - gap, badge_width, badge_height),
            QRectF(region_rect.right() + gap, centered_y, badge_width, badge_height),
            QRectF(region_rect.left() - badge_width - gap, centered_y, badge_width, badge_height),
            QRectF(region_rect.right() + gap, region_rect.top(), badge_width, badge_height),
            QRectF(region_rect.left() - badge_width - gap, region_rect.top(), badge_width, badge_height),
            QRectF(region_rect.left(), region_rect.bottom() + gap, badge_width, badge_height),
            QRectF(region_rect.right() - badge_width, region_rect.bottom() + gap, badge_width, badge_height),
        )

        ranked: list[tuple[float, QRectF]] = []
        badge_area = max(1.0, badge_width * badge_height)
        for order, candidate in enumerate(candidates):
            if not viewport_rect.contains(candidate):
                continue
            region_overlap = sum(
                cls._overlap_area(candidate, obstacle.adjusted(-1, -1, 1, 1))
                for obstacle in other_regions
            )
            badge_overlap = sum(
                cls._overlap_area(candidate, obstacle.adjusted(-2, -2, 2, 2))
                for obstacle in placed_badges
            )
            image_coverage = cls._overlap_area(candidate, image_rect) / badge_area
            score = (
                badge_overlap * 1_000_000.0
                + region_overlap * 100_000.0
                + image_coverage * 100.0
                + order
            )
            ranked.append((score, candidate))
        if ranked:
            return min(ranked, key=lambda item: item[0])[1]

        fallback = QRectF(
            min(
                max(viewport_rect.left(), region_rect.left()),
                viewport_rect.right() - badge_width,
            ),
            min(
                max(viewport_rect.top(), region_rect.top()),
                viewport_rect.bottom() - badge_height,
            ),
            badge_width,
            badge_height,
        )
        return fallback

    def _draw_regions(self, painter: QPainter) -> None:
        regions = list(self.window.session.get("regions", []))
        region_rects = [self._region_rect_view(region) for region in regions]
        region_styles: list[tuple[QColor, bool]] = []
        for index, (region, rect) in enumerate(zip(regions, region_rects)):
            selected = index == self.window.selected_index
            if not region.get("enabled", True):
                color = QColor(150, 150, 150, 190)
            elif selected:
                color = QColor(33, 150, 243, 235)
            else:
                color = QColor(255, 193, 7, 185)
            region_styles.append((color, selected))
            painter.setPen(QPen(color, 2 if selected else 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)
            if selected:
                painter.setBrush(QColor(33, 150, 243))
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                for handle in self._handle_rects(rect).values():
                    painter.drawRect(handle)

        painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        metrics = painter.fontMetrics()
        viewport_rect = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        image = self.window.display_image
        image_rect = QRectF(
            self.offset.x(),
            self.offset.y(),
            image.width * self.scale,
            image.height * self.scale,
        )
        placed_badges: list[QRectF] = []
        for index, rect in enumerate(region_rects):
            label = str(index + 1)
            badge_width = max(20.0, float(metrics.horizontalAdvance(label) + 10))
            badge_height = 17.0
            badge_rect = self._choose_region_badge_rect(
                rect,
                badge_width,
                badge_height,
                [other for position, other in enumerate(region_rects) if position != index],
                placed_badges,
                viewport_rect,
                image_rect,
            )
            placed_badges.append(badge_rect)
            _color, selected = region_styles[index]
            painter.setBrush(QColor(23, 32, 51, 235 if selected else 210))
            painter.setPen(QPen(QColor(15, 23, 42), 1))
            painter.drawRoundedRect(badge_rect, 5, 5)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, label)

    def _draw_stroke_preview(self, painter: QPainter) -> None:
        if len(self.stroke_points) == 0:
            return
        if self.tool == "paint_brush":
            color = QColor(self.window.paint_color)
            color.setAlpha(210)
        elif self.tool == "paint_eraser":
            color = QColor(0, 200, 255, 150)
        elif self.tool == "inpaint_brush":
            color = QColor(0, 190, 255, 175)
        elif self.tool == "inpaint_eraser":
            color = QColor(255, 255, 255, 145)
        elif self.tool == "restore_brush":
            color = QColor(76, 217, 100, 190)
        elif self.tool == "restore_eraser":
            color = QColor(255, 214, 10, 180)
        else:
            color = QColor(0, 150, 255, 150)
        painter.setPen(
            QPen(
                color,
                max(1.0, self.window.brush_size * self.scale),
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        points = [self.image_to_view(QPointF(x, y)) for x, y in self.stroke_points]
        if len(points) == 1:
            r = max(1.0, self.window.brush_size * self.scale / 2.0)
            painter.drawEllipse(points[0], r, r)
        else:
            for i in range(1, len(points)):
                painter.drawLine(points[i - 1], points[i])

    def _draw_brush_cursor(self, painter: QPainter) -> None:
        if self.hover_image_pos is None or self.tool not in {
            "inpaint_brush",
            "inpaint_eraser",
            "paint_brush",
            "paint_eraser",
            "restore_brush",
            "restore_eraser",
        }:
            return
        center = self.image_to_view(self.hover_image_pos)
        radius = max(2.0, self.window.brush_size * self.scale / 2.0)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(0, 0, 0, 210), 3.0))
        painter.drawEllipse(center, radius, radius)
        painter.setPen(QPen(QColor(255, 255, 255, 235), 1.0))
        painter.drawEllipse(center, radius, radius)

    def _draw_textbox_preview(self, painter: QPainter) -> None:
        if self.textbox_start is None or self.textbox_current is None:
            return
        p1 = self.image_to_view(self.textbox_start)
        p2 = self.image_to_view(self.textbox_current)
        rect = QRectF(p1, p2).normalized()
        painter.setPen(QPen(QColor(76, 175, 80), 2, Qt.PenStyle.DashLine))
        painter.drawRect(rect)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self.tool == "pan"
        ):
            self.pan_drag_start = QPointF(event.position())
            self.pan_drag_origin = QPointF(self.pan_offset)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        image_pos = self.view_to_image(QPointF(event.position()))
        self.hover_image_pos = image_pos
        if self.tool == "select":
            index, mode = self._hit_test(QPointF(event.position()))
            self.window.set_selected_index(index)
            self.drag_mode = mode
            self.drag_start_image = image_pos
            self.drag_start_view = QPointF(event.position())
            self.drag_has_moved = False
            self.drag_history_pushed = False
            region = self.window.selected_region()
            self.drag_start_bbox = _bbox(region) if region is not None else None
            self.update()
            return
        if self.tool == "textbox":
            self.textbox_start = image_pos
            self.textbox_current = image_pos
            self.update()
            return
        if self.tool in {
            "inpaint_brush",
            "inpaint_eraser",
            "paint_brush",
            "paint_eraser",
            "restore_brush",
            "restore_eraser",
        }:
            self.stroke_points = [(int(round(image_pos.x())), int(round(image_pos.y())))]
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.pan_drag_start is not None:
            movement = QPointF(event.position()) - self.pan_drag_start
            self.pan_offset = self.pan_drag_origin + movement
            self._fit_transform()
            self.update()
            event.accept()
            return
        image_pos = self.view_to_image(QPointF(event.position()))
        self.hover_image_pos = image_pos
        if self.drag_mode and self.drag_start_bbox is not None and self.tool == "select":
            movement = QPointF(event.position()) - self.drag_start_view
            if not self.drag_has_moved and (
                abs(movement.x()) + abs(movement.y())
            ) < 4.0:
                return
            if not self.drag_has_moved:
                self.drag_has_moved = True
                if self.drag_mode in {"move", "nw", "ne", "sw", "se"}:
                    label = (
                        "텍스트 박스 이동"
                        if self.drag_mode == "move"
                        else "텍스트 박스 크기 조절"
                    )
                    self.window._push_history(label)
                    self.drag_history_pushed = True
            self.window.update_selected_bbox_from_drag(self.drag_mode, self.drag_start_bbox, self.drag_start_image, image_pos)
            self.update()
            return
        if self.textbox_start is not None:
            self.textbox_current = image_pos
            self.update()
            return
        if self.stroke_points and self.tool in {
            "inpaint_brush",
            "inpaint_eraser",
            "paint_brush",
            "paint_eraser",
            "restore_brush",
            "restore_eraser",
        }:
            point = (int(round(image_pos.x())), int(round(image_pos.y())))
            if point != self.stroke_points[-1]:
                self.stroke_points.append(point)
                self.update()
            return
        self.update()

    def leaveEvent(self, event) -> None:
        self.hover_image_pos = None
        self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if (
            event.button()
            in {Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton}
            and self.pan_drag_start is not None
        ):
            self.pan_drag_start = None
            if self.tool == "pan":
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.unsetCursor()
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.drag_mode:
            self.window.commit_region_form(mark_dirty=self.drag_has_moved)
            self.drag_mode = None
            self.drag_start_bbox = None
            self.drag_has_moved = False
            self.drag_history_pushed = False
            return
        if self.textbox_start is not None and self.textbox_current is not None:
            self.window.create_textbox_from_drag(self.textbox_start, self.textbox_current)
            self.textbox_start = None
            self.textbox_current = None
            self.update()
            return
        if self.stroke_points:
            points = list(self.stroke_points)
            self.stroke_points = []
            self.window.apply_stroke(self.tool, points)
            self.update()


# THESIS: 캔버스를 중심에 두고 편집 도구와 설정을 왼쪽 한 방향으로 모은다.
# OWN-WORLD: 잉크색 작업 막대, 밝은 설정지, 파란색 선택 상태, 장식 없는 Qt 기본 조작어.
# STORY: 사용자는 도구를 고르고 같은 쪽에서 값을 조정한 뒤 중앙 결과를 즉시 확인한다.
# FIRST VIEWPORT: 얇은 왼쪽 도구 레일, 설정 패널, 넓은 중앙 캔버스, 최소 상단 작업 표시줄.
# FORM: Tubik/Ribbet식 데스크톱 편집기 표준 구조, canon:tubik-left-editor.
# FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, and DESIGN.md
class AdvancedEditorWindow(QMainWindow):
    def __init__(self, session_path: str):
        super().__init__()
        self.session_path = Path(session_path)
        self.session = load_session(self.session_path)
        self.related_session_paths = related_edit_session_paths(
            self.session_path, self.session
        )
        self.editor_settings = _load_editor_settings()
        self.help_widgets: list[QWidget] = []
        self._dirty_region_ids: set[str] = {
            str(region_id)
            for region_id in self.session.get("editor_dirty_region_ids", []) or []
            if str(region_id)
        }
        self.session["editor_dirty_region_ids"] = sorted(self._dirty_region_ids)
        self.startup_warnings: list[str] = []
        self.base_image = self._load_base_image()
        self.display_image = self.base_image.copy()
        self.display_qimage: Optional[QImage] = _pil_to_qimage(self.display_image)
        self.image_size = self.base_image.size
        self.zoom_factor = 1.0
        self.displaying_rendered_result = False
        self._start_maximized = False
        self.selected_index: Optional[int] = 0 if self.session.get("regions") else None
        self.brush_size = 30
        self.paint_color = "#ffffff"
        self.history: list[dict[str, Any]] = []
        self.redo_stack: list[dict[str, Any]] = []
        self._syncing = False
        self._switching_session = False
        self._syncing_canvas_scrollbars = False
        self._font_family_cache: dict[str, str] = {}
        self._font_style_cache: dict[str, str] = {}
        self._font_catalog: dict[str, dict[str, Any]] = {}
        self._font_path_index: dict[str, tuple[str, str]] = {}
        self._font_change_requested = False
        self._task_signals: list[TaskSignals] = []
        self._background_job_count = 0
        self._inpainting_running = False
        self._preview_request_serial = 0
        self._preview_running = False
        self._preview_pending = False
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(250)
        self._preview_timer.timeout.connect(self.refresh_exact_preview)
        self._translation_history_active = False
        self._translation_history_timer = QTimer(self)
        self._translation_history_timer.setSingleShot(True)
        self._translation_history_timer.setInterval(800)
        self._translation_history_timer.timeout.connect(
            lambda: setattr(self, "_translation_history_active", False)
        )
        self._save_toast_timer = QTimer(self)
        self._save_toast_timer.setSingleShot(True)
        self._save_toast_timer.setInterval(2400)
        self._save_toast_timer.timeout.connect(self._hide_save_toast)

        self.setWindowTitle(f"MangaTranslator 고급 편집기 - {self.session_path.name}")
        self.setMinimumSize(1200 if len(self.related_session_paths) > 1 else 1120, 680)
        self._build_ui()
        self.apply_editor_preset(1, persist=False, show_status=False)
        self._fit_initial_window()
        self.reload_display_image()
        self.refresh_region_list()
        self.load_selected_region_to_form()
        recovery_warning = str(self.session.get("asset_recovery_warning") or "").strip()
        if recovery_warning:
            self.startup_warnings.append(recovery_warning)
        if self.startup_warnings:
            self.status_label.setText("\n".join(self.startup_warnings))

    def _fit_initial_window(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1480, 920)
            if hasattr(self, "splitter"):
                if len(self.related_session_paths) > 1:
                    self.splitter.setSizes([410, 860, 210])
                else:
                    self.splitter.setSizes([410, 1070, 0])
            return
        available = screen.availableGeometry()
        minimum_width = 1200 if len(self.related_session_paths) > 1 else 1120
        width = min(1680, max(minimum_width, int(available.width() * 0.96)))
        height = min(1040, max(680, int(available.height() * 0.94)))
        width = min(width, available.width())
        height = min(height, available.height())
        self.resize(width, height)
        if hasattr(self, "splitter"):
            inspector_width = min(450, max(360, int(width * 0.25)))
            list_width = 210 if len(self.related_session_paths) > 1 else 0
            self.splitter.setSizes(
                [
                    inspector_width,
                    max(480, width - inspector_width - list_width),
                    list_width,
                ]
            )
        self.move(
            available.x() + max(0, (available.width() - width) // 2),
            available.y() + max(0, (available.height() - height) // 2),
        )
        self._start_maximized = available.width() < 1280 or available.height() < 760

    def _load_base_image(self) -> Image.Image:
        candidates = (
            ("inpainted_image_path", "인페인팅 결과"),
            ("base_image_path", "편집용 배경 이미지"),
            ("rendered_snapshot_path", "보존된 번역 결과"),
            ("translated_image_path", "번역 결과"),
            ("original_snapshot_path", "보존된 원본"),
            ("source_image_path", "원본 이미지"),
        )
        for key, label in candidates:
            value = self.session.get(key)
            if value and Path(value).exists():
                if key not in {"base_image_path", "inpainted_image_path"}:
                    self.startup_warnings.append(
                        f"편집용 배경 이미지가 없어 {label}을 사용합니다."
                    )
                return Image.open(value).convert("RGBA")
        raise FileNotFoundError(
            "편집에 필요한 이미지 파일을 찾을 수 없습니다. 편집 세션을 다시 만들어 주세요."
        )

    def _load_ocr_image(self) -> Image.Image:
        for key in (
            "original_snapshot_path",
            "source_image_path",
            "rendered_snapshot_path",
            "translated_image_path",
            "base_image_path",
        ):
            value = self.session.get(key)
            if value and Path(value).is_file():
                image = Image.open(value).convert("RGBA")
                if image.size != self.image_size:
                    image = image.resize(self.image_size, Image.Resampling.LANCZOS)
                return image
        return self._load_base_image().copy()

    def _build_ui(self) -> None:
        self._apply_theme()

        tool_toolbar = QToolBar("편집 도구", self)
        tool_toolbar.setObjectName("ToolRail")
        tool_toolbar.setMovable(False)
        tool_toolbar.setFloatable(False)
        tool_toolbar.setAllowedAreas(Qt.ToolBarArea.LeftToolBarArea)
        tool_toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tool_toolbar.setIconSize(QSize(20, 20))
        tool_toolbar.setMinimumWidth(160)
        tool_toolbar.setMaximumWidth(180)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, tool_toolbar)
        self.tool_toolbar = tool_toolbar

        self.tool_actions: dict[str, QAction] = {}
        tool_icons = {
            "select": "select",
            "textbox": "textbox",
            "inpaint_brush": "brush",
            "inpaint_eraser": "eraser",
            "paint_brush": "brush",
            "paint_eraser": "eraser",
            "restore_brush": "restore",
            "restore_eraser": "eraser",
        }
        for key, label in TOOLS.items():
            action = QAction(
                _editor_icon(tool_icons[key]), TOOL_RAIL_LABELS[key], self
            )
            action.setToolTip(label)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, tool=key: self.set_tool(tool))
            self.tool_actions[key] = action
            tool_toolbar.addAction(action)
        self.tool_actions["select"].setChecked(True)

        action_toolbar = QToolBar("작업", self)
        action_toolbar.setObjectName("TopBar")
        action_toolbar.setMovable(False)
        action_toolbar.setFloatable(False)
        action_toolbar.setAllowedAreas(Qt.ToolBarArea.TopToolBarArea)
        action_toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        action_toolbar.setIconSize(QSize(18, 18))
        action_toolbar.setMinimumHeight(50)
        action_toolbar.setMaximumHeight(50)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, action_toolbar)
        self.action_toolbar = action_toolbar

        self.undo_action = QAction(_editor_icon("undo"), "실행 취소", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(self.undo)
        action_toolbar.addAction(self.undo_action)
        self.redo_action = QAction(_editor_icon("redo"), "다시 실행", self)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.triggered.connect(self.redo)
        action_toolbar.addAction(self.redo_action)

        self.history_menu = QMenu(self)
        self.history_button = QToolButton(self)
        self.history_button.setText("작업 기록")
        self.history_button.setIcon(_editor_icon("history"))
        self.history_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.history_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.history_button.setMenu(self.history_menu)
        action_toolbar.addWidget(self.history_button)
        action_toolbar.addSeparator()

        brush_label = QLabel("브러시 크기")
        brush_label.setObjectName("TopBarLabel")
        action_toolbar.addWidget(brush_label)
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setObjectName("TopBarBrushSlider")
        self.brush_slider.setRange(5, 200)
        self.brush_slider.setValue(self.brush_size)
        self.brush_slider.setFixedWidth(120)
        self.brush_slider.valueChanged.connect(self._on_brush_size_changed)
        action_toolbar.addWidget(self.brush_slider)
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(5, 200)
        self.brush_size_spin.setSuffix(" px")
        self.brush_size_spin.setValue(self.brush_size)
        self.brush_size_spin.setFixedWidth(76)
        self.brush_size_spin.valueChanged.connect(self.brush_slider.setValue)
        self.brush_slider.valueChanged.connect(self.brush_size_spin.setValue)
        action_toolbar.addWidget(self.brush_size_spin)

        toolbar_spacer = QWidget(self)
        toolbar_spacer.setObjectName("TopBarSpacer")
        toolbar_spacer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        action_toolbar.addWidget(toolbar_spacer)

        self.render_action = QAction(_editor_icon("save"), "저장  Ctrl+S", self)
        self.render_action.setShortcut(QKeySequence("Ctrl+S"))
        self.render_action.setToolTip("편집 세션과 결과 저장 (Ctrl+S)")
        self.render_action.triggered.connect(self.render_and_save)
        action_toolbar.addAction(self.render_action)

        self.plain_render_action = QAction(_editor_icon("export"), "내보내기", self)
        self.plain_render_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.plain_render_action.setToolTip("편집 정보가 없는 이미지로 내보내기 (Ctrl+Shift+S)")
        self.plain_render_action.triggered.connect(self.save_plain_rendered_image)
        action_toolbar.addAction(self.plain_render_action)
        save_button = action_toolbar.widgetForAction(self.render_action)
        if save_button is not None:
            save_button.setProperty("toolbarRole", "primary")

        central = QWidget(self)
        central.setObjectName("CentralSurface")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(1)
        root.addWidget(self.splitter)

        self.splitter.addWidget(self._build_side_panel())

        canvas_panel = QWidget(self)
        canvas_panel.setObjectName("CanvasPanel")
        canvas_layout = QVBoxLayout(canvas_panel)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)
        zoom_bar = QWidget(canvas_panel)
        zoom_bar.setObjectName("ZoomBar")
        zoom_layout = QHBoxLayout(zoom_bar)
        zoom_layout.setContentsMargins(8, 5, 8, 5)
        zoom_layout.setSpacing(6)

        self.pan_tool_button = QPushButton("화면 이동")
        self.pan_tool_button.setCheckable(True)
        self.pan_tool_button.setToolTip(
            "선택한 뒤 이미지 위에서 왼쪽 마우스로 드래그해 화면을 이동합니다."
        )
        self.pan_tool_button.clicked.connect(lambda: self.set_tool("pan"))
        zoom_layout.addWidget(self.pan_tool_button)

        zoom_out_button = QPushButton("-")
        zoom_out_button.setFixedSize(34, 30)
        zoom_out_button.setShortcut(QKeySequence("Ctrl+-"))
        zoom_out_button.setToolTip("축소 (마우스 휠 아래)")
        zoom_out_button.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(zoom_out_button)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(58)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setToolTip(
            "현재 표시 배율입니다. 휠로 확대/축소하고 화면 이동 도구 또는 스크롤바로 이동합니다."
        )
        zoom_layout.addWidget(self.zoom_label)

        zoom_in_button = QPushButton("+")
        zoom_in_button.setFixedSize(34, 30)
        zoom_in_button.setShortcut(QKeySequence("Ctrl+="))
        zoom_in_button.setToolTip("확대 (마우스 휠 위)")
        zoom_in_button.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(zoom_in_button)

        fit_button = QPushButton("맞춤")
        fit_button.setToolTip("이미지 전체가 캔버스에 보이도록 맞춥니다.")
        fit_button.clicked.connect(self.zoom_fit)
        zoom_layout.addWidget(fit_button)

        actual_size_button = QPushButton("100%")
        actual_size_button.setShortcut(QKeySequence("Ctrl+1"))
        actual_size_button.setToolTip(
            "이미지 1픽셀을 화면 1픽셀 크기로 표시합니다."
        )
        actual_size_button.clicked.connect(self.zoom_actual_size)
        zoom_layout.addWidget(actual_size_button)
        zoom_layout.addStretch(1)

        canvas_layout.addWidget(zoom_bar)
        canvas_view = QWidget(canvas_panel)
        canvas_grid = QGridLayout(canvas_view)
        canvas_grid.setContentsMargins(0, 0, 0, 0)
        canvas_grid.setSpacing(0)

        self.canvas = EditorCanvas(self)
        self.canvas_vertical_scrollbar = QScrollBar(Qt.Orientation.Vertical)
        self.canvas_vertical_scrollbar.setObjectName("CanvasScrollBar")
        self.canvas_vertical_scrollbar.setSingleStep(40)
        self.canvas_vertical_scrollbar.setPageStep(300)
        self.canvas_vertical_scrollbar.valueChanged.connect(
            self._on_canvas_vertical_scroll
        )
        self.canvas_horizontal_scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.canvas_horizontal_scrollbar.setObjectName("CanvasScrollBar")
        self.canvas_horizontal_scrollbar.setSingleStep(40)
        self.canvas_horizontal_scrollbar.setPageStep(300)
        self.canvas_horizontal_scrollbar.valueChanged.connect(
            self._on_canvas_horizontal_scroll
        )
        scroll_corner = QWidget(canvas_view)
        scroll_corner.setObjectName("ScrollCorner")
        scroll_corner.setFixedSize(
            self.canvas_vertical_scrollbar.sizeHint().width(),
            self.canvas_horizontal_scrollbar.sizeHint().height(),
        )

        canvas_grid.addWidget(self.canvas, 0, 0)
        canvas_grid.addWidget(self.canvas_vertical_scrollbar, 0, 1)
        canvas_grid.addWidget(self.canvas_horizontal_scrollbar, 1, 0)
        canvas_grid.addWidget(scroll_corner, 1, 1)
        canvas_grid.setRowStretch(0, 1)
        canvas_grid.setColumnStretch(0, 1)
        canvas_layout.addWidget(canvas_view, 1)
        self.splitter.addWidget(canvas_panel)
        self.session_panel = self._build_session_panel()
        self.splitter.addWidget(self.session_panel)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self._apply_button_icons(central)
        self._refresh_history_menu()

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #edf2f7;
            }
            QWidget {
                color: #172033;
                font-family: "Segoe UI", "Malgun Gothic";
                font-size: 12px;
            }
            QWidget#CentralSurface {
                background: #edf2f7;
            }
            QToolBar#TopBar {
                background: #0f172a;
                border: 0;
                border-bottom: 1px solid #263244;
                spacing: 5px;
                padding: 6px 12px;
            }
            QToolBar#TopBar QToolButton {
                color: #e5e7eb;
                background: transparent;
                border: 1px solid transparent;
                border-radius: 6px;
                min-height: 22px;
                padding: 6px 9px;
            }
            QToolBar#TopBar QToolButton:hover {
                background: #273449;
                border-color: #3b4a60;
            }
            QToolBar#TopBar QToolButton:disabled {
                color: #64748b;
                background: transparent;
                border-color: transparent;
            }
            QToolBar#TopBar QToolButton[toolbarRole="primary"] {
                background: #2563eb;
                border-color: #60a5fa;
                color: #ffffff;
                font-weight: 700;
                padding-left: 12px;
                padding-right: 12px;
            }
            QToolBar#TopBar QToolButton[toolbarRole="primary"]:hover {
                background: #1d4ed8;
                border-color: #93c5fd;
            }
            QToolBar#TopBar QLabel#TopBarLabel {
                color: #e5e7eb;
                background: transparent;
                padding-left: 4px;
            }
            QToolBar#TopBar QWidget#TopBarSpacer {
                background: transparent;
                border: 0;
            }
            QToolBar#TopBar QSpinBox {
                min-height: 20px;
                padding: 4px 6px;
                background: #ffffff;
                border: 1px solid #64748b;
                border-radius: 5px;
            }
            QToolBar#TopBar QSlider::groove:horizontal {
                height: 4px;
                background: #475569;
                border-radius: 2px;
            }
            QToolBar#TopBar QSlider::sub-page:horizontal {
                background: #60a5fa;
                border-radius: 2px;
            }
            QToolBar#TopBar QSlider::handle:horizontal {
                width: 14px;
                margin: -5px 0;
                background: #ffffff;
                border: 2px solid #2563eb;
                border-radius: 7px;
            }
            QToolBar#ToolRail {
                background: #111827;
                border: 0;
                border-right: 1px solid #263244;
                spacing: 4px;
                padding: 12px 9px;
            }
            QToolBar#ToolRail QToolButton {
                color: #cbd5e1;
                background: transparent;
                border: 1px solid transparent;
                border-radius: 6px;
                min-height: 26px;
                padding: 8px 10px;
                min-width: 142px;
                text-align: left;
            }
            QToolBar#ToolRail QToolButton:hover {
                color: #ffffff;
                background: #1f2937;
                border-color: #334155;
            }
            QToolBar#ToolRail QToolButton:checked {
                background: #2563eb;
                border-color: #60a5fa;
                color: #ffffff;
                font-weight: 700;
            }
            QToolBar#ToolRail QToolButton:focus,
            QToolBar#TopBar QToolButton:focus {
                border-color: #93c5fd;
            }
            QWidget#InspectorShell,
            QScrollArea#InspectorScroll,
            QWidget#InspectorPanel {
                background: #f4f7fb;
            }
            QWidget#InspectorShell {
                border-right: 1px solid #d5dde8;
            }
            QScrollArea#InspectorScroll {
                border: 0;
            }
            QWidget#CanvasPanel {
                background: #1c222b;
            }
            QSplitter::handle {
                background: #d5dde8;
            }
            QSplitter::handle:hover {
                background: #60a5fa;
            }
            QWidget#ZoomBar {
                background: #252d3a;
                border-bottom: 1px solid #3a4657;
            }
            QWidget#ZoomBar QLabel {
                color: #f8fafc;
                background: transparent;
                font-weight: 700;
            }
            QWidget#ZoomBar QPushButton {
                min-height: 20px;
                padding: 4px 9px;
                color: #f8fafc;
                background: #364152;
                border: 1px solid #4b5a6f;
                border-radius: 5px;
            }
            QWidget#ZoomBar QPushButton:hover {
                background: #475569;
                border-color: #93c5fd;
            }
            QWidget#ZoomBar QPushButton:checked {
                background: #2563eb;
                border-color: #93c5fd;
                color: #ffffff;
            }
            QScrollBar#CanvasScrollBar:vertical {
                width: 18px;
                background: #d7dde6;
                border-left: 1px solid #aeb8c6;
            }
            QScrollBar#CanvasScrollBar:horizontal {
                height: 18px;
                background: #d7dde6;
                border-top: 1px solid #aeb8c6;
            }
            QScrollBar#CanvasScrollBar::handle:vertical {
                min-height: 36px;
                margin: 2px;
                background: #64748b;
                border-radius: 6px;
            }
            QScrollBar#CanvasScrollBar::handle:horizontal {
                min-width: 36px;
                margin: 2px;
                background: #64748b;
                border-radius: 6px;
            }
            QScrollBar#CanvasScrollBar::handle:hover {
                background: #2563eb;
            }
            QScrollBar#CanvasScrollBar::add-line,
            QScrollBar#CanvasScrollBar::sub-line {
                width: 0;
                height: 0;
            }
            QWidget#ScrollCorner {
                background: #d7dde6;
                border-top: 1px solid #aeb8c6;
                border-left: 1px solid #aeb8c6;
            }
            QScrollArea {
                border: 0;
                background: #f4f7fb;
            }
            QLabel {
                color: #334155;
                font-weight: 500;
            }
            QLabel#SectionTitle,
            QLabel#PanelTitle {
                color: #172033;
                font-size: 13px;
                font-weight: 700;
                padding-top: 3px;
                padding-bottom: 6px;
                border-bottom: 1px solid #d5dde8;
            }
            QTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #ffffff;
                color: #172033;
                border: 1px solid #c4cfdd;
                border-radius: 6px;
                min-height: 20px;
                padding: 6px 8px;
                selection-background-color: #bfdbfe;
            }
            QTextEdit:focus, QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #2563eb;
            }
            QPushButton {
                background: #ffffff;
                color: #172033;
                border: 1px solid #c4cfdd;
                border-radius: 6px;
                min-height: 20px;
                padding: 7px 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #eff6ff;
                border-color: #60a5fa;
            }
            QPushButton:pressed {
                background: #dbeafe;
            }
            QPushButton:focus {
                border: 1px solid #2563eb;
            }
            QPushButton:disabled {
                background: #edf1f6;
                border-color: #d5dde8;
                color: #94a3b8;
            }
            QTabWidget::pane {
                border: 1px solid #d5dde8;
                border-radius: 8px;
                background: #ffffff;
                margin-top: -1px;
                padding: 2px;
            }
            QTabBar::tab {
                background: transparent;
                color: #64748b;
                border: 1px solid transparent;
                border-bottom: 2px solid transparent;
                padding: 7px 9px;
                margin-right: 1px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:hover {
                color: #1d4ed8;
                background: #eff6ff;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #1d4ed8;
                border-color: #d5dde8;
                border-bottom-color: #2563eb;
                font-weight: 700;
            }
            QCheckBox {
                spacing: 8px;
                color: #1f2937;
                min-height: 22px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #cbd5e1;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
                background: #2563eb;
            }
            QLabel#StatusLabel {
                background: #eef6ff;
                color: #1e3a8a;
                border: 1px solid #bfdbfe;
                border-radius: 7px;
                margin: 8px 10px 10px 10px;
                padding: 9px 10px;
                font-weight: 500;
            }
            QLabel#HelpLabel {
                background: #ffffff;
                color: #475569;
                border: 1px solid #d5dde8;
                border-radius: 6px;
                padding: 9px 10px;
                font-size: 11px;
                font-weight: 400;
            }
            QWidget#SessionPanel {
                background: #f8fafc;
                border-left: 1px solid #d5dde8;
            }
            QListWidget {
                background: #ffffff;
                color: #334155;
                border: 1px solid #d5dde8;
                border-radius: 7px;
                outline: 0;
                padding: 4px;
            }
            QListWidget::item {
                min-height: 26px;
                padding: 6px 8px;
                border-radius: 5px;
            }
            QListWidget::item:hover {
                background: #eff6ff;
            }
            QListWidget::item:selected {
                background: #dbeafe;
                color: #1d4ed8;
                font-weight: 700;
            }
            QMenu {
                background: #ffffff;
                color: #172033;
                border: 1px solid #c4cfdd;
                padding: 5px;
            }
            QMenu::item {
                padding: 7px 24px 7px 10px;
                border-radius: 5px;
            }
            QMenu::item:selected {
                background: #eff6ff;
                color: #1d4ed8;
            }
            QToolTip {
                background: #172033;
                color: #ffffff;
                border: 1px solid #334155;
                padding: 6px 8px;
            }
            QFrame#SaveToast {
                background: #172033;
                border: 1px solid #334155;
                border-radius: 8px;
            }
            QFrame#SaveToast QLabel {
                background: transparent;
                border: 0;
            }
            QLabel#SaveToastText {
                color: #ffffff;
                font-size: 12px;
                font-weight: 600;
            }
            QScrollBar:vertical {
                width: 10px;
                background: transparent;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                min-height: 32px;
                background: #a8b4c4;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #64748b;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                height: 0;
                background: transparent;
            }
            QScrollBar:horizontal {
                height: 10px;
                background: transparent;
                margin: 2px;
            }
            QScrollBar::handle:horizontal {
                min-width: 32px;
                background: #a8b4c4;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #64748b;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                width: 0;
                background: transparent;
            }
            """
        )

    def _help_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("HelpLabel")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.help_widgets.append(label)
        return label

    def _position_save_toast(self) -> None:
        if not hasattr(self, "save_toast"):
            return
        margin = 18
        self.save_toast.move(
            max(margin, self.width() - self.save_toast.width() - margin),
            max(margin, self.height() - self.save_toast.height() - margin),
        )

    def _show_save_toast(self, text: str) -> None:
        if not hasattr(self, "save_toast"):
            self.save_toast = QFrame(self)
            self.save_toast.setObjectName("SaveToast")
            toast_layout = QHBoxLayout(self.save_toast)
            toast_layout.setContentsMargins(12, 9, 14, 9)
            toast_layout.setSpacing(8)
            icon_label = QLabel(self.save_toast)
            icon_label.setPixmap(_editor_icon("save", "#ffffff").pixmap(18, 18))
            toast_layout.addWidget(icon_label)
            self.save_toast_text = QLabel(self.save_toast)
            self.save_toast_text.setObjectName("SaveToastText")
            toast_layout.addWidget(self.save_toast_text)
        self.save_toast_text.setText(text)
        self.save_toast.adjustSize()
        self._position_save_toast()
        self.save_toast.raise_()
        self.save_toast.show()
        self._save_toast_timer.start()

    def _hide_save_toast(self) -> None:
        if hasattr(self, "save_toast"):
            self.save_toast.hide()

    @staticmethod
    def _button_icon_name(text: str) -> str:
        label = str(text or "").strip()
        rules = (
            (("편집 정보 제거", "내보내기"), "export"),
            (("새로고침",), "refresh"),
            (("초기화", "되돌리기"), "reset"),
            (("지우개",), "eraser"),
            (("삭제", "지우기", "비우기"), "trash"),
            (("추가", "+"), "add"),
            (("찾기", "폴더", "경로"), "folder"),
            (("저장",), "save"),
            (("OCR", "번역", "검출", "실행", "적용"), "play"),
            (("색",), "palette"),
            (("굵",), "bold"),
            (("기울",), "italic"),
            (("화면 이동",), "hand"),
            (("맞춤", "100%"), "fit"),
            (("-",), "minus"),
        )
        for keywords, icon_name in rules:
            if any(keyword == label or keyword in label for keyword in keywords):
                return icon_name
        return "action"

    def _apply_button_icons(self, root: QWidget) -> None:
        for button in root.findChildren(QPushButton):
            color = (
                "#f8fafc"
                if button.parentWidget()
                and button.parentWidget().objectName() == "ZoomBar"
                else "#172033"
            )
            button.setIcon(_editor_icon(self._button_icon_name(button.text()), color))
            button.setIconSize(QSize(17, 17))

    def _apply_help_visibility(self) -> None:
        visible = bool(
            not hasattr(self, "show_help_checkbox")
            or self.show_help_checkbox.isChecked()
        )
        for widget in self.help_widgets:
            widget.setVisible(visible)
        if hasattr(self, "inpaint_method_combo"):
            self._update_inpaint_option_visibility()

    def _on_show_help_changed(self, *_args) -> None:
        self._apply_help_visibility()
        if self._syncing:
            return
        self.editor_settings["show_help"] = self.show_help_checkbox.isChecked()
        _save_editor_settings(self.editor_settings)
        self._store_editor_controls()

    def _saved_section(self, name: str) -> dict[str, Any]:
        value = self.editor_settings.get(name, {})
        return dict(value) if isinstance(value, dict) else {}

    def _remember_settings(self, name: str, values: dict[str, Any]) -> None:
        stored = dict(values)
        stored.pop("huggingface_token", None)
        stored.pop("api_key_override", None)
        self.editor_settings[name] = stored
        _save_editor_settings(self.editor_settings)

    def _editor_setting_value(self, key: str, default: Any) -> Any:
        editor = self._saved_section("editor")
        if key in editor:
            return editor[key]
        if key == "show_help":
            return self.editor_settings.get("show_help", default)
        return default

    def _editor_ui_settings(self) -> dict[str, Any]:
        return {
            "show_help": bool(
                self.show_help_checkbox.isChecked()
                if hasattr(self, "show_help_checkbox")
                else self._editor_setting_value("show_help", True)
            ),
            "show_inpaint_mask": bool(
                self.show_inpaint_mask_checkbox.isChecked()
                if hasattr(self, "show_inpaint_mask_checkbox")
                else self._editor_setting_value("show_inpaint_mask", True)
            ),
            "show_restore_mask": bool(
                self.show_restore_mask_checkbox.isChecked()
                if hasattr(self, "show_restore_mask_checkbox")
                else self._editor_setting_value("show_restore_mask", True)
            ),
        }

    def _saved_output_settings(self) -> dict[str, Any]:
        output = self.editor_settings.get("output", {})
        return dict(output) if isinstance(output, dict) else {}

    def _initial_custom_output_path(self) -> str:
        saved = self._saved_output_settings()
        if "custom_output_path" in saved:
            return str(saved.get("custom_output_path") or "").strip()
        return str(self.session.get("custom_output_path") or "").strip()

    def _initial_overwrite_source_on_save(self) -> bool:
        saved = self._saved_output_settings()
        if "overwrite_source_on_save" in saved:
            return bool(saved.get("overwrite_source_on_save"))
        return bool(self.session.get("overwrite_source_on_save", False))

    def _restore_output_widgets_from_saved_settings(self) -> None:
        if hasattr(self, "output_path_edit"):
            self.output_path_edit.setText(self._initial_custom_output_path())
        if hasattr(self, "overwrite_source_checkbox"):
            self.overwrite_source_checkbox.setChecked(
                self._initial_overwrite_source_on_save()
            )

    def _store_editor_controls(self, *_args) -> None:
        if self._syncing:
            return
        settings = self._editor_ui_settings()
        self.editor_settings["editor"] = settings
        self.editor_settings["show_help"] = settings["show_help"]
        _save_editor_settings(self.editor_settings)
        if hasattr(self, "canvas"):
            self.canvas.update()

    def _mark_region_dirty(self, region: Optional[dict[str, Any]] = None) -> None:
        target = region or self.selected_region()
        if target is None:
            return
        region_id = str(target.get("id") or "")
        if not region_id:
            return
        self._dirty_region_ids.add(region_id)
        self.session["editor_dirty_region_ids"] = sorted(self._dirty_region_ids)

    def _session_display_name(
        self, path: Path, collection_root: Optional[Path] = None
    ) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            display_path = str(data.get("source_display_path") or "").strip()
            if display_path:
                return display_path
            for key in ("translated_image_path", "source_image_path"):
                value = data.get(key)
                if value:
                    image_path = Path(str(value))
                    if collection_root is not None:
                        try:
                            return str(image_path.resolve().relative_to(collection_root))
                        except Exception:
                            pass
                    return image_path.name
        except Exception:
            pass
        name = path.stem
        return name[: -len("_edit_session")] if name.endswith("_edit_session") else name

    def _build_session_panel(self) -> QWidget:
        panel = QWidget(self)
        panel.setObjectName("SessionPanel")
        panel.setMinimumWidth(190)
        panel.setMaximumWidth(280)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 16, 12, 16)
        layout.setSpacing(10)

        title = QLabel(f"편집 이미지 ({len(self.related_session_paths)}장)")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        self.session_list = QListWidget()
        try:
            collection_root = Path(
                os.path.commonpath(
                    [str(path.parent.resolve()) for path in self.related_session_paths]
                )
            )
        except Exception:
            collection_root = None
        for index, path in enumerate(self.related_session_paths, start=1):
            self.session_list.addItem(
                f"{index}. {self._session_display_name(path, collection_root)}"
            )
            item = self.session_list.item(self.session_list.count() - 1)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
        self.session_list.currentRowChanged.connect(self._on_session_list_changed)
        layout.addWidget(self.session_list, 1)

        self._set_session_list_current(self.session_path)
        panel.setVisible(len(self.related_session_paths) > 1)
        return panel

    def _set_session_list_current(self, path: Path) -> None:
        if not hasattr(self, "session_list"):
            return
        target = str(path.resolve()).casefold()
        self.session_list.blockSignals(True)
        try:
            for row in range(self.session_list.count()):
                item = self.session_list.item(row)
                value = str(item.data(Qt.ItemDataRole.UserRole) or "")
                if str(Path(value).resolve()).casefold() == target:
                    self.session_list.setCurrentRow(row)
                    break
        finally:
            self.session_list.blockSignals(False)

    def _on_session_list_changed(self, row: int) -> None:
        if self._switching_session or row < 0:
            return
        item = self.session_list.item(row)
        if item is None:
            return
        target = Path(str(item.data(Qt.ItemDataRole.UserRole) or "")).resolve()
        if target == self.session_path.resolve():
            return
        if self._background_job_count > 0 or self._inpainting_running:
            self._set_session_list_current(self.session_path)
            QMessageBox.information(
                self,
                "작업 진행 중",
                "OCR, 번역 또는 인페인팅 작업이 끝난 뒤 다른 이미지를 선택해 주세요.",
            )
            return
        self._switch_editor_session(target)

    def _switch_editor_session(self, target: Path) -> None:
        previous = self.session_path
        self._switching_session = True
        try:
            self.commit_region_form(mark_dirty=False)
            self._persist_editor_control_settings()
            self._persist_output_settings()
            save_session(self.session_path, self.session)

            self._preview_timer.stop()
            self._preview_request_serial += 1
            self.session_path = target
            self.session = load_session(target)
            self._dirty_region_ids = {
                str(region_id)
                for region_id in self.session.get("editor_dirty_region_ids", []) or []
                if str(region_id)
            }
            self.session["editor_dirty_region_ids"] = sorted(self._dirty_region_ids)
            self.startup_warnings = []
            self.base_image = self._load_base_image()
            self.display_image = self.base_image.copy()
            self.display_qimage = _pil_to_qimage(self.display_image)
            self.image_size = self.base_image.size
            self.displaying_rendered_result = False
            self.selected_index = 0 if self.session.get("regions") else None
            self.zoom_factor = 1.0
            self.canvas.pan_offset = QPointF(0.0, 0.0)
            self.history.clear()
            self.redo_stack.clear()

            self._restore_output_widgets_from_saved_settings()
            self.reload_display_image()
            self.refresh_region_list()
            self.load_selected_region_to_form()
            self._refresh_history_menu()
            self.setWindowTitle(
                f"MangaTranslator 고급 편집기 - {self.session_path.name}"
            )
            self._set_session_list_current(self.session_path)
            self.schedule_exact_preview()
            self.status_label.setText(
                f"편집 이미지를 전환했습니다: {self._session_display_name(target)}"
            )
            self.canvas.setFocus()
        except Exception as exc:
            try:
                self.session_path = previous
                self.session = load_session(previous)
                self._dirty_region_ids = {
                    str(region_id)
                    for region_id in self.session.get("editor_dirty_region_ids", []) or []
                    if str(region_id)
                }
                self.startup_warnings = []
                self.base_image = self._load_base_image()
                self.display_image = self.base_image.copy()
                self.display_qimage = _pil_to_qimage(self.display_image)
                self.image_size = self.base_image.size
                self.selected_index = 0 if self.session.get("regions") else None
                self.reload_display_image()
                self.refresh_region_list()
                self.load_selected_region_to_form()
                self._restore_output_widgets_from_saved_settings()
            except Exception:
                pass
            finally:
                self._set_session_list_current(previous)
            QMessageBox.critical(self, "이미지 전환 실패", str(exc))
        finally:
            self._switching_session = False

    def _build_side_panel(self) -> QWidget:
        panel = QWidget(self)
        panel.setObjectName("InspectorPanel")
        panel.setMinimumWidth(340)
        panel.setMaximumWidth(450)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 18)
        layout.setSpacing(10)

        region_title = QLabel("영역과 텍스트")
        region_title.setObjectName("SectionTitle")
        layout.addWidget(region_title)
        layout.addWidget(QLabel("텍스트 영역"))
        self.region_combo = QComboBox()
        self.region_combo.currentIndexChanged.connect(self._on_region_combo_changed)
        layout.addWidget(self.region_combo)

        button_row = QHBoxLayout()
        add_button = QPushButton("추가")
        add_button.clicked.connect(self.add_text_region)
        delete_button = QPushButton("삭제")
        delete_button.clicked.connect(self.delete_selected_region)
        button_row.addWidget(add_button)
        button_row.addWidget(delete_button)
        layout.addLayout(button_row)

        self.source_edit = ResizableTextEdit(68)
        self.source_edit.setReadOnly(True)
        layout.addWidget(QLabel("원문"))
        layout.addWidget(self.source_edit)

        layout.addWidget(QLabel("편집할 텍스트"))
        self.translation_edit = ResizableTextEdit(82)
        self.translation_edit.textChanged.connect(self._on_translation_text_changed)
        layout.addWidget(self.translation_edit)

        style_row = QHBoxLayout()
        bold_button = QPushButton("굵게")
        bold_button.setToolTip("선택한 텍스트를 굵게 표시합니다.")
        bold_button.clicked.connect(lambda: self.wrap_translation_selection("**"))
        italic_button = QPushButton("기울임")
        italic_button.setToolTip("선택한 텍스트를 기울임으로 표시합니다.")
        italic_button.clicked.connect(lambda: self.wrap_translation_selection("*"))
        bold_italic_button = QPushButton("굵게+기울임")
        bold_italic_button.setToolTip("선택한 텍스트를 굵게+기울임으로 표시합니다.")
        bold_italic_button.clicked.connect(lambda: self.wrap_translation_selection("***"))
        style_row.addWidget(bold_button)
        style_row.addWidget(italic_button)
        style_row.addWidget(bold_italic_button)
        layout.addLayout(style_row)

        align_row = QHBoxLayout()
        align_row.addWidget(QLabel("정렬"))
        self.alignment_combo = QComboBox()
        self.alignment_combo.addItem("왼쪽", "left")
        self.alignment_combo.addItem("가운데", "center")
        self.alignment_combo.addItem("오른쪽", "right")
        self.alignment_combo.currentIndexChanged.connect(self.commit_region_form)
        align_row.addWidget(self.alignment_combo, 1)
        layout.addLayout(align_row)

        self.enabled_checkbox = QCheckBox("이 영역 렌더링")
        self.vertical_checkbox = QCheckBox("세로쓰기")
        self.enabled_checkbox.stateChanged.connect(self.commit_region_form)
        self.vertical_checkbox.stateChanged.connect(self.commit_region_form)
        layout.addWidget(self.enabled_checkbox)
        layout.addWidget(self.vertical_checkbox)

        layout.addSpacing(8)
        style_title = QLabel("글자 모양과 배치")
        style_title.setObjectName("SectionTitle")
        layout.addWidget(style_title)

        self.x_spin = self._spin("X", layout, 0, 99999)
        self.y_spin = self._spin("Y", layout, 0, 99999)
        self.w_spin = self._spin("너비", layout, 1, 99999)
        self.h_spin = self._spin("높이", layout, 1, 99999)
        self.font_size_spin = self._spin("글자 크기(px)", layout, 1, 999)
        self.font_size_spin.setToolTip(
            "번역 결과에 실제 사용된 글자 크기를 유지합니다. "
            "값을 변경하면 자동 맞춤 없이 지정한 크기로 렌더링하며, "
            "박스에 들어가지 않는 큰 글자는 박스 바깥까지 표시합니다."
        )
        self.rotation_spin = self._double_spin("회전", layout, -180.0, 180.0)
        self.line_spacing_spin = self._double_spin("줄/글자 간격", layout, 0.2, 4.0)
        self.padding_spin = self._double_spin("여백", layout, 0.0, 80.0)
        self.stroke_width_spin = self._double_spin("외곽선 굵기", layout, 0.0, 20.0)

        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("글꼴 가족"))
        self.font_family_combo = QComboBox()
        self.font_family_combo.setMinimumContentsLength(18)
        self.font_family_combo.setToolTip(
            "fonts 폴더와 모든 하위 폴더의 글꼴을 이름이 같은 가족끼리 묶어서 표시합니다."
        )
        self.font_family_combo.currentIndexChanged.connect(
            self._on_font_family_changed
        )
        font_row.addWidget(self.font_family_combo, 1)
        refresh_fonts_button = QPushButton("새로고침")
        refresh_fonts_button.clicked.connect(self.populate_font_combo)
        font_row.addWidget(refresh_fonts_button)
        layout.addLayout(font_row)

        font_style_row = QHBoxLayout()
        font_style_row.addWidget(QLabel("굵기/스타일"))
        self.font_style_combo = QComboBox()
        self.font_style_combo.setMinimumContentsLength(18)
        self.font_style_combo.setToolTip(
            "선택한 글꼴 가족에서 Regular, Light, Bold, Italic 등의 파일을 고릅니다."
        )
        self.font_style_combo.currentIndexChanged.connect(self._on_font_style_changed)
        font_style_row.addWidget(self.font_style_combo, 1)
        layout.addLayout(font_style_row)
        self.populate_font_combo()

        color_row = QHBoxLayout()
        self.font_color_button = QPushButton("글자 색")
        self.font_color_button.clicked.connect(self.choose_font_color)
        self.stroke_color_button = QPushButton("외곽선 색")
        self.stroke_color_button.clicked.connect(self.choose_stroke_color)
        self.paint_color_button = QPushButton("브러시 색")
        self.paint_color_button.clicked.connect(self.choose_paint_color)
        color_row.addWidget(self.font_color_button)
        color_row.addWidget(self.stroke_color_button)
        color_row.addWidget(self.paint_color_button)
        layout.addLayout(color_row)

        layout.addSpacing(8)
        settings_title = QLabel("도구별 상세 설정")
        settings_title.setObjectName("SectionTitle")
        layout.addWidget(settings_title)
        settings_tabs = QTabWidget()
        settings_tabs.setObjectName("InspectorTabs")
        settings_tabs.addTab(self._build_ocr_tab(), "OCR")
        settings_tabs.addTab(self._build_inpaint_tab(), "인페인팅")
        settings_tabs.addTab(self._build_translation_tab(), "번역")
        settings_tabs.addTab(self._build_editor_settings_tab(), "편집기 설정")
        layout.addWidget(settings_tabs)

        layout.addSpacing(8)
        output_title = QLabel("출력")
        output_title.setObjectName("SectionTitle")
        layout.addWidget(output_title)
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("저장 위치"))
        self.output_path_edit = QLineEdit(self._initial_custom_output_path())
        self.output_path_edit.setPlaceholderText(
            "저장할 폴더를 선택하세요. 비워두면 설정의 기본 저장 폴더 또는 편집 세션 기본 위치에 저장"
        )
        output_row.addWidget(self.output_path_edit, 1)
        browse_output_button = QPushButton("폴더 선택")
        browse_output_button.clicked.connect(self.choose_output_path)
        output_row.addWidget(browse_output_button)
        layout.addLayout(output_row)

        self.overwrite_source_checkbox = QCheckBox("불러온 파일에 그대로 덮어쓰기")
        self.overwrite_source_checkbox.setChecked(
            self._initial_overwrite_source_on_save()
        )
        layout.addWidget(self.overwrite_source_checkbox)
        self.output_path_edit.editingFinished.connect(self._persist_output_settings)
        self.overwrite_source_checkbox.stateChanged.connect(
            self._persist_output_settings
        )

        self.status_label = AutoHideStatusLabel("")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setObjectName("InspectorScroll")
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(470)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(panel)

        shell = QWidget(self)
        shell.setObjectName("InspectorShell")
        shell.setMinimumWidth(360)
        shell.setMaximumWidth(470)
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        shell_layout.addWidget(scroll, 1)
        shell_layout.addWidget(self.status_label)
        self._apply_help_visibility()
        return shell

    def _build_ocr_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 14)
        layout.setSpacing(9)
        saved = self._saved_section("ocr")
        saved.update(self.session.get("ocr_settings", {}) or {})

        self.auto_ocr_checkbox = QCheckBox("텍스트 박스를 그리면 자동 OCR")
        self.auto_ocr_checkbox.setChecked(bool(saved.get("auto_ocr", True)))
        layout.addWidget(self.auto_ocr_checkbox)

        layout.addWidget(QLabel("박스 검출 방식"))
        self.detector_combo = QComboBox()
        self.detector_combo.addItem("기본값 - DBNet+ResNet34", "default")
        self.detector_combo.addItem("YOLO 1 말풍선 (내장)", "yolo_1")
        self.detector_combo.addItem("YOLO 2 말풍선 (내장)", "yolo_2")
        self.detector_combo.addItem("OSB 외부 텍스트 (내장)", "osb")
        layout.addWidget(self.detector_combo)
        layout.addWidget(self._help_label(
            "기본값은 DBNet+ResNet34입니다.\n"
            "말풍선 안과 밖의 텍스트를 모두 검출할 수 있는 범용 모델입니다.\n"
            "YOLO 1 / YOLO 2 / OSB는 프로그램 내장 검출 모델이며, YOLO 1과 YOLO 2는 "
            "말풍선 안 텍스트, OSB는 말풍선 밖 텍스트 검출에 적합합니다.\n"
            "처음 실행 시 모델 로딩 또는 다운로드가 진행될 수 있습니다.\n"
            "텍스트 박스가 감지되지 않는다면 다른 모델로 변경해보세요."
        ))

        detector_row = QHBoxLayout()
        detector_row.addWidget(QLabel("민감도"))
        self.detector_confidence_spin = QDoubleSpinBox()
        self.detector_confidence_spin.setRange(0.01, 0.99)
        self.detector_confidence_spin.setSingleStep(0.05)
        self.detector_confidence_spin.setDecimals(2)
        self.detector_confidence_spin.setValue(
            float(saved.get("detector_confidence", 0.30))
        )
        detector_row.addWidget(self.detector_confidence_spin)
        layout.addLayout(detector_row)
        layout.addWidget(self._help_label(
            "민감도 값을 낮추면 더 많은 영역을 감지하지만 오검출이 늘어날 수 있습니다.\n"
            "값을 높이면 확실한 영역만 감지하지만 일부 텍스트가 누락될 수 있습니다.\n"
            "기본값 DBNet+ResNet34와 YOLO 1 / YOLO 2 / OSB에 모두 적용됩니다."
        ))

        self.merge_detected_boxes_checkbox = QCheckBox("선택 영역 안 검출 결과 합치기")
        self.merge_detected_boxes_checkbox.setChecked(
            bool(saved.get("merge_detected_boxes", True))
        )
        layout.addWidget(self.merge_detected_boxes_checkbox)
        layout.addWidget(self._help_label(
            "켜면 크게 드래그한 선택 영역 안에서 DBNet+ResNet34가 여러 조각으로 검출한 텍스트를 "
            "하나의 큰 텍스트 박스로 합칩니다.\n"
            "말풍선 하나를 크게 잡고 한 번에 OCR/수정하고 싶을 때 권장합니다. "
            "여러 말풍선을 한 번에 잡은 경우에는 하나로 합쳐질 수 있으니 꺼 주세요."
        ))

        layout.addWidget(QLabel("OCR 모델"))
        self.ocr_model_combo = QComboBox()
        self.ocr_model_combo.addItem("MOCR", "manga_ocr")
        self.ocr_model_combo.addItem("48px OCR", "48px")
        self.ocr_model_combo.addItem("LLM OCR", "llm_ocr")
        layout.addWidget(self.ocr_model_combo)
        layout.addWidget(self._help_label(
            "MOCR과 48px는 범용적으로 사용되는 로컬 OCR입니다.\n"
            "LLM OCR은 WebUI 번역 탭에 저장된 번역 제공자, 모델, API 키와 언어 설정을 "
            "그대로 사용합니다."
        ))

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("OCR 임계값"))
        self.ocr_threshold_spin = QDoubleSpinBox()
        self.ocr_threshold_spin.setRange(0.0, 1.0)
        self.ocr_threshold_spin.setSingleStep(0.05)
        self.ocr_threshold_spin.setDecimals(2)
        self.ocr_threshold_spin.setValue(float(saved.get("ocr_threshold", 0.30)))
        threshold_row.addWidget(self.ocr_threshold_spin)
        layout.addLayout(threshold_row)
        layout.addWidget(self._help_label(
            "OCR 임계값은 이미지에서 글자로 인식할 기준값입니다.\n"
            "값을 낮추면 작은 글자까지 잘 잡지만 오탐이 늘어날 수 있고, 값을 높이면 확실한 "
            "글자만 잡지만 누락이 생길 수 있습니다.\n"
            "MOCR과 48px OCR에서만 적용됩니다."
        ))

        ocr_button = QPushButton("선택 영역 OCR 실행")
        ocr_button.clicked.connect(self.run_ocr_for_selected_region)
        layout.addWidget(ocr_button)
        layout.addStretch(1)
        self._set_combo_data(
            self.detector_combo, saved.get("detector", "default")
        )
        self._set_combo_data(
            self.ocr_model_combo, saved.get("ocr_model", "manga_ocr")
        )
        self.auto_ocr_checkbox.stateChanged.connect(self._store_ocr_controls)
        self.detector_combo.currentIndexChanged.connect(self._store_ocr_controls)
        self.detector_confidence_spin.valueChanged.connect(self._store_ocr_controls)
        self.merge_detected_boxes_checkbox.stateChanged.connect(self._store_ocr_controls)
        self.ocr_model_combo.currentIndexChanged.connect(self._store_ocr_controls)
        self.ocr_threshold_spin.valueChanged.connect(self._store_ocr_controls)
        return tab

    def _build_inpaint_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 14)
        layout.setSpacing(9)
        saved = self._saved_section("inpaint")
        saved.update(self.session.get("inpaint_settings", {}) or {})
        try:
            from ui import settings_manager
            webui_settings = settings_manager.get_saved_settings()
        except Exception:
            webui_settings = {}
        self._inpaint_option_widgets: dict[str, list[QWidget]] = {}

        def hint(text: str) -> QLabel:
            return self._help_label(text)

        def option_row(group: str, label: str, widget: QWidget) -> QWidget:
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(QLabel(label))
            row.addWidget(widget, 1)
            self._inpaint_option_widgets.setdefault(group, []).append(row_widget)
            layout.addWidget(row_widget)
            return row_widget

        def option_widget(group: str, widget: QWidget) -> QWidget:
            self._inpaint_option_widgets.setdefault(group, []).append(widget)
            layout.addWidget(widget)
            return widget

        layout.addWidget(QLabel("인페인팅 방식"))
        self.inpaint_method_combo = QComboBox()
        self.inpaint_method_combo.addItem("OpenCV", "opencv")
        self.inpaint_method_combo.addItem("LAMA Large", "lama_large")
        self.inpaint_method_combo.addItem("Flux.2 Klein 4B", "flux_klein_4b")
        self.inpaint_method_combo.addItem("Flux.2 Klein 9B", "flux_klein_9b")
        self.inpaint_method_combo.addItem("Flux.1 Kontext (12B)", "flux_kontext")
        layout.addWidget(self.inpaint_method_combo)
        layout.addWidget(hint(
            "OpenCV는 빠른 기본 복원 방식입니다. 하얀색 또는 검은색처럼 단순한 단일 배경일 때만 "
            "사용하는 것을 권장합니다.\n"
            "LAMA Large는 중간 수준의 인페인팅 방식으로, 가볍고 빠르지만 Flux보다 품질이 "
            "떨어질 수 있습니다.\n"
            "Flux 계열은 WebUI의 외부 텍스트 제거 설정과 동일한 방식으로 작동하는 고품질 복원 "
            "방식입니다. 고사양 그래픽카드가 필요합니다."
        ))

        self.inpaint_device_combo = QComboBox()
        self.inpaint_device_combo.addItem("자동", "auto")
        self.inpaint_device_combo.addItem("CPU", "cpu")
        self.inpaint_device_combo.addItem("CUDA", "cuda")
        option_row("model_device", "연산 장치", self.inpaint_device_combo)

        self.inpaint_seed_spin = self._standalone_spin(
            -1,
            2147483647,
            int(saved.get("seed", webui_settings.get("outside_text_seed", -1))),
        )
        self.inpaint_steps_spin = self._standalone_spin(
            1,
            50,
            int(saved.get("steps", webui_settings.get("outside_text_flux_num_inference_steps", 4))),
        )
        self.inpainting_size_spin = self._standalone_spin(
            256, 4096, int(saved.get("inpainting_size", 1024)), step=256
        )
        option_row("flux", "시드", self.inpaint_seed_spin)
        option_widget("flux", hint("Flux 시드입니다. -1은 매번 랜덤 시드를 사용합니다."))
        option_row("flux", "추론 단계", self.inpaint_steps_spin)
        option_widget("flux", hint(
            "Flux 추론 단계입니다. 값이 낮을수록 빠르고 높을수록 더 오래 걸립니다.\n"
            "Klein은 4를 권장합니다. Kontext는 6-15를 권장합니다."
        ))
        option_row("model_size", "처리 크기", self.inpainting_size_spin)

        prompt_widget = QWidget()
        prompt_layout = QVBoxLayout(prompt_widget)
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_layout.addWidget(QLabel("프롬프트"))
        self.inpaint_prompt_edit = QTextEdit()
        self.inpaint_prompt_edit.setMinimumHeight(72)
        default_prompt = (
            str(webui_settings.get("outside_text_flux_kontext_prompt") or DEFAULT_FLUX_KONTEXT_PROMPT)
            if str(saved.get("method") or "") == "flux_kontext"
            else str(webui_settings.get("outside_text_flux_klein_prompt") or DEFAULT_FLUX_KLEIN_PROMPT)
        )
        self.inpaint_prompt_edit.setPlainText(
            str(saved.get("prompt") or default_prompt)
        )
        prompt_layout.addWidget(self.inpaint_prompt_edit)
        option_widget("flux", prompt_widget)

        self.hf_token_edit = QLineEdit(str(
            saved.get("huggingface_token")
            or webui_settings.get("outside_text_huggingface_token")
            or ""
        ))
        self.hf_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        option_row("flux", "Hugging Face 토큰", self.hf_token_edit)

        self.low_vram_checkbox = QCheckBox("저 VRAM 모드")
        self.low_vram_checkbox.setChecked(bool(
            saved.get("low_vram", webui_settings.get("outside_text_flux_low_vram", True))
        ))
        self.luminance_checkbox = QCheckBox("휘도 보정")
        self.luminance_checkbox.setChecked(bool(
            saved.get("luminance_correction", webui_settings.get("outside_text_flux_luminance_correction", True))
        ))
        self.upscale_crop_checkbox = QCheckBox("작은 영역 확대 처리")
        self.upscale_crop_checkbox.setChecked(bool(
            saved.get("upscale_small_crops", webui_settings.get("outside_text_flux_upscale_small_crops", True))
        ))
        self.strict_clip_checkbox = QCheckBox("마스크 밖 합성 차단")
        self.strict_clip_checkbox.setChecked(bool(saved.get("strict_mask_clipping", True)))
        self.force_torch_checkbox = QCheckBox("LAMA에서 PyTorch 강제 사용")
        self.force_torch_checkbox.setChecked(bool(saved.get("force_torch", False)))
        option_widget("flux", self.low_vram_checkbox)
        option_widget("flux_klein", self.luminance_checkbox)
        option_widget("flux_klein", self.upscale_crop_checkbox)
        option_widget("lama", self.force_torch_checkbox)

        self.lama_precision_combo = QComboBox()
        self.lama_precision_combo.addItem("BF16", "bf16")
        self.lama_precision_combo.addItem("FP16", "fp16")
        self.lama_precision_combo.addItem("FP32", "fp32")
        option_row("lama", "LAMA 정밀도", self.lama_precision_combo)

        self.kontext_backend_combo = QComboBox()
        self.kontext_backend_combo.addItem("SDNQ", "sdnq")
        self.kontext_backend_combo.addItem("Nunchaku", "nunchaku")
        option_row("flux_kontext", "Flux Kontext 백엔드", self.kontext_backend_combo)

        self.residual_threshold_spin = QDoubleSpinBox()
        self.residual_threshold_spin.setRange(0.01, 1.0)
        self.residual_threshold_spin.setDecimals(2)
        self.residual_threshold_spin.setSingleStep(0.05)
        self.residual_threshold_spin.setValue(float(
            saved.get(
                "residual_diff_threshold",
                webui_settings.get("outside_text_flux_residual_diff_threshold", 0.15),
            )
        ))
        option_row("flux_kontext", "잔차 차이 임계값", self.residual_threshold_spin)

        self.opencv_radius_spin = QDoubleSpinBox()
        self.opencv_radius_spin.setRange(1.0, 30.0)
        self.opencv_radius_spin.setValue(float(saved.get("opencv_radius", 3.0)))
        option_row("opencv", "OpenCV 반경", self.opencv_radius_spin)
        self.opencv_algorithm_combo = QComboBox()
        self.opencv_algorithm_combo.addItem("Telea", "telea")
        self.opencv_algorithm_combo.addItem("Navier-Stokes", "ns")
        option_row("opencv", "OpenCV 알고리즘", self.opencv_algorithm_combo)

        self.auto_inpaint_checkbox = QCheckBox("인페인팅 브러시 후 자동 실행")
        self.auto_inpaint_checkbox.setChecked(bool(saved.get("auto_run_after_brush", False)))
        layout.addWidget(self.auto_inpaint_checkbox)
        layout.addWidget(hint(
            "켜면 인페인팅 브러시 획을 놓을 때 현재 마스크로 인페인팅을 자동 실행합니다. "
            "마스크 지우개를 사용할 때는 자동 실행하지 않습니다."
        ))

        run_button = QPushButton("브러시 마스크 인페인팅 실행")
        run_button.clicked.connect(self.run_inpainting_task)
        clear_mask_button = QPushButton("브러시 마스크 지우기")
        clear_mask_button.clicked.connect(self.clear_inpaint_mask)
        reset_button = QPushButton("인페인팅 결과 초기화")
        reset_button.clicked.connect(self.reset_inpainting_result)
        layout.addWidget(run_button)
        layout.addWidget(clear_mask_button)
        layout.addWidget(reset_button)
        layout.addStretch(1)

        self._set_combo_data(
            self.inpaint_method_combo,
            saved.get("method", webui_settings.get("outside_text_inpainting_method", "opencv")),
        )
        self._set_combo_data(
            self.inpaint_device_combo, saved.get("device", "auto")
        )
        self._set_combo_data(
            self.kontext_backend_combo,
            saved.get("kontext_backend", webui_settings.get("outside_text_kontext_backend", "sdnq")),
        )
        self._set_combo_data(
            self.opencv_algorithm_combo, saved.get("opencv_algorithm", "telea")
        )
        self._set_combo_data(
            self.lama_precision_combo, saved.get("precision", "bf16")
        )
        self.inpaint_method_combo.currentIndexChanged.connect(
            self._update_inpaint_option_visibility
        )
        self.auto_inpaint_checkbox.stateChanged.connect(
            self._store_inpaint_controls
        )
        self._update_inpaint_option_visibility()
        return tab

    def _update_inpaint_option_visibility(self, *_args) -> None:
        method = str(self.inpaint_method_combo.currentData() or "opencv")
        visible_groups = {"opencv"}
        if method == "lama_large":
            visible_groups = {"lama", "model_device", "model_size"}
        elif method in {"flux_klein_4b", "flux_klein_9b"}:
            visible_groups = {"flux", "flux_klein", "model_device"}
        elif method == "flux_kontext":
            visible_groups = {"flux", "flux_kontext", "model_device"}
        for group, widgets in getattr(self, "_inpaint_option_widgets", {}).items():
            show = group in visible_groups
            for widget in widgets:
                if widget.objectName() == "HelpLabel":
                    help_visible = bool(
                        not hasattr(self, "show_help_checkbox")
                        or self.show_help_checkbox.isChecked()
                    )
                    widget.setVisible(show and help_visible)
                else:
                    widget.setVisible(show)

    def _build_translation_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 14)
        layout.setSpacing(9)
        defaults = get_translation_defaults()
        saved = self._saved_section("translation")
        saved.update(self.session.get("editor_translation_settings", {}) or {})

        layout.addWidget(QLabel("WebUI 번역 설정"))
        layout.addWidget(self._help_label(
            "편집기 번역은 WebUI 번역 탭에 저장된 번역 제공자, 모델, API 키, 원문 언어와 "
            "번역 언어 설정을 그대로 사용합니다.\n"
            "LLM OCR도 동일한 WebUI 번역 탭 설정을 사용합니다.\n"
            "WebUI에서 설정을 변경하면 번역 또는 LLM OCR 실행 시 최신 저장값을 다시 읽습니다."
        ))
        summary = QLabel(
            f"현재 저장값: {defaults.get('provider') or '-'} / "
            f"{defaults.get('model_name') or '-'}\n"
            f"{defaults.get('input_language') or '-'} → "
            f"{defaults.get('output_language') or '-'}"
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        self.editor_translation_source_combo = QComboBox()
        self.editor_translation_source_combo.addItem("원문 사용", "source")
        self.editor_translation_source_combo.addItem("현재 편집 텍스트 사용", "translation")
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("입력"))
        source_row.addWidget(self.editor_translation_source_combo, 1)
        layout.addLayout(source_row)

        selected_button = QPushButton("선택 영역 번역")
        selected_button.clicked.connect(self.run_translate_selected_region)
        all_button = QPushButton("전체 영역 번역")
        all_button.clicked.connect(self.run_translate_all_regions)
        layout.addWidget(selected_button)
        layout.addWidget(all_button)
        layout.addStretch(1)

        self._set_combo_data(
            self.editor_translation_source_combo,
            saved.get("input_source") or "source",
        )
        self.editor_translation_source_combo.currentIndexChanged.connect(
            self._store_translation_controls
        )
        return tab

    def _build_editor_settings_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 14)
        layout.setSpacing(9)

        self.show_help_checkbox = QCheckBox("설명 보기")
        self.show_help_checkbox.setChecked(
            bool(self._editor_setting_value("show_help", True))
        )
        self.show_help_checkbox.stateChanged.connect(self._on_show_help_changed)
        layout.addWidget(self.show_help_checkbox)

        self.show_inpaint_mask_checkbox = QCheckBox("인페인팅 마스크 표시")
        self.show_inpaint_mask_checkbox.setChecked(
            bool(self._editor_setting_value("show_inpaint_mask", True))
        )
        self.show_inpaint_mask_checkbox.stateChanged.connect(self._store_editor_controls)
        layout.addWidget(self.show_inpaint_mask_checkbox)

        self.show_restore_mask_checkbox = QCheckBox("원본 복구 마스크 표시")
        self.show_restore_mask_checkbox.setChecked(
            bool(self._editor_setting_value("show_restore_mask", True))
        )
        self.show_restore_mask_checkbox.stateChanged.connect(self._store_editor_controls)
        layout.addWidget(self.show_restore_mask_checkbox)

        layout.addWidget(self._help_label(
            "편집기 설정값은 자동 저장됩니다.\n"
            "프리셋1은 편집기를 열 때 자동으로 불러오는 기본 프리셋입니다.\n"
            "프리셋1도 다른 프리셋과 똑같이 원하는 값으로 다시 저장할 수 있습니다.\n"
            "프리셋1~5는 OCR, 인페인팅, 편집기 옵션, 저장 위치와 불러온 파일 덮어쓰기 값을 "
            "바꾼 뒤 해당 줄의 현재값 저장 버튼을 누르면 저장됩니다."
        ))

        layout.addWidget(QLabel("프리셋"))
        for preset_id in self._editor_presets().keys():
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            button = QPushButton(f"프리셋{preset_id}")
            button.clicked.connect(
                lambda _checked=False, value=preset_id: self.apply_editor_preset(value)
            )
            row.addWidget(button, 1)
            save_button = QPushButton("현재값 저장")
            save_button.clicked.connect(
                lambda _checked=False, value=preset_id: self.save_current_as_preset(value)
            )
            row.addWidget(save_button, 1)
            layout.addWidget(row_widget)

        settings_path = QLabel(f"설정 저장 위치\n{EDITOR_SETTINGS_FILE}")
        settings_path.setWordWrap(True)
        settings_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(settings_path)

        layout.addStretch(1)
        return tab

    def _default_editor_presets(self) -> dict[int, dict[str, Any]]:
        return {
            1: {
                "auto_ocr": True,
                "detector": "default",
                "detector_confidence": 0.30,
                "merge_detected_boxes": True,
                "ocr_model": "manga_ocr",
                "ocr_threshold": 0.30,
                "inpaint_method": "opencv",
                "inpaint_steps": 4,
                "inpaint_seed": -1,
                "auto_run_after_brush": False,
                "show_inpaint_mask": True,
                "show_restore_mask": True,
            },
            2: {
                "auto_ocr": True,
                "detector": "default",
                "detector_confidence": 0.20,
                "merge_detected_boxes": True,
                "ocr_model": "48px",
                "ocr_threshold": 0.20,
                "inpaint_method": "opencv",
                "inpaint_steps": 4,
                "inpaint_seed": -1,
                "auto_run_after_brush": False,
                "show_inpaint_mask": True,
                "show_restore_mask": True,
            },
            3: {
                "auto_ocr": True,
                "detector": "yolo_2",
                "detector_confidence": 0.45,
                "merge_detected_boxes": False,
                "ocr_model": "manga_ocr",
                "ocr_threshold": 0.30,
                "inpaint_method": "lama_large",
                "inpaint_steps": 4,
                "inpaint_seed": -1,
                "auto_run_after_brush": False,
                "show_inpaint_mask": True,
                "show_restore_mask": True,
            },
            4: {
                "auto_ocr": True,
                "detector": "osb",
                "detector_confidence": 0.55,
                "merge_detected_boxes": False,
                "ocr_model": "llm_ocr",
                "ocr_threshold": 0.35,
                "inpaint_method": "flux_klein_4b",
                "inpaint_steps": 4,
                "inpaint_seed": -1,
                "auto_run_after_brush": False,
                "show_inpaint_mask": True,
                "show_restore_mask": True,
            },
            5: {
                "auto_ocr": True,
                "detector": "default",
                "detector_confidence": 0.25,
                "merge_detected_boxes": True,
                "ocr_model": "llm_ocr",
                "ocr_threshold": 0.30,
                "inpaint_method": "flux_kontext",
                "inpaint_steps": 8,
                "inpaint_seed": -1,
                "auto_run_after_brush": False,
                "show_inpaint_mask": True,
                "show_restore_mask": True,
            },
        }

    def _editor_presets(self) -> dict[int, dict[str, Any]]:
        presets = copy.deepcopy(self._default_editor_presets())
        stored = self.editor_settings.get("presets", {})
        output_fallback = {
            "custom_output_path": self._initial_custom_output_path(),
            "overwrite_source_on_save": self._initial_overwrite_source_on_save(),
        }
        if isinstance(stored, dict):
            for preset_id in range(1, 6):
                value = stored.get(str(preset_id), stored.get(preset_id))
                if isinstance(value, dict):
                    presets[preset_id].update(value)
        for preset in presets.values():
            for key, value in output_fallback.items():
                preset.setdefault(key, value)
        return presets

    def _current_preset_settings(self) -> dict[str, Any]:
        settings: dict[str, Any] = {}
        if hasattr(self, "auto_ocr_checkbox"):
            settings["auto_ocr"] = self.auto_ocr_checkbox.isChecked()
        if hasattr(self, "detector_combo"):
            settings["detector"] = self.detector_combo.currentData()
        if hasattr(self, "detector_confidence_spin"):
            settings["detector_confidence"] = float(self.detector_confidence_spin.value())
        if hasattr(self, "merge_detected_boxes_checkbox"):
            settings["merge_detected_boxes"] = self.merge_detected_boxes_checkbox.isChecked()
        if hasattr(self, "ocr_model_combo"):
            settings["ocr_model"] = self.ocr_model_combo.currentData()
        if hasattr(self, "ocr_threshold_spin"):
            settings["ocr_threshold"] = float(self.ocr_threshold_spin.value())
        if hasattr(self, "inpaint_method_combo"):
            settings["inpaint_method"] = self.inpaint_method_combo.currentData()
        if hasattr(self, "inpaint_device_combo"):
            settings["inpaint_device"] = self.inpaint_device_combo.currentData()
        if hasattr(self, "inpaint_seed_spin"):
            settings["inpaint_seed"] = int(self.inpaint_seed_spin.value())
        if hasattr(self, "inpaint_steps_spin"):
            settings["inpaint_steps"] = int(self.inpaint_steps_spin.value())
        if hasattr(self, "auto_inpaint_checkbox"):
            settings["auto_run_after_brush"] = self.auto_inpaint_checkbox.isChecked()
        if hasattr(self, "inpainting_size_spin"):
            settings["inpainting_size"] = int(self.inpainting_size_spin.value())
        if hasattr(self, "inpaint_prompt_edit"):
            settings["inpaint_prompt"] = self.inpaint_prompt_edit.toPlainText().strip()
        if hasattr(self, "low_vram_checkbox"):
            settings["low_vram"] = self.low_vram_checkbox.isChecked()
        if hasattr(self, "luminance_checkbox"):
            settings["luminance_correction"] = self.luminance_checkbox.isChecked()
        if hasattr(self, "upscale_crop_checkbox"):
            settings["upscale_small_crops"] = self.upscale_crop_checkbox.isChecked()
        if hasattr(self, "strict_clip_checkbox"):
            settings["strict_mask_clipping"] = self.strict_clip_checkbox.isChecked()
        if hasattr(self, "force_torch_checkbox"):
            settings["force_torch"] = self.force_torch_checkbox.isChecked()
        if hasattr(self, "lama_precision_combo"):
            settings["precision"] = self.lama_precision_combo.currentData()
        if hasattr(self, "kontext_backend_combo"):
            settings["kontext_backend"] = self.kontext_backend_combo.currentData()
        if hasattr(self, "residual_threshold_spin"):
            settings["residual_diff_threshold"] = float(self.residual_threshold_spin.value())
        if hasattr(self, "opencv_radius_spin"):
            settings["opencv_radius"] = float(self.opencv_radius_spin.value())
        if hasattr(self, "opencv_algorithm_combo"):
            settings["opencv_algorithm"] = self.opencv_algorithm_combo.currentData()
        if hasattr(self, "editor_translation_source_combo"):
            settings["translation_input_source"] = self.editor_translation_source_combo.currentData()
        if hasattr(self, "output_path_edit"):
            settings["custom_output_path"] = self.output_path_edit.text().strip()
        if hasattr(self, "overwrite_source_checkbox"):
            settings["overwrite_source_on_save"] = (
                self.overwrite_source_checkbox.isChecked()
            )
        settings.update(self._editor_ui_settings())
        return settings

    def apply_editor_preset(
        self,
        preset_id: int,
        *,
        persist: bool = True,
        show_status: bool = True,
    ) -> None:
        preset = self._editor_presets().get(int(preset_id))
        if not preset:
            return
        previous_syncing = self._syncing
        self._syncing = True
        try:
            if hasattr(self, "auto_ocr_checkbox"):
                self.auto_ocr_checkbox.setChecked(bool(preset["auto_ocr"]))
            if hasattr(self, "detector_combo"):
                self._set_combo_data(self.detector_combo, preset["detector"])
            if hasattr(self, "detector_confidence_spin"):
                self.detector_confidence_spin.setValue(
                    float(preset["detector_confidence"])
                )
            if hasattr(self, "merge_detected_boxes_checkbox"):
                self.merge_detected_boxes_checkbox.setChecked(
                    bool(preset.get("merge_detected_boxes", True))
                )
            if hasattr(self, "ocr_model_combo"):
                self._set_combo_data(self.ocr_model_combo, preset["ocr_model"])
            if hasattr(self, "ocr_threshold_spin"):
                self.ocr_threshold_spin.setValue(float(preset["ocr_threshold"]))
            if hasattr(self, "inpaint_method_combo"):
                self._set_combo_data(
                    self.inpaint_method_combo, preset["inpaint_method"]
                )
            if hasattr(self, "inpaint_device_combo") and "inpaint_device" in preset:
                self._set_combo_data(
                    self.inpaint_device_combo, preset["inpaint_device"]
                )
            if hasattr(self, "inpaint_steps_spin"):
                self.inpaint_steps_spin.setValue(int(preset["inpaint_steps"]))
            if hasattr(self, "inpaint_seed_spin"):
                self.inpaint_seed_spin.setValue(int(preset["inpaint_seed"]))
            if hasattr(self, "auto_inpaint_checkbox") and "auto_run_after_brush" in preset:
                self.auto_inpaint_checkbox.setChecked(
                    bool(preset["auto_run_after_brush"])
                )
            if hasattr(self, "inpainting_size_spin") and "inpainting_size" in preset:
                self.inpainting_size_spin.setValue(int(preset["inpainting_size"]))
            if hasattr(self, "inpaint_prompt_edit") and "inpaint_prompt" in preset:
                self.inpaint_prompt_edit.setPlainText(
                    str(preset["inpaint_prompt"] or "")
                )
            if hasattr(self, "low_vram_checkbox") and "low_vram" in preset:
                self.low_vram_checkbox.setChecked(bool(preset["low_vram"]))
            if hasattr(self, "luminance_checkbox") and "luminance_correction" in preset:
                self.luminance_checkbox.setChecked(
                    bool(preset["luminance_correction"])
                )
            if hasattr(self, "upscale_crop_checkbox") and "upscale_small_crops" in preset:
                self.upscale_crop_checkbox.setChecked(
                    bool(preset["upscale_small_crops"])
                )
            if hasattr(self, "strict_clip_checkbox") and "strict_mask_clipping" in preset:
                self.strict_clip_checkbox.setChecked(
                    bool(preset["strict_mask_clipping"])
                )
            if hasattr(self, "force_torch_checkbox") and "force_torch" in preset:
                self.force_torch_checkbox.setChecked(bool(preset["force_torch"]))
            if hasattr(self, "lama_precision_combo") and "precision" in preset:
                self._set_combo_data(
                    self.lama_precision_combo, preset["precision"]
                )
            if hasattr(self, "kontext_backend_combo") and "kontext_backend" in preset:
                self._set_combo_data(
                    self.kontext_backend_combo, preset["kontext_backend"]
                )
            if hasattr(self, "residual_threshold_spin") and "residual_diff_threshold" in preset:
                self.residual_threshold_spin.setValue(
                    float(preset["residual_diff_threshold"])
                )
            if hasattr(self, "opencv_radius_spin") and "opencv_radius" in preset:
                self.opencv_radius_spin.setValue(float(preset["opencv_radius"]))
            if hasattr(self, "opencv_algorithm_combo") and "opencv_algorithm" in preset:
                self._set_combo_data(
                    self.opencv_algorithm_combo, preset["opencv_algorithm"]
                )
            if hasattr(self, "editor_translation_source_combo") and "translation_input_source" in preset:
                self._set_combo_data(
                    self.editor_translation_source_combo,
                    preset["translation_input_source"],
                )
            if hasattr(self, "show_help_checkbox") and "show_help" in preset:
                self.show_help_checkbox.setChecked(bool(preset["show_help"]))
            if hasattr(self, "show_inpaint_mask_checkbox"):
                self.show_inpaint_mask_checkbox.setChecked(
                    bool(preset["show_inpaint_mask"])
                )
            if hasattr(self, "show_restore_mask_checkbox"):
                self.show_restore_mask_checkbox.setChecked(
                    bool(preset.get("show_restore_mask", True))
                )
            if hasattr(self, "output_path_edit") and "custom_output_path" in preset:
                self.output_path_edit.setText(
                    str(preset["custom_output_path"] or "")
                )
            if (
                hasattr(self, "overwrite_source_checkbox")
                and "overwrite_source_on_save" in preset
            ):
                self.overwrite_source_checkbox.setChecked(
                    bool(preset["overwrite_source_on_save"])
                )
        finally:
            self._syncing = previous_syncing
        self._apply_help_visibility()
        self._update_inpaint_option_visibility()
        if persist:
            self._persist_editor_control_settings()
            self._persist_output_settings()
            save_session(self.session_path, self.session)
        if show_status:
            self.status_label.setText(f"프리셋{preset_id} 설정을 적용했습니다.")
        self.canvas.update()

    def save_current_as_preset(self, preset_id: int) -> None:
        preset_id = int(preset_id)
        if preset_id not in range(1, 6):
            return
        presets = self.editor_settings.setdefault("presets", {})
        if not isinstance(presets, dict):
            presets = {}
            self.editor_settings["presets"] = presets
        presets[str(preset_id)] = self._current_preset_settings()
        self._persist_editor_control_settings()
        self._persist_output_settings()
        _save_editor_settings(self.editor_settings)
        save_session(self.session_path, self.session)
        self.status_label.setText(
            f"현재 설정값을 프리셋{preset_id}에 저장했습니다."
        )

    @staticmethod
    def _standalone_spin(
        minimum: int, maximum: int, value: int, *, step: int = 1
    ) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(max(minimum, min(maximum, value)))
        return spin

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: Any) -> None:
        index = combo.findData(value)
        if index < 0 and value:
            try:
                target = os.path.normcase(str(Path(str(value)).resolve()))
                for item_index in range(combo.count()):
                    data = combo.itemData(item_index)
                    if data and os.path.normcase(str(Path(str(data)).resolve())) == target:
                        index = item_index
                        break
            except Exception:
                pass
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _preferred_font_file(paths: list[Path]) -> Optional[Path]:
        if not paths:
            return None

        def score(path: Path) -> tuple[int, int, str]:
            _family, style = _split_font_filename(path.stem)
            style_score = _font_style_sort_key(style)
            regular_priority = 0 if style in {"Regular", "Normal", "Roman"} else 1
            return regular_priority, style_score[0], path.name.casefold()

        return sorted(paths, key=score)[0]

    @staticmethod
    def _font_files_under(path: Path) -> list[Path]:
        try:
            if path.is_file() and path.suffix.lower() in SUPPORTED_FONT_SUFFIXES:
                return [path.resolve()]
            if path.is_dir():
                return sorted(
                    (
                        item.resolve()
                        for item in path.rglob("*")
                        if item.is_file()
                        and item.suffix.lower() in SUPPORTED_FONT_SUFFIXES
                    ),
                    key=lambda item: str(item).casefold(),
                )
        except Exception:
            pass
        return []

    def _default_font_choice(self) -> str:
        fonts_root = Path(__file__).resolve().parents[1] / "fonts"
        try:
            from ui import settings_manager

            saved_settings = settings_manager.get_saved_settings()
        except Exception:
            saved_settings = {}
        for key in (
            "font_pack",
            "batch_font_pack",
            "outside_text_osb_font_pack",
            "outside_text_osb_emphasis_font_pack",
        ):
            value = str(saved_settings.get(key) or "").strip()
            if not value:
                continue
            candidates = []
            candidate = Path(value)
            candidates.append(candidate)
            if not candidate.is_absolute():
                candidates.append(fonts_root / value)
            for path in candidates:
                selected = self._preferred_font_file(self._font_files_under(path))
                if selected is not None:
                    return str(selected)

        preferred = (
            fonts_root / "나눔고딕",
            fonts_root / "Nanum_Gothic",
            fonts_root / "Gowun_Dodum",
            fonts_root / "Noto_Serif_KR",
            fonts_root / "Komika Hand",
        )
        for path in preferred:
            selected = self._preferred_font_file(self._font_files_under(path))
            if selected is not None:
                return str(selected)
        selected = self._preferred_font_file(self._font_files_under(fonts_root))
        if selected is not None:
            return str(selected)
        return ""

    def _font_combo_target(self, value: Any) -> str:
        text = _resolve_repo_relative_path(value)
        if text:
            try:
                path = Path(text)
                if not path.is_absolute():
                    path = (Path(__file__).resolve().parents[1] / path).resolve()
                if path.is_file() and path.suffix.lower() in SUPPORTED_FONT_SUFFIXES:
                    return str(path.resolve())
                if path.is_dir():
                    selected = self._preferred_font_file(
                        self._font_files_under(path)
                    )
                    if selected is not None:
                        return str(selected)
            except Exception:
                pass
        return self._default_font_choice()

    def _spin(self, label: str, layout: QVBoxLayout, minimum: int, maximum: int) -> QSpinBox:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.valueChanged.connect(self.commit_region_form)
        row.addWidget(spin)
        layout.addLayout(row)
        return spin

    def _double_spin(self, label: str, layout: QVBoxLayout, minimum: float, maximum: float) -> QDoubleSpinBox:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(1)
        spin.valueChanged.connect(self.commit_region_form)
        row.addWidget(spin)
        layout.addLayout(row)
        return spin

    def _register_font_descriptor(self, descriptor: dict[str, str]) -> None:
        path = str(descriptor.get("path") or "")
        family_key = str(descriptor.get("family_key") or "")
        if not path or not family_key:
            return
        normalized_path = os.path.normcase(str(Path(path).resolve()))
        if normalized_path in self._font_path_index:
            return
        family = self._font_catalog.setdefault(
            family_key,
            {
                "label": descriptor.get("family") or Path(path).stem,
                "items": [],
            },
        )
        family["items"].append(descriptor)
        self._font_path_index[normalized_path] = (family_key, path)

    def _rebuild_font_catalog(self, extra_target: Any = "") -> None:
        fonts_root = (Path(__file__).resolve().parents[1] / "fonts").resolve()
        self._font_catalog = {}
        self._font_path_index = {}
        for path in self._font_files_under(fonts_root):
            self._register_font_descriptor(_font_descriptor(path, fonts_root))

        target = self._font_combo_target(extra_target)
        if target:
            target_path = Path(target)
            normalized = os.path.normcase(str(target_path.resolve()))
            if (
                normalized not in self._font_path_index
                and target_path.is_file()
                and target_path.suffix.lower() in SUPPORTED_FONT_SUFFIXES
            ):
                self._register_font_descriptor(
                    _font_descriptor(target_path, fonts_root)
                )

        for family in self._font_catalog.values():
            family["items"].sort(
                key=lambda item: (
                    _font_style_sort_key(str(item.get("style") or "Regular")),
                    str(item.get("relative") or "").casefold(),
                )
            )

    def _populate_font_style_combo(
        self,
        family_key: Any,
        preferred_path: Any = "",
    ) -> str:
        family = self._font_catalog.get(str(family_key or ""), {})
        items = list(family.get("items") or [])
        preferred = self._font_combo_target(preferred_path) if preferred_path else ""
        normalized_preferred = (
            os.path.normcase(str(Path(preferred).resolve())) if preferred else ""
        )
        style_counts: dict[str, int] = {}
        for item in items:
            style_key = str(item.get("style") or "Regular").casefold()
            style_counts[style_key] = style_counts.get(style_key, 0) + 1

        selected_index = -1
        self.font_style_combo.blockSignals(True)
        try:
            self.font_style_combo.clear()
            for index, item in enumerate(items):
                style = str(item.get("style") or "Regular")
                label = style
                if style_counts.get(style.casefold(), 0) > 1:
                    label = f"{style} · {Path(str(item.get('path') or '')).name}"
                path = str(item.get("path") or "")
                self.font_style_combo.addItem(label, path)
                self.font_style_combo.setItemData(
                    index,
                    str(item.get("relative") or path),
                    Qt.ItemDataRole.ToolTipRole,
                )
                if (
                    normalized_preferred
                    and os.path.normcase(str(Path(path).resolve()))
                    == normalized_preferred
                ):
                    selected_index = index
            if selected_index < 0 and items:
                selected_index = min(
                    range(len(items)),
                    key=lambda index: (
                        0
                        if str(items[index].get("style") or "")
                        in {"Regular", "Normal", "Roman"}
                        else 1,
                        _font_style_sort_key(
                            str(items[index].get("style") or "Regular")
                        ),
                    ),
                )
            if selected_index >= 0:
                self.font_style_combo.setCurrentIndex(selected_index)
                return str(self.font_style_combo.currentData() or "")
        finally:
            self.font_style_combo.blockSignals(False)
        return ""

    def _select_font_target(self, value: Any) -> str:
        target = self._font_combo_target(value)
        if not target:
            self.font_style_combo.clear()
            return ""
        normalized = os.path.normcase(str(Path(target).resolve()))
        indexed = self._font_path_index.get(normalized)
        if indexed is None:
            return ""
        family_key, indexed_path = indexed
        family_index = self.font_family_combo.findData(family_key)
        family_blocker = QSignalBlocker(self.font_family_combo)
        style_blocker = QSignalBlocker(self.font_style_combo)
        try:
            if family_index >= 0:
                self.font_family_combo.setCurrentIndex(family_index)
            return self._populate_font_style_combo(family_key, indexed_path)
        finally:
            del style_blocker
            del family_blocker

    def _on_font_family_changed(self, *_args) -> None:
        family_key = self.font_family_combo.currentData()
        self._populate_font_style_combo(family_key)
        if not self._syncing:
            self._font_change_requested = True
            self.commit_region_form()

    def _on_font_style_changed(self, *_args) -> None:
        if not self._syncing:
            self._font_change_requested = True
            self.commit_region_form()

    def populate_font_combo(self) -> None:
        current = (
            self.font_style_combo.currentData()
            if self.font_style_combo.count()
            else ""
        )
        region = self.selected_region()
        if not current and region is not None:
            current = region.get("render", {}).get("font_dir", "")
        self._rebuild_font_catalog(current)

        self.font_family_combo.blockSignals(True)
        try:
            self.font_family_combo.clear()
            families = sorted(
                self._font_catalog.items(),
                key=lambda item: str(item[1].get("label") or "").casefold(),
            )
            for family_key, family in families:
                count = len(family.get("items") or [])
                self.font_family_combo.addItem(
                    f"{family.get('label') or family_key} ({count})",
                    family_key,
                )
            selected = self._select_font_target(current)
            if not selected and families:
                self.font_family_combo.setCurrentIndex(0)
                self._populate_font_style_combo(families[0][0])
        finally:
            self.font_family_combo.blockSignals(False)

    def update_zoom_display(self, scale: float) -> None:
        if hasattr(self, "zoom_label"):
            percent = max(2, min(800, int(round(float(scale) * 100.0))))
            text = f"{percent}%"
            if self.zoom_label.text() != text:
                self.zoom_label.setText(text)

    def update_canvas_scrollbars(
        self,
        horizontal_max: int,
        horizontal_value: int,
        vertical_max: int,
        vertical_value: int,
    ) -> None:
        if not hasattr(self, "canvas_horizontal_scrollbar"):
            return
        self._syncing_canvas_scrollbars = True
        try:
            horizontal = self.canvas_horizontal_scrollbar
            vertical = self.canvas_vertical_scrollbar
            horizontal.blockSignals(True)
            vertical.blockSignals(True)
            try:
                horizontal.setRange(0, max(0, int(horizontal_max)))
                horizontal.setPageStep(max(1, self.canvas.width()))
                horizontal.setValue(
                    max(0, min(horizontal.maximum(), int(horizontal_value)))
                )
                vertical.setRange(0, max(0, int(vertical_max)))
                vertical.setPageStep(max(1, self.canvas.height()))
                vertical.setValue(
                    max(0, min(vertical.maximum(), int(vertical_value)))
                )
                horizontal.setEnabled(horizontal.maximum() > 0)
                vertical.setEnabled(vertical.maximum() > 0)
            finally:
                horizontal.blockSignals(False)
                vertical.blockSignals(False)
        finally:
            self._syncing_canvas_scrollbars = False

    def _on_canvas_horizontal_scroll(self, value: int) -> None:
        if self._syncing_canvas_scrollbars or not hasattr(self, "canvas"):
            return
        self.canvas.set_scroll_position(horizontal_value=value)

    def _on_canvas_vertical_scroll(self, value: int) -> None:
        if self._syncing_canvas_scrollbars or not hasattr(self, "canvas"):
            return
        self.canvas.set_scroll_position(vertical_value=value)

    def zoom_in(self) -> None:
        if hasattr(self, "canvas"):
            self.canvas.zoom_at(
                QPointF(self.canvas.width() / 2.0, self.canvas.height() / 2.0),
                1.2,
            )

    def zoom_out(self) -> None:
        if hasattr(self, "canvas"):
            self.canvas.zoom_at(
                QPointF(self.canvas.width() / 2.0, self.canvas.height() / 2.0),
                1.0 / 1.2,
            )

    def zoom_fit(self) -> None:
        if hasattr(self, "canvas"):
            self.canvas.reset_view()

    def zoom_actual_size(self) -> None:
        if hasattr(self, "canvas"):
            self.canvas.set_absolute_scale(1.0)

    def set_tool(self, tool: str) -> None:
        for key, action in self.tool_actions.items():
            action.setChecked(key == tool)
        if hasattr(self, "pan_tool_button"):
            self.pan_tool_button.blockSignals(True)
            self.pan_tool_button.setChecked(tool == "pan")
            self.pan_tool_button.blockSignals(False)
        self.canvas.set_tool(tool)
        tool_label = "화면 이동" if tool == "pan" else TOOLS.get(tool, tool)
        if tool == "pan":
            self.status_label.setText(
                "화면 이동: 이미지 위에서 왼쪽 마우스로 드래그하거나 스크롤바를 사용하세요."
            )
        else:
            self.status_label.setText(f"선택한 도구: {tool_label}")

    def schedule_exact_preview(self) -> None:
        self._preview_request_serial += 1
        if self._preview_running:
            self._preview_pending = True
        self._preview_timer.start()

    def refresh_exact_preview(self) -> None:
        if self._preview_running:
            self._preview_pending = True
            return
        request_serial = self._preview_request_serial
        preview_session = copy.deepcopy(self.session)
        signals = TaskSignals()
        self._task_signals.append(signals)
        self._preview_running = True
        self._preview_pending = False

        def _finish_preview() -> None:
            self._preview_running = False
            if signals in self._task_signals:
                self._task_signals.remove(signals)
            if self._preview_pending or request_serial != self._preview_request_serial:
                self._preview_pending = False
                self._preview_timer.start(40)

        def _success(result: object) -> None:
            try:
                image, errors = result
                if request_serial != self._preview_request_serial:
                    return
                self.displaying_rendered_result = True
                self.display_image = image.convert("RGBA")
                self.display_qimage = _pil_to_qimage(self.display_image)
                self.image_size = self.display_image.size
                self.canvas.update()
                if errors:
                    self.status_label.setText(
                        "일부 영역 미리보기 렌더링 경고:\n"
                        + "\n".join(errors[:3])
                    )
            finally:
                _finish_preview()

        def _failure(message: str) -> None:
            try:
                if request_serial == self._preview_request_serial:
                    self.status_label.setText(
                        "미리보기 렌더링 실패:\n" + message.splitlines()[-1]
                    )
            finally:
                _finish_preview()

        signals.finished.connect(_success)
        signals.failed.connect(_failure)

        def _worker() -> None:
            try:
                signals.finished.emit(render_session_image(preview_session))
            except Exception:
                signals.failed.emit(traceback.format_exc())

        threading.Thread(target=_worker, daemon=True).start()

    def _run_background(
        self,
        task,
        on_success,
        *,
        working_message: str,
        on_failure=None,
    ) -> None:
        signals = TaskSignals()
        self._task_signals.append(signals)
        self._background_job_count += 1
        if hasattr(self, "session_list"):
            self.session_list.setEnabled(False)
        signals.finished.connect(on_success)
        signals.failed.connect(
            lambda message: QMessageBox.critical(self, "작업 실패", message)
        )
        signals.failed.connect(lambda message: self.status_label.setText(message))
        if on_failure is not None:
            signals.failed.connect(on_failure)

        def _cleanup(*_args) -> None:
            if signals in self._task_signals:
                self._task_signals.remove(signals)
            self._background_job_count = max(0, self._background_job_count - 1)
            if hasattr(self, "session_list"):
                self.session_list.setEnabled(self._background_job_count == 0)

        signals.finished.connect(_cleanup)
        signals.failed.connect(_cleanup)
        self.status_label.setText(working_message)

        def _worker() -> None:
            try:
                signals.finished.emit(task())
            except Exception:
                signals.failed.emit(traceback.format_exc())

        threading.Thread(target=_worker, daemon=True).start()

    def _on_brush_size_changed(self, value: int) -> None:
        self.brush_size = int(value)
        self.canvas.update()

    def _on_region_combo_changed(self, index: int) -> None:
        if self._syncing:
            return
        if index < 0:
            self.set_selected_index(None)
        else:
            self.set_selected_index(index)

    def region_at(self, index: Optional[int]) -> Optional[dict[str, Any]]:
        if index is None:
            return None
        regions = self.session.get("regions", [])
        if 0 <= index < len(regions):
            return regions[index]
        return None

    def selected_region(self) -> Optional[dict[str, Any]]:
        return self.region_at(self.selected_index)

    @staticmethod
    def _region_effective_vertical_stack(region: dict[str, Any]) -> bool:
        orientation = str(region.get("resolved_orientation") or "").lower()
        if orientation in {"vertical", "horizontal"}:
            return orientation == "vertical"
        if bool(region.get("vertical_stack", False)):
            return True
        baseline = region.get("render_baseline")
        if isinstance(baseline, dict):
            return str(baseline.get("resolved_orientation") or "").lower() == "vertical"
        return False

    def set_selected_index(self, index: Optional[int]) -> None:
        self.selected_index = index
        self._syncing = True
        try:
            self.region_combo.setCurrentIndex(index if index is not None else -1)
        finally:
            self._syncing = False
        self.load_selected_region_to_form()
        self.canvas.update()

    def keyPressEvent(self, event) -> None:
        if self.handle_editor_key_event(event):
            return
        super().keyPressEvent(event)

    def handle_editor_key_event(self, event) -> bool:
        if event.key() in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace}:
            focus = self.focusWidget()
            if isinstance(focus, (QTextEdit, QLineEdit)):
                return False
            if (
                focus is self.canvas
                and self.canvas.tool == "select"
                and self.selected_region() is not None
            ):
                self.delete_selected_region()
                event.accept()
                return True

        arrow_deltas = {
            Qt.Key.Key_Left: (-1, 0),
            Qt.Key.Key_Right: (1, 0),
            Qt.Key.Key_Up: (0, -1),
            Qt.Key.Key_Down: (0, 1),
        }
        if (
            event.key() in arrow_deltas
            and self.focusWidget() is self.canvas
            and self.canvas.tool == "select"
            and self.selected_region() is not None
            and not (
                event.modifiers()
                & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier)
            )
        ):
            dx, dy = arrow_deltas[event.key()]
            step = 10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
            self.move_selected_region(dx * step, dy * step)
            event.accept()
            return True
        return False

    def move_selected_region(self, dx: float, dy: float) -> None:
        region = self.selected_region()
        if region is None:
            return
        self._push_history("텍스트 박스 이동")
        x1, y1, x2, y2 = _bbox(region)
        box_width = x2 - x1
        box_height = y2 - y1
        image_width, image_height = self.image_size
        new_x1 = max(0.0, min(float(image_width) - box_width, x1 + float(dx)))
        new_y1 = max(0.0, min(float(image_height) - box_height, y1 + float(dy)))
        region["bbox"] = [
            new_x1,
            new_y1,
            new_x1 + box_width,
            new_y1 + box_height,
        ]
        self._mark_region_dirty(region)
        self.load_selected_region_to_form()
        self.canvas.update()
        self.schedule_exact_preview()
        self.status_label.setText(
            f"텍스트 박스를 이동했습니다: X {int(round(new_x1))}, Y {int(round(new_y1))}"
        )

    def _on_translation_text_changed(self) -> None:
        if not self._syncing:
            if not self._translation_history_active:
                self._push_history("텍스트 수정")
                self._translation_history_active = True
            self._translation_history_timer.start()
        self.commit_region_form()

    def refresh_region_list(self) -> None:
        self._syncing = True
        try:
            self.region_combo.clear()
            for index, region in enumerate(self.session.get("regions", []), start=1):
                text = str(region.get("translation") or "").replace("\n", " ").strip()
                if len(text) > 34:
                    text = text[:34] + "..."
                region_type = str(region.get("type") or "region")
                type_label = REGION_TYPE_LABELS.get(region_type, region_type)
                self.region_combo.addItem(f"{index}. {type_label} | {text or '(비어 있음)'}")
            if self.selected_index is not None:
                self.region_combo.setCurrentIndex(self.selected_index)
        finally:
            self._syncing = False

    def load_selected_region_to_form(self) -> None:
        region = self.selected_region()
        self._syncing = True
        self._font_change_requested = False
        try:
            enabled = region is not None
            for widget in (
                self.translation_edit,
                self.source_edit,
                self.enabled_checkbox,
                self.vertical_checkbox,
                self.x_spin,
                self.y_spin,
                self.w_spin,
                self.h_spin,
                self.font_size_spin,
                self.rotation_spin,
                self.line_spacing_spin,
                self.padding_spin,
                self.stroke_width_spin,
                self.font_color_button,
                self.stroke_color_button,
                self.font_family_combo,
                self.font_style_combo,
                self.alignment_combo,
            ):
                widget.setEnabled(enabled)
            if region is None:
                self.source_edit.setPlainText("")
                self.translation_edit.setPlainText("")
                return
            x1, y1, x2, y2 = _bbox(region)
            render = region.setdefault("render", {})
            self.source_edit.setPlainText(str(region.get("source_text") or ""))
            self.translation_edit.setPlainText(str(region.get("translation") or ""))
            self.enabled_checkbox.setChecked(bool(region.get("enabled", True)))
            self.vertical_checkbox.setChecked(self._region_effective_vertical_stack(region))
            self.x_spin.setValue(int(round(x1)))
            self.y_spin.setValue(int(round(y1)))
            self.w_spin.setValue(max(1, int(round(x2 - x1))))
            self.h_spin.setValue(max(1, int(round(y2 - y1))))
            font_size = resolve_region_font_size(
                self.session,
                region,
                (self.selected_index or 0) + 1,
            )
            self.font_size_spin.setValue(font_size)
            self.rotation_spin.setValue(float(region.get("rotation_deg") or 0.0))
            spacing_value = (
                render.get("vertical_line_spacing_mult")
                if region.get("vertical_stack", False) and render.get("vertical_line_spacing_mult") is not None
                else render.get("line_spacing_mult")
            )
            self.line_spacing_spin.setValue(float(spacing_value or 1.0))
            self.padding_spin.setValue(float(render.get("padding_pixels") or 5.0))
            self.stroke_width_spin.setValue(float(render.get("outline_width") or 0.0))
            self.font_color_button.setStyleSheet(f"background: {region.get('font_color') or '#000000'};")
            outline_rgb = render.get("outline_color_rgb")
            if isinstance(outline_rgb, (list, tuple)) and len(outline_rgb) >= 3:
                outline_hex = f"#{int(outline_rgb[0]):02x}{int(outline_rgb[1]):02x}{int(outline_rgb[2]):02x}"
            else:
                outline_hex = "#ffffff"
            self.stroke_color_button.setStyleSheet(f"background: {outline_hex};")
            font_target = self._font_combo_target(render.get("font_dir") or "")
            self._select_font_target(font_target)
            self._set_combo_data(self.alignment_combo, render.get("text_alignment") or "center")
        finally:
            self._syncing = False

    def commit_region_form(self, *_args, mark_dirty: bool = True) -> None:
        if self._syncing:
            return
        region = self.selected_region()
        if region is None:
            return
        x = float(self.x_spin.value())
        y = float(self.y_spin.value())
        w = float(self.w_spin.value())
        h = float(self.h_spin.value())
        region["translation"] = self.translation_edit.toPlainText()
        region["enabled"] = self.enabled_checkbox.isChecked()
        region["vertical_stack"] = self.vertical_checkbox.isChecked()
        region["resolved_orientation"] = (
            "vertical" if region["vertical_stack"] else "horizontal"
        )
        region["bbox"] = [x, y, x + w, y + h]
        region["rotation_deg"] = float(self.rotation_spin.value())
        render = region.setdefault("render", {})
        set_region_font_size(region, int(self.font_size_spin.value()))
        line_spacing = float(self.line_spacing_spin.value())
        render["line_spacing_mult"] = line_spacing
        render["emphasis_line_spacing_mult"] = line_spacing
        render["vertical_line_spacing_mult"] = line_spacing
        render["vertical_emphasis_line_spacing_mult"] = line_spacing
        render["padding_pixels"] = float(self.padding_spin.value())
        render["outline_width"] = float(self.stroke_width_spin.value())
        render["text_alignment"] = self.alignment_combo.currentData() or "center"
        if region["vertical_stack"]:
            render["auto_vertical_text"] = False
            render["compare_forced_vertical_layout"] = False
        else:
            render["auto_vertical_text"] = False
        selected_font = str(self.font_style_combo.currentData() or "")
        if self._font_change_requested and selected_font:
            render["font_dir"] = str(selected_font)
        self._font_change_requested = False
        if mark_dirty:
            self._mark_region_dirty(region)
        self.schedule_exact_preview()

    def update_selected_bbox_from_drag(
        self,
        mode: str,
        start_bbox: list[float],
        start_point: QPointF,
        current_point: QPointF,
    ) -> None:
        region = self.selected_region()
        if region is None:
            return
        x1, y1, x2, y2 = start_bbox
        dx = current_point.x() - start_point.x()
        dy = current_point.y() - start_point.y()
        if mode == "move":
            new_box = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
        else:
            new_box = [x1, y1, x2, y2]
            if "w" in mode:
                new_box[0] = min(x2 - 1, x1 + dx)
            if "e" in mode:
                new_box[2] = max(x1 + 1, x2 + dx)
            if "n" in mode:
                new_box[1] = min(y2 - 1, y1 + dy)
            if "s" in mode:
                new_box[3] = max(y1 + 1, y2 + dy)
        width, height = self.image_size
        new_box[0] = max(0.0, min(float(width - 1), new_box[0]))
        new_box[1] = max(0.0, min(float(height - 1), new_box[1]))
        new_box[2] = max(new_box[0] + 1.0, min(float(width), new_box[2]))
        new_box[3] = max(new_box[1] + 1.0, min(float(height), new_box[3]))
        region["bbox"] = new_box
        self._mark_region_dirty(region)
        self.load_selected_region_to_form()

    def add_text_region(self) -> None:
        self._push_history("텍스트 영역 추가")
        self.commit_region_form(mark_dirty=False)
        save_session(self.session_path, self.session)
        session, region_id, _preview = add_region(self.session_path)
        self.session = session
        self.selected_index = max(0, len(self.session.get("regions", [])) - 1)
        self._mark_region_dirty(self.selected_region())
        self.refresh_region_list()
        self.load_selected_region_to_form()
        self.canvas.update()
        self.status_label.setText("텍스트 영역을 추가했습니다.")

    def create_textbox_from_drag(self, start: QPointF, end: QPointF) -> None:
        x1, x2 = sorted([start.x(), end.x()])
        y1, y2 = sorted([start.y(), end.y()])
        if x2 - x1 < 8 or y2 - y1 < 8:
            return
        self.add_text_region()
        region = self.selected_region()
        if region is None:
            return
        region["bbox"] = [float(x1), float(y1), float(x2), float(y2)]
        region["translation"] = "새 텍스트"
        self._mark_region_dirty(region)
        self.load_selected_region_to_form()
        save_session(self.session_path, self.session)
        self.schedule_exact_preview()
        if self.auto_ocr_checkbox.isChecked():
            self.run_ocr_for_roi(region["bbox"], replace_selected=True)

    def _ocr_settings(self) -> dict[str, Any]:
        settings = {
            "auto_ocr": self.auto_ocr_checkbox.isChecked(),
            "detector": self.detector_combo.currentData(),
            "detector_confidence": float(self.detector_confidence_spin.value()),
            "merge_detected_boxes": (
                self.merge_detected_boxes_checkbox.isChecked()
                if hasattr(self, "merge_detected_boxes_checkbox")
                else True
            ),
            "ocr_model": self.ocr_model_combo.currentData(),
            "ocr_threshold": float(self.ocr_threshold_spin.value()),
            "huggingface_token": (
                self.hf_token_edit.text().strip()
                if hasattr(self, "hf_token_edit")
                else ""
            ),
        }
        self.session["ocr_settings"] = settings
        self._remember_settings("ocr", settings)
        return settings

    def _store_ocr_controls(self, *_args) -> None:
        if not self._syncing:
            self._ocr_settings()

    def run_ocr_for_selected_region(self) -> None:
        region = self.selected_region()
        if region is None:
            QMessageBox.warning(self, "OCR", "OCR을 실행할 텍스트 영역을 선택해 주세요.")
            return
        self.run_ocr_for_roi(_bbox(region), replace_selected=True)

    def run_ocr_for_roi(
        self, roi: list[float], *, replace_selected: bool
    ) -> None:
        settings = self._ocr_settings()
        image = self._load_ocr_image()
        selected_index = self.selected_index

        def _task():
            return detect_and_recognize(image, list(roi), settings)

        def _success(results: list[dict[str, Any]]) -> None:
            if not results:
                self.status_label.setText(
                    "선택한 범위에서 텍스트를 찾지 못했습니다. 민감도나 OCR 임계값을 낮춰 보세요."
                )
                return
            self._push_history("OCR 결과 적용")
            regions = self.session.setdefault("regions", [])
            target_index = (
                selected_index
                if replace_selected
                and selected_index is not None
                and 0 <= selected_index < len(regions)
                else None
            )
            template = (
                copy.deepcopy(regions[target_index])
                if target_index is not None
                else copy.deepcopy(regions[-1]) if regions else {"render": {}}
            )
            inserted_indices = []
            for result_index, result in enumerate(results):
                if result_index == 0 and target_index is not None:
                    region = regions[target_index]
                    index = target_index
                else:
                    region = copy.deepcopy(template)
                    region.pop("render_baseline", None)
                    for key in list(region):
                        if str(key).startswith("_editor_"):
                            region.pop(key, None)
                    region["id"] = (
                        f"ocr_{datetime.now().strftime('%H%M%S%f')}_{result_index}"
                    )
                    region["type"] = "added_text"
                    regions.append(region)
                    index = len(regions) - 1
                text = str(result.get("text") or "").strip()
                region["bbox"] = [float(v) for v in result["bbox"]]
                region["source_text"] = text
                region["translation"] = text
                region["enabled"] = True
                region["ocr_confidence"] = float(result.get("confidence") or 0.0)
                self._mark_region_dirty(region)
                inserted_indices.append(index)
            self.selected_index = inserted_indices[0]
            save_session(self.session_path, self.session)
            self.refresh_region_list()
            self.load_selected_region_to_form()
            self.schedule_exact_preview()
            self.status_label.setText(
                f"OCR을 완료했습니다. 인식 영역: {len(results)}개"
            )

        self._run_background(
            _task,
            _success,
            working_message="텍스트 검출과 OCR을 실행 중입니다. 최초 실행은 모델 다운로드로 시간이 걸릴 수 있습니다.",
        )

    def _translation_settings(self) -> dict[str, Any]:
        settings = {
            "input_source": self.editor_translation_source_combo.currentData() or "source",
        }
        self.session["editor_translation_settings"] = dict(settings)
        self._remember_settings("translation", settings)
        return settings

    def _store_translation_controls(self, *_args) -> None:
        if not self._syncing:
            self._translation_settings()

    def _region_translation_input(self, region: dict[str, Any], input_source: str) -> str:
        if input_source == "translation":
            return str(region.get("translation") or "").strip()
        source_text = str(region.get("source_text") or "").strip()
        if source_text:
            return source_text
        return str(region.get("translation") or "").strip()

    @staticmethod
    def _clean_editor_translation(text: str) -> tuple[str, str]:
        cleaned = str(text or "").strip()
        if "||" in cleaned:
            cleaned = cleaned.rsplit("||", 1)[1].strip()
        tag, content = extract_translation_tag(cleaned)
        return (content.strip() if content.strip() else cleaned, tag or "")

    def _apply_editor_translations(
        self,
        indexes: list[int],
        translations: list[str],
        *,
        settings: dict[str, Any],
    ) -> None:
        self._push_history("번역 결과 적용")
        regions = self.session.get("regions", [])
        for index, translated in zip(indexes, translations):
            if not (0 <= index < len(regions)):
                continue
            text, tag = self._clean_editor_translation(translated)
            regions[index]["translation"] = text
            if tag:
                regions[index]["render_tag"] = tag
            self._mark_region_dirty(regions[index])
        self.session["editor_translation_settings"] = {
            key: value
            for key, value in settings.items()
            if key != "api_key_override"
        }
        save_session(self.session_path, self.session)
        self.refresh_region_list()
        self.load_selected_region_to_form()
        self.schedule_exact_preview()
        self.status_label.setText(f"번역을 적용했습니다. 영역: {len(indexes)}개")

    def run_translate_selected_region(self) -> None:
        self.commit_region_form(mark_dirty=False)
        region = self.selected_region()
        if region is None or self.selected_index is None:
            QMessageBox.warning(self, "번역", "번역할 텍스트 영역을 선택해 주세요.")
            return
        settings = self._translation_settings()
        source_text = self._region_translation_input(
            region,
            str(settings.get("input_source") or "source"),
        )
        if not source_text:
            QMessageBox.warning(self, "번역", "선택 영역에 번역할 텍스트가 없습니다.")
            return
        selected_index = self.selected_index

        def _task():
            return translate_texts([source_text], settings)

        def _success(results: list[str]) -> None:
            self._apply_editor_translations([selected_index], results, settings=settings)

        self._run_background(
            _task,
            _success,
            working_message="선택 영역을 번역 중입니다.",
        )

    def run_translate_all_regions(self) -> None:
        self.commit_region_form(mark_dirty=False)
        settings = self._translation_settings()
        input_source = str(settings.get("input_source") or "source")
        indexes: list[int] = []
        texts: list[str] = []
        for index, region in enumerate(self.session.get("regions", [])):
            text = self._region_translation_input(region, input_source)
            if text:
                indexes.append(index)
                texts.append(text)
        if not texts:
            QMessageBox.warning(self, "번역", "번역할 텍스트가 있는 영역이 없습니다.")
            return

        def _task():
            return translate_texts(texts, settings)

        def _success(results: list[str]) -> None:
            self._apply_editor_translations(indexes, results, settings=settings)

        self._run_background(
            _task,
            _success,
            working_message=f"전체 텍스트 영역 {len(texts)}개를 번역 중입니다.",
        )

    def delete_selected_region(self) -> None:
        region = self.selected_region()
        if region is None:
            return
        self._push_history("텍스트 영역 삭제")
        try:
            erase_box = self._fast_erase_bbox(region, self.image_size)
            self.session.setdefault("deleted_region_bboxes", []).append(list(erase_box))
        except Exception:
            pass
        save_session(self.session_path, self.session)
        session, next_id, _preview = delete_region(self.session_path, region.get("id"))
        self.session = session
        self._dirty_region_ids.discard(str(region.get("id") or ""))
        self.session["editor_dirty_region_ids"] = sorted(self._dirty_region_ids)
        save_session(self.session_path, self.session)
        if next_id is None:
            self.selected_index = None
        else:
            ids = [item.get("id") for item in self.session.get("regions", [])]
            self.selected_index = ids.index(next_id) if next_id in ids else 0
        self.refresh_region_list()
        self.load_selected_region_to_form()
        self.canvas.update()
        self.schedule_exact_preview()
        self.status_label.setText("선택한 영역을 삭제했습니다.")

    def choose_font_color(self) -> None:
        region = self.selected_region()
        if region is None:
            return
        current = QColor(region.get("font_color") or "#000000")
        color = QColorDialog.getColor(current, self, "글자 색 선택")
        if not color.isValid():
            return
        self._push_history("글자 색 변경")
        value = _hex_from_qcolor(color)
        region["font_color"] = value
        region["text_color_rgb"] = list(_rgb_from_hex(value))
        self._mark_region_dirty(region)
        self.font_color_button.setStyleSheet(f"background: {value};")
        self.schedule_exact_preview()

    def choose_stroke_color(self) -> None:
        region = self.selected_region()
        if region is None:
            return
        render = region.setdefault("render", {})
        outline_rgb = render.get("outline_color_rgb")
        if isinstance(outline_rgb, (list, tuple)) and len(outline_rgb) >= 3:
            current = QColor(
                int(outline_rgb[0]),
                int(outline_rgb[1]),
                int(outline_rgb[2]),
            )
        else:
            current = QColor("#ffffff")
        color = QColorDialog.getColor(current, self, "외곽선 색 선택")
        if not color.isValid():
            return
        self._push_history("외곽선 색 변경")
        value = _hex_from_qcolor(color)
        render["outline_color_rgb"] = list(_rgb_from_hex(value, (255, 255, 255)))
        if float(render.get("outline_width") or 0.0) <= 0.0:
            render["outline_width"] = 1.0
            self.stroke_width_spin.setValue(1.0)
        self._mark_region_dirty(region)
        self.stroke_color_button.setStyleSheet(f"background: {value};")
        self.schedule_exact_preview()

    def wrap_translation_selection(self, marker: str) -> None:
        region = self.selected_region()
        if region is None:
            return
        marker = str(marker)
        if marker not in {"*", "**", "***"}:
            return
        self._push_history("텍스트 강조 변경")
        cursor = self.translation_edit.textCursor()
        if cursor.hasSelection():
            selected = cursor.selectedText().replace("\u2029", "\n")
            if (
                len(selected) >= len(marker) * 2
                and selected.startswith(marker)
                and selected.endswith(marker)
            ):
                replacement = selected[len(marker) : -len(marker)]
            else:
                replacement = f"{marker}{selected}{marker}"
            cursor.insertText(replacement)
            self.translation_edit.setTextCursor(cursor)
            return
        text = self.translation_edit.toPlainText()
        if text.strip():
            if (
                len(text) >= len(marker) * 2
                and text.startswith(marker)
                and text.endswith(marker)
            ):
                text = text[len(marker) : -len(marker)]
            else:
                text = f"{marker}{text}{marker}"
            self.translation_edit.setPlainText(text)
            return
        cursor.insertText(marker + marker)
        cursor.movePosition(
            QTextCursor.MoveOperation.Left,
            QTextCursor.MoveMode.MoveAnchor,
            len(marker),
        )
        self.translation_edit.setTextCursor(cursor)

    def choose_paint_color(self) -> None:
        color = QColorDialog.getColor(QColor(self.paint_color), self, "브러시 색 선택")
        if not color.isValid():
            return
        self.paint_color = _hex_from_qcolor(color)
        self.paint_color_button.setStyleSheet(f"background: {self.paint_color};")
        self.canvas.update()

    def apply_stroke(self, tool: str, points: list[tuple[int, int]]) -> None:
        if not points:
            return
        stroke_labels = {
            "inpaint_brush": "인페인팅 브러시",
            "inpaint_eraser": "인페인팅 마스크 지우개",
            "paint_brush": "그리기 브러시",
            "paint_eraser": "그리기 지우개",
            "restore_brush": "원본 복구 브러시",
            "restore_eraser": "원본 복구 지우개",
        }
        self._push_history(stroke_labels.get(tool, "브러시 편집"))
        try:
            if tool in {"inpaint_brush", "inpaint_eraser"}:
                self._apply_inpaint_mask_stroke(tool, points)
            elif tool in {"restore_brush", "restore_eraser"}:
                self._apply_restore_mask_stroke(tool, points)
            else:
                self._apply_paint_stroke(tool, points)
        except Exception as exc:
            QMessageBox.warning(self, "브러시 작업 실패", str(exc))
            return
        self.save_current_session(push_history=False)
        self.reload_display_image()
        if tool in {"restore_brush", "restore_eraser"}:
            self.schedule_exact_preview()
        if (
            tool == "inpaint_brush"
            and hasattr(self, "auto_inpaint_checkbox")
            and self.auto_inpaint_checkbox.isChecked()
        ):
            self.status_label.setText("인페인팅 브러시 마스크를 수정했습니다. 자동 인페인팅을 실행합니다.")
            QTimer.singleShot(120, self.run_inpainting_task)

    def _stroke_mask(self, points: list[tuple[int, int]]) -> np.ndarray:
        width, height = self.image_size
        stroke = np.zeros((height, width), dtype=np.uint8)
        radius = max(1, int(round(self.brush_size / 2)))
        thickness = max(1, int(round(self.brush_size)))
        if len(points) == 1:
            cv2.circle(stroke, points[0], radius, 255, thickness=-1, lineType=cv2.LINE_8)
        else:
            for index in range(1, len(points)):
                cv2.line(stroke, points[index - 1], points[index], 255, thickness=thickness, lineType=cv2.LINE_8)
            for point in points:
                cv2.circle(stroke, point, radius, 255, thickness=-1, lineType=cv2.LINE_8)
        return stroke

    def _apply_inpaint_mask_stroke(
        self, tool: str, points: list[tuple[int, int]]
    ) -> None:
        assets_dir = Path(
            self.session.get("assets_dir")
            or self.session_path.with_name(f"{self.session_path.stem}_assets")
        )
        mask_path = Path(
            self.session.get("inpaint_mask_path")
            or assets_dir / "inpaint_brush_mask.png"
        )
        mask = _load_mask(mask_path, self.image_size)
        stroke = self._stroke_mask(points)
        if tool == "inpaint_brush":
            mask[stroke > 0] = 255
        else:
            mask[stroke > 0] = 0
        _save_mask(mask_path, mask)
        self.session["inpaint_mask_path"] = str(mask_path.resolve())
        self.status_label.setText(f"인페인팅 마스크를 수정했습니다: {mask_path.name}")

    def _apply_restore_mask_stroke(
        self, tool: str, points: list[tuple[int, int]]
    ) -> None:
        original_path = Path(
            self.session.get("original_snapshot_path")
            or self.session.get("source_image_path")
            or ""
        )
        if not original_path.is_file():
            raise FileNotFoundError(
                "번역 전 원본 이미지를 찾을 수 없어 원본 복구 브러시를 사용할 수 없습니다."
            )
        assets_dir = Path(
            self.session.get("assets_dir")
            or self.session_path.with_name(f"{self.session_path.stem}_assets")
        )
        mask_path = Path(
            self.session.get("restore_mask_path")
            or assets_dir / "original_restore_mask.png"
        )
        mask = _load_mask(mask_path, self.image_size)
        stroke = self._stroke_mask(points)
        if tool == "restore_brush":
            mask[stroke > 0] = 255
            message = "칠한 부분을 번역 전 원본으로 복구했습니다."
        else:
            mask[stroke > 0] = 0
            message = "칠한 부분의 원본 복구를 취소했습니다."
        _save_mask(mask_path, mask)
        self.session["restore_mask_path"] = str(mask_path.resolve())
        self.status_label.setText(message)

    def _apply_paint_stroke(self, tool: str, points: list[tuple[int, int]]) -> None:
        assets_dir = Path(self.session.get("assets_dir") or self.session_path.with_name(f"{self.session_path.stem}_assets"))
        overlay_path = Path(self.session.get("paint_overlay_path") or assets_dir / "paint_overlay.png")
        overlay = _overlay_to_array(overlay_path, self.image_size)
        stroke = self._stroke_mask(points)
        pixels = stroke > 0
        if tool == "paint_brush":
            r, g, b = _rgb_from_hex(self.paint_color, (255, 255, 255))
            overlay[pixels, 0] = r
            overlay[pixels, 1] = g
            overlay[pixels, 2] = b
            overlay[pixels, 3] = 255
        else:
            overlay[pixels] = 0
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        _array_to_overlay_image(overlay).save(overlay_path)
        self.session["paint_overlay_path"] = str(overlay_path)
        self.status_label.setText(f"브러시 레이어를 수정했습니다: {overlay_path.name}")

    def _inpaint_settings(self) -> dict[str, Any]:
        settings = {
            "method": self.inpaint_method_combo.currentData(),
            "device": self.inpaint_device_combo.currentData(),
            "seed": int(self.inpaint_seed_spin.value()),
            "steps": int(self.inpaint_steps_spin.value()),
            "inpainting_size": int(self.inpainting_size_spin.value()),
            "prompt": self.inpaint_prompt_edit.toPlainText().strip(),
            "huggingface_token": self.hf_token_edit.text().strip(),
            "low_vram": self.low_vram_checkbox.isChecked(),
            "luminance_correction": self.luminance_checkbox.isChecked(),
            "upscale_small_crops": self.upscale_crop_checkbox.isChecked(),
            "strict_mask_clipping": self.strict_clip_checkbox.isChecked(),
            "force_torch": self.force_torch_checkbox.isChecked(),
            "auto_run_after_brush": self.auto_inpaint_checkbox.isChecked(),
            "precision": self.lama_precision_combo.currentData() or "bf16",
            "kontext_backend": self.kontext_backend_combo.currentData(),
            "residual_diff_threshold": float(
                self.residual_threshold_spin.value()
            ),
            "opencv_radius": float(self.opencv_radius_spin.value()),
            "opencv_algorithm": self.opencv_algorithm_combo.currentData(),
        }
        self.session["inpaint_settings"] = settings
        self._remember_settings("inpaint", settings)
        return settings

    def _store_inpaint_controls(self, *_args) -> None:
        if not self._syncing:
            self._inpaint_settings()

    def run_inpainting_task(self) -> None:
        if self._inpainting_running:
            self.status_label.setText("인페인팅이 이미 실행 중입니다.")
            return
        mask_path = self.session.get("inpaint_mask_path")
        if not mask_path or not Path(mask_path).exists():
            QMessageBox.warning(
                self,
                "인페인팅",
                "먼저 도구 모음의 '인페인팅 브러시'로 지울 부분을 칠해 주세요.",
            )
            return
        mask = _load_mask(mask_path, self.image_size)
        if not np.any(mask):
            QMessageBox.warning(self, "인페인팅", "인페인팅 마스크가 비어 있습니다.")
            return
        settings = self._inpaint_settings()
        image = self._inpainting_source_image()
        self._inpainting_running = True

        def _task():
            return run_inpainting(image, mask, settings)

        def _success(result: Image.Image) -> None:
            try:
                self._push_history("인페인팅 결과 적용")
                assets_dir = Path(
                    self.session.get("assets_dir")
                    or self.session_path.with_name(f"{self.session_path.stem}_assets")
                )
                assets_dir.mkdir(parents=True, exist_ok=True)
                output_path = assets_dir / (
                    f"inpainted_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
                )
                result.convert("RGBA").save(output_path)
                self.session["inpainted_image_path"] = str(output_path.resolve())
                self.session["inpainted_background_text_free"] = True
                self.session["paint_overlay_path"] = ""
                self.session["inpaint_settings"] = settings
                save_session(self.session_path, self.session)
                self.reload_display_image()
                self.schedule_exact_preview()
                self.status_label.setText(
                    f"인페인팅을 완료했습니다: {output_path.name}"
                )
            finally:
                self._inpainting_running = False

        self._run_background(
            _task,
            _success,
            working_message="인페인팅을 실행 중입니다. 최초 모델 다운로드 중에는 시간이 걸릴 수 있습니다.",
            on_failure=lambda _message: setattr(self, "_inpainting_running", False),
        )

    def clear_inpaint_mask(self) -> None:
        mask_path = self.session.get("inpaint_mask_path")
        if not mask_path:
            self.status_label.setText("지울 인페인팅 마스크가 없습니다.")
            return
        self._push_history("인페인팅 마스크 비우기")
        path = Path(mask_path)
        if path.exists():
            _save_mask(path, np.zeros((self.image_size[1], self.image_size[0]), dtype=np.uint8))
        self.session["inpaint_mask_path"] = str(path.resolve())
        save_session(self.session_path, self.session)
        self.canvas.update()
        self.status_label.setText("인페인팅 브러시 마스크를 비웠습니다.")

    def reset_inpainting_result(self) -> None:
        self._push_history("인페인팅 결과 초기화")
        self.session["inpainted_image_path"] = ""
        self.session["inpainted_background_text_free"] = False
        save_session(self.session_path, self.session)
        self.reload_display_image()
        self.schedule_exact_preview()
        self.status_label.setText("인페인팅 결과를 초기화했습니다.")

    def _region_changed_for_fast_preview(self, region: dict[str, Any]) -> bool:
        return str(region.get("id") or "") in self._dirty_region_ids

    def fast_preview_regions(self) -> list[dict[str, Any]]:
        return []

    def _editor_background_source(self) -> tuple[Optional[Path], str]:
        candidates = (
            ("inpainted_image_path", "inpainted"),
            ("rendered_snapshot_path", "rendered"),
            ("translated_image_path", "translated"),
            ("edited_image_path", "edited"),
            ("base_image_path", "base"),
            ("original_snapshot_path", "original"),
            ("source_image_path", "source"),
        )
        for key, source_kind in candidates:
            value = self.session.get(key)
            if value and Path(value).is_file():
                return Path(value), source_kind
        return None, ""

    def _fast_erase_bbox(
        self,
        region: dict[str, Any],
        image_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        boxes = [_bbox(region)]
        baseline = region.get("render_baseline")
        if isinstance(baseline, dict):
            try:
                boxes.append([float(v) for v in baseline.get("bbox", [])[:4]])
            except Exception:
                pass
        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        padding = max(6.0, min(22.0, max(width, height) * 0.05))
        image_width, image_height = image_size
        left = max(0, min(image_width - 1, int(round(x1 - padding))))
        top = max(0, min(image_height - 1, int(round(y1 - padding))))
        right = max(left + 1, min(image_width, int(round(x2 + padding))))
        bottom = max(top + 1, min(image_height, int(round(y2 + padding))))
        return left, top, right, bottom

    def _paste_clean_regions_for_fast_edit(
        self,
        image: Image.Image,
        *,
        include_dirty_regions: bool,
    ) -> Image.Image:
        boxes = list(self.session.get("deleted_region_bboxes", []) or [])
        if include_dirty_regions:
            for region in self.session.get("regions", []):
                if self._region_changed_for_fast_preview(region):
                    boxes.append(self._fast_erase_bbox(region, image.size))
        if not boxes:
            return image
        clean_base = self._clean_base_image()
        if clean_base.size != image.size:
            clean_base = clean_base.resize(image.size, Image.Resampling.LANCZOS)
        for box in boxes:
            try:
                erase_box = tuple(int(v) for v in box[:4])
            except Exception:
                continue
            if len(erase_box) != 4:
                continue
            left, top, right, bottom = erase_box
            if right <= left or bottom <= top:
                continue
            image.paste(clean_base.crop(erase_box), erase_box)
        return image

    def _apply_paint_overlay_to(self, image: Image.Image) -> Image.Image:
        overlay_path = self.session.get("paint_overlay_path")
        if overlay_path and Path(overlay_path).exists():
            overlay = _load_rgba(overlay_path, image.size)
            image.alpha_composite(overlay)
        return image

    def _flatten_fast_editor_layers_after_save(self, saved_background_path: Path) -> None:
        baseline_path = saved_background_path.resolve()
        restore_mask_path = Path(self.session.get("restore_mask_path") or "")
        if restore_mask_path.is_file():
            session_without_restore = copy.deepcopy(self.session)
            session_without_restore["restore_mask_path"] = ""
            unmasked_image, unmasked_errors = render_session_image(
                session_without_restore
            )
            if not unmasked_errors:
                assets_dir = Path(
                    self.session.get("assets_dir")
                    or self.session_path.with_name(f"{self.session_path.stem}_assets")
                )
                assets_dir.mkdir(parents=True, exist_ok=True)
                baseline_path = assets_dir / "rendered_snapshot_without_restore.png"
                unmasked_image.convert("RGBA").save(baseline_path)
                baseline_path = baseline_path.resolve()
        self.session["rendered_snapshot_path"] = str(baseline_path)
        self._dirty_region_ids.clear()
        self.session["editor_dirty_region_ids"] = []
        self.session["deleted_region_bboxes"] = []
        self.session["paint_overlay_path"] = ""
        self.session["inpainted_image_path"] = ""
        self.session["inpainted_background_text_free"] = False
        for region in self.session.get("regions", []):
            region["render_baseline"] = _region_render_signature(region)

    def _preview_background_image(self) -> Image.Image:
        background_path, source_kind = self._editor_background_source()
        if background_path is not None:
            image = Image.open(background_path).convert("RGBA")
            return self._apply_paint_overlay_to(image)

        image = self._load_base_image().copy()
        return self._apply_paint_overlay_to(image)

    def _inpainting_source_image(self) -> Image.Image:
        inpainted_path = Path(self.session.get("inpainted_image_path") or "")
        if (
            self.session.get("inpainted_background_text_free")
            and inpainted_path.is_file()
        ):
            image = Image.open(inpainted_path).convert("RGBA")
        else:
            image = self._clean_base_image().copy()
        if image.size != self.image_size:
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        return self._apply_paint_overlay_to(image)

    def _text_restore_source_image(self) -> Optional[Image.Image]:
        session_copy = copy.deepcopy(self.session)
        session_copy["inpainted_image_path"] = ""
        session_copy["paint_overlay_path"] = ""
        try:
            image, errors = render_session_image(session_copy)
            if not errors:
                if image.size != self.image_size:
                    image = image.resize(self.image_size, Image.Resampling.LANCZOS)
                return image.convert("RGBA")
        except Exception:
            pass
        for key in ("rendered_snapshot_path", "translated_image_path", "edited_image_path"):
            value = self.session.get(key)
            if value and Path(value).is_file():
                return _load_rgba(value, self.image_size)
        return None

    def _translated_text_area_mask(self, size: tuple[int, int]) -> np.ndarray:
        width, height = size
        mask = np.zeros((height, width), dtype=bool)
        for region in self.session.get("regions", []) or []:
            if region.get("enabled", True) is False:
                continue
            x1, y1, x2, y2 = _bbox(region)
            pad = max(6, int(round(max(x2 - x1, y2 - y1) * 0.04)))
            left = max(0, min(width - 1, int(round(x1 - pad))))
            top = max(0, min(height - 1, int(round(y1 - pad))))
            right = max(left + 1, min(width, int(round(x2 + pad))))
            bottom = max(top + 1, min(height, int(round(y2 + pad))))
            mask[top:bottom, left:right] = True
        return mask

    def _restore_translated_text_pixels(
        self,
        image: Image.Image,
        inpaint_mask: np.ndarray,
    ) -> Image.Image:
        text_source = self._text_restore_source_image()
        if text_source is None:
            return image
        result = image.convert("RGBA")
        if result.size != self.image_size:
            result = result.resize(self.image_size, Image.Resampling.LANCZOS)
        if text_source.size != result.size:
            text_source = text_source.resize(result.size, Image.Resampling.LANCZOS)
        try:
            clean_base = self._clean_base_image()
        except Exception:
            return result
        if clean_base.size != result.size:
            clean_base = clean_base.resize(result.size, Image.Resampling.LANCZOS)

        mask = np.asarray(inpaint_mask)
        if mask.ndim > 2:
            mask = mask[..., 0]
        width, height = result.size
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8), iterations=1)

        source_arr = np.asarray(text_source, dtype=np.uint8)
        base_arr = np.asarray(clean_base, dtype=np.uint8)
        result_arr = np.asarray(result, dtype=np.uint8).copy()
        diff = np.max(
            np.abs(source_arr[..., :3].astype(np.int16) - base_arr[..., :3].astype(np.int16)),
            axis=2,
        )
        text_pixels = diff > 30
        region_area = self._translated_text_area_mask(result.size)
        restore_mask = text_pixels & region_area & (mask > 0)
        if not np.any(restore_mask):
            return result
        result_arr[restore_mask] = source_arr[restore_mask]
        return Image.fromarray(result_arr, mode="RGBA")

    def _clean_base_image(self) -> Image.Image:
        base_path = self.session.get("base_image_path")
        if base_path and Path(base_path).exists():
            return Image.open(base_path).convert("RGBA")
        return self._load_base_image().copy()

    def _font_file_for_render(self, render: dict[str, Any]) -> Optional[Path]:
        value = _resolve_repo_relative_path(render.get("font_dir"))
        if not value:
            return None
        path = Path(str(value))
        if path.is_file() and path.suffix.lower() in SUPPORTED_FONT_SUFFIXES:
            return path
        if path.is_dir():
            fonts = [
                item
                for item in path.rglob("*")
                if item.is_file()
                and item.suffix.lower() in SUPPORTED_FONT_SUFFIXES
            ]
            if not fonts:
                return None
            def score(item: Path) -> tuple[int, str]:
                name = item.stem.lower()
                if "regular" in name or "normal" in name:
                    return (0, item.name.casefold())
                if not any(token in name for token in ("bold", "italic", "oblique", "black")):
                    return (1, item.name.casefold())
                return (2, item.name.casefold())
            return sorted(fonts, key=score)[0]
        return None

    def _qfont_family_for_region(self, region: dict[str, Any]) -> str:
        render = region.get("render", {}) if isinstance(region.get("render"), dict) else {}
        font_file = self._font_file_for_render(render)
        if font_file is None:
            return "Malgun Gothic"
        key = str(font_file.resolve())
        cached = self._font_family_cache.get(key)
        if cached:
            return cached
        font_id = QFontDatabase.addApplicationFont(key)
        if font_id >= 0:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                self._font_family_cache[key] = families[0]
                return families[0]
        return "Malgun Gothic"

    def _qfont_style_for_region(self, region: dict[str, Any], family: str) -> str:
        render = region.get("render", {}) if isinstance(region.get("render"), dict) else {}
        font_file = self._font_file_for_render(render)
        if font_file is None:
            return ""
        key = str(font_file.resolve())
        if key in self._font_style_cache:
            return self._font_style_cache[key]
        fonts_root = Path(__file__).resolve().parents[1] / "fonts"
        desired = str(_font_descriptor(font_file, fonts_root).get("style") or "")
        desired_key = "".join(char for char in desired.casefold() if char.isalnum())
        matched = ""
        for style in QFontDatabase.styles(family):
            style_key = "".join(char for char in style.casefold() if char.isalnum())
            if style_key == desired_key:
                matched = style
                break
        self._font_style_cache[key] = matched or desired
        return self._font_style_cache[key]

    def _preview_text(self, region: dict[str, Any]) -> str:
        text = str(region.get("translation") or "").strip()
        _tag, text = extract_translation_tag(text)
        text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
        return text.replace("\\n", "\n")

    def draw_region_fast_text(
        self,
        painter: QPainter,
        region: dict[str, Any],
        rect: QRectF,
        scale: float,
    ) -> None:
        text = self._preview_text(region)
        if not text:
            return
        render = region.get("render", {}) if isinstance(region.get("render"), dict) else {}
        fixed_font_size = max(
            1,
            int(
                render.get("font_size")
                or render.get("max_font_size")
                or 24
            ),
        )
        pixel_size = max(1, int(round(fixed_font_size * scale)))
        font_family = self._qfont_family_for_region(region)
        font = QFont(font_family)
        font_style = self._qfont_style_for_region(region, font_family)
        if font_style:
            font.setStyleName(font_style)
        font.setPixelSize(pixel_size)
        raw_text = str(region.get("translation") or "")
        if "**" in raw_text:
            font.setBold(True)
        if re.search(r"(?<!\*)\*[^*].*?\*(?!\*)", raw_text):
            font.setItalic(True)

        align_key = str(render.get("text_alignment") or "center")
        if align_key == "left":
            horizontal_alignment = Qt.AlignmentFlag.AlignLeft
        elif align_key == "right":
            horizontal_alignment = Qt.AlignmentFlag.AlignRight
        else:
            horizontal_alignment = Qt.AlignmentFlag.AlignHCenter

        padding = max(0.0, float(render.get("padding_pixels") or 0.0) * scale)
        available_width = max(1, int(rect.width() - padding * 2.0))
        available_height = max(1, int(rect.height() - padding * 2.0))
        vertical = bool(region.get("vertical_stack", False))
        spacing_value = (
            render.get("vertical_line_spacing_mult")
            if vertical and render.get("vertical_line_spacing_mult") is not None
            else render.get("line_spacing_mult")
        )
        try:
            line_spacing_mult = float(spacing_value if spacing_value is not None else 1.0)
        except Exception:
            line_spacing_mult = 1.0
        line_spacing_mult = max(0.05, min(6.0, line_spacing_mult))

        def wrapped_horizontal_lines(metrics: QFontMetrics) -> list[str]:
            lines: list[str] = []
            paragraphs = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            for paragraph in paragraphs:
                if paragraph == "":
                    lines.append("")
                    continue
                current = ""
                for char in paragraph:
                    candidate = current + char
                    if current and metrics.horizontalAdvance(candidate) > available_width:
                        lines.append(current.rstrip())
                        current = char.lstrip()
                    else:
                        current = candidate
                if current:
                    lines.append(current.rstrip())
            return lines or [""]

        font.setPixelSize(pixel_size)

        color = QColor(region.get("font_color") or "#000000")
        outline_rgb = render.get("outline_color_rgb")
        outline = QColor(255, 255, 255)
        if isinstance(outline_rgb, (list, tuple)) and len(outline_rgb) >= 3:
            outline = QColor(int(outline_rgb[0]), int(outline_rgb[1]), int(outline_rgb[2]))
        outline_width = max(0, int(round(float(render.get("outline_width") or 0.0) * scale)))

        painter.save()
        center = rect.center()
        rotation = float(region.get("rotation_deg") or 0.0)
        if abs(rotation) > 0.01:
            painter.translate(center)
            painter.rotate(rotation)
            draw_rect = QRectF(
                -rect.width() / 2.0 + padding,
                -rect.height() / 2.0 + padding,
                max(1.0, rect.width() - padding * 2.0),
                max(1.0, rect.height() - padding * 2.0),
            )
        else:
            draw_rect = rect.adjusted(padding, padding, -padding, -padding)

        painter.setFont(font)
        painter.setClipRect(draw_rect.adjusted(-outline_width, -outline_width, outline_width, outline_width))
        metrics = QFontMetrics(font)
        line_step = max(1.0, metrics.height() * line_spacing_mult)

        def draw_text_with_outline(target_rect: QRectF, flags, value: str) -> None:
            if outline_width > 0:
                painter.setPen(outline)
                for dx, dy in (
                    (-outline_width, 0),
                    (outline_width, 0),
                    (0, -outline_width),
                    (0, outline_width),
                ):
                    painter.drawText(target_rect.translated(dx, dy), flags, value)
            painter.setPen(color)
            painter.drawText(target_rect, flags, value)

        if vertical:
            chars = [ch for ch in text if ch not in "\r\n"]
            if chars:
                total_h = metrics.height() + line_step * max(0, len(chars) - 1)
                y = draw_rect.center().y() - total_h / 2.0
                for ch in chars:
                    char_rect = QRectF(draw_rect.left(), y, draw_rect.width(), metrics.height())
                    draw_text_with_outline(char_rect, Qt.AlignmentFlag.AlignCenter, ch)
                    y += line_step
        else:
            lines = wrapped_horizontal_lines(metrics)
            total_h = metrics.height() + line_step * max(0, len(lines) - 1)
            y = draw_rect.center().y() - total_h / 2.0
            flags = horizontal_alignment | Qt.AlignmentFlag.AlignTop
            for line in lines:
                line_rect = QRectF(draw_rect.left(), y, draw_rect.width(), metrics.height())
                draw_text_with_outline(line_rect, flags, line)
                y += line_step
        painter.restore()

    def _qimage_to_pil(self, qimage: QImage) -> Image.Image:
        converted = qimage.convertToFormat(QImage.Format.Format_RGBA8888)
        width = converted.width()
        height = converted.height()
        ptr = converted.bits()
        ptr.setsize(converted.sizeInBytes())
        arr = np.frombuffer(ptr, np.uint8).reshape((height, converted.bytesPerLine()))
        arr = arr[:, : width * 4].reshape((height, width, 4)).copy()
        return Image.fromarray(arr, mode="RGBA")

    def compose_preview_image(self) -> Image.Image:
        return self._preview_background_image().convert("RGBA")

    def reload_display_image(self) -> None:
        image = self._preview_background_image()
        self.base_image = image.copy()
        self.displaying_rendered_result = False
        self.display_image = image
        self.display_qimage = _pil_to_qimage(image)
        self.image_size = image.size
        self.canvas.update()

    def save_current_session(self, *, push_history: bool = True) -> None:
        if push_history:
            self._push_history("편집 설정 변경")
        self.commit_region_form(mark_dirty=False)
        self._persist_editor_control_settings()
        self._persist_output_settings()
        save_session(self.session_path, self.session)
        self.refresh_region_list()
        self.status_label.setText("변경 내용을 저장했습니다.")

    def _persist_editor_control_settings(self) -> None:
        if hasattr(self, "show_help_checkbox"):
            self._store_editor_controls()
        if hasattr(self, "detector_combo"):
            self._ocr_settings()
        if hasattr(self, "inpaint_method_combo"):
            self._inpaint_settings()
        if hasattr(self, "editor_translation_source_combo"):
            self._translation_settings()

    def _persist_output_settings(self, *_args) -> None:
        if self._syncing:
            return
        output_settings = self.editor_settings.setdefault("output", {})
        if not isinstance(output_settings, dict):
            output_settings = {}
            self.editor_settings["output"] = output_settings
        if hasattr(self, "output_path_edit"):
            custom_output_path = self.output_path_edit.text().strip()
            self.session["custom_output_path"] = custom_output_path
            output_settings["custom_output_path"] = custom_output_path
        if hasattr(self, "overwrite_source_checkbox"):
            overwrite_source = self.overwrite_source_checkbox.isChecked()
            self.session["overwrite_source_on_save"] = overwrite_source
            output_settings["overwrite_source_on_save"] = overwrite_source
        _save_editor_settings(self.editor_settings)

    def choose_default_output_dir(self) -> None:
        current = ""
        if hasattr(self, "default_output_dir_edit"):
            current = self.default_output_dir_edit.text().strip()
        if not current:
            current = str(Path(self.session.get("edited_image_path") or self.session_path).parent)
        directory = QFileDialog.getExistingDirectory(
            self,
            "기본 저장 폴더 선택",
            current,
        )
        if not directory:
            return
        self.default_output_dir_edit.setText(directory)
        self._store_editor_controls()
        self.status_label.setText(f"기본 저장 폴더를 설정했습니다: {directory}")

    def choose_output_path(self) -> None:
        current = self.output_path_edit.text().strip()
        if not current:
            edited_path = self.session.get("edited_image_path")
            if edited_path:
                current = str(Path(edited_path).parent)
            else:
                current = str(self.session_path.parent)
        current_path = Path(current)
        if current_path.suffix:
            current = str(current_path.parent)
        target = QFileDialog.getExistingDirectory(
            self,
            "렌더링 결과 저장 폴더",
            current,
        )
        if not target:
            return
        self.output_path_edit.setText(target)
        self._persist_output_settings()
        save_session(self.session_path, self.session)

    def _default_export_filename(self) -> str:
        candidates = (
            self.session.get("source_display_path"),
            self.session.get("source_image_path"),
            self.session.get("translated_image_path"),
            self.session.get("rendered_snapshot_path"),
        )
        for value in candidates:
            text = str(value or "").strip()
            if not text:
                continue
            name = Path(text).name
            decoded = _decode_uploaded_relative_path(name)
            if decoded is not None:
                name = decoded.name
            stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", Path(name).stem).strip(" ._")
            if not stem:
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in SUPPORTED_IMAGE_SUFFIXES:
                suffix = ".png"
            return f"edited_{stem}{suffix}"
        stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", self.session_path.stem).strip(" ._")
        return f"edited_{stem or 'image'}.png"

    def _safe_relative_output_parts(self, value: Any) -> list[str]:
        raw = str(value or "").replace("\\", "/").strip().strip("/")
        if not raw:
            return []
        parts: list[str] = []
        for part in raw.split("/"):
            cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", part).strip(" .")
            if cleaned and cleaned not in {".", ".."}:
                parts.append(cleaned)
        return parts

    def _source_relative_output_name(self) -> Path:
        candidates = (
            self.session.get("source_display_path"),
            self.session.get("translated_image_path"),
            self.session.get("source_image_path"),
            self.session.get("rendered_snapshot_path"),
        )
        for value in candidates:
            text = str(value or "").strip()
            if not text:
                continue
            parts = self._safe_relative_output_parts(text)
            if not parts:
                continue
            last = Path(parts[-1])
            if last.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                parts[-1] = f"{last.stem or 'image'}.png"
            return Path(*parts)
        default_name = self._default_export_filename()
        if default_name.startswith("edited_"):
            default_name = default_name[len("edited_") :]
        return Path(default_name)

    def _custom_output_target(self, *, preserve_source_name: bool) -> Optional[Path]:
        if not hasattr(self, "output_path_edit"):
            return None
        custom_path = self.output_path_edit.text().strip()
        if not custom_path:
            return None
        selected = Path(custom_path).expanduser().resolve()
        if selected.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            return selected
        if preserve_source_name:
            return selected / self._source_relative_output_name()
        return selected / self._default_export_filename()

    def _overwrite_target_path(self) -> Path:
        candidates: list[Any] = []
        if self.session.get("standalone_image_session"):
            source_path = self.session.get("source_image_path")
            source_overwrite_available = bool(
                self.session.get(
                    "source_overwrite_available",
                    not _is_temporary_upload_path(source_path),
                )
            )
            if not source_overwrite_available:
                raise FileNotFoundError(
                    "이 세션은 브라우저 업로드/ZIP 임시 파일에서 만들어져 실제 원본 경로를 알 수 없습니다. "
                    "원본 파일에 그대로 덮어쓰려면 WebUI 고급 이미지 편집기의 '원본 위치 연결'에서 "
                    "실제 원본 폴더, 이미지 또는 ZIP을 연결한 뒤 편집 세션을 다시 만들거나, "
                    "체크를 해제한 뒤 저장 위치를 지정해 저장하세요."
                )
            explicit_target = self.session.get("overwrite_target_path")
            if explicit_target:
                candidates.append(explicit_target)
            candidates.append(self.session.get("source_image_path"))
        else:
            explicit_target = self.session.get("overwrite_target_path")
            if explicit_target:
                candidates.append(explicit_target)
            candidates.append(self.session.get("translated_image_path"))
            candidates.append(self.session.get("source_image_path"))

        candidates.append(self.session.get("translated_image_path"))
        candidates.append(self.session.get("source_image_path"))

        seen: set[str] = set()
        for value in candidates:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            target = Path(text).expanduser().resolve()
            if target.is_file():
                return target

        raise FileNotFoundError(
            "덮어쓸 파일을 찾을 수 없습니다. 번역 결과 또는 불러온 이미지 파일 경로가 세션에 없습니다."
        )

    def _overwrite_archive_reference(self) -> Optional[tuple[Path, str]]:
        archive_value = str(self.session.get("source_archive_path") or "").strip()
        member = (
            str(self.session.get("source_archive_member") or "")
            .replace("\\", "/")
            .strip("/")
        )
        if not archive_value and not member:
            return None
        if not archive_value or not member:
            raise FileNotFoundError(
                "원본 ZIP 연결 정보가 불완전합니다. WebUI에서 원본 ZIP을 다시 연결해 편집 세션을 만들어 주세요."
            )
        archive_path = Path(archive_value).expanduser().resolve()
        if not archive_path.is_file():
            raise FileNotFoundError(f"연결한 원본 ZIP을 찾을 수 없습니다: {archive_path}")
        try:
            with zipfile.ZipFile(archive_path, "r") as archive:
                normalized_names = {
                    name.replace("\\", "/").strip("/")
                    for name in archive.namelist()
                }
        except zipfile.BadZipFile as exc:
            raise ValueError(f"연결한 원본 ZIP이 손상되었습니다: {archive_path}") from exc
        if member not in normalized_names:
            raise FileNotFoundError(
                f"원본 ZIP에서 덮어쓸 이미지를 찾을 수 없습니다: {member}"
            )
        return archive_path, member

    def _selected_output_target(self) -> Optional[Path]:
        self._persist_output_settings()
        if self.overwrite_source_checkbox.isChecked():
            if self._overwrite_archive_reference() is not None:
                return None
            return self._overwrite_target_path()
        return self._custom_output_target(preserve_source_name=False)

    @staticmethod
    def _save_image_to_target(image: Image.Image, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = target.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".bmp"}:
            image.convert("RGB").save(target, quality=95)
        else:
            image.convert("RGBA").save(target)

    @staticmethod
    def _image_bytes_for_name(image: Image.Image, name: str) -> bytes:
        suffix = Path(name).suffix.lower()
        output = io.BytesIO()
        if suffix in {".jpg", ".jpeg"}:
            image.convert("RGB").save(output, format="JPEG", quality=95)
        elif suffix == ".webp":
            image.convert("RGBA").save(output, format="WEBP", quality=95)
        elif suffix == ".bmp":
            image.convert("RGB").save(output, format="BMP")
        elif suffix in {".tif", ".tiff"}:
            image.convert("RGBA").save(output, format="TIFF")
        else:
            image.convert("RGBA").save(output, format="PNG", compress_level=2)
        return output.getvalue()

    @classmethod
    def _save_image_to_zip(
        cls,
        image: Image.Image,
        archive_path: Path,
        member: str,
    ) -> None:
        replacement = cls._image_bytes_for_name(image, member)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{archive_path.stem}_",
            suffix=".tmp.zip",
            dir=str(archive_path.parent),
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        replaced = False
        try:
            with zipfile.ZipFile(archive_path, "r") as source_archive:
                archive_comment = source_archive.comment
                with zipfile.ZipFile(temporary_path, "w") as target_archive:
                    target_archive.comment = archive_comment
                    for info in source_archive.infolist():
                        normalized = info.filename.replace("\\", "/").strip("/")
                        if normalized == member:
                            target_archive.writestr(info, replacement)
                            replaced = True
                        else:
                            target_archive.writestr(info, source_archive.read(info.filename))
            if not replaced:
                raise FileNotFoundError(
                    f"원본 ZIP에서 덮어쓸 이미지를 찾을 수 없습니다: {member}"
                )
            os.replace(temporary_path, archive_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def save_plain_rendered_image(self, _checked: bool = False) -> None:
        self.render_and_save(embed_editor_info=False)

    def render_and_save(
        self,
        _checked: bool = False,
        *,
        embed_editor_info: bool = True,
    ) -> None:
        self.commit_region_form(mark_dirty=False)
        self._persist_editor_control_settings()
        self._persist_output_settings()
        save_session(self.session_path, self.session)
        try:
            archive_target = (
                self._overwrite_archive_reference()
                if self.overwrite_source_checkbox.isChecked()
                else None
            )
            export_target = (
                None if archive_target is not None else self._selected_output_target()
            )
            direct_output_path = (
                export_target
                if export_target is not None
                and self.overwrite_source_checkbox.isChecked()
                else None
            )
            output_path = Path(
                direct_output_path
                or self.session.get("edited_image_path")
                or self.session_path.with_name(f"{self.session_path.stem}_edited.png")
            ).resolve()
            self.session["edited_image_path"] = str(output_path)
            if direct_output_path is not None:
                self.session["overwrite_target_path"] = str(output_path)
            save_session(self.session_path, self.session)
            image, output_path, render_errors = rerender_session(self.session_path)
            self.session = load_session(self.session_path)
            if archive_target is not None:
                archive_path, archive_member = archive_target
                self._save_image_to_zip(image, archive_path, archive_member)
                self.session["last_export_path"] = (
                    f"{archive_path}!/{archive_member}"
                )
            elif export_target is not None and direct_output_path is None:
                self._save_image_to_target(image, export_target)
                self.session["last_export_path"] = str(export_target)
            elif direct_output_path is not None:
                self.session["last_export_path"] = str(output_path)
                self.session["overwrite_target_path"] = str(output_path)
            if not render_errors:
                self._flatten_fast_editor_layers_after_save(output_path)
            self._persist_output_settings()
            save_session(self.session_path, self.session)
            package_warnings: list[str] = []
            if embed_editor_info and archive_target is None:
                embedded_targets: list[Path] = [output_path]
                if export_target is not None and export_target.resolve() != output_path.resolve():
                    embedded_targets.append(export_target.resolve())
                for target in embedded_targets:
                    try:
                        embed_editor_package(target, self.session_path)
                    except Exception as exc:
                        package_warnings.append(f"{target}: {exc}")
            self.reload_display_image()
            if embed_editor_info:
                message = f"렌더링 결과를 저장했습니다: {output_path}"
            else:
                message = f"편집 정보 없는 일반 이미지로 저장했습니다: {output_path}"
            if archive_target is not None:
                archive_path, archive_member = archive_target
                message += (
                    f"\n원본 ZIP에 덮어썼습니다: {archive_path}"
                    f"\nZIP 내부 파일: {archive_member}"
                )
            elif export_target is not None:
                message += f"\n지정한 위치에도 저장했습니다: {export_target}"
            if render_errors:
                message += (
                    "\n일부 영역 렌더링 경고가 있어 기준값은 갱신하지 않았습니다.\n"
                    + "\n".join(render_errors[:5])
                )
            if package_warnings:
                message += (
                    "\nEditor package embedding warnings:\n"
                    + "\n".join(package_warnings[:3])
                )
            self.status_label.setText(message)
            saved_name = (
                Path(archive_target[1]).name
                if archive_target is not None
                else Path(export_target or output_path).name
            )
            self._show_save_toast(f"{saved_name} 저장 완료")
        except Exception as exc:
            QMessageBox.critical(self, "렌더링 실패", str(exc))

    def _capture_state(self) -> dict[str, Any]:
        masks: dict[str, np.ndarray] = {}
        for region in self.session.get("regions", []):
            path = region.get("mask_path")
            if path and Path(path).exists():
                masks[str(path)] = _load_mask(path, self.image_size)
        inpaint_mask_path = self.session.get("inpaint_mask_path")
        if inpaint_mask_path and Path(inpaint_mask_path).exists():
            masks[str(inpaint_mask_path)] = _load_mask(
                inpaint_mask_path, self.image_size
            )
        restore_mask_path = self.session.get("restore_mask_path")
        if restore_mask_path and Path(restore_mask_path).exists():
            masks[str(restore_mask_path)] = _load_mask(
                restore_mask_path, self.image_size
            )
        overlay_path = self.session.get("paint_overlay_path")
        overlay = None
        if overlay_path and Path(overlay_path).exists():
            overlay = _overlay_to_array(overlay_path, self.image_size)
        return {
            "session": copy.deepcopy(self.session),
            "masks": masks,
            "overlay_path": overlay_path,
            "overlay": overlay,
            "selected_index": self.selected_index,
        }

    def _restore_state(self, state: dict[str, Any]) -> None:
        self.session = copy.deepcopy(state["session"])
        self._dirty_region_ids = {
            str(region_id)
            for region_id in self.session.get("editor_dirty_region_ids", [])
            if str(region_id)
        }
        for path, mask in state.get("masks", {}).items():
            _save_mask(Path(path), mask)
        overlay_path = state.get("overlay_path")
        overlay = state.get("overlay")
        if overlay_path and overlay is not None:
            Path(overlay_path).parent.mkdir(parents=True, exist_ok=True)
            _array_to_overlay_image(overlay).save(overlay_path)
        self.selected_index = state.get("selected_index")
        save_session(self.session_path, self.session)
        self.reload_display_image()
        self.refresh_region_list()
        self.load_selected_region_to_form()
        self.schedule_exact_preview()

    def _push_history(self, label: str = "편집") -> None:
        state = self._capture_state()
        state["history_label"] = str(label or "편집")
        self.history.append(state)
        if len(self.history) > 80:
            self.history.pop(0)
        self.redo_stack.clear()
        self._refresh_history_menu()

    def _refresh_history_menu(self) -> None:
        if not hasattr(self, "history_menu"):
            return
        self.history_menu.clear()
        if self.history:
            for steps, state in enumerate(reversed(self.history), start=1):
                label = str(state.get("history_label") or "편집")
                action = self.history_menu.addAction(f"{steps}. {label}")
                action.triggered.connect(
                    lambda _checked=False, count=steps: self.undo_steps(count)
                )
        else:
            empty_action = self.history_menu.addAction("실행 취소할 작업이 없습니다")
            empty_action.setEnabled(False)

        undo_label = (
            str(self.history[-1].get("history_label") or "편집")
            if self.history
            else ""
        )
        redo_label = (
            str(self.redo_stack[-1].get("history_label") or "편집")
            if self.redo_stack
            else ""
        )
        self.undo_action.setEnabled(bool(self.history))
        self.redo_action.setEnabled(bool(self.redo_stack))
        self.undo_action.setToolTip(
            f"실행 취소: {undo_label} (Ctrl+Z)"
            if undo_label
            else "실행 취소할 작업이 없습니다"
        )
        self.redo_action.setToolTip(
            f"다시 실행: {redo_label} (Ctrl+Y)"
            if redo_label
            else "다시 실행할 작업이 없습니다"
        )

    def undo(self) -> None:
        self.undo_steps(1)

    def undo_steps(self, count: int) -> None:
        steps = min(max(0, int(count)), len(self.history))
        if steps <= 0:
            return
        current = self._capture_state()
        undone_labels: list[str] = []
        for _ in range(steps):
            previous = self.history.pop()
            label = str(previous.get("history_label") or "편집")
            current["history_label"] = label
            self.redo_stack.append(current)
            current = previous
            undone_labels.append(label)
        self._restore_state(current)
        self._refresh_history_menu()
        if steps == 1:
            self.status_label.setText(f"실행 취소: {undone_labels[0]}")
        else:
            self.status_label.setText(
                f"{steps}개 작업을 실행 취소했습니다: {undone_labels[-1]}까지"
            )

    def redo(self) -> None:
        if not self.redo_stack:
            return
        state = self.redo_stack.pop()
        label = str(state.get("history_label") or "편집")
        current = self._capture_state()
        current["history_label"] = label
        self.history.append(current)
        self._restore_state(state)
        self._refresh_history_menu()
        self.status_label.setText(f"다시 실행: {label}")

    def closeEvent(self, event) -> None:
        try:
            self.commit_region_form(mark_dirty=False)
            self._persist_editor_control_settings()
            self._persist_output_settings()
            save_session(self.session_path, self.session)
        except Exception:
            pass
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "save_toast") and self.save_toast.isVisible():
            self._position_save_toast()


def _write_lifecycle_log(message: str) -> None:
    try:
        repo_root = Path(__file__).resolve().parents[1]
        log_dir = repo_root / "output" / "advanced_editor_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "advanced_editor_lifecycle.log"
        timestamp = datetime.now().isoformat(timespec="seconds")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} PID={os.getpid()} {message}\n")
    except Exception:
        pass


def launch(session_path: str) -> int:
    resolved_session_path = str(resolve_editor_session_input(session_path))
    _write_lifecycle_log(f"launch started session={resolved_session_path}")
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(True)
    window = AdvancedEditorWindow(resolved_session_path)
    if getattr(window, "_start_maximized", False):
        window.showMaximized()
    else:
        window.showNormal()
    window.raise_()
    window.activateWindow()

    if sys.platform == "win32":
        try:
            import ctypes

            hwnd = int(window.winId())
            ctypes.windll.user32.ShowWindow(hwnd, 9)
            ctypes.windll.user32.SetForegroundWindow(hwnd)
        except Exception:
            pass

    _write_lifecycle_log("window shown")
    app.aboutToQuit.connect(lambda: _write_lifecycle_log("application quitting"))
    exit_code = app.exec()
    _write_lifecycle_log(f"event loop exited code={exit_code}")
    return exit_code


def main() -> int:
    if len(sys.argv) < 2:
        print("사용법: python -m advanced_editor.launcher <edit_session.json>")
        return 2
    try:
        return launch(sys.argv[1])
    except Exception as exc:
        try:
            repo_root = Path(__file__).resolve().parents[1]
            log_dir = repo_root / "output" / "advanced_editor_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            crash_log = log_dir / "advanced_editor_crash.log"
            with open(crash_log, "a", encoding="utf-8") as f:
                f.write("\n--- Advanced editor crash ---\n")
                f.write(f"Session: {sys.argv[1] if len(sys.argv) > 1 else ''}\n")
                f.write(traceback.format_exc())
        except Exception:
            pass
        app = QApplication.instance() or QApplication(sys.argv[:1])
        QMessageBox.critical(None, "고급 편집기 실행 실패", str(exc))
        print(f"고급 편집기 실행 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
