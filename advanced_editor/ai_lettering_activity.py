from pathlib import Path
from random import shuffle

from PyQt6.QtCore import QEvent, QRect, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt6.QtWidgets import QSizePolicy, QWidget

from advanced_editor.standalone import allow_loading_animation


class AiLetteringActivity(QWidget):
    """Paint into the existing toolbar spacer while an AI lettering job runs."""

    _LABEL = "AI 식자 중"
    _ICON_SIZE = 22
    _SPRITE_SIZE = 44
    _FRAMES = 20
    _SHEET = Path(__file__).resolve().parents[1] / "assets" / "ai-lettering-status.png"

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("TopBarSpacer")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._active = False
        self._anchor = None
        self._sheet = None
        self._icon = None
        self._order = []
        self._frame = 0
        self._timer = QTimer(self, interval=105)
        self._timer.timeout.connect(self._advance)

    def align_above(self, anchor):
        self._anchor = anchor
        anchor.installEventFilter(self)

    def set_active(self, active: bool) -> None:
        if active == self._active:
            return
        self._active = active
        self.setAccessibleName(self._LABEL if active else "")
        self.setToolTip(self.accessibleName())
        if active:
            if self._sheet is None:
                self._sheet = QPixmap(str(self._SHEET))
            self._next_icon()
            self._start_motion()
        else:
            self._timer.stop()
        self.update()

    def _start_motion(self):
        if self._active and self.isVisible() and self._icon is not None and allow_loading_animation():
            self._timer.start()

    def _next_icon(self):
        self._frame = 0
        if self._sheet.isNull():
            return
        if not self._order:
            # Sprite rows: I, J, K, E, H, C, B. Play each once before reshuffling.
            self._order = list(range(self._sheet.height() // self._SPRITE_SIZE))
            shuffle(self._order)
            if self._order[-1] == self._icon:
                self._order[0], self._order[-1] = self._order[-1], self._order[0]
        self._icon = self._order.pop()

    def _advance(self):
        self._frame += 1
        if self._frame >= self._FRAMES:
            self._next_icon()
        icon_rect, _ = self._activity_rects()
        self.update(icon_rect)

    def _activity_rects(self):
        font = self.fontMetrics()
        text_width = font.horizontalAdvance(self._LABEL) + 5
        show_label = self.width() >= self._ICON_SIZE + text_width + 14
        width = self._ICON_SIZE + (text_width + 5 if show_label else 0)
        right_limit = max(0, self.width() - width - 6)
        x = right_limit
        if show_label and self._anchor is not None and self._anchor.isVisible():
            window = self.window()
            anchor_x = self._anchor.mapTo(window, self._anchor.rect().topLeft()).x()
            own_x = self.mapTo(window, self.rect().topLeft()).x()
            x = min(right_limit, max(2, anchor_x - own_x))
        icon = QRect(x, (self.height() - self._ICON_SIZE) // 2, self._ICON_SIZE, self._ICON_SIZE)
        label = QRect(icon.right() + 6, 0, text_width, self.height()) if show_label else QRect()
        return icon, label

    def eventFilter(self, watched, event):
        if self._active and event.type() in (QEvent.Type.Move, QEvent.Type.Resize):
            self.update()
        return super().eventFilter(watched, event)

    def showEvent(self, event):
        super().showEvent(event)
        self._start_motion()

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._active or self.width() < self._ICON_SIZE:
            return
        icon, label = self._activity_rects()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setPen(QColor("#d6cbfb"))
        if self._icon is not None:
            size = self._SPRITE_SIZE
            frame = QRect(self._frame * size, self._icon * size, size, size)
            painter.drawPixmap(icon, self._sheet, frame)
        else:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.drawEllipse(icon.adjusted(6, 6, -6, -6))
        if not label.isEmpty():
            text_font = self.font()
            text_font.setWeight(QFont.Weight.DemiBold)
            painter.setFont(text_font)
            painter.drawText(label, Qt.AlignmentFlag.AlignVCenter, self._LABEL)
        painter.end()
