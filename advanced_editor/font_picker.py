"""Compact, shared-scroll font palette; keys remain the editor's font keys."""
from pathlib import Path
import json

from PyQt6.QtCore import QEvent, QPoint, QRect, QSize, QSettings, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QListView, QListWidget, QListWidgetItem, QMessageBox,
    QColorDialog, QComboBox, QDialog, QDialogButtonBox, QHeaderView,
    QScrollArea, QTableWidget, QTableWidgetItem,
    QPushButton, QStyle, QStyledItemDelegate, QVBoxLayout, QWidget,
)


TAGS = ("일반대사", "남성", "여성", "강조", "작게", "쭈글", "울먹", "효과음", "손글씨")
GROUPS = {
    "전체": (), "일반대사": ("일반대사",),
    "특수대사": ("강조", "작게", "쭈글", "울먹"),
    "효과음/손글씨": ("효과음", "손글씨"),
}


def tag_ink(color):
    channels = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4
                for v in QColor(color).getRgbF()[:3]]
    luminance = sum(v * w for v, w in zip(channels, (.2126, .7152, .0722)))
    return "#000000" if luminance > .179 else "#ffffff"


class TagManager(QDialog):
    def __init__(self, picker):
        super().__init__(picker.combo.window())
        self.picker = picker
        self.setWindowTitle("태그 관리")
        self.setStyleSheet(picker.styleSheet() + "QDialog { background: #f4f7fb; } QTableWidget { background: white; color: #172033; gridline-color: #d5dde8; }")
        self.resize(520, 420)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("이름을 수정하거나 색상을 클릭하세요. 변경은 저장할 때 적용됩니다."))
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["태그 이름", "색상"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)
        for tag in picker.tag_definitions:
            self.add_row(tag)
        actions = QHBoxLayout()
        add = QPushButton("태그 추가")
        add.clicked.connect(lambda: self.add_row())
        remove = QPushButton("선택 삭제")
        remove.clicked.connect(self.remove_rows)
        actions.addWidget(add)
        actions.addWidget(remove)
        actions.addStretch()
        layout.addLayout(actions)
        note = QLabel("태그를 삭제하면 해당 태그의 폰트 지정도 함께 해제됩니다.")
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("저장")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def add_row(self, definition=None):
        definition = definition or {"name": "", "color": "#e8edf3", "group": ""}
        row = self.table.rowCount()
        self.table.insertRow(row)
        item = QTableWidgetItem(definition["name"])
        item.setData(Qt.ItemDataRole.UserRole, dict(definition))
        self.table.setItem(row, 0, item)
        button = QPushButton()
        self.set_color(button, definition["color"])
        button.clicked.connect(lambda: self.pick_color(button))
        self.table.setCellWidget(row, 1, button)
        if not definition["name"]:
            self.table.setCurrentItem(item)
            self.table.editItem(item)

    def set_color(self, button, color):
        button.setProperty("tagColor", color)
        button.setText(color.upper())
        button.setStyleSheet(f"background:{color}; color:{tag_ink(color)}; border:0; padding:5px;")
        button.setAccessibleName("태그 색상 " + color)

    def pick_color(self, button):
        color = QColorDialog.getColor(QColor(button.property("tagColor")), self, "태그 색상")
        if color.isValid():
            self.set_color(button, color.name())

    def remove_rows(self):
        for row in sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(row)

    def save(self):
        definitions, rename = [], {}
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            old = item.data(Qt.ItemDataRole.UserRole)
            name = item.text().strip()
            if not name or len(name) > 40 or any(ord(c) < 32 for c in name) or any(t["name"].casefold() == name.casefold() for t in definitions):
                QMessageBox.warning(self, "태그 이름 확인", "이름은 중복 없이 1~40자로 입력하세요. 줄바꿈은 사용할 수 없습니다.")
                return
            definitions.append({"name": name, "color": self.table.cellWidget(row, 1).property("tagColor"), "group": old.get("group", "")})
            if old["name"]:
                rename[old["name"]] = name
        if self.picker.replace_tags(definitions, rename):
            self.accept()


class FilterManager(QDialog):
    def __init__(self, picker):
        super().__init__(picker.combo.window())
        self.picker = picker
        self.setWindowTitle("상단 필터 관리")
        self.resize(560, 430)
        self.setStyleSheet(picker.styleSheet() + """
            QDialog { background: #f4f7fb; }
            QListWidget { border: 1px solid #d5dde8; border-radius: 5px; }
            QListWidget::item { padding: 6px 8px; }
            QListWidget::item:selected { background: #dbeafe; color: #172033; }
        """)
        layout = QVBoxLayout(self)
        description = QLabel("필터 이름은 더블클릭으로 수정하고, 끌어서 순서를 바꾸세요.\n포함할 태그 중 하나라도 있는 폰트를 표시합니다. ‘대사’는 첫 번째에 고정됩니다.")
        layout.addWidget(description)
        row = QHBoxLayout()
        self.entries = QListWidget()
        self.entries.setAccessibleName("상단 필터 목록")
        self.entries.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.tags = QListWidget()
        self.tags.setAccessibleName("필터에 포함할 태그")
        for title, widget in (("필터 · 표시 순서", self.entries), ("포함할 태그", self.tags)):
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            column.addWidget(widget)
            row.addLayout(column)
        layout.addLayout(row, 1)
        for definition in picker.filter_definitions:
            self.add_filter(definition)
        self.entries.currentItemChanged.connect(self.show_tags)
        self.tags.itemChanged.connect(self.update_tags)
        actions = QHBoxLayout()
        add = QPushButton("필터 추가")
        add.clicked.connect(lambda: self.add_filter())
        remove = QPushButton("선택 삭제")
        remove.clicked.connect(lambda: self.entries.takeItem(self.entries.currentRow()) if self.entries.currentRow() >= 0 else None)
        actions.addWidget(add)
        actions.addWidget(remove)
        actions.addStretch()
        layout.addLayout(actions)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("저장")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.entries.setCurrentRow(0)
        self.show_tags()

    def add_filter(self, definition=None):
        item = QListWidgetItem(definition["name"] if definition else "새 필터")
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        item.setData(Qt.ItemDataRole.UserRole, list(definition["tags"]) if definition else [])
        self.entries.addItem(item)
        if definition is None:
            self.entries.setCurrentItem(item)
            self.entries.editItem(item)

    def show_tags(self, *_args):
        self.tags.blockSignals(True)
        self.tags.clear()
        current = self.entries.currentItem()
        self.tags.setEnabled(current is not None)
        selected = current.data(Qt.ItemDataRole.UserRole) if current else []
        for tag in self.picker.tag_definitions:
            item = QListWidgetItem(tag["name"])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if tag["name"] in selected else Qt.CheckState.Unchecked)
            self.tags.addItem(item)
        self.tags.blockSignals(False)

    def update_tags(self, *_args):
        current = self.entries.currentItem()
        if current:
            current.setData(Qt.ItemDataRole.UserRole, [self.tags.item(i).text() for i in range(self.tags.count()) if self.tags.item(i).checkState() == Qt.CheckState.Checked])

    def save(self):
        definitions = []
        names = {"전체", "대사"}
        for i in range(self.entries.count()):
            item = self.entries.item(i)
            name = item.text().strip()
            if not name or len(name) > 40 or any(ord(c) < 32 for c in name) or name.casefold() in names:
                QMessageBox.warning(self, "필터 이름 확인", "이름은 ‘전체’, ‘대사’를 제외하고 중복 없이 1~40자로 입력하세요.")
                return
            names.add(name.casefold())
            definitions.append({"name": name, "tags": item.data(Qt.ItemDataRole.UserRole)})
        previous = self.picker.settings.value("filters")
        self.picker.settings.setValue("filters", json.dumps(definitions, ensure_ascii=False))
        self.picker.settings.sync()
        if self.picker.settings.status() != QSettings.Status.NoError:
            if previous is None:
                self.picker.settings.remove("filters")
            else:
                self.picker.settings.setValue("filters", previous)
            QMessageBox.warning(self, "저장 실패", "필터를 저장하지 못했습니다. 폴더 쓰기 권한을 확인하세요.")
            return
        self.picker.filter_definitions = definitions
        self.picker.rebuild_filters()
        self.picker.filter_items()
        self.accept()


def split_label(label):
    # Only known collection prefixes: do not guess a family from arbitrary words.
    for prefix in ("학교안심", "나눔손글씨", "나눔", "온글잎", "마포"):
        if label.startswith(prefix) and label[len(prefix):].strip():
            return prefix, label[len(prefix):].lstrip()
    return "", label


class FontDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        return self.parent().gridSize()

    def paint(self, painter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = option.rect.adjusted(3, 2, -3, -2)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        if selected or hovered:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#dbeafe" if selected else "#eff6ff"))
            painter.drawRoundedRect(rect, 5, 5)
        if option.state & QStyle.StateFlag.State_HasFocus:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#2563eb"), 1))
            painter.drawRoundedRect(rect, 5, 5)
        painter.setFont(option.font)
        metrics = painter.fontMetrics()
        prefix, name = split_label(index.data())
        x, y = rect.left() + 9, rect.top() + 21
        if prefix:
            painter.setPen(QColor("#586578"))
            painter.drawText(x, y, prefix)
            x += metrics.horizontalAdvance(prefix + " ")
        painter.setPen(QColor("#172033"))
        painter.drawText(x, y, metrics.elidedText(name, Qt.TextElideMode.ElideRight, max(0, rect.right() - x - 8)))
        x = rect.left() + 9
        tags = index.data(Qt.ItemDataRole.UserRole + 1) or []
        for tag in tags:
            width = metrics.horizontalAdvance(tag) + 10
            if x + width > rect.right() - 6:
                painter.setPen(QColor("#475569"))
                painter.drawText(x, y + 22, "+")
                break
            badge = QRect(x, rect.top() + 29, width, 19)
            painter.setPen(Qt.PenStyle.NoPen)
            color = self.parent().property("tagColors").get(tag, "#e8edf3")
            painter.setBrush(QColor(color))
            painter.drawRoundedRect(badge, 3, 3)
            painter.setPen(QColor(tag_ink(color)))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, tag)
            x += width + 4
        painter.restore()


class FontPicker(QFrame):
    def __init__(self, combo, settings_path=None):
        super().__init__(combo, Qt.WindowType.Popup)
        self.combo = combo
        self.settings = QSettings(str(settings_path or Path(__file__).resolve().parents[1] / "font_tags.ini"), QSettings.Format.IniFormat)
        self.tag_definitions = json.loads(self.settings.value("definitions", "null"))
        if self.tag_definitions is None:
            self.tag_definitions = [{"name": tag, "color": "#e8edf3", "group": next((group for group, tags in GROUPS.items() if tag in tags), "")} for tag in TAGS]
        self.filter_definitions = json.loads(self.settings.value("filters", "null"))
        if self.filter_definitions is None:
            self.filter_definitions = [{"name": name, "tags": [tag["name"] for tag in self.tag_definitions if tag.get("group") == name]} for name in GROUPS if name not in ("전체", "일반대사")]
        self.filter_definitions = [entry for entry in self.filter_definitions if entry["name"] not in ("전체", "대사")]
        self.group = "대사"
        self.setObjectName("FontPicker")
        self.setStyleSheet("""
            QFrame#FontPicker { background: #ffffff; border: 1px solid #c4cfdd; border-radius: 6px; }
            QLineEdit { background: #f4f7fb; color: #172033; border: 1px solid #c4cfdd; border-radius: 4px; padding: 5px 8px; }
            QPushButton { color: #475569; background: transparent; border: 0; border-radius: 4px; padding: 5px 9px; }
            QPushButton:hover { background: #eff6ff; }
            QPushButton:checked { color: #1d4ed8; background: #dbeafe; }
            QPushButton:focus { border: 1px solid #2563eb; }
            QListWidget { background: white; border: 0; outline: 0; }
            QLabel, QCheckBox { color: #475569; background: transparent; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(6)
        top = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("폰트 이름 검색")
        self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName("폰트 이름 검색")
        self.search.textChanged.connect(self.filter_items)
        self.search.installEventFilter(self)
        top.addWidget(self.search, 1)
        self.edit = QPushButton("태그 편집")
        self.edit.setCheckable(True)
        self.edit.toggled.connect(self.edit_mode)
        top.addWidget(self.edit)
        layout.addLayout(top)
        filters = QHBoxLayout()
        filters.setSpacing(3)
        self.filters = QButtonGroup(self)
        self.all_button = QPushButton()
        self.all_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogResetButton))
        self.all_button.setFixedSize(26, 26)
        self.all_button.setCheckable(True)
        self.all_button.setAccessibleName("전체 폰트 보기")
        self.all_button.setToolTip("전체 폰트 보기 · 검색과 필터 초기화")
        self.all_button.clicked.connect(self.reset_filters)
        self.all_button.installEventFilter(self)
        self.filters.addButton(self.all_button)
        filters.addWidget(self.all_button)
        filter_scroll = QScrollArea()
        filter_scroll.setWidgetResizable(True)
        filter_scroll.setFrameShape(QFrame.Shape.NoFrame)
        filter_scroll.setFixedHeight(32)
        filter_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        filter_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        filter_scroll.setStyleSheet("""
            QScrollArea, QScrollArea > QWidget > QWidget { background: #ffffff; border: 0; }
            QScrollBar:horizontal { height: 4px; background: #ffffff; }
            QScrollBar::handle:horizontal { background: #c4cfdd; border-radius: 2px; min-width: 24px; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
        """)
        filter_scroll.viewport().installEventFilter(self)
        filter_content = QWidget()
        self.filter_layout = QHBoxLayout(filter_content)
        self.filter_layout.setContentsMargins(0, 0, 0, 0)
        self.filter_layout.setSpacing(3)
        self.filter_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        filter_scroll.setWidget(filter_content)
        filters.addWidget(filter_scroll, 1)
        self.tag_filter = QComboBox()
        self.tag_filter.setAccessibleName("개별 태그 필터")
        self.tag_filter.setMaximumWidth(135)
        self.tag_filter.setFixedHeight(26)
        self.tag_filter.setStyleSheet("QComboBox { background: #ffffff; color: #475569; border: 1px solid #d5dde8; border-radius: 4px; padding: 3px 7px; }")
        self.tag_filter.currentIndexChanged.connect(self.filter_items)
        filters.addWidget(self.tag_filter)
        self.count_label = QLabel()
        filters.addWidget(self.count_label)
        layout.addLayout(filters)
        self.list = QListWidget()
        self.list.setAccessibleName("폰트 목록")
        self.list.setViewMode(QListView.ViewMode.IconMode)
        self.list.setFlow(QListView.Flow.LeftToRight)
        self.list.setWrapping(True)
        self.list.setResizeMode(QListView.ResizeMode.Adjust)
        self.list.setMovement(QListView.Movement.Static)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setMouseTracking(True)
        self.list.setItemDelegate(FontDelegate(self.list))
        self.list.itemClicked.connect(self.choose)
        self.list.itemActivated.connect(self.choose)
        self.list.currentItemChanged.connect(self.sync_checks)
        layout.addWidget(self.list, 1)
        self.empty = QLabel("일치하는 폰트가 없습니다. 검색어나 용도를 바꿔 보세요.")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.empty)
        self.tag_panel = QWidget()
        panel_layout = QVBoxLayout(self.tag_panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        management = QHBoxLayout()
        manage = QPushButton("태그 관리")
        manage.setToolTip("태그 추가·삭제·이름·색상 변경")
        manage.clicked.connect(self.manage_tags)
        management.addWidget(manage)
        manage_filters = QPushButton("필터 관리")
        manage_filters.clicked.connect(self.manage_filters)
        management.addWidget(manage_filters)
        management.addStretch()
        panel_layout.addLayout(management)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(100)
        scroll.setMinimumHeight(64)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        self.tags_layout = QGridLayout(content)
        self.tags_layout.setContentsMargins(0, 3, 0, 0)
        self.tags_layout.setSpacing(7)
        scroll.setWidget(content)
        panel_layout.addWidget(scroll)
        self.checks = {}
        self.tag_panel.hide()
        layout.addWidget(self.tag_panel)
        self.hint = QLabel("휠로 이동 · 클릭 / Enter 적용 · Esc 닫기")
        layout.addWidget(self.hint)
        # Route wheel events from the fixed header to the same scroll surface.
        for widget in (self, self.edit, self.tag_filter, self.count_label, self.hint, *self.filters.buttons(), *self.checks.values()):
            widget.installEventFilter(self)
        self.rebuild_tags()
        self.rebuild_filters()

    def rebuild_filters(self):
        for button in self.filters.buttons():
            if button is self.all_button:
                continue
            self.filters.removeButton(button)
            self.filter_layout.removeWidget(button)
            button.deleteLater()
        names = ["대사"] + [entry["name"] for entry in self.filter_definitions]
        if self.group not in ["전체"] + names:
            self.group = "대사"
        self.all_button.setChecked(self.group == "전체")
        for name in names:
            button = QPushButton(name)
            button.setFixedHeight(26)
            button.setFixedWidth(min(180, button.fontMetrics().horizontalAdvance(name) + 20))
            button.setText(button.fontMetrics().elidedText(name, Qt.TextElideMode.ElideRight, button.width() - 20))
            button.setCheckable(True)
            button.setChecked(name == self.group)
            button.setToolTip("특수·강조·효과음·손글씨 태그가 없는 폰트 (미지정 포함)" if name == "대사" else name)
            button.clicked.connect(lambda checked, value=name: self.set_group(value))
            button.installEventFilter(self)
            self.filters.addButton(button)
            self.filter_layout.addWidget(button)

    def manage_filters(self):
        self.hide()
        dialog = FilterManager(self)
        dialog.exec()
        self.open()
        dialog.deleteLater()

    def rebuild_tags(self):
        for check in self.checks.values():
            self.tags_layout.removeWidget(check)
            check.deleteLater()
        self.checks = {}
        self.list.setProperty("tagColors", {tag["name"]: tag["color"] for tag in self.tag_definitions})
        current = self.tag_filter.currentData()
        self.tag_filter.blockSignals(True)
        self.tag_filter.clear()
        self.tag_filter.addItem("모든 태그", "")
        for position, definition in enumerate(self.tag_definitions):
            tag, color = definition["name"], definition["color"]
            check = QCheckBox(tag)
            check.setToolTip(tag)
            check.setStyleSheet(f"QCheckBox {{ background:{color}; color:{tag_ink(color)}; padding:3px; border-radius:3px; }}")
            check.toggled.connect(self.save_tags)
            self.tags_layout.addWidget(check, position // 3, position % 3)
            self.checks[tag] = check
            self.tag_filter.addItem(tag, tag)
        self.tag_filter.setCurrentIndex(max(0, self.tag_filter.findData(current)))
        self.tag_filter.blockSignals(False)
        self.sync_checks()

    def manage_tags(self):
        self.hide()
        dialog = TagManager(self)
        dialog.exec()
        self.open()
        dialog.deleteLater()

    def replace_tags(self, definitions, rename):
        # QSettings atomically syncs the INI, including assignments of absent fonts.
        before = {key: self.settings.value(key) for key in self.settings.allKeys()}
        for key in self.settings.allKeys():
            if key.startswith("tags/"):
                values = self.settings.value(key, [], type=list)
                self.settings.setValue(key, [rename[tag] for tag in values if tag in rename])
        self.settings.setValue("definitions", json.dumps(definitions, ensure_ascii=False))
        filters = [{"name": entry["name"], "tags": [rename[tag] for tag in entry["tags"] if tag in rename]} for entry in self.filter_definitions]
        self.settings.setValue("filters", json.dumps(filters, ensure_ascii=False))
        self.settings.sync()
        if self.settings.status() != QSettings.Status.NoError:
            self.settings.clear()
            for key, value in before.items():
                self.settings.setValue(key, value)
            QMessageBox.warning(self.combo.window(), "태그 저장 실패", "태그를 저장하지 못했습니다. 폴더 쓰기 권한을 확인하세요.")
            return False
        self.tag_definitions = definitions
        self.filter_definitions = filters
        self.rebuild_tags()
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setData(Qt.ItemDataRole.UserRole + 1, [rename[tag] for tag in item.data(Qt.ItemDataRole.UserRole + 1) or [] if tag in rename])
            self.update_tooltip(item)
        self.filter_items()
        self.list.viewport().update()
        return True

    def open(self):
        self.list.clear()
        for i in range(self.combo.count()):
            item = QListWidgetItem(self.combo.itemText(i))
            key = str(self.combo.itemData(i))
            item.setData(Qt.ItemDataRole.UserRole, key)
            tags = self.settings.value("tags/" + key, [], type=list)
            item.setData(Qt.ItemDataRole.UserRole + 1, [d["name"] for d in self.tag_definitions if d["name"] in tags])
            self.list.addItem(item)
            self.update_tooltip(item)
            if i == self.combo.currentIndex():
                self.list.setCurrentItem(item)
        self.search.clear()
        self.filter_items()
        screen = self.combo.screen().availableGeometry()
        width = min(760, screen.width() - 20)
        height = min(420, screen.height() - 20)
        self.resize(width, height)
        anchor = self.combo.mapToGlobal(QPoint(0, self.combo.height() + 4))
        x = max(screen.left() + 10, min(anchor.x(), screen.right() - width - 9))
        y = max(screen.top() + 10, min(anchor.y(), screen.bottom() - height - 9))
        self.move(x, y)
        self.show()
        self.resize_grid()
        if self.list.currentItem() and not self.list.currentItem().isHidden():
            self.list.scrollToItem(self.list.currentItem())
        self.search.setFocus()

    def resize_grid(self):
        width = self.list.viewport().width()
        columns = 3 if width >= 690 else 2 if width >= 440 else 1
        self.list.setGridSize(QSize(max(1, (width - 4) // columns), 56))
        self.list.doItemsLayout()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resize_grid()

    def set_group(self, name):
        self.group = name
        self.filter_items()

    def reset_filters(self):
        self.group = "전체"
        self.search.clear()
        self.tag_filter.setCurrentIndex(0)
        self.filter_items()

    def filter_items(self):
        query = self.search.text().strip().casefold()
        required = next((entry["tags"] for entry in self.filter_definitions if entry["name"] == self.group), [])
        specific = self.tag_filter.currentData()
        visible = 0
        for i in range(self.list.count()):
            item = self.list.item(i)
            tags = item.data(Qt.ItemDataRole.UserRole + 1) or []
            group_matches = not {"특수", "강조", "효과음", "손글씨"}.intersection(tags) if self.group == "대사" else self.group == "전체" or any(t in tags for t in required)
            matches = query in item.text().casefold() and group_matches and (not specific or specific in tags)
            item.setHidden(not matches)
            visible += int(matches)
        self.count_label.setText(f"{visible}개")
        self.empty.setVisible(visible == 0)
        current = self.list.currentItem()
        if current is None or current.isHidden():
            self.list.setCurrentItem(next((self.list.item(i) for i in range(self.list.count()) if not self.list.item(i).isHidden()), None))
        self.sync_checks()

    def choose(self, item):
        if self.edit.isChecked() or item.isHidden():
            return
        index = self.combo.findData(item.data(Qt.ItemDataRole.UserRole))
        if index >= 0:
            self.combo.setCurrentIndex(index)
            self.combo.activated.emit(index)
        self.close()

    def edit_mode(self, checked):
        self.tag_panel.setVisible(checked)
        self.hint.setText("폰트를 선택한 뒤 용도를 체크하세요 · 자동 저장" if checked else "휠로 이동 · 클릭 / Enter 적용 · Esc 닫기")
        self.sync_checks()
        if not checked:
            self.filter_items()

    def sync_checks(self, *_args):
        item = self.list.currentItem()
        valid = item is not None and not item.isHidden()
        tags = item.data(Qt.ItemDataRole.UserRole + 1) if valid else []
        for tag, check in self.checks.items():
            check.blockSignals(True)
            check.setChecked(tag in tags)
            check.setEnabled(valid)
            check.blockSignals(False)

    def save_tags(self):
        item = self.list.currentItem()
        if item is None or item.isHidden():
            return
        tags = [tag for tag, check in self.checks.items() if check.isChecked()]
        self.settings.setValue("tags/" + item.data(Qt.ItemDataRole.UserRole), tags)
        self.settings.sync()
        if self.settings.status() != QSettings.Status.NoError:
            QMessageBox.warning(self, "태그 저장 실패", "폰트 태그를 저장하지 못했습니다. 폴더 쓰기 권한을 확인하세요.")
            self.sync_checks()
            return
        item.setData(Qt.ItemDataRole.UserRole + 1, tags)
        self.update_tooltip(item)
        self.list.viewport().update()

    def update_tooltip(self, item):
        tags = item.data(Qt.ItemDataRole.UserRole + 1) or []
        item.setToolTip(item.text() + "\n" + (" · ".join(tags) if tags else "용도 미지정 · 태그 편집에서 지정"))

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Wheel:
            bar = self.list.verticalScrollBar()
            delta = event.pixelDelta().y() or event.angleDelta().y() // 2
            bar.setValue(bar.value() - delta)
            return True
        if watched is self.search and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Down:
                self.list.setFocus()
                return True
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if self.list.currentItem():
                    self.choose(self.list.currentItem())
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.combo.hidePopup()
