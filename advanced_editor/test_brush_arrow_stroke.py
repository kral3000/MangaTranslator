import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import QApplication

from advanced_editor.launcher import AdvancedEditorWindow, EditorCanvas


class BrushArrowStrokeTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = SimpleNamespace(
            image_size=(120, 160), display_image=None, brush_size=1, brush_shape="circle",
            handle_editor_key_event=Mock(return_value=True), apply_stroke=Mock(),
        )
        self.canvas = EditorCanvas(self.window)
        self.canvas.tool = "inpaint_brush"
        self.addCleanup(self.canvas.close)

    def arrow(self, key, *, repeat=False, modifiers=Qt.KeyboardModifier.NoModifier):
        event = QKeyEvent(QEvent.Type.KeyPress, key, modifiers, "", repeat)
        self.canvas.keyPressEvent(event)
        return event

    def test_all_directions_reach_image_edge_and_commit_as_one_stroke(self):
        for key, end in ((Qt.Key.Key_Left, (0, 60)),
                         (Qt.Key.Key_Right, (119, 60)),
                         (Qt.Key.Key_Up, (40, 0)),
                         (Qt.Key.Key_Down, (40, 159))):
            with self.subTest(key=key):
                self.window.apply_stroke.reset_mock()
                self.canvas.mousePressEvent(QMouseEvent(
                    QEvent.Type.MouseButtonPress, QPointF(40, 60),
                    Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier))
                event = self.arrow(key)
                self.assertTrue(event.isAccepted())
                self.window.handle_editor_key_event.assert_not_called()
                self.window.apply_stroke.assert_not_called()
                points = list(self.canvas.stroke_points)
                mask = AdvancedEditorWindow._stroke_mask(self.window, points)
                self.assertEqual(mask[end[1], end[0]], 255)
                self.assertEqual(mask[60, 40], 255)
                expected_length = abs(end[0] - 40) + abs(end[1] - 60) + 1
                self.assertEqual(int((mask > 0).sum()), expected_length)
                self.canvas.mouseReleaseEvent(QMouseEvent(
                    QEvent.Type.MouseButtonRelease, QPointF(40, 60),
                    Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier))
                self.window.apply_stroke.assert_called_once_with("inpaint_brush", points)
                self.assertEqual(self.canvas.stroke_points, [])

    def test_next_mouse_move_has_no_diagonal_back_from_edge(self):
        self.canvas.stroke_points = [(40, 60)]
        self.arrow(Qt.Key.Key_Down)
        self.canvas.mouseMoveEvent(QMouseEvent(
            QEvent.Type.MouseMove, QPointF(45, 60), Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
        mask = AdvancedEditorWindow._stroke_mask(self.window, self.canvas.stroke_points)
        self.assertEqual(mask[159, 40], 255)
        self.assertEqual(mask[60, 45], 255)
        self.assertEqual(mask[110, 42], 0)

    def test_repeat_does_not_grow_stroke_and_arrows_can_share_one_anchor(self):
        self.canvas.stroke_points = [(40, 60)]
        self.arrow(Qt.Key.Key_Down)
        points = list(self.canvas.stroke_points)
        self.arrow(Qt.Key.Key_Down, repeat=True)
        self.assertEqual(self.canvas.stroke_points, points)
        self.arrow(Qt.Key.Key_Right)
        self.assertIn((119, 60), self.canvas.stroke_points)
        self.assertEqual(self.canvas.stroke_points[-1], (40, 60))

    def test_other_tools_idle_and_modified_shortcuts_keep_existing_handler(self):
        for tool, points, modifiers in (
            ("inpaint_brush", [], Qt.KeyboardModifier.NoModifier),
            ("select", [(40, 60)], Qt.KeyboardModifier.NoModifier),
            ("inpaint_eraser", [(40, 60)], Qt.KeyboardModifier.NoModifier),
            ("inpaint_brush", [(40, 60)], Qt.KeyboardModifier.ControlModifier),
        ):
            with self.subTest(tool=tool, points=points, modifiers=modifiers):
                self.window.handle_editor_key_event.reset_mock()
                self.canvas.tool = tool
                self.canvas.stroke_points = list(points)
                self.arrow(Qt.Key.Key_Down, modifiers=modifiers)
                self.assertEqual(self.canvas.stroke_points, points)
                self.window.handle_editor_key_event.assert_called_once()
