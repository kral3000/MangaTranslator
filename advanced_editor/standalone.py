"""Show startup feedback before importing the editor and its render dependencies."""

import sys
from pathlib import Path

from PyQt6.QtCore import QEventLoop, Qt, QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QMessageBox, QProgressBar, QVBoxLayout


EDITOR_ICON = Path(__file__).resolve().parents[1] / "assets" / "nene-himekawa-editor.ico"


def startup_dialog(message="고급 편집기 시작 중…", parent=None):
    dialog = QDialog(parent, Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
    dialog.setFixedWidth(330)
    dialog.setModal(True)
    dialog.setStyleSheet("""
        QDialog { background: #f4f7fb; border: 1px solid #c4cfdd; }
        QLabel { font: 600 14px "Malgun Gothic"; color: #172033; }
        QProgressBar { border: none; background: #dbeafe; }
        QProgressBar::chunk { background: #2563eb; }
    """)
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(20, 18, 20, 18)
    layout.setSpacing(12)
    layout.addWidget(QLabel(message))
    progress = QProgressBar()
    progress.setRange(0, 0)
    progress.setTextVisible(False)
    progress.setFixedHeight(4)
    layout.addWidget(progress)
    return dialog


def main():
    app = QApplication(sys.argv[:1])
    app.setWindowIcon(QIcon(str(EDITOR_ICON)))
    app.setQuitOnLastWindowClosed(False)
    startup = startup_dialog()
    startup.show()

    first_paint = QEventLoop()
    QTimer.singleShot(75, first_paint.quit)
    first_paint.exec()
    try:
        draft = None
        session = sys.argv[1] if len(sys.argv) > 1 else None
        if session is None:
            from advanced_editor.job_picker import OpenEditorJobDialog

            picker = OpenEditorJobDialog()
            startup.hide()
            if picker.exec() != QDialog.DialogCode.Accepted:
                return 0
            session = str(picker.session_path)
            draft = getattr(picker, "draft", None)
            startup.show()
            app.processEvents()
        from advanced_editor.launcher import launch

        startup.hide()
        return launch(session, draft=draft)
    except Exception as exc:
        QMessageBox.critical(startup, "고급 편집기 실행 실패", str(exc))
        return 1
    finally:
        startup.hide()


if __name__ == "__main__":
    raise SystemExit(main())
