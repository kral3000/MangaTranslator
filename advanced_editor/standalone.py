"""Show startup feedback before importing the editor and its render dependencies."""

import os
import sys
from pathlib import Path

from PyQt6.QtCore import QEventLoop, Qt, QTimer
from PyQt6.QtGui import QFontDatabase, QIcon
from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QMessageBox, QProgressBar, QVBoxLayout


EDITOR_ICON = Path(__file__).resolve().parents[1] / "assets" / "nene-himekawa-editor.ico"


def editor_application() -> QApplication:
    app = QApplication.instance()
    if app is not None:
        return app
    args = sys.argv[:1]
    if sys.platform == "win32" and not os.environ.get("QT_QPA_PLATFORM"):
        # FreeType gives small KoPubWorld UI text neutral, grayscale edges on Windows.
        args += ["-platform", "windows:fontengine=freetype"]
    return QApplication(args)


def register_editor_ui_font():
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return
    # Qt may not enumerate fonts installed for the current Windows user.
    font_dir = Path(local_app_data) / "Microsoft" / "Windows" / "Fonts"
    styles = QFontDatabase.styles("KoPubWorldDotum_Pro")
    for style in ("Medium", "Bold"):
        font_path = font_dir / f"KoPubWorld Dotum_Pro {style}.otf"
        if style not in styles and font_path.is_file():
            QFontDatabase.addApplicationFont(str(font_path))


def startup_dialog(message="고급 편집기 시작 중…", parent=None):
    register_editor_ui_font()
    dialog = QDialog(parent, Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
    dialog.setFixedWidth(330)
    dialog.setModal(True)
    dialog.setStyleSheet("""
        QDialog { background: #f4f7fb; border: 1px solid #c4cfdd; }
        QLabel {
            font-family: "KoPubWorldDotum_Pro", "Malgun Gothic";
            font-size: 14px; font-weight: 500; color: #172033;
        }
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
    app = editor_application()
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
