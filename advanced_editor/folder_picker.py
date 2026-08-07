from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QFileDialog


def main() -> int:
    initial = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    title = sys.argv[2] if len(sys.argv) > 2 else "폴더 선택"
    app = QApplication.instance() or QApplication(sys.argv[:1])
    dialog = QFileDialog()
    dialog.setWindowTitle(title)
    dialog.setDirectory(str(initial))
    dialog.setFileMode(QFileDialog.FileMode.Directory)
    dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
    dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    if dialog.exec() == QFileDialog.DialogCode.Accepted:
        selected = dialog.selectedFiles()
        if selected:
            print(str(Path(selected[0]).resolve()))
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
