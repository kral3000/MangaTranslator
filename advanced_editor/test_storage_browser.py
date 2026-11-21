import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from advanced_editor.storage_dialog import StorageDialog
from advanced_editor.storage_inventory import inventory


class StorageBrowserTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait(self, fn):
        timeout = time.monotonic() + 5
        while not fn():
            self.app.processEvents()
            self.assertLess(time.monotonic(), timeout)
            time.sleep(0.005)

    def test_shared_scan_group_identity_selection_and_preview(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            image = repo / "preview.png"
            Image.new("RGB", (52, 31), "red").save(image)
            rows = [
                dict(
                    id=key,
                    kind="history",
                    paths=[key],
                    title="만화 · 01페이지",
                    detail="01페이지",
                    state="이전 생성",
                    group_id=group,
                    group_title="같은 작품",
                    modified=1,
                    size=2,
                    before=image,
                    after=image if key != "3" else None,
                )
                for key, group in (("1", "a"), ("2", "a"), ("3", "b"))
            ]
            gate = threading.Event()

            def discover(*args, **kwargs):
                gate.wait(3)
                return rows

            with patch(
                "advanced_editor.storage_inventory.list_storage", side_effect=discover
            ) as scan:
                first = StorageDialog(repo=repo, history=True)
                first.reject()
                view = StorageDialog(repo=repo, history=True)
                gate.set()
                self.wait(lambda: len(view.groups) == 2)
                self.assertEqual(scan.call_count, 1)
                self.assertEqual(view.groups["a"].childCount(), 2)
                view.groups["a"].setExpanded(True)
                view.tree.setCurrentItem(view.items["1"])
                self.wait(lambda: all(p.original for p in view.preview.panes))
                view.items["1"].setCheckState(0, Qt.CheckState.Checked)
                self.assertEqual([r["id"] for r in view._selected()], ["1"])
                view.items["1"].setCheckState(0, Qt.CheckState.Unchecked)
                view.tree.setCurrentItem(view.groups["b"])
                self.assertFalse(view._selected())
                self.wait(
                    lambda: (
                        view.preview.panes[0].original
                        and view.preview.panes[1].text() == "저장된 이미지 없음"
                    )
                )
                view.search.setText("무관한 검색")
                self.assertTrue(all(x.isHidden() for x in view.groups.values()))
                self.assertIsNone(view.preview.row)
                view.reject()
                reopened = StorageDialog(repo=repo, history=True)
                self.assertEqual(len(reopened.rows), 3)
                self.assertEqual(scan.call_count, 1)
                reopened._load(force=True)
                self.wait(lambda: not reopened.catalog.running)
                self.assertEqual(scan.call_count, 2)
                reopened.reject()
            inventory.cache_clear()

    def test_picker_shares_finished_catalog_including_empty_results(self):
        with (
            TemporaryDirectory() as tmp,
            patch("advanced_editor.storage_inventory.list_storage") as scan,
        ):
            view = StorageDialog(repo=Path(tmp), jobs=[])
            self.wait(lambda: not view.catalog.running)
            scan.assert_not_called()
            view.reject()
            inventory.cache_clear()

    def test_session_gallery_keeps_page_order_and_opens_the_selected_page(self):
        import json

        from advanced_editor.storage_preview import StoragePreview

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for name, color in (("z", "red"), ("a", "blue")):
                image = root / (name + ".png")
                Image.new("RGB", (35, 46), color).save(image)
                session = root / (name + "_edit_session.json")
                session.write_text(
                    json.dumps({"rendered_snapshot_path": str(image)}), encoding="utf-8"
                )
                paths.append(str(session))
            row = dict(kind="sessions", id=paths[0], title="테스트", paths=paths)
            view = StoragePreview()
            view.show()
            view.select(row)
            gallery = view.gallery
            self.wait(lambda: gallery.large.original and len(gallery.loaded_small) == 2)
            self.assertEqual(
                gallery.large.original.toImage().pixelColor(0, 0).name(), "#ff0000"
            )
            gallery.pages.setCurrentRow(1)
            self.wait(lambda: gallery.large.original)
            self.assertEqual(
                gallery.large.original.toImage().pixelColor(0, 0).name(), "#0000ff"
            )
            self.assertEqual(gallery.selected_path(), paths[1])
            view.select(dict(row, paths=[*paths, str(root / "missing.json")]))
            self.assertEqual(gallery.pages.currentRow(), 1)
            gallery.pages.setCurrentRow(2)
            self.wait(lambda: gallery.large.text() == "저장된 이미지 없음")
            view.select(None)
            self.assertEqual(gallery.pages.count(), 0)
            view.close()

    def test_library_navigation_keeps_page_and_separates_storage_from_history(self):
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QDialog

        from advanced_editor.job_picker import OpenEditorJobDialog

        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            pages = [str(repo / ("page%d.json" % i)) for i in range(3)]
            jobs = [
                dict(
                    id=pages[0],
                    paths=pages[:2],
                    title="첫 작업",
                    label="첫 작업",
                    modified=1,
                    page_count=2,
                    total_pages=5,
                    translating=True,
                ),
                dict(
                    id=pages[-1],
                    paths=pages[2:],
                    title="둘째 작업",
                    label="둘째 작업",
                    modified=1,
                    page_count=1,
                ),
            ]
            main = OpenEditorJobDialog(repo=repo, jobs=jobs)
            main.show()
            self.assertTrue(main.tree.isColumnHidden(0))
            self.assertEqual(main.items[pages[0]].text(2), "2 / 5")
            main.preview.gallery.pages.setCurrentRow(1)
            sections = []

            def close_child():
                child = self.app.activeModalWidget()
                sections.append((child.kind, child.scope))
                child.reject()

            QTimer.singleShot(0, close_child)
            main.history_button.click()
            QTimer.singleShot(0, close_child)
            main.storage_button.click()
            self.assertEqual(
                sections, [("history", set(map(Path, pages[:2]))), ("outputs", set())]
            )
            self.assertEqual(main.preview.gallery.selected_path(), pages[1])
            main.select_mode.click()
            self.assertFalse(main.tree.isColumnHidden(0))
            main.select_mode.click()
            main.search.setText("둘째")
            self.assertEqual(main.preview.gallery.selected_path(), pages[-1])
            main._open()
            self.assertEqual(main.result(), QDialog.DialogCode.Accepted)
            self.assertEqual(main.session_path, Path(pages[-1]))
            inventory.cache_clear()
