from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from core.editor_batch_progress import EditorBatchProgress
from core.editor_session import related_edit_session_paths
from core.editor_session_catalog import discover_existing_editor_sessions


class EditorBatchProgressTest(TestCase):
    def test_ready_pages_stay_in_input_order_and_do_not_rewrite_sessions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            folder = root / "editor_sessions" / "translation_sessions" / "batch"
            folder.mkdir(parents=True)
            paths = [folder / f"{i}_edit_session.json" for i in (1, 2, 3)]
            batch = EditorBatchProgress(root, "Series · Chapter 5", root / "out", paths)
            self.assertTrue(batch.publish())
            paths[0].write_text("{}", encoding="utf-8")

            def jobs(active=True):
                return discover_existing_editor_sessions(
                    root, include_members=True, include_active=active
                )[2]

            batch.page_ready(paths[0])
            self.assertEqual(jobs()[0]["paths"], [])
            self.assertEqual(jobs()[0]["total_pages"], 3)
            self.assertFalse(jobs(active=False))

            for index in (2, 0):
                page = paths[index]
                page.write_text('{"local_edit": true}', encoding="utf-8")
                # The publisher uses the same finalized asset path as the pipeline.
                assets = folder / f"assets_{index}"
                assets.mkdir()
                (assets / "rendered_snapshot.png").touch()
                with patch("core.editor_session._asset_paths", return_value=(assets, None, None)):
                    batch.page_ready(page)

            expected = [paths[0], paths[2]]
            self.assertEqual(jobs()[0]["paths"], list(map(str, expected)))
            self.assertEqual(related_edit_session_paths(paths[2]), expected)
            batch.finish()
            archived = jobs(active=False)[0]
            self.assertFalse(archived["translating"])
            self.assertEqual(archived["page_count"], 2)
            self.assertEqual(paths[2].read_text(encoding="utf-8"), '{"local_edit": true}')
