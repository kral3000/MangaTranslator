import json
import os
import time
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from core import workspace_cleanup as cleanup


class WorkspaceCleanupTest(TestCase):
    def test_custom_dates_include_both_calendar_days_and_reject_reversed_range(self):
        items = [
            dict(
                id=str(i),
                label=f"작업 {i}",
                modified=datetime.fromisoformat(stamp).timestamp(),
            )
            for i, stamp in enumerate(
                [
                    "2026-09-01T00:00:00+09:00",
                    "2026-09-02T23:59:59+09:00",
                    "2026-09-03T00:00:00+09:00",
                    "2026-08-31T23:59:59+09:00",
                ]
            )
        ]
        self.assertEqual(
            cleanup.filtered_choices(
                items, -1, "작업", start_date="2026-09-01", end_date="2026-09-02"
            ),
            [("작업 0", "0"), ("작업 1", "1")],
        )
        self.assertEqual(
            len(cleanup.filtered_choices(items, -1, "", end_date="2026-09-02")), 3
        )
        self.assertEqual(len(cleanup.filtered_choices(items, -1, "")), 4)
        with self.assertRaisesRegex(ValueError, "시작일"):
            cleanup.filtered_choices(
                items, -1, "", start_date="2026-09-03", end_date="2026-09-02"
            )
        with self.assertRaisesRegex(ValueError, "올바른 날짜"):
            cleanup.filtered_choices(items, -1, "", start_date="wrong")
        # Custom dates must not affect a quick preset.
        self.assertEqual(
            len(cleanup.filtered_choices(items, 0, "", start_date="wrong")), 4
        )

    def test_review_and_trash_keep_originals_shared_assets_and_changed_files_safe(self):
        with TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            managed = repo / "editor_sessions"
            managed.mkdir()
            original = repo / "original.png"
            original.write_bytes(b"original")
            session = managed / "one_edit_session.json"
            assets = managed / session.stem
            assets.mkdir()
            (assets / "base.png").write_bytes(b"background")
            edited = managed / (session.stem + "_edited.png")
            edited.write_bytes(b"edited")
            session.write_text(
                json.dumps(
                    dict(
                        assets_dir=str(assets),
                        edited_image_path=str(edited),
                        source_image_path=str(original),
                    )
                ),
                encoding="utf-8",
            )
            records = [
                dict(id="one", label="one", paths=[str(session)], modified=time.time())
            ]
            plan = cleanup.prepare_cleanup(
                "sessions", records, ["one"], repo, repo / "output"
            )
            self.assertEqual(
                set(plan["fingerprints"]), {str(session), str(assets), str(edited)}
            )
            with patch.object(cleanup, "_trash") as trash:
                done, errors = cleanup.execute_cleanup(plan)
                self.assertFalse(errors)
                self.assertEqual(set(done), set(plan["fingerprints"]))
                self.assertEqual(trash.call_count, 3)
                self.assertEqual(original.read_bytes(), b"original")

                (assets / "base.png").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "変更|변경"):
                    cleanup.execute_cleanup(plan)
                self.assertEqual(trash.call_count, 3)

            other = managed / "other_edit_session.json"
            other.write_text(
                json.dumps(dict(base_image_path=str(assets / "base.png"))),
                encoding="utf-8",
            )
            shared = cleanup.prepare_cleanup(
                "sessions", records, ["one"], repo, repo / "output"
            )
            self.assertNotIn(str(assets), shared["fingerprints"])
            with patch.object(
                cleanup, "_active_editor_sessions", return_value=[session]
            ):
                with self.assertRaisesRegex(ValueError, "열려 있는"):
                    cleanup.prepare_cleanup(
                        "sessions", records, ["one"], repo, repo / "output"
                    )
            unsafe = [dict(id="original", paths=[str(original)])]
            with self.assertRaisesRegex(ValueError, "범위"):
                cleanup.prepare_cleanup(
                    "sessions", unsafe, ["original"], repo, repo / "output"
                )

    def test_output_filtering_expiry_job_guard_and_failure_never_delete_permanently(
        self,
    ):
        with TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            output = repo / "output"
            output.mkdir()
            batch = output / "20260901_120000"
            batch.mkdir()
            image = batch / "01_translated.png"
            image.write_bytes(b"result")
            os.utime(image, (100, 100))
            os.utime(batch, (100, 100))
            (output / "models").mkdir()
            (output / "ordinary.png").write_bytes(b"untouched")
            single = output / "page_translated_20260901_120000.png"
            single.write_bytes(b"single")
            items = cleanup.output_records(output)
            self.assertEqual(len(items), 2)
            self.assertEqual(
                cleanup.filtered_choices(items, 30, "20260901", now=86400 * 31),
                [
                    (item["label"], item["id"])
                    for item in items
                    if item["paths"] == [str(batch)]
                ],
            )
            selected = [str(batch)]
            plan = cleanup.prepare_cleanup("outputs", items, selected, repo, output)
            with patch.object(cleanup, "_trash", side_effect=OSError("locked")):
                completed, errors = cleanup.execute_cleanup(plan)
                self.assertEqual(completed, [])
                self.assertEqual(errors, ["locked"])
                self.assertTrue(image.is_file())
            plan["created"] -= 301
            with self.assertRaisesRegex(ValueError, "만료"):
                cleanup.execute_cleanup(plan)

            @cleanup.workspace_job
            def running():
                with self.assertRaisesRegex(ValueError, "번역이 진행"):
                    cleanup.prepare_cleanup("outputs", items, selected, repo, output)
                raise RuntimeError("job failed")

            with self.assertRaises(RuntimeError):
                running()
            self.assertEqual(cleanup._ACTIVE_JOBS, 0)
