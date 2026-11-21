import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from uuid import uuid4

from PIL import Image

from advanced_editor.lettering_history import (
    attempt_directory,
    preserve_applied_files,
    save_record,
)
from advanced_editor.storage_records import list_storage
from core.editor_package import build_editor_package
from core.editor_session import load_session, save_session
from core.workspace_cleanup import execute_cleanup, prepare_cleanup


class StorageHistoryTest(TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.repo = Path(temp.name).resolve()
        self.page = self.repo / "editor_sessions" / "chapter" / "01_edit_session.json"
        self.assets = self.page.with_suffix("")
        self.assets.mkdir(parents=True)
        self.source = self.assets / "source_original.png"
        Image.new("RGB", (28, 32), "white").save(self.source)
        self.session = dict(
            source_display_path="Chapter/01.png",
            regions=[],
            assets_dir=str(self.assets),
            original_snapshot_path=str(self.source),
            base_image_path=str(self.source),
        )
        save_session(self.page, self.session)
        self.attempt = attempt_directory(self.page, self.assets, self.repo)
        self.meta = dict(
            session_path=str(self.page),
            region_index=0,
            stamp=["r1", [1, 2, 11, 12], "A", "B", None, "B"],
            background_box=[1, 2, 11, 12],
            background_digest="test-digest",
        )
        self.attempt.mkdir(parents=True)
        self.patch = {
            "ai_image": {"id": uuid4().hex, "box": self.meta["background_box"]}
        }
        for key, filename in [
            ("ai_original_path", "original.png"),
            ("ai_processed_path", "processed.png"),
            ("ai_mask_path", "mask.png"),
        ]:
            path = self.attempt / filename
            Image.new("RGB", (10, 10), "white").save(path)
            self.patch[key] = str(path)
        save_record(self.attempt, self.meta, "ready", patch=self.patch)

    def records(self):
        return list_storage(self.repo, "history", self.repo / "output", {self.page})

    def test_saved_draft_survives_reopen_and_applied_files_become_protected(self):
        (row,) = self.records()
        self.assertEqual(row["state"], "미적용")
        self.assertEqual(row["draft"]["stamp"][1], (1, 2, 11, 12))
        self.assertEqual(row["draft"]["patch"], self.patch)
        preserved = preserve_applied_files(self.patch, self.assets)
        self.session["regions"] = [dict(id="r1", **preserved)]
        save_session(self.page, self.session)
        for key in ("ai_original_path", "ai_processed_path", "ai_mask_path"):
            self.assertEqual(
                Path(self.patch[key]).read_bytes(), Path(preserved[key]).read_bytes()
            )
        (new,) = self.records()
        self.assertEqual(new["state"], "적용 중")
        with self.assertRaises(ValueError):
            prepare_cleanup(
                "history", [row], [row["id"]], self.repo, self.repo / "output"
            )

    def test_review_rechecks_changes_and_never_removes_render_assets(self):
        manifest = self.attempt / "history.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["pid"] = 0
        manifest.write_text(json.dumps(data), encoding="utf-8")
        (row,) = self.records()
        plan = prepare_cleanup(
            "history", [row], [row["id"]], self.repo, self.repo / "output"
        )
        self.assertEqual(list(plan["fingerprints"]), [str(self.attempt)])
        (self.attempt / "new.png").write_bytes(b"new")
        with patch("core.workspace_cleanup._trash") as trash:
            with self.assertRaisesRegex(ValueError, "변경"):
                execute_cleanup(plan)
            trash.assert_not_called()

    def test_read_and_repeat_save_preserve_mtime_and_portable_files(self):
        base = self.assets / "base_cleaned.png"
        base.write_bytes(self.source.read_bytes())
        self.session["base_image_path"] = str(base)
        save_session(self.page, self.session)
        with patch(
            "core.editor_session._asset_paths",
            return_value=(self.assets, base, self.page.with_suffix(".png")),
        ):
            load_session(self.page)
            old = 1_600_000_000_123_000_000
            os.utime(self.page, ns=(old, old))
            image_hash = hashlib.sha256(self.source.read_bytes()).digest()
            reopened = load_session(self.page)
            save_session(self.page, reopened)
            self.assertEqual(self.page.stat().st_mtime_ns, old)
            self.assertEqual(
                hashlib.sha256(self.source.read_bytes()).digest(), image_hash
            )
        legacy = self.assets / ("ai_sfx_" + uuid4().hex)
        legacy.mkdir()
        (legacy / "generated.png").write_bytes(b"unused draft")
        active = legacy / "original.png"
        active.write_bytes(self.source.read_bytes())
        reopened["regions"] = [{"ai_original_path": str(active)}]
        save_session(self.page, reopened)
        import io, zipfile

        with zipfile.ZipFile(io.BytesIO(build_editor_package(self.page))) as z:
            originals = {
                v["original"] for v in json.loads(z.read("manifest.json"))["asset_map"]
            }
        self.assertIn(str(active), originals)
        self.assertNotIn(str(legacy / "generated.png"), originals)
