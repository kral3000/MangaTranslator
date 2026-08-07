import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from ui.callbacks import _discover_existing_editor_sessions


class ExistingEditorSessionDiscoveryTest(TestCase):
    def test_lists_valid_sessions_recent_first_and_skips_invalid_json(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            older = root / "editor_sessions" / "older_edit_session.json"
            newer = root / "editor_imports" / "newer_edit_session.json"
            invalid = root / "editor_sessions" / "broken_edit_session.json"
            older.parent.mkdir(parents=True)
            newer.parent.mkdir(parents=True)
            older.write_text(
                json.dumps({"source_display_path": "older.png", "regions": [{}]}),
                encoding="utf-8",
            )
            newer.write_text(
                json.dumps({"source_display_path": "newer.png", "regions": [{}, {}]}),
                encoding="utf-8",
            )
            invalid.write_text("not json", encoding="utf-8")
            os.utime(older, (100, 100))
            os.utime(newer, (200, 200))

            choices, invalid_count = _discover_existing_editor_sessions(root)

            self.assertEqual([Path(value).name for _label, value in choices], [
                "newer_edit_session.json",
                "older_edit_session.json",
            ])
            self.assertIn("영역 2개", choices[0][0])
            self.assertEqual(invalid_count, 1)
