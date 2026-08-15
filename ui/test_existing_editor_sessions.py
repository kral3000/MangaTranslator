import base64
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from core.editor_session import related_edit_session_paths
from ui.callbacks import _discover_existing_editor_sessions
from ui.logic import _batch_source_folder_name


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
                json.dumps(
                    {
                        "source_display_path": "newer.png",
                        "editor_character_profile_name": "지붕 밑 아르테미스",
                        "editor_source_folder_name": "Chapter 55",
                        "regions": [{}, {}],
                    }
                ),
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
            self.assertIn("〔지붕 밑 아르테미스 · Chapter 55〕", choices[0][0])
            self.assertEqual(invalid_count, 1)

    def test_reads_chapter_name_from_folder_upload_paths(self) -> None:
        relative_path = "Chapter 55/01.png"
        encoded = base64.urlsafe_b64encode(relative_path.encode()).decode().rstrip("=")
        upload = Path(f"__MT_RELATIVE_PATH__{encoded}.png")

        self.assertEqual(_batch_source_folder_name([upload]), "Chapter 55")

    def test_keeps_zip_file_name_for_session_context(self) -> None:
        with TemporaryDirectory() as temp_dir:
            archive = Path(temp_dir) / "아르테미스 125-126화.zip"
            archive.touch()

            self.assertEqual(
                _batch_source_folder_name(str(archive)),
                "아르테미스 125-126화.zip",
            )

    def test_restores_a_group_after_the_repository_folder_was_renamed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_dir = root / "editor_sessions" / "translation_sessions" / "batch"
            session_dir.mkdir(parents=True)
            first = session_dir / "01_edit_session.json"
            second = session_dir / "02_edit_session.json"
            old_root = Path("C:/old/MangaTranslator")
            old_group = [
                str(old_root / path.relative_to(root)) for path in (first, second)
            ]
            for index, path in enumerate((first, second), start=1):
                path.write_text(
                    json.dumps({
                        "source_display_path": f"{index:02d}.png",
                        "regions": [],
                        "editor_session_group": old_group,
                    }),
                    encoding="utf-8",
                )

            self.assertEqual(related_edit_session_paths(first), [first, second])
            choices, invalid_count = _discover_existing_editor_sessions(root)
            self.assertEqual(len(choices), 1)
            self.assertIn("묶음 2장", choices[0][0])
            self.assertEqual(invalid_count, 0)
