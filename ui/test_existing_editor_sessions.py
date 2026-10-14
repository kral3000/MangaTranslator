import base64
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from core.editor_session import related_edit_session_paths
from ui.callbacks import (
    DEFAULT_MANA_DIRECTORY,
    _discover_existing_editor_sessions,
    _resolve_native_batch_inputs,
    handle_browse_batch_folder,
)
from ui.logic import _batch_source_folder_name


class ExistingEditorSessionDiscoveryTest(TestCase):
    def test_unchanged_list_metadata_is_cached_and_file_changes_are_visible(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "editor_sessions" / "one_edit_session.json"
            path.parent.mkdir(parents=True)
            data = {"source_display_path": "one.png", "regions": [{}],
                    "editor_session_group": [str(path)]}
            path.write_text(json.dumps(data), encoding="utf-8")
            read_text = Path.read_text
            with patch.object(Path, "read_text",
                              side_effect=read_text, autospec=True) as read:
                first = _discover_existing_editor_sessions(root)
                self.assertEqual(_discover_existing_editor_sessions(root), first)
                choices, invalid, records = _discover_existing_editor_sessions(
                    root, include_members=True
                )
                self.assertEqual((choices, invalid), first)
                self.assertEqual(records[0]["paths"], [str(path.resolve())])
                self.assertEqual(read.call_count, 1)
                original_stat = path.stat()
                data["source_display_path"] = "two.png"
                path.write_text(json.dumps(data), encoding="utf-8")
                os.utime(path, ns=(original_stat.st_atime_ns,
                                  original_stat.st_mtime_ns + 1_000_000_000))
                updated, invalid = _discover_existing_editor_sessions(root)
                self.assertIn("two.png", updated[0][0])
                self.assertEqual(invalid, 0)
                self.assertEqual(read.call_count, 2)
                path.unlink()
                self.assertEqual(_discover_existing_editor_sessions(root), ([], 0))

    def test_legacy_import_is_discovered_once_per_folder(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for folder_name in ("imported_first", "imported_second"):
                folder = root / "editor_imports" / folder_name
                folder.mkdir(parents=True)
                for number in (10, 2, 1):
                    (folder / f"{number}_edit_session.json").write_text(
                        json.dumps({"source_display_path": f"{number}.png", "regions": []}),
                        encoding="utf-8",
                    )
                (folder / "broken_edit_session.json").write_text("broken", encoding="utf-8")
            with patch("core.editor_session.related_edit_session_paths",
                       wraps=related_edit_session_paths) as discover:
                choices, invalid_count = _discover_existing_editor_sessions(root)
            self.assertEqual(discover.call_count, 2)
            self.assertEqual(len(choices), 2)
            self.assertEqual(invalid_count, 2)
            for label, value in choices:
                self.assertIn("1.png → 10.png", label)
                self.assertIn("묶음 3장", label)
                self.assertEqual(Path(value).name, "1_edit_session.json")

    def test_batch_folder_picker_always_starts_from_mana(self) -> None:
        selected = DEFAULT_MANA_DIRECTORY / "테스트 만화"
        with patch(
            "ui.callbacks._select_folder_with_explorer",
            return_value=str(selected),
        ) as picker:
            path_update, _file_update = handle_browse_batch_folder()

        picker.assert_called_once_with(
            DEFAULT_MANA_DIRECTORY,
            "번역할 만화 폴더 선택",
        )
        self.assertEqual(path_update["value"], str(selected))

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

    def test_native_folder_selection_supplies_nested_images(self) -> None:
        with TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir) / "Chapter 55"
            nested = folder / "pages"
            nested.mkdir(parents=True)
            image = nested / "01.png"
            image.touch()
            (nested / "memo.txt").touch()

            files, archive = _resolve_native_batch_inputs(
                None, None, folder, None
            )

            self.assertEqual(files, [str(image)])
            self.assertIsNone(archive)

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
            with patch(
                "core.editor_session.related_edit_session_paths",
                wraps=related_edit_session_paths,
            ) as related_paths:
                choices, invalid_count = _discover_existing_editor_sessions(root)
            self.assertEqual(len(choices), 1)
            self.assertIn("묶음 2장", choices[0][0])
            self.assertEqual(invalid_count, 0)
            self.assertEqual(related_paths.call_count, 0)
