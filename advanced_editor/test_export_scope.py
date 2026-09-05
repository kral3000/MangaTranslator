from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType, SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from PIL import Image

from advanced_editor.launcher import AdvancedEditorWindow, DEFAULT_EXPORT_DIRECTORY


class ExportScopeTest(TestCase):
    def test_saved_original_restore_is_baked_for_the_next_inpaint(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            saved = root / "saved.png"
            Image.new("RGBA", (3, 2), "green").save(saved)
            original = root / "original.png"
            Image.new("RGBA", (3, 2), (210, 30, 20, 255)).save(original)
            restore_mask = root / "restore.png"
            mask = Image.new("L", (3, 2), 0)
            mask.putpixel((0, 0), 255)
            mask.save(restore_mask)
            clean_base = Mock(return_value=Image.new("RGBA", (3, 2), "blue"))
            window = SimpleNamespace(
                session={
                    "rendered_snapshot_path": "",
                    "edited_image_path": str(saved),
                    "original_snapshot_path": str(original),
                    "restore_mask_path": str(restore_mask),
                    "regions": [],
                },
                session_path=root / "session.json",
                image_size=(3, 2),
                _dirty_region_ids={"changed"},
                _clean_base_image=clean_base,
                _apply_paint_overlay_to=lambda image: image,
            )
            window._inpainting_source_image = MethodType(
                AdvancedEditorWindow._inpainting_source_image, window
            )

            AdvancedEditorWindow._flatten_fast_editor_layers_after_save(window, saved)
            image = AdvancedEditorWindow._inpainting_source_image(window)

            self.assertEqual(window.session["restore_mask_path"], "")
            self.assertEqual(image.getpixel((0, 0)), (210, 30, 20, 255))
            self.assertEqual(image.getpixel((1, 0)), (0, 0, 255, 255))
            self.assertEqual(
                Path(window.session["rendered_snapshot_path"]), saved.resolve()
            )

    def test_export_dialogs_start_in_mana_folder(self) -> None:
        self.assertEqual(
            DEFAULT_EXPORT_DIRECTORY,
            Path(r"C:\Users\dongw\Documents\Mana"),
        )

    def test_save_updates_the_translation_session_output(self) -> None:
        output = Path("output/run/01_translated.png").resolve()
        window = SimpleNamespace(
            session={
                "translated_image_path": str(output),
                "edited_image_path": str(Path("editor_sessions/work.png").resolve()),
            },
            session_path=Path("editor_sessions/01_edit_session.json").resolve(),
        )

        target = AdvancedEditorWindow._session_save_target(window)

        self.assertEqual(target, output)

    def test_standalone_session_keeps_its_editor_work_file(self) -> None:
        work_file = Path("editor_imports/01_edit_session_edited.png").resolve()
        window = SimpleNamespace(
            session={
                "standalone_image_session": True,
                "translated_image_path": str(Path("source.png").resolve()),
                "edited_image_path": str(work_file),
            },
            session_path=Path("editor_imports/01_edit_session.json").resolve(),
        )

        target = AdvancedEditorWindow._session_save_target(window)

        self.assertEqual(target, work_file)

    def test_session_switch_does_not_reset_multi_selection(self) -> None:
        target = Path("target.json")
        window = SimpleNamespace(
            session_path=Path("current.json"),
            session={"regions": []},
            _switching_session=False,
            _preview_timer=SimpleNamespace(stop=Mock()),
            _preview_request_serial=0,
            canvas=SimpleNamespace(pan_offset=None, setFocus=Mock()),
            history=[],
            redo_stack=[],
            commit_region_form=Mock(),
            _persist_editor_control_settings=Mock(),
            _persist_output_settings=Mock(),
            _load_base_image=Mock(return_value=Image.new("RGBA", (2, 2))),
            _restore_output_widgets_from_saved_settings=Mock(),
            reload_display_image=Mock(),
            refresh_region_list=Mock(),
            load_selected_region_to_form=Mock(),
            _refresh_history_menu=Mock(),
            setWindowTitle=Mock(),
            _set_session_list_current=Mock(),
            schedule_exact_preview=Mock(),
            status_label=SimpleNamespace(setText=Mock()),
            _session_display_name=Mock(return_value="target"),
        )

        with (
            patch("advanced_editor.launcher.save_session"),
            patch(
                "advanced_editor.launcher.load_session",
                return_value={"regions": [], "editor_dirty_region_ids": []},
            ),
        ):
            AdvancedEditorWindow._switch_editor_session(window, target)

        window._set_session_list_current.assert_not_called()

    def test_bulk_export_writes_new_files_without_rerendering_work_files(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            session_paths = [root / "one.json", root / "two.json"]
            window = SimpleNamespace(
                status_label=SimpleNamespace(setText=Mock(), text=Mock(return_value="")),
                _show_save_toast=Mock(),
                _save_image_to_target=Mock(),
            )
            window._safe_relative_output_parts = MethodType(
                AdvancedEditorWindow._safe_relative_output_parts, window
            )
            window._session_export_relative_path = MethodType(
                AdvancedEditorWindow._session_export_relative_path, window
            )
            window._unique_export_target = AdvancedEditorWindow._unique_export_target

            sessions = {
                session_paths[0]: {"source_display_path": "chapter/01.png"},
                session_paths[1]: {"source_display_path": "chapter/02.png"},
            }
            with (
                patch(
                    "advanced_editor.launcher.load_session",
                    side_effect=lambda path: sessions[path],
                ),
                patch(
                    "advanced_editor.launcher.render_session_image",
                    return_value=(Image.new("RGBA", (2, 2)), []),
                ),
                patch("advanced_editor.launcher.rerender_session") as rerender,
            ):
                progress = Mock()
                exported, failures, warnings = AdvancedEditorWindow._export_sessions_work(
                    window,
                    session_paths,
                    root / "export",
                    single_file=False,
                    progress=progress,
                )

            rerender.assert_not_called()
            self.assertEqual(failures, [])
            self.assertEqual(warnings, [])
            self.assertEqual(len(exported), 2)
            targets = [call.args[1] for call in window._save_image_to_target.call_args_list]
            self.assertEqual(
                targets,
                [root / "export/chapter/01.png", root / "export/chapter/02.png"],
            )
            self.assertEqual(
                [call.args[:2] for call in progress.call_args_list],
                [(0, 2), (1, 2), (1, 2), (2, 2)],
            )
