from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType, SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from PIL import Image

from advanced_editor.launcher import AdvancedEditorWindow


class ExportScopeTest(TestCase):
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
                AdvancedEditorWindow._export_sessions(
                    window, session_paths, root / "export", single_file=False
                )

            rerender.assert_not_called()
            targets = [call.args[1] for call in window._save_image_to_target.call_args_list]
            self.assertEqual(
                targets,
                [root / "export/chapter/01.png", root / "export/chapter/02.png"],
            )

