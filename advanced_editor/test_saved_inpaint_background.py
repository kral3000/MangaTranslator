from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace, MethodType
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from PIL import Image

from advanced_editor.launcher import AdvancedEditorWindow
from core.editor_session import render_session_image, _editor_base_image_path


class SavedInpaintBackgroundTest(TestCase):
    def test_saved_background_survives_rerender_and_new_restore_stroke(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base, saved, new = [root / name for name in ('base.png', 'saved.png', 'new.png')]
            for path, color in ((base, 'white'), (saved, 'red'), (new, 'blue')):
                Image.new('RGBA', (12, 12), color).save(path)
            mask = root / 'original_restore_mask.png'
            Image.new('L', (12, 12), 255).save(mask)
            session = dict(base_image_path=str(base), source_image_path=str(base),
                           rendered_snapshot_path=str(saved), inpaint_source_path=str(new),
                           restore_mask_path='', assets_dir=str(root), regions=[],
                           deleted_region_bboxes=[[0, 0, 12, 12]])
            image, errors = render_session_image(session)
            self.assertFalse(errors)
            self.assertEqual(image.getpixel((6, 6)), (0, 0, 255, 255))
            self.assertEqual(_editor_base_image_path(session), new)
            stroke = np.zeros((12, 12), dtype=np.uint8)
            stroke[1, 1] = 255
            window = SimpleNamespace(session=session, image_size=(12, 12),
                session_path=root / 'session.json', _stroke_mask=Mock(return_value=stroke),
                status_label=Mock(), _dirty_region_ids=set())
            self.assertEqual(AdvancedEditorWindow._clean_base_image(window).getpixel((6, 6)), (0, 0, 255, 255))
            AdvancedEditorWindow._apply_restore_mask_stroke(window, 'restore_brush', [])
            with Image.open(mask) as result:
                self.assertEqual(np.count_nonzero(np.asarray(result)), 1)
            window._inpainting_source_image = MethodType(AdvancedEditorWindow._inpainting_source_image, window)
            AdvancedEditorWindow._flatten_fast_editor_layers_after_save(window, saved)
            first = Path(session['inpaint_source_path'])
            first_bytes = first.read_bytes()
            session['restore_mask_path'] = str(mask)
            session['inpainted_image_path'] = str(new)
            session['inpainted_background_text_free'] = True
            AdvancedEditorWindow._flatten_fast_editor_layers_after_save(window, saved)
            self.assertNotEqual(first, Path(session['inpaint_source_path']))
            self.assertEqual(first.read_bytes(), first_bytes)
