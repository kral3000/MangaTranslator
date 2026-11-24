from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from core.config import MangaTranslatorConfig
from core.pipeline import batch_translate_images
from ui.translation_wait import TranslationWait


class TranslationWaitTest(TestCase):
    def test_batch_counts_only_successes_in_serial_parallel_and_retry(self):
        for workers in (1, 3):
            with self.subTest(workers=workers), TemporaryDirectory() as folder:
                root = Path(folder)
                sources = root / "inputs"
                sources.mkdir()
                for number in range(1, 5):
                    (sources / f"{number}.png").touch()

                attempts = {}

                def render(source, _config, output, **_options):
                    page = Path(source).stem
                    attempts[page] = attempts.get(page, 0) + 1
                    if page == "2" or (page == "3" and attempts[page] == 1):
                        raise ValueError("test translation failure")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.touch()
                    return object()

                counts = []
                config = MangaTranslatorConfig(
                    yolo_model_path="", parallel_requests=workers, retry_failed_once=True
                )
                with patch("core.pipeline.translate_and_render", side_effect=render):
                    result = batch_translate_images(
                        sources, config, root / "output",
                        page_progress=lambda done, total: counts.append((done, total)),
                    )
                self.assertEqual(counts, [(0, 4), (1, 4), (2, 4), (3, 4)])
                self.assertEqual(result["success_count"], 3)
                self.assertEqual(result["error_count"], 1)
                self.assertEqual(result["retry_success_count"], 1)

    def test_cancelled_run_does_not_count_late_outputs_or_refill_the_bar(self):
        wait = TranslationWait(batch=True)
        self.assertIn("전체 장수 확인 중", wait.render())
        self.assertIsNone(wait.render(only_changed=True))
        wait.report(1, 3)
        self.assertIn('value="1" max="3"', wait.render(only_changed=True))
        wait.cancellation.cancel()
        wait.finish("cancelled")
        wait.report(2, 3)
        wait.finish("complete")
        self.assertIn("번역 중단", wait.render())
        self.assertIn("1 / 3장 완료", wait.render())
