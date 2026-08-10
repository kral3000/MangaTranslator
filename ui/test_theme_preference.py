from unittest import TestCase

from ui.layout import theme_preference_head


class ThemePreferenceHeadTest(TestCase):
    def test_remembers_all_gradio_theme_modes(self) -> None:
        self.assertIn("mangatranslator.theme-preference", theme_preference_head)
        self.assertIn("['light', 'dark', 'system']", theme_preference_head)
        self.assertIn("window.location.replace", theme_preference_head)
        self.assertIn("document.addEventListener('click'", theme_preference_head)
