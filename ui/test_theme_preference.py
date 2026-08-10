from unittest import TestCase

from ui.layout import startup_preference_head


class ThemePreferenceHeadTest(TestCase):
    def test_remembers_all_gradio_theme_modes(self) -> None:
        self.assertIn("mangatranslator.theme-preference", startup_preference_head)
        self.assertIn("['light', 'dark', 'system']", startup_preference_head)
        self.assertIn("window.location.replace", startup_preference_head)
        self.assertIn("document.addEventListener('click'", startup_preference_head)

    def test_remembers_gradio_display_language(self) -> None:
        self.assertIn("mangatranslator.language-preference", startup_preference_head)
        self.assertIn("Object.defineProperties(navigator", startup_preference_head)
        self.assertIn("attributeFilter: ['lang']", startup_preference_head)
