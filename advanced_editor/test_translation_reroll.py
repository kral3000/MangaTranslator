import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from advanced_editor.launcher import AdvancedEditorWindow
from advanced_editor.translation_reroll import reroll_translation
from core.config import TranslationConfig


class RerollBackendTest(unittest.TestCase):
    def setUp(self):
        self.config = TranslationConfig()
        self.config.provider = 'Codex'
        self.config.model_name = 'saved-model'
        self.config.special_instructions = '真白 = 마시로\nさん = 씨'
        self.builder = patch('advanced_editor.translation_reroll._build_translation_config', return_value=self.config)
        self.builder.start()
        self.addCleanup(self.builder.stop)
        self.endpoint_patch = patch('core.services.translation._call_llm_endpoint', return_value='{"1":"뭔가 한 방이 더 필요한데."}')
        self.endpoint = self.endpoint_patch.start()
        self.addCleanup(self.endpoint_patch.stop)

    def test_uses_saved_provider_and_rules_without_images_or_changing_settings(self):
        settings = {'input_source': 'source'}
        result = reroll_translation('<BN>他に何かあと一押し…</BN>', '다른 무언가 한 번 더 밀기…', settings)
        args, kwargs = self.endpoint.call_args
        self.assertIs(args[0], self.config)
        self.assertEqual(args[1], [])
        self.assertIn('真白 = 마시로', args[2])
        self.assertIn('さん = 씨', args[2])
        self.assertIn('다른 무언가 한 번 더 밀기…', args[2])
        self.assertEqual(result['text'], '뭔가 한 방이 더 필요한데')
        self.assertEqual(settings, {'input_source': 'source'})
        self.assertEqual(self.config.model_name, 'saved-model')
        self.assertIn('system_prompt', kwargs)

    def test_dialect_direction_and_local_context_are_sent_in_one_request(self):
        result = reroll_translation('ありがとう', '고맙다', {}, style='gyeongsang', context=[{'source':'やあ', 'translation':'안녕'}])
        self.endpoint.assert_called_once()
        self.assertIn('youthful Gyeongsang', self.endpoint.call_args.args[2])
        self.assertIn('やあ', self.endpoint.call_args.args[2])
        self.assertEqual(result['style'], 'gyeongsang')

    def test_custom_direction_is_one_off(self):
        request = '난처해서 얼버무리는 느낌으로, 존댓말은 유지해 줘'
        reroll_translation('ありがとう', '감사합니다', {}, style='request', request=request)
        self.assertIn(request, self.endpoint.call_args.args[2])
        self.assertEqual(self.config.special_instructions, '真白 = 마시로\nさん = 씨')

    def test_invalid_inputs_do_not_call_endpoint(self):
        for source, style, request in [('', 'natural', ''), ('x','unknown',''), ('x','request','')]:
            with self.subTest(style=style), self.assertRaises(ValueError):
                reroll_translation(source, '', {}, style=style, request=request)
        self.endpoint.assert_not_called()

    def test_invalid_or_missing_response_is_not_a_translation(self):
        for response in ['', None, 'not indexed', '{"2":"wrong index"}', '{"1":""}']:
            with self.subTest(response=response), self.assertRaises(ValueError):
                self.endpoint.return_value = response
                reroll_translation('ありがとう', '', {})

    def test_usage_is_returned_and_exclamations_are_preserved(self):
        def endpoint(*args, **kwargs):
            kwargs['usage_output'].update(input_tokens=150, output_tokens=14)
            return '{"1":"고마워!?"}'
        self.endpoint.side_effect = endpoint
        result = reroll_translation('ありがとう!?', '', {})
        self.assertEqual(result['text'], '고마워!?')
        self.assertEqual(result['usage']['input_tokens'], 150)


class RerollSafetyTest(unittest.TestCase):
    def setUp(self):
        self.region = {'id':'1', 'source_text':'ありがとう', 'translation':'고마워'}
        self.window = SimpleNamespace(session_path=Path('page1.json'), selected_region=lambda: self.region,
            session={'editor_translation_settings': {'input_source':'current'}}, selected_index=0,
            status_label=Mock(), _apply_editor_translations=Mock(), _sync_ai_sfx_controls=Mock())
        self.window._translation_reroll_stamp = AdvancedEditorWindow._translation_reroll_stamp
        self.window._translation_reroll_draft_is_current = lambda draft=None: AdvancedEditorWindow._translation_reroll_draft_is_current(self.window, draft)
        self.draft = {'session_path':str(self.window.session_path.resolve()), 'stamp':self.window._translation_reroll_stamp(self.region)}
        self.window._translation_reroll_draft = self.draft

    def test_changed_page_region_source_or_translation_rejects_apply(self):
        changes = [('id','2'), ('source_text','別'), ('translation','편집'), ('ai_image', {'id':'ai'})]
        for key, value in changes:
            with self.subTest(key=key):
                original = dict(self.region)
                self.region[key] = value
                self.assertFalse(AdvancedEditorWindow._apply_translation_reroll(self.window, self.draft, '고마워요'))
                self.region.clear()
                self.region.update(original)
        self.window.session_path = Path('page2.json')
        self.assertFalse(AdvancedEditorWindow._apply_translation_reroll(self.window, self.draft, '고마워요'))
        self.window._apply_editor_translations.assert_not_called()

    def test_apply_uses_existing_undo_and_save_path_without_changing_editor_options(self):
        self.assertTrue(AdvancedEditorWindow._apply_translation_reroll(self.window, self.draft, '고마워요'))
        self.window._apply_editor_translations.assert_called_once_with([0], ['고마워요'], settings={'input_source':'current'})
        self.assertIsNone(self.window._translation_reroll_draft)


if __name__ == '__main__':
    unittest.main()
