import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from ui import character_profiles


class CharacterProfilesTest(TestCase):
    def test_profiles_can_be_saved_selected_and_merged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_file = Path(temp_dir) / "character_name_profiles.json"
            with patch.object(character_profiles, "PROFILE_FILE", profile_file):
                store = character_profiles.save_profile(
                    "", "테스트 만화", "秋吉 直 = 아키요시 나오"
                )
                self.assertEqual(store["active"], "테스트 만화")
                self.assertEqual(
                    character_profiles.load_profiles()["profiles"]["테스트 만화"],
                    "秋吉 直 = 아키요시 나오",
                )

                merged = character_profiles.merge_with_special_instructions(
                    "테스트 만화", "1. Tag Classification"
                )
                self.assertIn("0. Character Name", merged)
                self.assertIn("秋吉 直 = 아키요시 나오", merged)
                self.assertTrue(merged.endswith("1. Tag Classification"))

                character_profiles.delete_profile("테스트 만화")
                self.assertEqual(character_profiles.load_profiles()["profiles"], {})

    def test_missing_profile_leaves_instructions_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_file = Path(temp_dir) / "character_name_profiles.json"
            with patch.object(character_profiles, "PROFILE_FILE", profile_file):
                self.assertEqual(
                    character_profiles.merge_with_special_instructions(
                        "없는 작품", "기존 지시"
                    ),
                    "기존 지시",
                )
