from core.config import (
    normalize_outside_text_processing_mode,
    should_process_outside_text_tag,
)


def test_outside_text_processing_modes():
    assert normalize_outside_text_processing_mode(True) == "all"
    assert normalize_outside_text_processing_mode(False) == "off"
    assert should_process_outside_text_tag("prose", "ON")
    assert should_process_outside_text_tag("prose", "OM")
    assert should_process_outside_text_tag("prose", "OD")
    assert should_process_outside_text_tag("prose", "OE")
    assert not should_process_outside_text_tag("prose", "OT")
    assert not should_process_outside_text_tag("prose", "OH")
    assert not should_process_outside_text_tag("prose", "SFX")
    assert not should_process_outside_text_tag("prose", "SFXI")
    assert not should_process_outside_text_tag("prose", "SFXS")
    assert should_process_outside_text_tag("all", "OH")
    assert not should_process_outside_text_tag("off", "ON")
