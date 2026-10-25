import re
import unicodedata
from typing import Callable, List, Optional, Tuple

import numpy as np

# Markdown-like style pattern: ***bold italic***, **bold**, *italic*
STYLE_PATTERN = re.compile(r"(\*{1,3})(.*?)(\1)")
NO_SPACE_BEFORE_MARKER = "\uf000"
EXPLICIT_SPACE_MARKER = "\uf001"
KOREAN_POSTPOSITION_PARTICLES = {
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "의",
    "도",
    "만",
    "로",
    "와",
    "과",
    "랑",
    "께",
    "란",
    "게",
    "서",
    "럼",
    "면",
    "요",
}
KOREAN_POSTPOSITION_PATTERN = "|".join(
    sorted(KOREAN_POSTPOSITION_PARTICLES, key=len, reverse=True)
)
KOREAN_POSTPOSITION_RE = re.compile(
    rf"^({KOREAN_POSTPOSITION_PATTERN})(\W*)$"
)


def is_rtl_script(text: str) -> bool:
    """Check if text contains dominant RTL script characters (Arabic, Hebrew, etc.)."""
    rtl_count = 0
    ltr_count = 0
    for ch in text:
        cp = ord(ch)
        if ch.isspace() or ch in ("*",):
            continue
        # Arabic (0600–06FF), Arabic Supplement (0750–077F), Arabic Extended-A (08A0–08FF),
        # Arabic Presentation Forms A/B (FB50–FDFF, FE70–FEFF)
        if (
            0x0600 <= cp <= 0x06FF
            or 0x0750 <= cp <= 0x077F
            or 0x08A0 <= cp <= 0x08FF
            or 0xFB50 <= cp <= 0xFDFF
            or 0xFE70 <= cp <= 0xFEFF
        ):
            rtl_count += 1
        # Hebrew (0590–05FF, FB1D–FB4F)
        elif 0x0590 <= cp <= 0x05FF or 0xFB1D <= cp <= 0xFB4F:
            rtl_count += 1
        # Thaana (0780–07BF) — Maldivian RTL
        elif 0x0780 <= cp <= 0x07BF:
            rtl_count += 1
        # NKo (07C0–07FA) — Mande RTL
        elif 0x07C0 <= cp <= 0x07FA:
            rtl_count += 1
        else:
            ltr_count += 1
    return rtl_count > ltr_count


def is_latin_style_language(language_name: str) -> bool:
    """
    Determines if a language typically uses Latin script and hyphenation.
    This is used to decide whether to apply automatic hyphenation logic.
    """
    latin_style_languages = {
        "english",
        "french",
        "spanish",
        "german",
        "italian",
        "portuguese",
        "dutch",
        "polish",
        "czech",
        "swedish",
        "danish",
        "norwegian",
        "finnish",
        "hungarian",
        "romanian",
        "turkish",
        "vietnamese",
        "indonesian",
        "malay",
        "tagalog",
    }
    return language_name.lower() in latin_style_languages


def _is_hangul_character(char: str) -> bool:
    """Check if a character is Korean Hangul (syllables or jamo)."""
    if len(char) != 1:
        return False
    code = ord(char)
    return (
        (0xAC00 <= code <= 0xD7AF)  # Hangul Syllables
        or (0x1100 <= code <= 0x11FF)  # Hangul Jamo
        or (0x3130 <= code <= 0x318F)  # Hangul Compatibility Jamo
    )


def _is_single_hangul_unit(text: str) -> bool:
    return len(text) == 1 and _is_hangul_character(text)


def strip_no_space_before_marker(token: str) -> str:
    if token.startswith(NO_SPACE_BEFORE_MARKER):
        return token[len(NO_SPACE_BEFORE_MARKER) :]
    return token


def preserve_explicit_spaces(text: str) -> str:
    """Protect every space after the first one in a consecutive run."""
    return re.sub(
        r" {2,}",
        lambda match: " " + EXPLICIT_SPACE_MARKER * (len(match.group(0)) - 1),
        text,
    )


def restore_explicit_spaces(text: str) -> str:
    return text.replace(EXPLICIT_SPACE_MARKER, " ")


def _token_content_without_style(token: str) -> str:
    clean_token = strip_no_space_before_marker(token)
    match = STYLE_PATTERN.fullmatch(clean_token)
    return match.group(2) if match else clean_token


def _is_korean_postposition_token(token: str) -> bool:
    content = _token_content_without_style(token)
    return bool(KOREAN_POSTPOSITION_RE.match(content))


def glue_korean_postposition_particles(tokens: List[str]) -> List[str]:
    """Leave user/LLM spacing intact.

    The older behavior joined separate tokens such as "게다가 이" into
    "게다가이" because "이" can also be a particle.  Particle protection now
    happens only when splitting an already-attached Hangul word for wrapping.
    """
    return tokens


def attach_korean_postposition_particles(text: str) -> str:
    """Leave user/LLM spacing intact."""
    return text


def hyphenation_token_length(token: str) -> int:
    clean_token = strip_no_space_before_marker(token)
    return sum(1 for ch in clean_token if ch.isalnum())


def split_hangul_word_for_wrapping(token: str) -> Optional[List[str]]:
    """
    Split a Hangul-containing word into breakable units without adding hyphens.

    The first unit keeps normal spacing. Later units are marked so the wrapper
    can break before them without inserting spaces into the rendered line.
    """
    normalized = unicodedata.normalize("NFC", token)
    match = re.match(r"^(\W*)([\w\-]+)(\W*)$", normalized)
    if match:
        leading_punc, core_word, trailing_punc = match.groups()
    else:
        leading_punc, core_word, trailing_punc = "", normalized, ""

    if not any(_is_hangul_character(ch) for ch in core_word):
        return None

    units: List[str] = []
    current_non_hangul = ""
    for ch in core_word:
        if _is_hangul_character(ch):
            if current_non_hangul:
                units.append(current_non_hangul)
                current_non_hangul = ""
            units.append(ch)
        elif unicodedata.combining(ch) and units:
            units[-1] += ch
        else:
            current_non_hangul += ch

    if current_non_hangul:
        units.append(current_non_hangul)

    if len(units) < 2:
        return None

    if _is_korean_postposition_token(units[-1]) and units[-2]:
        units[-2] += units[-1]
        units.pop()
        if len(units) >= 2 and _is_single_hangul_unit(units[-2]):
            units[-2] += units[-1]
            units.pop()

    if len(units) < 2:
        return None

    units[0] = leading_punc + units[0]
    units[-1] = units[-1] + trailing_punc
    return [units[0]] + [f"{NO_SPACE_BEFORE_MARKER}{unit}" for unit in units[1:]]


def is_cjk_character(char: str) -> bool:
    """Check if a character is CJK (Chinese/Japanese/Korean)."""
    if len(char) != 1:
        return False
    code = ord(char)
    return (
        (0x4E00 <= code <= 0x9FFF)  # CJK Unified Ideographs
        or (0x3400 <= code <= 0x4DBF)  # CJK Extension A
        or (0x20000 <= code <= 0x2CEAF)  # CJK Extension B-F
        or (0xF900 <= code <= 0xFAFF)  # CJK Compatibility
        or (0x3040 <= code <= 0x309F)  # Hiragana
        or (0x30A0 <= code <= 0x30FF)  # Katakana
        or (0x31F0 <= code <= 0x31FF)  # Katakana Extensions
        or (0xAC00 <= code <= 0xD7AF)  # Hangul Syllables
        or (0x1100 <= code <= 0x11FF)  # Hangul Jamo
        or (0x3130 <= code <= 0x318F)  # Hangul Compatibility Jamo
        or (0x3000 <= code <= 0x303F)  # CJK Symbols/Punctuation
        or (0xFF00 <= code <= 0xFFEF)  # Fullwidth Forms
    )


def parse_styled_segments(text: str) -> List[Tuple[str, str]]:
    """
    Parses text with markdown-like style markers into segments.

    Args:
        text (str): Input text potentially containing ***bold italic***, **bold**, *italic*.

    Returns:
        List[Tuple[str, str]]: List of (segment_text, style_name) tuples.
                               style_name is one of "regular", "italic", "bold", "bold_italic".
    """
    segments = []
    last_end = 0
    for match in STYLE_PATTERN.finditer(text):
        start, end = match.span()
        marker = match.group(1)
        content = match.group(2)

        if start > last_end:
            segments.append((text[last_end:start], "regular"))

        style = "regular"
        if len(marker) == 3:
            style = "bold_italic"
        elif len(marker) == 2:
            style = "bold"
        elif len(marker) == 1:
            style = "italic"

        segments.append((content, style))
        last_end = end

    if last_end < len(text):
        segments.append((text[last_end:], "regular"))

    return [(txt, style) for txt, style in segments if txt]


# Kinsoku Shori (禁則処理) - CJK line-breaking rules
KINSOKU_NOT_AT_START = set(  # Cannot start a line
    "、。，．！？）】」』〕〉》，．！？）］｝,.)!?;:…‥ー"
    "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ"
)
KINSOKU_NOT_AT_END = set("（【「『〔〈《（［｛([")  # Cannot end a line

TRAILING_PUNCT_CLOSERS = r"\)\]\}\>\u2019\u201D'\""
DETACHABLE_TRAILING_PUNCT_CORE = (
    r"(?:[.!?]{2,}|\u2026+|[\u3161\u30fc\u2500\u2013\u2014\u2015-]{2,})"
)
DETACHABLE_LEADING_PUNCT_RE = re.compile(
    rf"^({DETACHABLE_TRAILING_PUNCT_CORE})(.+)$"
)
DETACHABLE_TRAILING_PUNCT_RE = re.compile(
    rf"^(.*?)({DETACHABLE_TRAILING_PUNCT_CORE}[{TRAILING_PUNCT_CLOSERS}]*)$"
)
DETACHED_TRAILING_PUNCT_RE = re.compile(
    rf"^{DETACHABLE_TRAILING_PUNCT_CORE}[{TRAILING_PUNCT_CLOSERS}]*$"
)

WRAPPED_SYMBOL_ORPHAN_MAX_CHARS = 1
WRAPPED_SYMBOL_KEEP_TOGETHER_MAX_CHARS = 8
WRAPPED_SYMBOL_SYMMETRIC_QUOTES = {'"', "'"}
WRAPPED_SYMBOL_OPEN_TO_CLOSE = {
    "(": ")",
    "[": "]",
    "{": "}",
    "<": ">",
    "\u2018": "\u2019",
    "\u201c": "\u201d",
    "\u3008": "\u3009",
    "\u300a": "\u300b",
    "\u300c": "\u300d",
    "\u300e": "\u300f",
    "\u3010": "\u3011",
    "\u3014": "\u3015",
    "\u3016": "\u3017",
    "\u3018": "\u3019",
    "\u301a": "\u301b",
    "\uff08": "\uff09",
    "\uff3b": "\uff3d",
    "\uff5b": "\uff5d",
    "\uff1c": "\uff1e",
}
WRAPPED_SYMBOL_OPENERS = (
    set(WRAPPED_SYMBOL_OPEN_TO_CLOSE) | WRAPPED_SYMBOL_SYMMETRIC_QUOTES
)
WRAPPED_SYMBOL_CLOSERS = (
    set(WRAPPED_SYMBOL_OPEN_TO_CLOSE.values()) | WRAPPED_SYMBOL_SYMMETRIC_QUOTES
)


def strip_translation_terminal_periods(text: str) -> str:
    """Remove sentence-ending full stops while retaining ellipses and OCR effects."""
    def replace(match: re.Match) -> str:
        prefix = text[:match.start()].rsplit("\n", 1)[-1]
        visible = re.sub(r"</?[A-Z][A-Z0-9]*>", "", prefix).rstrip()
        if visible.endswith((".", "。", "．", "…", "⋯", "‥")):
            return match.group()
        return "" if any(char.isalnum() for char in visible) else match.group()

    return re.sub(
        r"(?<![.。．…⋯‥])[.。．](?=[ \t]*(?:[\"'”’」』】）》〉\]\)*_][ \t]*)*(?:</[A-Z][A-Z0-9]*>|\r?$))",
        replace,
        text,
        flags=re.MULTILINE,
    )


def is_detached_trailing_punctuation(token: str) -> bool:
    return bool(DETACHED_TRAILING_PUNCT_RE.match(token))


def _is_detached_ellipsis(token: str) -> bool:
    return is_detached_trailing_punctuation(token) and (
        token.startswith("..") or token.startswith("\u2026")
    )


def _split_detachable_boundary_punctuation(token: str) -> List[str]:
    pieces: List[str] = []
    remaining = token

    leading_match = DETACHABLE_LEADING_PUNCT_RE.match(remaining)
    if leading_match:
        pieces.append(leading_match.group(1))
        remaining = leading_match.group(2)

    trailing_match = DETACHABLE_TRAILING_PUNCT_RE.match(remaining)
    if trailing_match and trailing_match.group(1):
        pieces.append(trailing_match.group(1))
        pieces.append(trailing_match.group(2))
    elif remaining:
        pieces.append(remaining)

    return pieces


def _is_symmetric_quote_context(text: str, index: int) -> bool:
    ch = text[index]
    if ch not in WRAPPED_SYMBOL_SYMMETRIC_QUOTES:
        return False
    prev_ch = text[index - 1] if index > 0 else ""
    next_ch = text[index + 1] if index + 1 < len(text) else ""
    return not (prev_ch.isalnum() and next_ch.isalnum())


def _update_wrapped_symbol_stack(
    stack: List[str],
    ch: str,
    token: str,
    index: int,
) -> bool:
    if ch in WRAPPED_SYMBOL_SYMMETRIC_QUOTES:
        if stack and stack[-1] == ch:
            stack.pop()
            return True
        if not _is_symmetric_quote_context(token, index):
            return False
        stack.append(ch)
        return False

    closer = WRAPPED_SYMBOL_OPEN_TO_CLOSE.get(ch)
    if closer:
        stack.append(closer)
        return False

    if ch in WRAPPED_SYMBOL_CLOSERS:
        for stack_index in range(len(stack) - 1, -1, -1):
            if stack[stack_index] == ch:
                del stack[stack_index:]
                return True

    return False


def _split_attached_text_after_wrapped_symbol(
    token: str,
    stack: List[str],
) -> List[str]:
    pieces: List[str] = []
    current = token

    while current:
        split_at: Optional[int] = None
        local_stack = stack
        for index, ch in enumerate(current):
            closed_wrapper = _update_wrapped_symbol_stack(
                local_stack,
                ch,
                current,
                index,
            )
            if not closed_wrapper:
                continue
            suffix = current[index + 1 :]
            if suffix and not suffix[0].isspace() and suffix[0] not in WRAPPED_SYMBOL_CLOSERS:
                split_at = index + 1
                break

        if split_at is None:
            pieces.append(current)
            break

        pieces.append(current[:split_at])
        current = f"{NO_SPACE_BEFORE_MARKER}{current[split_at:]}"
        pieces.append(current)
        break

    return pieces


def _split_attached_wrapped_symbol_suffixes(tokens: List[str]) -> List[str]:
    split_tokens: List[str] = []
    stack: List[str] = []

    for token in tokens:
        parts = _split_attached_text_after_wrapped_symbol(token, stack)
        split_tokens.extend(parts)

    return split_tokens


def _split_with_cjk_awareness(
    text: str, detach_trailing_punctuation: bool = True
) -> List[str]:
    """Split text into tokens. Each CJK char is a token; kinsoku rules apply.

    Hangul (Korean) is excluded from per-character splitting because Korean
    uses spaces between words — syllables accumulate into word-level tokens
    like Latin characters, preserving inter-word spacing.
    """
    tokens: List[str] = []
    current_token = ""

    for char in text:
        if char.isspace():
            if current_token:
                tokens.append(current_token)
                current_token = ""
        elif is_cjk_character(char) and not _is_hangul_character(char):
            if char in KINSOKU_NOT_AT_START:
                if current_token:
                    current_token += char
                elif tokens:
                    tokens[-1] += char
                else:
                    current_token = char
            elif char in KINSOKU_NOT_AT_END:
                if current_token:
                    tokens.append(current_token)
                current_token = char
            else:
                if current_token:
                    if current_token[-1] in KINSOKU_NOT_AT_END:
                        current_token += char
                        tokens.append(current_token)
                        current_token = ""
                    else:
                        tokens.append(current_token)
                        current_token = ""
                        tokens.append(char)
                else:
                    tokens.append(char)
        else:
            current_token += char

    if current_token:
        tokens.append(current_token)

    if detach_trailing_punctuation:
        final_tokens = []
        for t in tokens:
            final_tokens.extend(_split_detachable_boundary_punctuation(t))
        return _split_attached_wrapped_symbol_suffixes(final_tokens)

    return _split_attached_wrapped_symbol_suffixes(tokens)


def tokenize_styled_text(
    text: str, detach_trailing_punctuation: bool = True
) -> List[Tuple[str, bool]]:
    """
    Tokenizes text into atomic units for wrapping where styled blocks are
    preserved as single, unbreakable tokens.

    Returns: List[Tuple[str, bool]] where each tuple is (token_text, is_styled).
    - Styled tokens are split into per-word tokens (CJK-aware), each wrapped
      with the same markers, to allow wrapping at word/character boundaries
      while preserving style.
    - Plain text outside markers is split with CJK awareness into word/character tokens.
    """
    text = preserve_explicit_spaces(text)
    tokens: List[Tuple[str, bool]] = []
    last_end = 0
    for match in STYLE_PATTERN.finditer(text):
        start, end = match.span()
        if start > last_end:
            preceding = text[last_end:start]
            for w in _split_with_cjk_awareness(preceding, detach_trailing_punctuation):
                tokens.append((w, False))

        marker = match.group(1)
        content = match.group(2)
        if content:
            for w in _split_with_cjk_awareness(content, detach_trailing_punctuation):
                tokens.append((f"{marker}{w}{marker}", True))

        last_end = end

    if last_end < len(text):
        trailing = text[last_end:]
        for w in _split_with_cjk_awareness(trailing, detach_trailing_punctuation):
            tokens.append((w, False))

    return tokens


def try_hyphenate_word(
    word_str: str,
    min_word_length: int,
    width_test_func: Callable[[str], bool],
) -> Optional[List[str]]:
    """
    Attempts to split a word into two parts with a hyphen such that each part passes the width test.

    This is a generic hyphenation function that doesn't know about fonts or rendering.
    It uses a callback function to test if each part fits.

    Args:
        word_str: The word to potentially hyphenate
        min_word_length: Minimum word length to attempt hyphenation
        width_test_func: Function that takes a string and returns True if it fits

    Returns:
        List of two strings (the split parts) if successful, None otherwise
    """
    no_space_before = word_str.startswith(NO_SPACE_BEFORE_MARKER)
    core_word = strip_no_space_before_marker(word_str)
    leading_punc = NO_SPACE_BEFORE_MARKER if no_space_before else ""
    trailing_punc = ""

    if hyphenation_token_length(word_str) < min_word_length:
        return None

    def _split_with_single_hyphen(base: str, idx: int) -> Tuple[str, str]:
        ch_before = base[idx - 1] if idx > 0 else ""
        ch_at = base[idx] if idx < len(base) else ""
        if ch_at == "-":
            left = base[: idx + 1]
            right = base[idx + 1 :]
        elif ch_before == "-":
            left = base[:idx]
            right = base[idx:]
        else:
            left = base[:idx] + "-"
            right = base[idx:]
        if left.endswith("-") and right.startswith("-"):
            right = right[1:]
        return left, right

    # Try splitting at existing hyphens first
    if "-" in core_word:
        hyphen_positions = [i for i, ch in enumerate(core_word) if ch == "-"]
        mid = len(core_word) // 2
        hyphen_positions.sort(key=lambda i: abs(i - mid))
        for pos in hyphen_positions:
            if pos <= 0 or pos >= len(core_word) - 1:
                continue
            left_part = core_word[: pos + 1]
            right_part = core_word[pos + 1 :]

            final_left_part = leading_punc + left_part
            final_right_part = right_part + trailing_punc

            if width_test_func(final_left_part) and width_test_func(final_right_part):
                return [final_left_part, final_right_part]

    # Try splitting at various positions
    mid = len(core_word) // 2
    candidate_indices: List[int] = []
    max_d = max(mid, len(core_word) - mid)
    for d in range(0, max_d):
        left_idx = mid - d
        right_idx = mid + d
        if 2 <= left_idx < len(core_word) - 2:
            candidate_indices.append(left_idx)
        if 2 <= right_idx < len(core_word) - 2 and right_idx != left_idx:
            candidate_indices.append(right_idx)

    for idx in candidate_indices:
        left_part, right_part = _split_with_single_hyphen(core_word, idx)

        final_left_part = leading_punc + left_part
        final_right_part = right_part + trailing_punc

        if width_test_func(final_left_part) and width_test_func(final_right_part):
            return [final_left_part, final_right_part]

    return None


def _is_cjk_token(token: str) -> bool:
    """Check if token consists entirely of spaceless CJK (Chinese/Japanese, not Hangul)."""
    token = strip_no_space_before_marker(token)
    match = STYLE_PATTERN.match(token)
    content = match.group(2) if match else token
    return len(content) > 0 and all(
        is_cjk_character(c) and not _is_hangul_character(c) for c in content
    )


def _needs_space_between(
    left_token: str, right_token: str, detach_trailing_punctuation: bool = True
) -> bool:
    """No space needed between adjacent CJK tokens or before separated punctuation."""
    if right_token.startswith(NO_SPACE_BEFORE_MARKER):
        return False

    if _token_content_without_style(left_token).endswith(EXPLICIT_SPACE_MARKER):
        return False

    left_token = strip_no_space_before_marker(left_token)
    right_token = strip_no_space_before_marker(right_token)

    if _is_cjk_token(left_token) and _is_cjk_token(right_token):
        return False

    if detach_trailing_punctuation:
        match = STYLE_PATTERN.match(right_token)
        r_content = match.group(2) if match else right_token
        if is_detached_trailing_punctuation(r_content):
            return False

    return True


def _join_tokens_smart(
    tokens: List[str], detach_trailing_punctuation: bool = True
) -> str:
    """Join tokens with smart spacing (no space between adjacent CJK tokens)."""
    if not tokens:
        return ""
    result = strip_no_space_before_marker(tokens[0])
    for i in range(1, len(tokens)):
        starts_with_detached_punctuation = False
        if detach_trailing_punctuation and i == 1:
            first_token = strip_no_space_before_marker(tokens[0])
            match = STYLE_PATTERN.match(first_token)
            content = match.group(2) if match else first_token
            starts_with_detached_punctuation = _is_detached_ellipsis(content)

        clean_token = strip_no_space_before_marker(tokens[i])
        if starts_with_detached_punctuation:
            result += clean_token
        elif _needs_space_between(
            tokens[i - 1], tokens[i], detach_trailing_punctuation
        ):
            result += " " + clean_token
        else:
            result += clean_token
    return result


def _plain_wrapping_token_text(token: str) -> str:
    return _token_content_without_style(token).replace(NO_SPACE_BEFORE_MARKER, "")


def _wrapped_symbol_stack_for_text(text: str) -> List[Tuple[str, int]]:
    stack: List[Tuple[str, int]] = []
    for ch in text:
        if ch in WRAPPED_SYMBOL_SYMMETRIC_QUOTES:
            if stack and stack[-1][0] == ch:
                stack.pop()
            else:
                stack.append((ch, 0))
            continue

        closer = WRAPPED_SYMBOL_OPEN_TO_CLOSE.get(ch)
        if closer:
            stack.append((closer, 0))
            continue

        if ch in WRAPPED_SYMBOL_CLOSERS:
            for idx in range(len(stack) - 1, -1, -1):
                if stack[idx][0] == ch:
                    del stack[idx:]
                    break
            continue

        if stack and not ch.isspace():
            closer, char_count = stack[-1]
            stack[-1] = (closer, char_count + 1)
    return stack


def _wrapped_symbol_tail_closer(token: str) -> Optional[str]:
    visible_chars = 0
    for ch in _plain_wrapping_token_text(token).lstrip():
        if ch.isspace():
            continue
        if ch in WRAPPED_SYMBOL_CLOSERS:
            if visible_chars <= WRAPPED_SYMBOL_ORPHAN_MAX_CHARS:
                return ch
            return None
        if ch in WRAPPED_SYMBOL_OPENERS:
            return None
        visible_chars += 1
        if visible_chars > WRAPPED_SYMBOL_ORPHAN_MAX_CHARS:
            return None
    return None


def _open_wrapper_closes_within_short_span(
    tokens: List[str],
    break_index: int,
    expected_closer: str,
    chars_before_break: int,
) -> bool:
    visible_after_break = 0
    for token in tokens[break_index:]:
        for ch in _plain_wrapping_token_text(token):
            if ch == expected_closer:
                return (
                    chars_before_break + visible_after_break
                    <= WRAPPED_SYMBOL_KEEP_TOGETHER_MAX_CHARS
                )
            if ch.isspace() or ch in WRAPPED_SYMBOL_OPENERS or ch in WRAPPED_SYMBOL_CLOSERS:
                continue
            visible_after_break += 1
            if (
                chars_before_break + visible_after_break
                > WRAPPED_SYMBOL_KEEP_TOGETHER_MAX_CHARS
            ):
                return False
    return False


def _break_splits_wrapped_symbol_orphan(
    tokens: List[str],
    break_index: int,
    protect_short_wrapped_symbols: bool = True,
) -> bool:
    if break_index <= 0 or break_index >= len(tokens):
        return False

    text_before_break = "".join(
        _plain_wrapping_token_text(token) for token in tokens[:break_index]
    )
    open_stack = _wrapped_symbol_stack_for_text(text_before_break)
    if not open_stack:
        return False

    # Do not leave an opener plus only one character at the end of a line.
    if open_stack[-1][1] <= WRAPPED_SYMBOL_ORPHAN_MAX_CHARS:
        return True

    if protect_short_wrapped_symbols:
        for closer, char_count in reversed(open_stack):
            if _open_wrapper_closes_within_short_span(
                tokens, break_index, closer, char_count
            ):
                return True

    # Do not start the next line with one character plus the matching closer.
    tail_closer = _wrapped_symbol_tail_closer(tokens[break_index])
    if tail_closer is None:
        return False
    return any(closer == tail_closer for closer, _ in open_stack)


def find_optimal_breaks_dp(
    tokens: List[str],
    max_width: float,
    word_width_func: Callable[[str], float],
    space_width: float,
    badness_exponent: float = 3.0,
    hyphen_penalty: float = 1000.0,
    detach_trailing_punctuation: bool = True,
    protect_short_wrapped_symbols: bool = True,
) -> Optional[List[str]]:
    """
    Pragmatic Knuth-Plass style DP to find globally optimal line breaks.

    This is a pure algorithm that doesn't know about fonts or rendering.
    It uses callback functions to get widths.

    Args:
        tokens: List of word tokens
        max_width: Maximum allowed line width
        word_width_func: Function that takes a word and returns its width
        space_width: Width of a space character
        badness_exponent: Exponent for badness calculation (higher = prefer tighter lines)
        hyphen_penalty: Penalty for lines ending with hyphens

    Returns:
        List of lines (strings) if successful, None if impossible to fit
    """
    try:
        if not tokens:
            return []

        # Calculate widths for all tokens
        token_w: List[float] = [word_width_func(t) for t in tokens]

        N = len(tokens)
        forbidden_breaks = {
            idx
            for idx in range(1, N)
            if _break_splits_wrapped_symbol_orphan(
                tokens, idx, protect_short_wrapped_symbols
            )
        }
        min_cost: List[float] = [float("inf")] * (N + 1)
        path: List[int] = [0] * (N + 1)
        min_cost[0] = 0.0
        # Slack badness is measured in pixel units and can easily reach
        # millions, so scale the UI penalty into the same order of magnitude.
        effective_hyphen_penalty = hyphen_penalty * max(1000.0, max_width * 25.0)

        for i in range(1, N + 1):
            line_width = 0.0
            for j in range(i - 1, -1, -1):
                # Add space only if needed between this token and the previous one on the line
                if j < i - 1:
                    # Check if we need space between tokens[j] and tokens[j+1]
                    if _needs_space_between(
                        tokens[j], tokens[j + 1], detach_trailing_punctuation
                    ):
                        line_width += space_width
                line_width += token_w[j]

                if line_width > max_width:
                    break
                if j in forbidden_breaks:
                    continue

                slack = max_width - line_width
                badness = pow(slack, badness_exponent)

                # Add hyphen penalty if line ends with hyphen (support styled markers)
                last_token = strip_no_space_before_marker(
                    tokens[i - 1] if i > 0 else ""
                )
                ends_with_hyphen = last_token.endswith("-")
                if not ends_with_hyphen:
                    styled_match = STYLE_PATTERN.match(last_token)
                    if styled_match:
                        ends_with_hyphen = styled_match.group(2).endswith("-")
                if ends_with_hyphen:
                    badness += effective_hyphen_penalty

                total_cost = min_cost[j] + badness
                if total_cost < min_cost[i]:
                    min_cost[i] = total_cost
                    path[i] = j

        if not np.isfinite(min_cost[N]):
            return None

        lines: List[str] = []
        current_break = N
        while current_break > 0:
            prev_break = path[current_break]
            line = _join_tokens_smart(
                tokens[prev_break:current_break], detach_trailing_punctuation
            )
            lines.insert(0, line)
            current_break = prev_break

        return lines

    except Exception:
        return None


# LLM tag classification wrappers used for per-tag rendering settings.
TRANSLATION_TAGS = {
    "BN",
    "BE",
    "BS",
    "BT",
    "BW",
    "ON",
    "OT",
    "OM",
    "OD",
    "OE",
    "OH",
    "SFX",
    "SFXI",
    "SFXS",
}
TAG_WRAPPER_PATTERN = re.compile(
    r"^\s*<\s*([A-Za-z0-9]+)\s*>\s*(.*?)\s*<\s*/\s*\1\s*>\s*$",
    re.DOTALL,
)
TAG_ANY_PATTERN = re.compile(r"<\s*/?\s*([A-Za-z0-9]+)\s*>")


def extract_translation_tag(text: str) -> Tuple[Optional[str], str]:
    """Return (TAG, inner_text) for <BN>...</BN> style model output.

    Unknown tags are left untouched. Known tags are removed before rendering so
    the visible manga text never includes classification markup.
    """
    raw = str(text or "")
    match = TAG_WRAPPER_PATTERN.match(raw)
    if match:
        tag = match.group(1).upper().strip()
        if tag in TRANSLATION_TAGS:
            return tag, match.group(2).strip()

    found: Optional[str] = None

    def _replace_tag(m: re.Match) -> str:
        nonlocal found
        tag = m.group(1).upper().strip()
        if tag in TRANSLATION_TAGS:
            if found is None:
                found = tag
            return ""
        return m.group(0)

    cleaned = TAG_ANY_PATTERN.sub(_replace_tag, raw).strip()
    return found, cleaned
