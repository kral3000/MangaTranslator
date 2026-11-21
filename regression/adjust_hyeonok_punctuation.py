"""Adjust only Hyeonok punctuation outlines; --apply keeps the original backup."""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.ttProgram import Program
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = ROOT / "fonts" / "강원교육현옥샘" / "강원교육현옥샘.ttf"
SAMPLE = "가나다라마바사아.,!?…·:;\"'()[]{}~—–-。！？、‘’“”〈〉《》「」『』"
BACKUP_PATH = ROOT / "fonts/_BACKUP/강원교육현옥샘_문장부호원본/강원교육현옥샘.ttf"
REVIEW_DIR = ROOT / "output/font_review"


def preview(font_path, output_path, label):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1500, 1100), "white")
    draw = ImageDraw.Draw(image)
    labels = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 20)
    draw.text((35, 20), label, font=labels, fill="#333333")
    for index, size in enumerate((24, 36, 60)):
        font = ImageFont.truetype(str(font_path), size)
        y = 70 + index * 105
        draw.text((35, y), f"{size}px", font=labels, fill="#777777")
        draw.text((140, y), "정말 그래? 나는 좋아! 잠깐... 음… 괜찮아, 고마워.", font=font, fill="black")
    font = ImageFont.truetype(str(font_path), 70)
    for index, char in enumerate(SAMPLE[8:]):
        x = 35 + (index % 12) * 122
        y = 440 + (index // 12) * 190
        draw.text((x, y), f"U+{ord(char):04X}", font=labels, fill="#777777")
        draw.text((x, y + 32), "가" + char, font=font, fill="black")
    image.save(output_path)


def comparison():
    image = Image.new("RGB", (1300, 330), "white")
    draw = ImageDraw.Draw(image)
    label_font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 22)
    text = "정말 그래? 나는 좋아! 잠깐... 음… 괜찮아, 고마워."
    for index, (path, label) in enumerate(((BACKUP_PATH, "수정 전"), (FONT_PATH, "수정 후"))):
        y = 30 + index * 145
        draw.text((30, y + 15), label, font=label_font, fill="#555555")
        font = ImageFont.truetype(str(path), 56)
        draw.text((150, y), text, font=font, fill="black")
        small = ImageFont.truetype(str(path), 28)
        draw.text((150, y + 70), "가. 나, 다! 라? 마… 바· 사: 아; ‘따옴표’ “큰따옴표” (괄호)", font=small, fill="black")
    image.save(REVIEW_DIR / "hyeonok_comparison.png")


def adjust(font):
    cmap = font.getBestCmap()
    changed = set()

    def transform(char, sx, sy, anchor_bottom=False, dy=0, small_only=False):
        name = cmap.get(ord(char))
        if not name or name in changed:
            return
        glyph = font["glyf"][name]
        assert not glyph.isComposite(), f"Unexpected composite: {char}"
        coords, ends, _flags = glyph.getCoordinates(font["glyf"])
        start = 0
        for end in ends:
            points = coords[start:end + 1]
            xmin = min(x for x, y in points)
            xmax = max(x for x, y in points)
            ymin = min(y for x, y in points)
            ymax = max(y for x, y in points)
            # Treat each dot separately to preserve ellipsis spacing and stems.
            if not small_only or (xmax - xmin <= 110 and ymax - ymin <= 110):
                cx = (xmin + xmax) / 2
                cy = ymin if anchor_bottom else (ymin + ymax) / 2
                for index in range(start, end + 1):
                    x, y = coords[index]
                    coords[index] = (round(cx + (x - cx) * sx), round(cy + (y - cy) * sy + dy))
            start = end + 1
        glyph.coordinates = coords
        glyph.program = Program()
        glyph.program.fromBytecode([])
        glyph.recalcBounds(font["glyf"])
        advance, _lsb = font["hmtx"][name]
        font["hmtx"][name] = advance, glyph.xMin
        changed.add(name)

    for char in ".．…‥·:：":
        transform(char, 0.66, 0.66)
    for char in "!！?？":
        transform(char, 0.58, 0.58, small_only=True)
        # Shorten the stem/curve from its lower edge, leaving the dot intact.
        glyph = font["glyf"][cmap[ord(char)]]
        coords, ends, _flags = glyph.getCoordinates(font["glyf"])
        start = 0
        for end in ends:
            points = coords[start:end + 1]
            ymin = min(y for x, y in points)
            ymax = max(y for x, y in points)
            if ymax - ymin > 110:
                for index in range(start, end + 1):
                    x, y = coords[index]
                    coords[index] = (x, round(ymin + (y - ymin) * 0.90))
            start = end + 1
        glyph.coordinates = coords
        glyph.recalcBounds(font["glyf"])
    for char in ",，;；":
        transform(char, 0.78, 0.78)
    for char in "\"'‘’“”＂＇":
        transform(char, 0.85, 0.85, anchor_bottom=True, dy=-45)
    for char in "{}｛｝":
        transform(char, 0.85, 0.85)
    transform("。", 0.72, 0.72)
    transform("、", 0.80, 0.80)
    transform("~", 0.90, 0.85)
    transform("-", 1.0, 0.60)
    return changed


def verify(original, adjusted, changed):
    assert original.getBestCmap() == adjusted.getBestCmap()
    assert original["head"].unitsPerEm == adjusted["head"].unitsPerEm
    for name in original.getGlyphOrder():
        assert original["hmtx"][name][0] == adjusted["hmtx"][name][0]
        if name not in changed:
            assert original["hmtx"][name] == adjusted["hmtx"][name]
            assert original["glyf"][name].compile(original["glyf"]) == adjusted["glyf"][name].compile(adjusted["glyf"]), name
    for char in ".…·:!?":
        name = original.getBestCmap()[ord(char)]
        assert original["glyf"][name].compile(original["glyf"]) != adjusted["glyf"][name].compile(adjusted["glyf"])
    print(f"Verified {len(changed)} modified punctuation glyphs; all other glyphs and advances unchanged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    source = BACKUP_PATH if BACKUP_PATH.exists() else FONT_PATH
    with TTFont(source) as font:
        cmap = font.getBestCmap()
        print("unitsPerEm:", font["head"].unitsPerEm)
        print("tables:", list(font.keys()))
        for char in SAMPLE:
            name = cmap.get(ord(char))
            if not name:
                print(repr(char), "MISSING")
                continue
            glyph = font["glyf"][name]
            if hasattr(glyph, "xMin"):
                print(repr(char), name, font["hmtx"][name], (glyph.xMin, glyph.yMin, glyph.xMax, glyph.yMax))
    preview(source, REVIEW_DIR / "hyeonok_before.png", "강원교육현옥샘 — 원본")
    if not args.apply:
        return
    BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not BACKUP_PATH.exists():
        shutil.copy2(FONT_PATH, BACKUP_PATH)
    with TTFont(BACKUP_PATH) as original, TTFont(BACKUP_PATH) as adjusted:
        changed = adjust(adjusted)
        with tempfile.NamedTemporaryFile(dir=FONT_PATH.parent, suffix=".tmp", delete=False) as temp:
            candidate = Path(temp.name)
        try:
            adjusted.save(candidate)
            with TTFont(candidate) as reloaded:
                verify(original, reloaded, changed)
            os.replace(candidate, FONT_PATH)
        finally:
            candidate.unlink(missing_ok=True)
    preview(FONT_PATH, REVIEW_DIR / "hyeonok_after.png", "강원교육현옥샘 — 문장부호 보정")
    comparison()
    print("Original preserved:", BACKUP_PATH)
    print("Adjusted:", FONT_PATH)


if __name__ == "__main__":
    main()
