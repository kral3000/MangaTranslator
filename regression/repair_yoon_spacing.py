"""Build and verify Yoon spacing from an untouched source: script.py source.ttf output.ttf."""
import json
import struct
import sys
import unicodedata
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.DefaultTable import DefaultTable


def repair(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Use an untouched source and a separate destination")
    original = TTFont(source, lazy=True, recalcBBoxes=False, recalcTimestamp=False)
    cmap = original.getBestCmap()
    metrics = dict(original['hmtx'].metrics)
    raw = original.getTableData('glyf')
    locations = original['loca'].locations
    hangul, punctuation = set(), set()
    for cp, glyph in cmap.items():
        if 0xAC00 <= cp <= 0xD7A3 or 0x3130 <= cp <= 0x318F or 0x1100 <= cp <= 0x11FF:
            hangul.add(glyph)
        elif unicodedata.category(chr(cp)).startswith('P'):
            punctuation.add(glyph)
    changed = {}
    for glyph in hangul | punctuation:
        index = original.getGlyphID(glyph)
        start, end = locations[index:index + 2]
        if end <= start:
            continue  # Keep blank/missing glyphs untouched.
        _, xmin, _, xmax, _ = struct.unpack('>hhhhh', raw[start:start + 10])
        ink = xmax - xmin
        advance, left = metrics[glyph]
        right = advance - left - ink
        if glyph in hangul:
            new_left = max(0, left - 40)
            new_advance = max(advance - 80, new_left + ink + 40)
        else:
            # Cap existing side bearings; never expand punctuation or alter its outline.
            new_left = min(left, 90)
            new_advance = new_left + ink + min(right, 90)
        if (new_advance, new_left) != (advance, left):
            changed[glyph] = [advance, left, new_advance, new_left]
            metrics[glyph] = (new_advance, new_left)

    # A half-em word space is too wide beside the compact Hangul; use a quarter em.
    for cp in (0x20, 0xA0):
        glyph = cmap.get(cp)
        if glyph:
            advance, left = metrics[glyph]
            new_advance = round(original['head'].unitsPerEm * 0.25)
            metrics[glyph] = (new_advance, left)
            changed[glyph] = [advance, left, new_advance, left]

    result = TTFont(source, lazy=True, recalcBBoxes=False, recalcTimestamp=False)
    result['hmtx'].metrics = metrics
    result['head'].flags &= ~2  # Side bearings intentionally differ from outline xMin.
    extents = []
    for index, glyph in enumerate(original.getGlyphOrder()):
        start, end = locations[index:index + 2]
        if end <= start:
            continue
        _, xmin, _, xmax, _ = struct.unpack('>hhhhh', raw[start:start + 10])
        advance, left = metrics[glyph]
        extents.append((left, advance - left - (xmax - xmin), left + xmax - xmin))
    result['hhea'].advanceWidthMax = max(value[0] for value in metrics.values())
    result['hhea'].minLeftSideBearing = min(value[0] for value in extents)
    result['hhea'].minRightSideBearing = min(value[1] for value in extents)
    result['hhea'].xMaxExtent = max(value[2] for value in extents)
    # Preserve malformed legacy outlines and all untouched tables byte for byte.
    # Recompiling legacy cmap/name tables can change encoding records unnecessarily.
    for tag in original.keys():
        if tag in ('GlyphOrder', 'hmtx', 'head', 'hhea'):
            continue
        table = DefaultTable(tag)
        table.data = original.getTableData(tag)
        result[tag] = table
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination)
    result.close()
    verify = TTFont(destination, lazy=True)
    assert verify.getBestCmap() == cmap
    assert dict(verify['hmtx'].metrics) == metrics
    for tag in original.keys():
        if tag not in ('GlyphOrder', 'hmtx', 'head', 'hhea'):
            assert original.getTableData(tag) == verify.getTableData(tag), tag
    assert not verify['head'].flags & 2
    for glyph in hangul:
        index = original.getGlyphID(glyph)
        start, end = locations[index:index + 2]
        if end > start:
            _, xmin, _, xmax, _ = struct.unpack('>hhhhh', raw[start:start + 10])
            advance, left = metrics[glyph]
            assert advance - left - (xmax - xmin) >= 40
    assert verify['hmtx'][cmap[32]][0] == round(original['head'].unitsPerEm * 0.25)
    for cp in (0x3000, ord('A'), ord('0')):
        glyph = cmap.get(cp)
        if glyph:
            assert metrics[glyph] == original['hmtx'][glyph]
    report = {'changed_glyphs': len(changed), 'metrics': changed,
              'checks': 'Outlines, cmap, names, kerning, Latin and digits preserved; quarter-em word space verified.'}
    destination.with_suffix('.audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    verify.close()
    original.close()
    return report


if __name__ == '__main__':
    report = repair(*sys.argv[1:])
    print(report['changed_glyphs'], report['checks'])
