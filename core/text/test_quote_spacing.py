from io import BytesIO
from unittest import TestCase

import uharfbuzz as hb
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

from core.text.layout_engine import calculate_styled_line_width, shape_line
from core.text.text_processing import (
    attach_korean_postposition_particles,
    glue_korean_postposition_particles,
    tokenize_styled_text,
)


class QuoteSpacingTest(TestCase):
    def test_particle_stays_with_quotation_when_wrapped(self):
        for opening, closing in ("‘’", "“”", "「」", "『』", "''", '""'):
            with self.subTest(opening=opening):
                quoted = f"{opening}바레{closing}"
                text = f"누르는 {quoted} 는 검지로"
                normalized = attach_korean_postposition_particles(text)
                self.assertEqual(normalized, f"누르는 {quoted}는 검지로")
                tokens = [token for token, _ in tokenize_styled_text(normalized, False)]
                self.assertEqual(
                    glue_korean_postposition_particles(tokens),
                    ["누르는", f"{quoted}는", "검지로"],
                )

        styled = attach_korean_postposition_particles("**‘바레’** 는 말")
        tokens = [token for token, _ in tokenize_styled_text(styled, False)]
        self.assertEqual(glue_korean_postposition_particles(tokens), ["**‘바레’**는", "말"])
        self.assertEqual(attach_korean_postposition_particles("'don't' 는"), "'don't'는")
        for text in ("게다가 이 곡", "‘바레’  는", "‘바레’ 하고", "‘바레’\n는", "don't 는"):
            self.assertEqual(attach_korean_postposition_particles(text), text)

    def test_horizontal_quote_trims_padding_without_replacing_glyphs(self):
        builder = FontBuilder(1000, isTTF=True)
        metrics = {
            ".notdef": (500, 0),
            "left": (1000, 800),
            "right": (1000, 80),
            "ascii": (230, 45),
            "hangul": (1000, 50),
        }
        builder.setupGlyphOrder(list(metrics))
        builder.setupCharacterMap(
            {ord(ch): g for ch, g in zip("‘’'가", ("left", "right", "ascii", "hangul"))}
        )
        glyphs = {}
        for name, (_, left) in metrics.items():
            pen = TTGlyphPen(None)
            if name != ".notdef":
                width = 800 if name == "hangul" else 120
                pen.moveTo((left, 500))
                pen.lineTo((left + width, 500))
                pen.lineTo((left + width, 650))
                pen.lineTo((left, 650))
                pen.closePath()
            glyphs[name] = pen.glyph()
        builder.setupGlyf(glyphs)
        builder.setupHorizontalMetrics(metrics)
        builder.setupHorizontalHeader(ascent=800, descent=-200)
        builder.setupNameTable({"familyName": "Quote Probe", "styleName": "Regular"})
        builder.setupOS2(
            sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200
        )
        builder.setupPost()
        data = BytesIO()
        builder.save(data)
        face = hb.Face(data.getvalue())
        font = hb.Font(face)
        font.scale = (3200, 3200)
        text = "‘가’가'"
        old_infos, old, _ = shape_line(text, font, {})
        infos, compact, _ = shape_line(text, font, {}, compact_quotes=True)

        self.assertEqual(
            [info.codepoint for info in infos], [info.codepoint for info in old_infos]
        )
        self.assertEqual([p.y_offset for p in compact], [p.y_offset for p in old])
        self.assertEqual([p.x_advance for p in compact], [864, 3200, 864, 3200, 736])
        extents = font.get_glyph_extents(infos[0].codepoint)
        self.assertEqual(compact[0].x_offset + extents.x_bearing, 224)
        self.assertEqual(
            calculate_styled_line_width(text, 50, {"regular": face}, {}),
            sum(p.x_advance for p in compact) / 64,
        )
