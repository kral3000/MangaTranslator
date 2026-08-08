"""SD (chibi) Kita Ikuyo app icon, drawn with PIL primitives + bezier paths.

Renders each .ico size separately so detail drops out gracefully as the icon
shrinks instead of turning to mush.
"""
from PIL import Image, ImageDraw, ImageFont
import os

APP = r"C:\Users\dongw\Documents\Mana\MangaTranslator_portable\MangaTranslatorKo"
FONT = os.path.join(APP, "fonts", "Black_Han_Sans", "BlackHanSans-Regular.ttf")
ASSETS = os.path.join(APP, "assets")
OUT = os.path.join(ASSETS, "mangatranslator.ico")

# palette
BG_TOP = (126, 96, 255)
BG_BOT = (58, 30, 150)
HAIR = (232, 62, 46)          # Kita's red
HAIR_DK = (183, 38, 34)       # shadow / back mass
HAIR_HI = (255, 122, 92)      # sheen
SKIN = (255, 226, 208)
SKIN_SH = (244, 194, 176)
EYE_DK = (58, 30, 44)
IRIS = (176, 219, 40)         # chartreuse
IRIS_DK = (129, 176, 26)
MOUTH = (176, 48, 74)
BLUSH = (255, 138, 138)
WHITE = (255, 255, 255)
RIBBON = (222, 52, 62)


# ---------- geometry helpers (normalized 0..1 coords) ----------
def bez(p0, p1, p2, p3, n=36):
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append((
            u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0],
            u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1],
        ))
    return out


def path(*segs):
    """segs: (p0,p1,p2,p3) cubic tuples, chained."""
    pts = []
    for s in segs:
        seg = bez(*s)
        pts.extend(seg if not pts else seg[1:])
    return pts


def render(size, detail):
    ss = 4 if size >= 64 else 8
    S = size * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))

    # gradient background
    g = Image.new("RGB", (1, S))
    for y in range(S):
        t = y / (S - 1)
        g.putpixel((0, y), tuple(round(BG_TOP[i] + (BG_BOT[i] - BG_TOP[i]) * t) for i in range(3)))
    img.paste(g.resize((S, S), Image.BICUBIC))
    d = ImageDraw.Draw(img)

    def P(pts):
        return [(x * S, y * S) for x, y in pts]

    def poly(pts, fill):
        d.polygon(P(pts), fill=fill)

    def ell(cx, cy, rx, ry, fill):
        d.ellipse([(cx - rx) * S, (cy - ry) * S, (cx + rx) * S, (cy + ry) * S], fill=fill)

    # ---------------- lowest detail: chunky silhouette ----------------
    if detail == 0:
        # hair mass
        ell(0.50, 0.46, 0.335, 0.335, HAIR)
        d.rectangle([0.165 * S, 0.46 * S, 0.835 * S, 0.90 * S], fill=HAIR)
        ell(0.50, 0.86, 0.335, 0.115, HAIR)
        ell(0.82, 0.34, 0.115, 0.115, HAIR)            # side ponytail
        # face
        ell(0.50, 0.52, 0.225, 0.235, SKIN)
        # fringe over forehead
        d.chord([0.275 * S, 0.245 * S, 0.725 * S, 0.66 * S], 180, 360, fill=HAIR)
        # eyes + smile as blobs
        ell(0.408, 0.545, 0.062, 0.078, EYE_DK)
        ell(0.592, 0.545, 0.062, 0.078, EYE_DK)
        ell(0.408, 0.525, 0.045, 0.055, IRIS)
        ell(0.592, 0.525, 0.045, 0.055, IRIS)
        d.chord([0.455 * S, 0.60 * S, 0.545 * S, 0.70 * S], 0, 180, fill=MOUTH)
        return _finish(img, size)

    # ---------------- back hair mass ----------------
    back = path(
        ((0.155, 0.50), (0.155, 0.10), (0.845, 0.10), (0.845, 0.50)),   # over the top
        ((0.845, 0.50), (0.885, 0.70), (0.870, 0.83), (0.845, 0.93)),   # right fall
        ((0.845, 0.93), (0.70, 0.99), (0.30, 0.99), (0.155, 0.93)),     # hem
        ((0.155, 0.93), (0.130, 0.83), (0.115, 0.70), (0.155, 0.50)),   # left fall
    )
    poly(back, HAIR_DK)

    # side ponytail (her left = viewer's right): a tuft sweeping up and out
    pony = path(
        ((0.770, 0.402), (0.866, 0.410), (0.948, 0.336), (0.984, 0.236)),  # lower edge → tip
        ((0.984, 0.236), (0.932, 0.256), (0.856, 0.272), (0.776, 0.312)),  # upper edge back
        ((0.776, 0.312), (0.758, 0.348), (0.758, 0.378), (0.770, 0.402)),  # root
    )
    poly(pony, HAIR)
    if detail >= 2:
        d.line(P(path(((0.800, 0.378), (0.868, 0.362), (0.918, 0.318), (0.948, 0.268)))),
               fill=HAIR_DK, width=max(ss, round(S * 0.013)))

    # ---------------- uniform / shoulders ----------------
    body = path(
        ((0.255, 1.02), (0.275, 0.84), (0.38, 0.775), (0.50, 0.775)),
        ((0.50, 0.775), (0.62, 0.775), (0.725, 0.84), (0.745, 1.02)),
        ((0.745, 1.02), (0.60, 1.02), (0.40, 1.02), (0.255, 1.02)),
    )
    poly(body, WHITE)
    # sailor collar V + red ribbon
    poly([(0.50, 0.90), (0.415, 0.785), (0.585, 0.785)], (238, 238, 245))
    if detail >= 2:
        poly([(0.50, 0.845), (0.435, 0.925), (0.50, 0.955), (0.565, 0.925)], RIBBON)
        ell(0.50, 0.858, 0.022, 0.020, (255, 120, 126))
    else:
        poly([(0.50, 0.85), (0.44, 0.93), (0.56, 0.93)], RIBBON)

    # neck
    d.rectangle([0.455 * S, 0.70 * S, 0.545 * S, 0.81 * S], fill=SKIN_SH)

    # ---------------- face ----------------
    face = path(
        ((0.275, 0.44), (0.275, 0.26), (0.725, 0.26), (0.725, 0.44)),   # cranium
        ((0.725, 0.44), (0.730, 0.58), (0.660, 0.685), (0.50, 0.715)),  # right cheek → chin
        ((0.50, 0.715), (0.340, 0.685), (0.270, 0.58), (0.275, 0.44)),  # left cheek
    )
    poly(face, SKIN)

    # ---------------- bangs (grown-out fringe, scalloped) ----------------
    bangs = path(
        ((0.262, 0.500), (0.262, 0.235), (0.738, 0.235), (0.738, 0.500)),  # top
        ((0.738, 0.500), (0.716, 0.545), (0.678, 0.520), (0.640, 0.430)),  # long right tip
        ((0.640, 0.430), (0.612, 0.535), (0.560, 0.548), (0.528, 0.452)),  # scallop
        ((0.528, 0.452), (0.498, 0.545), (0.438, 0.545), (0.404, 0.442)),  # centre part
        ((0.404, 0.442), (0.368, 0.552), (0.310, 0.556), (0.284, 0.462)),  # scallop
        ((0.284, 0.462), (0.272, 0.492), (0.264, 0.498), (0.262, 0.500)),  # left tip
    )
    poly(bangs, HAIR)

    # long side locks framing the cheeks
    for sx in (1, -1):
        def m(p):
            return (0.5 + sx * (p[0] - 0.5), p[1])
        lock = [m(p) for p in path(
            ((0.735, 0.36), (0.790, 0.48), (0.792, 0.62), (0.760, 0.755)),
            ((0.760, 0.755), (0.735, 0.775), (0.700, 0.760), (0.690, 0.720)),
            ((0.690, 0.720), (0.700, 0.60), (0.700, 0.46), (0.680, 0.355)),
        )]
        poly(lock, HAIR)

    # hair sheen
    if detail >= 2:
        for seg in (((0.348, 0.372), (0.372, 0.318), (0.428, 0.290), (0.470, 0.288)),
                    ((0.552, 0.290), (0.598, 0.298), (0.632, 0.322), (0.648, 0.356))):
            d.line(P(path(seg)), fill=HAIR_HI, width=max(ss, round(S * 0.013)))

    # ---------------- face features ----------------
    if detail >= 2:
        ell(0.318, 0.605, 0.046, 0.027, BLUSH)
        ell(0.682, 0.605, 0.046, 0.027, BLUSH)

    for cx in (0.396, 0.604):
        rx, ry, cy = 0.055, 0.070, 0.556
        ell(cx, cy, rx, ry, WHITE)                          # sclera
        ell(cx, cy + 0.006, rx * 0.90, ry * 0.88, IRIS_DK)  # iris rim
        ell(cx, cy + 0.014, rx * 0.76, ry * 0.72, IRIS)     # iris
        ell(cx, cy - 0.022, rx * 0.40, ry * 0.36, EYE_DK)   # pupil
        # upper lash line — top arc only, kept thin so it reads as lashes
        d.arc([(cx - rx * 1.04) * S, (cy - ry * 1.08) * S,
               (cx + rx * 1.04) * S, (cy + ry * 0.80) * S],
              186, 354, fill=EYE_DK, width=max(ss, round(S * 0.016)))
        if detail >= 2:
            ell(cx - rx * 0.36, cy - 0.036, rx * 0.24, ry * 0.20, WHITE)   # highlight
            ell(cx + rx * 0.32, cy + 0.030, rx * 0.15, ry * 0.12, WHITE)   # catchlight

    # open, cheerful smile
    d.chord([0.455 * S, 0.632 * S, 0.545 * S, 0.706 * S], 0, 180, fill=MOUTH)
    if detail >= 2:
        d.chord([0.472 * S, 0.666 * S, 0.528 * S, 0.704 * S], 0, 180, fill=(232, 108, 128))

    # ---------------- speech bubble (largest sizes only) ----------------
    if detail >= 3:
        bx0, by0, bx1, by1 = 0.688, 0.660, 0.960, 0.808
        d.polygon(P([(0.742, by1 - 0.012), (0.812, by1 - 0.012), (0.736, 0.878)]), fill=WHITE)
        d.rounded_rectangle([bx0 * S, by0 * S, bx1 * S, by1 * S],
                            radius=(by1 - by0) * S * 0.30, fill=WHITE)
        f = ImageFont.truetype(FONT, int((by1 - by0) * S * 0.68))
        d.text(((bx0 + bx1) / 2 * S, (by0 + by1) / 2 * S), "번",
               font=f, fill=BG_BOT, anchor="mm")

    return _finish(img, size)


def _finish(img, size):
    S = img.size[0]
    img = img.resize((size, size), Image.LANCZOS)
    # rounded-square app mask
    ss = 4
    m = Image.new("L", (size * ss, size * ss), 0)
    ImageDraw.Draw(m).rounded_rectangle(
        [0, 0, size * ss - 1, size * ss - 1], radius=size * ss * 0.22, fill=255)
    img.putalpha(m.resize((size, size), Image.LANCZOS))
    return img


SIZES = [256, 128, 64, 48, 32, 16]
DETAIL = {256: 3, 128: 3, 64: 2, 48: 2, 32: 1, 16: 0}

os.makedirs(ASSETS, exist_ok=True)
imgs = [render(s, DETAIL[s]) for s in SIZES]
imgs[0].save(OUT, format="ICO", sizes=[(s, s) for s in SIZES], append_images=imgs[1:])
print("wrote", OUT, os.path.getsize(OUT), "bytes")

# --- preview: big render + actual-size strip on light and dark ---
big = render(512, 3)
pv = Image.new("RGBA", (512 + sum(s + 16 for s in SIZES) + 24, 560), (26, 26, 32, 255))
pv.alpha_composite(big, (8, 8))
x = 536
for im, s in zip(imgs, SIZES):
    pv.alpha_composite(im, (x, 20))
    pv.alpha_composite(im, (x, 300))
    x += s + 16
d = ImageDraw.Draw(pv)
d.rectangle([528, 280, pv.size[0], 560], fill=(240, 240, 245, 255))
x = 536
for im, s in zip(imgs, SIZES):
    pv.alpha_composite(im, (x, 300))
    x += s + 16
pv.save(os.path.join(ASSETS, "_icon_preview.png"))
print("preview ok")
