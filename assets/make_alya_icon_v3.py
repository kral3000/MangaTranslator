from pathlib import Path

from PIL import Image, ImageDraw


ASSETS = Path(__file__).resolve().parent
SOURCE = ASSETS / "alya-kujo-app-icon-v3-source.png"
PNG_OUTPUT = ASSETS / "alya-kujo-app-icon-v3.png"
ICO_OUTPUT = ASSETS / "alya-kujo-app-icon-v3.ico"
MASTER_SIZE = 1024
ICON_SIZES = (256, 128, 64, 48, 32, 16)


def rounded_mask(size: int) -> Image.Image:
    scale = 4
    mask = Image.new("L", (size * scale, size * scale), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size * scale - 1, size * scale - 1),
        radius=round(size * scale * 0.21),
        fill=255,
    )
    return mask.resize((size, size), Image.Resampling.LANCZOS)


source = Image.open(SOURCE).convert("RGB")
master = source.resize((MASTER_SIZE, MASTER_SIZE), Image.Resampling.LANCZOS)
master.putalpha(rounded_mask(MASTER_SIZE))
master.save(PNG_OUTPUT, optimize=True)

frames = [
    master.resize((size, size), Image.Resampling.LANCZOS) for size in ICON_SIZES
]
frames[0].save(
    ICO_OUTPUT,
    format="ICO",
    sizes=[(size, size) for size in ICON_SIZES],
    append_images=frames[1:],
)

print(f"wrote {PNG_OUTPUT}")
print(f"wrote {ICO_OUTPUT}")
