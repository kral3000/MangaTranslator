from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw, ImageFont


def _preview_font(repo_root: Path) -> str:
    for pattern in ("*.ttf", "*.otf", "*.ttc"):
        font = next((repo_root / "fonts").rglob(pattern), None)
        if font is not None:
            return str(font.resolve())
    return ""


def build_preview_session(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    font_dir = _preview_font(repo_root)
    image_path = root / "ui_preview_page.png"
    session_path = root / "ui_preview_edit_session.json"

    image = Image.new("RGBA", (900, 1200), "#f1f3f5")
    draw = ImageDraw.Draw(image)
    draw.rectangle((38, 32, 862, 1168), fill="white", outline="#1f2937", width=5)
    draw.rectangle((68, 64, 832, 540), fill="#d8dde3", outline="#111827", width=4)
    draw.rectangle((68, 568, 410, 1134), fill="#c8ced6", outline="#111827", width=4)
    draw.rectangle((438, 568, 832, 1134), fill="#e3e6ea", outline="#111827", width=4)
    for offset in range(-280, 760, 36):
        draw.line((70 + offset, 536, 420 + offset, 66), fill="#a8afb8", width=2)
    draw.ellipse((540, 92, 790, 360), fill="white", outline="#111827", width=4)
    draw.ellipse((106, 650, 346, 930), fill="white", outline="#111827", width=4)
    draw.rounded_rectangle((480, 760, 782, 930), radius=38, fill="white", outline="#111827", width=4)
    try:
        preview_font = ImageFont.truetype(font_dir, 34)
        effect_font = ImageFont.truetype(font_dir, 40)
        draw.text((665, 226), "UI 미리보기", fill="#111827", font=preview_font, anchor="mm")
        for index, character in enumerate("테스트"):
            draw.text(
                (226, 735 + index * 48),
                character,
                fill="#111827",
                font=effect_font,
                anchor="mm",
            )
        draw.multiline_text(
            (631, 845),
            "편집 화면\n테스트",
            fill="#111827",
            font=preview_font,
            anchor="mm",
            align="center",
            spacing=8,
        )
    except (OSError, ValueError):
        draw.text((610, 210), "UI PREVIEW", fill="#111827")
    image.save(image_path)

    default_render = {
        "font_dir": font_dir,
        "font_size": 32,
        "min_font_size": 32,
        "max_font_size": 32,
        "emphasis_max_font_size": 32,
        "line_spacing_mult": 1.0,
        "padding_pixels": 5.0,
        "padding_horizontal_pixels": 5.0,
        "padding_vertical_pixels": 5.0,
        "outline_width": 0.0,
        "outline_color_rgb": None,
        "text_alignment": "center",
        "supersampling_factor": 4,
        "auto_vertical_text": False,
    }

    def region(
        region_id: str,
        label: str,
        source: str,
        translation: str,
        bbox: list[int],
        *,
        vertical: bool = False,
        font_size: int = 32,
    ) -> dict:
        render = dict(default_render)
        render.update(
            font_size=font_size,
            min_font_size=font_size,
            max_font_size=font_size,
            emphasis_max_font_size=font_size,
        )
        return {
            "id": region_id,
            "label": label,
            "enabled": True,
            "type": "speech_bubble",
            "source_text": source,
            "translation": translation,
            "bbox": bbox,
            "original_bbox": bbox,
            "rotation_deg": 0.0,
            "vertical_stack": vertical,
            "resolved_orientation": "vertical" if vertical else "horizontal",
            "font_color": "#000000",
            "text_color_rgb": [0, 0, 0],
            "bubble_color_bgr": [255, 255, 255],
            "mask_path": None,
            "use_mask": False,
            "render": render,
        }

    session = {
        "version": 1,
        "standalone_image_session": True,
        "source_image_path": str(image_path),
        "original_snapshot_path": str(image_path),
        "base_image_path": str(image_path),
        "translated_image_path": "",
        "rendered_snapshot_path": "",
        "inpainted_image_path": "",
        "inpaint_mask_path": "",
        "paint_overlay_path": "",
        "restore_mask_path": "",
        "edited_image_path": str(root / "ui_preview_edited.png"),
        "assets_dir": str(root),
        "image_size": [900, 1200],
        "default_render": default_render,
        "output": {
            "jpeg_quality": 95,
            "png_compression": 2,
            "output_format": "png",
            "upscale_final_image": False,
        },
        "regions": [
            region(
                "r001",
                "1. 말풍선",
                "<BN>UIプレビュー</BN>",
                "UI 미리보기",
                [565, 110, 765, 330],
            ),
            region(
                "r002",
                "2. 세로 효과음",
                "<SFX>テスト</SFX>",
                "테스트",
                [138, 678, 310, 900],
                vertical=True,
                font_size=40,
            ),
            region(
                "r003",
                "3. 안내 문구",
                "<BN>EDITOR UI</BN>",
                "편집 화면 테스트",
                [500, 790, 760, 900],
                font_size=28,
            ),
        ],
        "editor_dirty_region_ids": [],
        "custom_output_path": str(root),
        "overwrite_source_on_save": False,
    }
    session_path.write_text(
        json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return session_path


def self_check() -> None:
    with TemporaryDirectory(prefix="mangatranslator_ui_preview_check_") as temp_dir:
        session_path = build_preview_session(Path(temp_dir))
        session = json.loads(session_path.read_text(encoding="utf-8"))
        assert Path(session["base_image_path"]).is_file()
        assert len(session["regions"]) == 3
        assert not session.get("editor_session_group")


def main() -> int:
    if sys.argv[1:] == ["--check"]:
        self_check()
        print("UI preview self-check passed")
        return 0

    try:
        with TemporaryDirectory(prefix="mangatranslator_ui_preview_") as temp_dir:
            session_path = build_preview_session(Path(temp_dir))
            from advanced_editor.launcher import launch

            return launch(str(session_path))
    except Exception as exc:
        try:
            from PyQt6.QtWidgets import QApplication, QMessageBox

            app = QApplication.instance() or QApplication(sys.argv[:1])
            QMessageBox.critical(None, "UI 미리보기 실행 실패", str(exc))
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
