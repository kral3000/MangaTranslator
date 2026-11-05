[English](README.md) | [简体中文](docs/translations/zh/README.md) | [한국어](docs/translations/ko/README.md) | [日本語](docs/translations/ja/README.md)

## MangaTranslator

Gradio-based web application for automating the translation of manga/comic page images using AI. Targets speech bubbles and text outside of speech bubbles. Supports 59 languages and custom font pack usage.

<div align="left">
  <table>
    <tr>
      <th style="text-align: left">Original</th>
      <th style="text-align: left">Translated (w/ a single click)</th>
    </tr>
    <tr>
      <td><img src="docs/images/example_original.jpg" width="400" /></td>
      <td><img src="docs/images/example_translation.jpg" width="400" /></td>
    </tr>
  </table>
</div>

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Install](#install)
- [Post-Install Setup](#post-install-setup)
- [Run](#run)
- [Documentation](#documentation)
- [Updating](#updating)
- [License & Credits](#license--credits)

## Features

- **Detection**: Speech bubble detection & segmentation (YOLO, SAM 2.1/3)
- **Cleaning**: Inpaint speech bubbles and OSB text (FLUX.2 Klein, FLUX.1 Kontext, or OpenCV)
- **Translation**: LLM-powered OCR & translation (59 languages)
- **Rendering**: Text rendering with alignment and custom font packs
- **Upscaling**: 2x-AnimeSharpV4 for enhanced output quality
- **Processing**: Single/batch processing with directory preservation and ZIP support
- **Interfaces**: Web UI (Gradio) and CLI
- **Automation**: One-click translation; no intervention required

### Emphasized dialogue and shouting tags

`BE` marks visually emphasized dialogue without an actual shout. `BS` marks yelling, emotional outcries, screams and cheers; size, punctuation or angry wording alone do not imply shouting. Both tags have independent font and layout rows in the tag-rendering settings. `BS` defaults to HY_POP Bold (`HY피오피B.TTF`), keeps existing emphasis sizing when upgrading old settings, and respects later font choices. Existing translated sessions retain their saved tags.

New pipeline sessions record whether a region uses an automatic tag font. Retranslation applies the current tag font only to automatic regions; selecting a family/style or applying a bulk font marks it as a manual choice, preserved across retranslation, undo and save/reopen. Older sessions without that record retain their existing fonts. Right-click **글꼴 가족 → 태그 기본 글꼴 사용** to explicitly restore automatic tag-font selection for a selected region. This changes the font only, preserving its size, position and other formatting.

### Outside-text inpainting: white outlines and seams

Successful Flux outside-text fills receive a conservative local CPU finishing pass. Background tone is matched only on consecutive rows with smooth, consistent reference strips outside the old glyph outline; the offset is capped at 32 RGB levels and applied only to background-like pixels. High-contrast/texture rows and strong generated linework/highlights are retained instead of spreading boundary colors through a gradient-domain solve. Narrow light outlines just beyond a detected rectangle are removed only where the background reference is supported. This adds no inference pass. Grouped and individual paths share the correction; skipped text, speech-bubble protection, exact edge-conflict masks, and brush tools retain their protections. Only compact, nearly rectangular regions are processed (up to 262,144 pixels); large or irregular masks retain their existing compositing. Previously saved background artifacts are not undone automatically.

### Advanced editor: AI lettering boundaries

New AI lettering requests include a bounded, read-only crop of nearby current artwork for anatomy and print-texture context, without attaching the whole page. The output selection remains unchanged. Prompts explicitly preserve native screentones, grain and visible anatomy. Boundary protection does not restore strongly erased ink inside matching artwork. Successful generations retain the raw `generated.png` and `generation.json` beside the patch for comparison with compositing; generated-background fidelity still requires review and is not guaranteed by prompts.

Older composite AI patches retain original pixels for nearly identical artwork that touches the selection edge, rather than joining a resampled copy to the surrounding page. Strong changes inside a shape (including white lettering on dark artwork) are kept. These legacy patches receive the conservative correction at render time without modifying their files; new separated layers do not need this composite-boundary repair. Preview rendering, exports and reopening use the shared compositor, and the original-image toggle remains available. Substantially redrawn artwork is not automatically reconstructed by this correction.

### Advanced editor: render performance

Decoded artwork and identical text-free page renders share a bounded 64 MiB image cache. Content-based keys detect overwritten assets even when their filename, size and timestamp are unchanged; returned images have independent pixels. Frame keys include region state, mask contents and image metadata, and frames changed during rendering are not cached. Patch composition reuses renderer-owned buffers while retaining copy semantics for other callers. Text layout, font shaping, interpolation and export quality are unchanged.

After a layered brush stroke, history and session settings are saved first, then the existing precise renderer runs immediately in its background worker. The last complete frame remains visible until the current result arrives; the UI no longer performs a duplicate full-page render before scheduling it. Existing stale-result checks, layer ordering and undo/redo remain in use.

### Advanced editor: text occlusion brush

The left tool rail's **글씨 덮기 브러시** places original-image pixels above the rendered text, so foreground artwork can obscure selected strokes or characters. **글씨 덮기 지우개** removes that occlusion locally. Size is controlled by the existing top brush field, down to 1px. **도구 설정 → 편집기 설정 → 글씨 덮기 전체 지우기** clears this page's occlusion with undo support. The independent mask survives saving, reopening and portable exports; ordinary original restoration remains below text. AI artwork references exclude this text-occlusion layer.

### Advanced editor: AI sound-effect lettering

New requests separate source removal and lettering while retaining one Codex image-generation call. A local source-ink mask covers outlined/solid lettering and its halo, excluding thin drawing lines and large edge-connected artwork. Uniform regions use a flat fill; other masked areas use the installed LaMa model (CUDA when memory permits, CPU fallback), with only masked pixels accepted. The single native generation returns a real RGBA lettering-only PNG; opaque, empty and overflowing results are rejected. The editor stores the repaired background, transparent lettering and erase mask independently, and composes them through the shared preview/export renderer. A flattened preview is retained for older readers. Original toggling, undo snapshots, save/reopen and portable exports preserve all layers. Already-saved composite artifacts are not rewritten automatically. Automatic source masks and the artwork hidden beneath original lettering still require visual review.

Select an existing text box or draw one with the ordinary text-box tool. **AI 식자** and **재번역** stay at the top right of the inspector's existing tabs, keeping their labels and positions while either job runs; both are disabled during a job. The text field supplies optional Korean wording; an empty field or the new-box placeholder lets the model read the Japanese effect. Generation sends the selected box and a bounded read-only neighborhood from the current edited artwork, excluding editable text overlays, with the same box from the original added when needed to read the effect. The full page is not uploaded for AI lettering. The right side of the canvas zoom toolbar shows the job type, target page and region, actual stage, elapsed time, and text-only **중단**. In a narrow canvas, a compact button opens the same feedback in a popup. Completion exposes **비교·적용** without automatically opening a preview; click it to compare the original and proposed result before applying it. Opening a prepared result returns to its target page and region. Closing the feedback preserves the prepared result.

Applied results are fixed image regions: the existing render toggle becomes **식자 표시**, switching between the stored result and the original Japanese. The existing text field becomes **다음 생성에 사용할 문구**; changing it leaves the current image intact until a new preview is explicitly applied. Geometry and font controls are disabled for these fixed image regions. The canvas toolbar's **결과 보기** reopens comparison of a saved applied result. Generation runs in the background; **중단** prevents its result from being applied, although an already submitted request may continue on the server. A failed or stopped regeneration preserves the previous result. Changed region content or artwork cannot receive stale results. Undo, reopening sessions and portable image exports retain the original/result assets.

This uses the locally signed-in Codex CLI's native image-generation tool, in a separate connection from translation. Image generation consumes Codex usage; toggling, saving and exporting do not regenerate images. A working image-generation-capable Codex login is required. The generated crop is aligned and composited around the selected lettering to limit background changes; inspect the preview because reconstruction and lettering quality can vary.

The first reference contains exactly the selected box, inside a square canvas with white padding; no full-page reference is sent. Reconstruction uses only the artwork visible within the selection. Coordinate instructions use the actual 1024×1024 reference pixels. Long, narrow regions are restored to that exact selected extent after generation. Output adding strokes beyond the box is rejected before compositing rather than silently clipping the text. Small output aspect-ratio differences are fitted uniformly with padding instead of stretching the lettering. Larger framing mismatches remain unapplied. If a generated image is rejected, **결과 보기** opens a read-only comparison with **결과 폴더 열기**; the generated image, references and `failure.json` are retained in that request's assets folder. The failure report includes requested and returned dimensions and the rejection reason. Failed output has no apply action. Cancelled requests still clean their temporary files.

### Advanced editor: translation reroll

Select an ordinary text region with source text and click **재번역** beside **AI 식자** at the inspector's top right for a natural rewrite from the original. Its dropdown offers **경상도 사투리** or **요청해서 재번역…** for a one-off direction. Requests use the saved translation provider, model and work rules. Progress, elapsed time, and **중단** appear on the right side of the canvas zoom toolbar. When the proposal is ready, click **비교·적용** to return to its target page and region and open the comparison. Compare the current sentence with the editable proposal, then explicitly choose **이 문장 적용** to apply and save it; undo restores the previous sentence. Changed source text or translation invalidates the proposal. The existing header slots and text-field positions are preserved.

## Requirements

- Python 3.10+
- PyTorch (CPU, CUDA, ROCm, XPU, MPS)
- Font pack with `.ttf`/`.otf` files; included with portable package
- LLM for Japanese source text; VLM for other languages (API or local)

## Install

### Portable Package (Recommended)

Download the standalone zip from the releases page: [Portable Build](https://github.com/meangrinch/MangaTranslator/releases/tag/portable)

**Requirements:**

- **Windows:** Bundled Python/Git included; no additional requirements
- **Linux/macOS:** Python 3.10+ and Git must be installed on your system

**Setup:**

1. Extract the zip file
2. Run the setup script for your platform:
   - **Windows:** Double-click `setup.bat`
   - **Linux/macOS:** Run `./setup.sh` in terminal
3. PyTorch version is automatically detected and installed based on your system
4. Open the launcher script created in `./MangaTranslator/`:
   - **Windows:** `start-webui.bat`
   - **Linux/macOS:** `start-webui.sh`

Included font packs:

- _Komika_ (normal text)
- _Comicka_ (normal/OSB text)
- _Roboto_ (supports accents)
- _Noto Sans SC_ (Simplified Chinese)
- _Noto Sans KR_ (Korean)
- _Noto Sans JP_ (Japanese)

> [!TIP]
> In the event that you need to transfer to a fresh portable package:
>
> - You can safely move the `fonts`, `models`, and `output` directories to the new portable package
> - You might be able to move the `runtime` directory over, assuming the same setup configuration is wanted

### Manual install

1. Clone and enter the repo

```bash
git clone https://github.com/meangrinch/MangaTranslator.git
cd MangaTranslator
```

2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
# Windows PowerShell/CMD
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

3. Install PyTorch (see: [PyTorch Install](https://pytorch.org/get-started/locally/))

```bash
# Example (CUDA 13.0)
pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 --extra-index-url https://download.pytorch.org/whl/cu130
# Example (ROCm 7.1)
pip install torch==2.11.0+rocm7.1 torchvision==0.26.0+rocm7.1 --extra-index-url https://download.pytorch.org/whl/rocm7.1
# Example (XPU)
pip install torch==2.11.0+xpu torchvision==0.26.0+xpu --extra-index-url https://download.pytorch.org/whl/xpu
# Example (MPS/CPU)
pip install torch==2.11.0 torchvision==0.26.0
```

4. Install Nunchaku (optional, for FLUX.1 Kontext via Nunchaku backend)

- Nunchaku wheels are not on PyPI. Install directly from the v1.3.0dev20260213 GitHub release URL, matching your OS and Python version. CUDA only, and requires a 2000-series card or newer.

```bash
# Example (Windows, Python 3.13, PyTorch 2.11.0, CUDA 13.0)
pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.3.0dev20260213/nunchaku-1.3.0.dev20260213+cu13.0torch2.11-cp313-cp313-win_amd64.whl

# Example (Linux, Python 3.13, PyTorch 2.11.0, CUDA 13.0)
pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.3.0dev20260213/nunchaku-1.3.0.dev20260213+cu13.0torch2.11-cp313-cp313-linux_x86_64.whl
```

> [!NOTE]
> Nunchaku is not necessary for the use of Flux models via the sd.cpp/SDNQ backends.

5. Install dependencies

```bash
pip install -r requirements.txt
```

## Post-Install Setup

### Models

- The application will automatically download and use all required models

### Fonts

- Put font packs as subfolders in `fonts/` with `.otf`/`.ttf` files
- Prefer filenames that include `italic`/`bold` or both so variants are detected
- Example structure:

```text
fonts/
├─ CC Wild Words/
│  ├─ CCWildWords-Regular.otf
│  ├─ CCWildWords-Italic.otf
│  ├─ CCWildWords-Bold.otf
│  └─ CCWildWords-BoldItalic.otf
└─ Komika/
   ├─ KOMIKA-HAND.ttf
   └─ KOMIKA-HANDBOLD.ttf
```

### LLM setup

- Providers: Codex (local ChatGPT sign-in), Google, OpenAI, Anthropic, xAI, DeepSeek, Z.ai, Moonshot AI, Xiaomi MiMo, OpenRouter, OpenAI-Compatible
- Web UI: configure provider/model/key in the Config tab (stored locally)
- Codex: install the latest `@openai/codex` CLI and run `codex login` with ChatGPT. Available models and reasoning levels load from your account. Translation reuses one App Server connection with independent ephemeral requests and in-memory images; coding tools, plugins, hooks, and skills are disabled for this connection. ChatGPT usage limits still apply.
- CLI: pass keys/URLs as flags or via env vars
- Env vars: `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`, `ZAI_API_KEY`, `MOONSHOT_API_KEY`, `MIMO_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`
- OpenAI-compatible default URL: `http://localhost:8080/v1`

> [!NOTE]
> YanoljaNEXT-Rosetta models (e.g., `yanolja/YanoljaNEXT-Rosetta-4B-2511-GGUF`) are automatically detected when used via the OpenAI-Compatible provider and receive optimized prompting. These are text-only models and require two-step + local OCR model. The Special Instructions field is mapped to Rosetta's translation glossary (one entry per line, e.g., `Yanolja NEXT -> 야놀자넥스트`).

### OSB text setup (optional)

If you want to use the OSB text pipeline, you need a Hugging Face token with access to the following repositories:

- `deepghs/AnimeText_yolo`

#### Steps to create a token:

1. Sign in or create a Hugging Face account
2. Visit and accept the terms on:
   - [AnimeText_yolo](https://huggingface.co/deepghs/AnimeText_yolo)
   - [FLUX.1 Kontext (dev)](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) (optional, if using FLUX.1 Kontext via Nunchaku backend)
   - [SAM 3](https://huggingface.co/facebook/sam3) (optional, if using SAM 3)
3. Create a new access token in your Hugging Face settings with read access to gated repos ("Read access to contents of public gated repos")
4. Add the token to the app:
   - Web UI: set `hf_token` in Config
   - Env var (alternative): set `HF_TOKEN`
5. Save config to preserve the token across sessions

## Run

### Web UI (Gradio)

- **Portable package:**
  - Windows: Double-click `start-webui.bat` inside the `MangaTranslator` folder
  - Linux/macOS: Run `./start-webui.sh` inside the `MangaTranslator` folder
- **Manual install:**
  - Windows: Run `python app.py --open-browser`

Options: `--models` (default `./models`), `--fonts` (default `./fonts`), `--port` (default `7676`), `--cpu`.
First launch can take ~1–2 minutes.

Once launched, configure your LLM provider in the Config tab, then upload images and click Translate.

### CLI

Examples:

```bash
# Single image, Japanese → English, Google provider
python main.py --input <image_path> \
  --font-dir "fonts/Komika" --provider Google --google-api-key <AI...>

# Batch folder, custom source/target languages, OpenAI-Compatible provider (llama.cpp)
python main.py --input <folder_path> --batch \
  --font-dir "fonts/Komika" \
  --input-language <src_lang> --output-language <tgt_lang> \
  --provider OpenAI-Compatible --openai-compatible-url http://localhost:8080/v1 \
  --output ./output

# Single Image, Japanese → English (Google), OSB text pipeline, custom OSB text font
python main.py --input <image_path> \
  --font-dir "fonts/Komika" --provider Google --google-api-key <AI...> \
  --osb-enable --osb-font-dir "fonts/Clementine"

# Cleaning-only mode (no translation/text rendering)
python main.py --input <image_path> --cleaning-only

# Upscaling-only mode (no detection/translation, only upscale)
python main.py --input <image_path> --upscaling-only --image-upscale-mode final --image-upscale-factor 2.0

# Test mode (no translation; render placeholder text)
python main.py --input <image_path> --test-mode

# Full options
python main.py --help
```

## Documentation

- [Hardware Requirements](docs/HARDWARE_REQUIREMENTS.md)
- [Recommended Fonts](docs/FONTS.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Updating

### Portable Package

- Windows: Run `update.bat` from the portable package root
- Linux/macOS: Run `./update.sh` from the portable package root

### Manual Install

From the repo root:

```bash
git pull
pip install -r requirements.txt  # Or activate venv first if present
```

## License & credits

- License: Apache-2.0 (see [LICENSE](LICENSE))
- Author: [grinnch](https://github.com/meangrinch)
<details>
<summary><b>ML Models & Libraries</b></summary>

- YOLOv8m Speech Bubble Detector: [kitsumed](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble)
- Manga109 Speech Bubble Detector: [huyvux3005](https://huggingface.co/huyvux3005/manga109-segmentation-bubble)
- Comic Speech Bubble Detector YOLOv8m: [ogkalu](https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m)
- Manga109 YOLO: [deepghs](https://huggingface.co/deepghs/manga109_yolo)
- AnimeText YOLO: [deepghs](https://huggingface.co/deepghs/AnimeText_yolo)
- SAM 2.1: Segment Anything in Images and Videos: [Meta AI](https://huggingface.co/facebook/sam2.1-hiera-large)
- SAM 3: [Meta AI](https://huggingface.co/facebook/sam3)
- Manga OCR: [kha-white](https://github.com/kha-white/manga-ocr)
- PaddleOCR-VL-1.5: [PaddlePaddle](https://github.com/PaddlePaddle/PaddleOCR-VL-1.5)
- FLUX.1 Kontext: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- FLUX.2 Klein 4B: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- FLUX.2 Klein 9B: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
- Nunchaku: [Nunchaku AI](https://github.com/nunchaku-ai/nunchaku)
- SDNQ Quants: [Disty0](https://huggingface.co/Disty0)
- Unsloth Quants: [Unsloth](https://huggingface.co/unsloth)
- stable-diffusion.cpp: [leejet](https://github.com/leejet/stable-diffusion.cpp)
- 2x-AnimeSharpV4: [Kim2091](https://huggingface.co/Kim2091/2x-AnimeSharpV4)

</details>
