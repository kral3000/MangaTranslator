import functools
from pathlib import Path
from typing import Any

import gradio as gr

from core.tag_styles import TAG_DEFINITIONS, TAG_STYLE_COLUMNS, tag_styles_to_table

from . import callbacks, character_profiles, korean, settings_manager, utils

korean.install_gradio_korean(gr)

_ALPHABETICAL_LANGUAGES = [
    "Afrikaans",
    "Albanian",
    "Arabic",
    "Armenian",
    "Bengali",
    "Bosnian",
    "Bulgarian",
    "Catalan",
    "Chinese (Simplified)",
    "Chinese (Traditional)",
    "Croatian",
    "Czech",
    "Danish",
    "Dutch",
    "English",
    "Estonian",
    "Persian (Farsi)",
    "Finnish",
    "French",
    "Galician",
    "Georgian",
    "German",
    "Greek",
    "Gujarati",
    "Hebrew",
    "Hindi",
    "Hungarian",
    "Icelandic",
    "Indonesian",
    "Italian",
    "Japanese",
    "Kannada",
    "Korean",
    "Latvian",
    "Lithuanian",
    "Malay",
    "Marathi",
    "Norwegian",
    "Polish",
    "Portuguese",
    "Punjabi",
    "Romanian",
    "Russian",
    "Serbian (Cyrillic)",
    "Serbian (Latin)",
    "Slovak",
    "Slovenian",
    "Spanish",
    "Swahili",
    "Swedish",
    "Tamil",
    "Telugu",
    "Filipino (Tagalog)",
    "Turkish",
    "Ukrainian",
    "Urdu",
    "Uzbek",
    "Vietnamese",
    "Welsh",
]

SOURCE_LANGUAGES = [
    "Japanese",
    "Korean",
    "Chinese (Simplified)",
    "Chinese (Traditional)",
] + [
    lang
    for lang in _ALPHABETICAL_LANGUAGES
    if lang
    not in ["Japanese", "Korean", "Chinese (Simplified)", "Chinese (Traditional)"]
]

TARGET_LANGUAGES = ["English"] + [
    lang for lang in _ALPHABETICAL_LANGUAGES if lang != "English"
]

# Languages supported by PaddleOCR-VL-1.5 (53 of the 59 in _ALPHABETICAL_LANGUAGES)
_PADDLE_OCR_VL_UNSUPPORTED = frozenset(
    ["Armenian", "Georgian", "Gujarati", "Hebrew", "Kannada", "Punjabi"]
)
PADDLE_OCR_VL_LANGUAGES = [
    lang for lang in _ALPHABETICAL_LANGUAGES if lang not in _PADDLE_OCR_VL_UNSUPPORTED
]

js_credits = """
function() {
    const editorStatus = document.getElementById('image_editor_status_message');
    if (editorStatus) {
        editorStatus.setAttribute('role', 'status');
        editorStatus.setAttribute('aria-live', 'polite');
        editorStatus.setAttribute('aria-atomic', 'true');
    }

    const footer = document.querySelector('footer');
    if (footer) {
        // Check if credits already exist
        if (footer.parentNode.querySelector('.mangatl-credits')) {
            return;
        }
        const newContent = document.createElement('div');
        newContent.className = 'mangatl-credits'; // Add a class for identification
        newContent.innerHTML = 'made by <a href="https://github.com/meangrinch">grinnch</a> with ❤️'; // credits

        newContent.style.textAlign = 'center';
        newContent.style.paddingTop = '50px';
        newContent.style.color = 'lightgray';

        // Style the hyperlink
        const link = newContent.querySelector('a');
        if (link) {
            link.style.color = 'gray';
            link.style.textDecoration = 'underline';
        }

        footer.parentNode.insertBefore(newContent, footer);
    }
}
"""

theme_preference_head = """
<script>
(() => {
    const storageKey = 'mangatranslator.theme-preference';
    const modes = ['light', 'dark', 'system'];
    const url = new URL(window.location.href);

    try {
        const requested = url.searchParams.get('__theme');
        if (modes.includes(requested)) {
            localStorage.setItem(storageKey, requested);
        } else {
            const saved = localStorage.getItem(storageKey);
            if (modes.includes(saved) && saved !== 'system') {
                url.searchParams.set('__theme', saved);
                window.location.replace(url.toString());
                return;
            }
        }

        document.addEventListener('click', (event) => {
            const button = event.target instanceof Element
                ? event.target.closest('.theme-buttons .theme-button')
                : null;
            const buttons = button?.closest('.theme-buttons')?.querySelectorAll('.theme-button');
            const index = buttons ? [...buttons].indexOf(button) : -1;
            if (index >= 0) {
                localStorage.setItem(storageKey, modes[index]);
            }
        }, true);
    } catch (_) {
        // The built-in Gradio "system" default remains available if storage is blocked.
    }
})();
</script>
"""

js_folder_upload = r"""
function() {
    const uploadElementIds = ['batch_folder_upload'];
    const encodedPathPrefix = '__MT_RELATIVE_PATH__';
    const supportedImage = /\.(?:jpe?g|png|webp|bmp|tiff?)$/i;

    const encodeRelativePath = (relativePath) => {
        const bytes = new TextEncoder().encode(relativePath.replaceAll('\\', '/'));
        let binary = '';
        for (const byte of bytes) {
            binary += String.fromCharCode(byte);
        }
        return btoa(binary)
            .replaceAll('+', '-')
            .replaceAll('/', '_')
            .replace(/=+$/, '');
    };

    const preserveRelativePath = (file, relativePath) => {
        const extensionMatch = file.name.match(/\.[^.]+$/);
        const extension = extensionMatch ? extensionMatch[0] : '';
        const encodedName = `${encodedPathPrefix}${encodeRelativePath(relativePath)}${extension}`;
        return new File([file], encodedName, {
            type: file.type,
            lastModified: file.lastModified,
        });
    };

    const replaceUploadDisplayNames = (root, replacements) => {
        const apply = () => {
            const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
            const textNodes = [];
            while (walker.nextNode()) {
                textNodes.push(walker.currentNode);
            }
            for (const node of textNodes) {
                let text = node.nodeValue || '';
                for (const [encodedName, displayName] of replacements.entries()) {
                    if (text.includes(encodedName)) {
                        text = text.replaceAll(encodedName, displayName);
                    }
                }
                node.nodeValue = text;
            }
            for (const element of root.querySelectorAll('[title], [aria-label]')) {
                for (const attr of ['title', 'aria-label']) {
                    const value = element.getAttribute(attr);
                    if (!value) {
                        continue;
                    }
                    let replaced = value;
                    for (const [encodedName, displayName] of replacements.entries()) {
                        replaced = replaced.replaceAll(encodedName, displayName);
                    }
                    if (replaced !== value) {
                        element.setAttribute(attr, replaced);
                    }
                }
            }
        };
        setTimeout(apply, 100);
        setTimeout(apply, 500);
        setTimeout(apply, 1500);
    };

    const readDirectory = async (directoryEntry) => {
        const reader = directoryEntry.createReader();
        const entries = [];
        while (true) {
            const batch = await new Promise((resolve, reject) => {
                reader.readEntries(resolve, reject);
            });
            if (!batch.length) {
                break;
            }
            entries.push(...batch);
        }

        const files = [];
        for (const entry of entries) {
            if (entry.isDirectory) {
                files.push(...await readDirectory(entry));
            } else if (entry.isFile && supportedImage.test(entry.name)) {
                const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
                files.push({
                    file,
                    relativePath: entry.fullPath.replace(/^\/+/, ''),
                });
            }
        }
        return files;
    };

    const setInputFiles = (input, filesWithPaths) => {
        const transfer = new DataTransfer();
        const replacements = new Map();
        for (const {file, relativePath} of filesWithPaths) {
            const preservedFile = preserveRelativePath(file, relativePath);
            replacements.set(preservedFile.name, relativePath.replaceAll('\\', '/'));
            transfer.items.add(preservedFile);
        }
        input.files = transfer.files;
        input.dataset.mtFolderPrepared = '1';
        input.dispatchEvent(new Event('change', {bubbles: true}));
        delete input.dataset.mtFolderPrepared;
        replaceUploadDisplayNames(input.closest('[id]') || input.parentElement || document.body, replacements);
    };

    const setupFolderUpload = (uploadElementId) => {
        const root = document.getElementById(uploadElementId);
        if (!root || root.dataset.mtFolderUploadReady === '1') {
            return Boolean(root);
        }
        root.dataset.mtFolderUploadReady = '1';

        root.addEventListener('change', (event) => {
            const input = event.target;
            if (!(input instanceof HTMLInputElement) || input.type !== 'file') {
                return;
            }
            if (input.dataset.mtFolderPrepared === '1') {
                return;
            }

            const folderFiles = Array.from(input.files || [])
                .filter((file) => file.webkitRelativePath && supportedImage.test(file.name))
                .map((file) => ({file, relativePath: file.webkitRelativePath}));
            if (!folderFiles.length) {
                return;
            }

            event.preventDefault();
            event.stopImmediatePropagation();
            setInputFiles(input, folderFiles);
        }, true);

        root.addEventListener('drop', async (event) => {
            const entries = Array.from(event.dataTransfer?.items || [])
                .map((item) => item.webkitGetAsEntry?.())
                .filter(Boolean);
            if (!entries.some((entry) => entry.isDirectory)) {
                return;
            }

            event.preventDefault();
            event.stopImmediatePropagation();
            const input = root.querySelector('input[type="file"]');
            if (!input) {
                return;
            }

            const filesWithPaths = [];
            for (const entry of entries) {
                if (entry.isDirectory) {
                    filesWithPaths.push(...await readDirectory(entry));
                } else if (entry.isFile && supportedImage.test(entry.name)) {
                    const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
                    filesWithPaths.push({
                        file,
                        relativePath: entry.fullPath.replace(/^\/+/, ''),
                    });
                }
            }
            if (filesWithPaths.length) {
                setInputFiles(input, filesWithPaths);
            }
        }, true);

        return true;
    };

    const setupAllFolderUploads = () => {
        let foundAny = false;
        let missingAny = false;
        for (const uploadElementId of uploadElementIds) {
            const found = setupFolderUpload(uploadElementId);
            foundAny = foundAny || found;
            missingAny = missingAny || !found;
        }
        return {foundAny, missingAny};
    };

    const initial = setupAllFolderUploads();
    if (initial.missingAny) {
        const observer = new MutationObserver(() => {
            const state = setupAllFolderUploads();
            if (!state.missingAny) {
                observer.disconnect();
            }
        });
        observer.observe(document.body, {childList: true, subtree: true});
    }
}
"""


js_status_fade = """
() => {
    // Find the specific config status element by its ID
    const statusElement = document.getElementById('config_status_message');  // Config status

    // Apply fade logic only to the config status element
    if (statusElement) {
        if (statusElement && statusElement.textContent.trim() !== "") {
            clearTimeout(statusElement.fadeTimer);
            clearTimeout(statusElement.resetTimer);

            statusElement.style.display = 'block';
            statusElement.style.transition = 'none';
            statusElement.style.opacity = '1';

            const fadeDelay = 3000;
            const fadeDuration = 1000;

            statusElement.fadeTimer = setTimeout(() => {
                statusElement.style.transition = `opacity ${fadeDuration}ms ease-out`;
                statusElement.style.opacity = '0';

                statusElement.resetTimer = setTimeout(() => {
                    statusElement.style.display = 'none';
                    statusElement.style.opacity = '1';
                    statusElement.style.transition = 'none';
                }, fadeDuration);

            }, fadeDelay);
        } else {
            // Ensure hidden if empty
            statusElement.style.display = 'none';
        }
    }
}
"""

js_refresh_button_reset = """
() => {
    setTimeout(() => {
        const refreshButton = document.querySelector('.config-refresh-button button');
         if (refreshButton) {
            refreshButton.textContent = '모델 / 글꼴 새로고침';
            refreshButton.disabled = false;
        }
    }, 100); // Small delay to ensure Gradio update cycle completes
}
"""

js_refresh_button_processing = """
() => {
    const refreshButton = document.querySelector('.config-refresh-button button');
    if (refreshButton) {
        refreshButton.textContent = '새로고침 중...';
        refreshButton.disabled = true;
    }
    return []; // Required for JS function input/output
}
"""


js_reset_status_height = """
() => {
    setTimeout(() => {
        const ids = ['#translator_status_message textarea', '#batch_status_message textarea'];
        ids.forEach(selector => {
            const el = document.querySelector(selector);
            if (el) {
                el.style.height = '';
                el.style.removeProperty('height');
            }
        });
    }, 100);
}
"""

js_copy_runtime_log = """
async () => {
    const output = document.querySelector('#runtime-log-output textarea');
    const button = document.getElementById('runtime-log-copy');
    if (!output || !button) return [];

    try {
        await navigator.clipboard.writeText(output.value || '');
    } catch (_) {
        output.focus();
        output.select();
        document.execCommand('copy');
        output.setSelectionRange(0, 0);
    }

    const original = button.textContent;
    button.textContent = '복사 완료';
    setTimeout(() => { button.textContent = original; }, 1200);
    return [];
}
"""

js_scroll_runtime_log = """
() => {
    const output = document.querySelector('#runtime-log-output textarea');
    if (output) output.scrollTop = output.scrollHeight;
    return [];
}
"""


def create_layout(
    models_dir: Path, fonts_base_dir: Path, target_device: Any
) -> gr.Blocks:
    """Creates the Gradio UI layout and connects callbacks."""

    css_path = Path(__file__).with_name("style.css")
    profile_store = character_profiles.load_profiles()
    profile_choices = character_profiles.profile_choices(profile_store)
    active_profile = profile_store["active"]
    active_profile_entries = profile_store["profiles"].get(active_profile, "")

    def build_character_profile_panel(prefix: str):
        with gr.Group(elem_classes=["character-profile-card"]):
            gr.Markdown(
                "#### 작품별 등장인물 이름\n"
                "선택한 작품의 이름 표기를 번역 지침에 자동으로 추가합니다.",
                elem_classes=["character-profile-heading"],
            )
            with gr.Row(elem_classes=["character-profile-picker"]):
                selector = gr.Dropdown(
                    choices=profile_choices,
                    value=active_profile,
                    label="작품 사전",
                    filterable=True,
                    elem_id=f"{prefix}-character-profile-select",
                )
                new_button = gr.Button(
                    "새 작품",
                    variant="secondary",
                    scale=0,
                    min_width=104,
                    elem_classes=["character-profile-new"],
                )
            status = gr.Markdown(
                (
                    f"이름 표기 {len([line for line in active_profile_entries.splitlines() if line.strip()])}개가 적용됩니다."
                    if active_profile
                    else "사용 안 함 — 기존 추가 지시사항만 적용됩니다."
                ),
                elem_classes=["character-profile-status"],
            )
            with gr.Group(
                visible=bool(active_profile),
                elem_classes=["character-profile-editor"],
            ) as editor:
                name = gr.Textbox(
                    label="작품 이름",
                    value=active_profile,
                    placeholder="예: 그리하여 계절은 푸름을 알게 된다",
                )
                entries = gr.Textbox(
                    label="일본어 이름과 사용할 표기",
                    value=active_profile_entries,
                    placeholder=(
                        "한 줄에 한 명씩 입력하세요.\n"
                        "佐野 夏生(さの なつき)=사노 나츠키\n"
                        "春宮 ひな(はるみや ひな)=하루미야 히나\n"
                        "鬼木祥子(おにき しょうこ)=오니키 쇼코\n"
                        "一徹龍二(いってつ りゅうじ)=잇테츠 류지"
                    ),
                    lines=5,
                    max_lines=16,
                    elem_id=f"{prefix}-character-profile-entries",
                )
                with gr.Row(elem_classes=["character-profile-actions"]):
                    save_button = gr.Button("사전 저장", variant="primary")
                    delete_button = gr.Button("삭제", variant="stop")
        return {
            "selector": selector,
            "new": new_button,
            "status": status,
            "editor": editor,
            "name": name,
            "entries": entries,
            "save": save_button,
            "delete": delete_button,
        }

    with gr.Blocks(
        title="MangaTranslator",
        js=js_credits,
        head=theme_preference_head,
        css_paths=str(css_path),
    ) as app:

        runtime_log_open_state = gr.State(False)
        with gr.Row(elem_id="app-header-row"):
            gr.Markdown("# MangaTranslator", elem_classes=["app-title"])
            runtime_log_toggle_button = gr.Button(
                "실행 로그",
                variant="secondary",
                scale=0,
                min_width=112,
                elem_id="runtime-log-toggle",
            )

        with gr.Column(visible=False, elem_id="runtime-log-drawer") as runtime_log_drawer:
            with gr.Row(elem_id="runtime-log-toolbar"):
                gr.Markdown(
                    "### 실행 로그\n"
                    "창을 닫아도 기록은 계속되며 전체 로그는 `output/logs`에 저장됩니다.",
                    elem_classes=["runtime-log-heading"],
                )
                runtime_log_refresh_button = gr.Button(
                    "새로고침", scale=0, min_width=96
                )
                runtime_log_copy_button = gr.Button(
                    "전체 복사",
                    scale=0,
                    min_width=96,
                    elem_id="runtime-log-copy",
                )
                runtime_log_folder_button = gr.Button(
                    "로그 폴더 열기", scale=0, min_width=128
                )
            runtime_log_output = gr.Textbox(
                value="",
                label="실행 로그 내용",
                show_label=False,
                interactive=False,
                lines=14,
                max_lines=24,
                elem_id="runtime-log-output",
            )
        runtime_log_timer = gr.Timer(value=1.0, active=False)

        font_choices, initial_default_font = utils.get_available_font_packs(
            fonts_base_dir
        )
        saved_settings = settings_manager.get_saved_settings()
        output_browser_root = (
            Path(__file__).resolve().anchor
            or str(Path(__file__).resolve().parents[1])
        )

        saved_font_pack = saved_settings.get("font_pack")
        default_font = (
            saved_font_pack
            if saved_font_pack in font_choices
            else (initial_default_font if initial_default_font else None)
        )
        batch_saved_font_pack = saved_settings.get("batch_font_pack")
        batch_default_font = (
            batch_saved_font_pack
            if batch_saved_font_pack in font_choices
            else (initial_default_font if initial_default_font else None)
        )

        saved_emphasis_font_pack = saved_settings.get("emphasis_font_pack", "")
        if saved_emphasis_font_pack not in ([""] + font_choices):
            saved_emphasis_font_pack = ""

        saved_tag_style_rows = tag_styles_to_table(
            saved_settings.get("tag_styles", saved_settings.get("tag_style_table"))
        )

        saved_osb_font_pack = saved_settings.get("outside_text_osb_font_pack", "")
        if saved_osb_font_pack not in ([""] + font_choices):
            saved_osb_font_pack = ""

        saved_osb_emphasis_font_pack = saved_settings.get(
            "outside_text_osb_emphasis_font_pack", ""
        )
        if saved_osb_emphasis_font_pack not in ([""] + font_choices):
            saved_osb_emphasis_font_pack = ""

        initial_provider = saved_settings.get(
            "provider", settings_manager.DEFAULT_SETTINGS["provider"]
        )
        initial_model_name = saved_settings.get("model_name")

        if initial_provider == "OpenRouter" or initial_provider == "OpenAI-Compatible":
            initial_models_choices = [initial_model_name] if initial_model_name else []
        else:
            initial_models_choices = settings_manager.PROVIDER_MODELS.get(
                initial_provider, []
            )

        saved_max_tokens = saved_settings.get("max_tokens")
        if saved_max_tokens is not None:
            initial_max_tokens = saved_max_tokens
        else:
            is_reasoning = utils.is_reasoning_model(
                initial_provider, initial_model_name
            )
            initial_max_tokens = 16384 if is_reasoning else 4096

        # Calculate initial max_tokens maximum based on provider/model
        initial_max_tokens_cap = utils.get_max_tokens_cap(
            initial_provider, initial_model_name
        )
        initial_max_tokens_maximum = (
            initial_max_tokens_cap if initial_max_tokens_cap is not None else 63488
        )

        # --- Define UI Components ---
        with gr.Tabs():
            with gr.TabItem("Translator"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            type="filepath",
                            label="Upload Image",
                            show_download_button=False,
                            image_mode=None,
                            elem_id="translator_input_image",
                        )
                        font_dropdown = gr.Dropdown(
                            choices=font_choices,
                            label="Text Font",
                            value=default_font,
                            filterable=False,
                            visible=False,
                        )
                        with gr.Accordion("Translation Settings", open=True):
                            # Hidden state to store original language selection before manga-ocr forces Japanese
                            original_language_state = gr.State(
                                value=saved_settings.get("input_language", "Japanese")
                            )
                            input_language = gr.Dropdown(
                                SOURCE_LANGUAGES,
                                label="Source Language",
                                value=saved_settings.get("input_language", "Japanese"),
                                allow_custom_value=True,
                            )
                            output_language = gr.Dropdown(
                                TARGET_LANGUAGES,
                                label="Target Language",
                                value=saved_settings.get("output_language", "English"),
                                allow_custom_value=True,
                            )
                        translator_character_profile = build_character_profile_panel(
                            "translator"
                        )
                        special_instructions = gr.Textbox(
                            label="Special Instructions",
                            placeholder="Give the LLM optional context, formatting instructions, etc.",
                            value=saved_settings.get("special_instructions", ""),
                            lines=8,
                            max_lines=20,
                            elem_id="translator_special_instructions",
                        )
                    with gr.Column(scale=1):
                        output_image = gr.Image(
                            type="pil",
                            label="Translated Image",
                            interactive=False,
                            elem_id="translator_output_image",
                        )
                        status_message = gr.Textbox(
                            label="Status",
                            interactive=False,
                            elem_id="translator_status_message",
                        )
                        with gr.Row():
                            translator_saved_file = gr.File(
                                label="저장된 결과 파일",
                                interactive=False,
                                visible=True,
                            )
                            translator_plain_file = gr.File(
                                label="일반 이미지 사본",
                                interactive=False,
                                visible=True,
                            )
                        editor_session_state = gr.State(value="")
                        with gr.Row():
                            translate_button = gr.Button("Translate", variant="primary")
                            clear_button = gr.Button("Clear")
                            cancel_button = gr.Button(
                                "Cancel", variant="stop", visible=False
                            )
                            open_advanced_editor_button = gr.Button(
                                "고급 편집기 열기", visible=False
                            )

            with gr.TabItem("고급 이미지 편집기", elem_id="advanced-editor-tab"):
                gr.Markdown(
                    "## 편집 작업 열기\n"
                    "이전에 번역한 작업을 이어서 열거나, 일반 이미지 폴더로 새 작업을 시작할 수 있습니다.",
                    elem_classes=["editor-launch-intro"],
                )
                with gr.Row(equal_height=False, elem_classes=["editor-launch-grid"]):
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["editor-launch-card"]):
                            gr.Markdown(
                                "### 기존 작업 계속하기\n"
                                "번역기를 다시 켠 뒤에도 저장된 작업을 찾습니다. 일괄 번역은 여러 장을 한 묶음으로 엽니다.",
                                elem_classes=["editor-card-heading"],
                            )
                            with gr.Row(elem_classes=["editor-session-picker-row"]):
                                existing_editor_session_select = gr.Dropdown(
                                    choices=[],
                                    label="저장된 편집 작업",
                                    info="‘묶음 N장’ 작업을 열면 편집기 이미지 목록에서 모든 이미지를 이동할 수 있습니다.",
                                    visible=True,
                                    interactive=False,
                                    filterable=True,
                                    scale=1,
                                )
                                existing_editor_refresh_button = gr.Button(
                                    "세션 새로고침",
                                    variant="secondary",
                                    scale=0,
                                    min_width=138,
                                    elem_classes=["editor-inline-action"],
                                )
                            with gr.Row(elem_classes=["editor-card-actions"]):
                                existing_editor_open_button = gr.Button(
                                    "선택한 작업 계속하기",
                                    variant="primary",
                                    visible=False,
                                    scale=0,
                                    min_width=210,
                                )
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["editor-launch-card"]):
                            gr.Markdown(
                                "### 새 이미지 가져오기\n"
                                "기존 세션이 없는 이미지 폴더를 새 편집 작업으로 가져옵니다.",
                                elem_classes=["editor-card-heading"],
                            )
                            with gr.Row(elem_classes=["editor-folder-picker-row"]):
                                image_editor_folder_path = gr.Textbox(
                                    label="이미지 폴더",
                                    value=saved_settings.get(
                                        "image_editor_source_path", ""
                                    ),
                                    placeholder=r"C:\새로 편집할 이미지 폴더",
                                    info=(
                                        "하위 폴더의 이미지도 함께 불러옵니다. "
                                        "원본 덮어쓰기를 켜면 이 폴더의 실제 파일이 변경될 수 있습니다."
                                    ),
                                    interactive=False,
                                    scale=1,
                                )
                                image_editor_folder_button = gr.Button(
                                    "폴더 선택",
                                    variant="secondary",
                                    scale=0,
                                    min_width=112,
                                    elem_classes=["editor-inline-action"],
                                )
                            image_editor_font_dropdown = gr.Dropdown(
                                choices=font_choices,
                                label="추가 텍스트 기본 글꼴",
                                value=default_font,
                                filterable=False,
                                visible=True,
                            )
                            with gr.Row(elem_classes=["editor-card-actions"]):
                                image_editor_create_button = gr.Button(
                                    "이미지로 새 작업 만들기",
                                    variant="primary",
                                    scale=0,
                                    min_width=210,
                                )
                                image_editor_clear_button = gr.Button(
                                    "가져오기 초기화",
                                    variant="secondary",
                                    scale=0,
                                    min_width=138,
                                )
                            image_editor_session_select = gr.Dropdown(
                                choices=[],
                                label="가져온 이미지",
                                visible=False,
                                interactive=True,
                                filterable=False,
                            )
                            image_editor_open_button = gr.Button(
                                "가져온 이미지 편집하기",
                                visible=False,
                            )
                image_editor_status = gr.Textbox(
                    label="상태",
                    interactive=False,
                    elem_id="image_editor_status_message",
                    elem_classes=["editor-status-callout"],
                )

            with gr.TabItem("Batch"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_files = gr.File(
                            label="Upload Entire Folder",
                            file_count="directory",
                            file_types=["image"],
                            type="filepath",
                            elem_id="batch_folder_upload",
                        )
                        input_zip = gr.File(
                            label="Upload ZIP Archive (preserves directory structure)",
                            file_count="single",
                            file_types=[".zip"],
                            type="filepath",
                        )
                        batch_font_dropdown = gr.Dropdown(
                            choices=font_choices,
                            label="Text Font",
                            value=batch_default_font,
                            filterable=False,
                            visible=False,
                        )
                        with gr.Accordion("Translation Settings", open=True):
                            # Hidden state to store original language selection before manga-ocr forces Japanese
                            batch_original_language_state = gr.State(
                                value=saved_settings.get(
                                    "batch_input_language", "Japanese"
                                )
                            )
                            batch_input_language = gr.Dropdown(
                                SOURCE_LANGUAGES,
                                label="Source Language",
                                value=saved_settings.get(
                                    "batch_input_language", "Japanese"
                                ),
                                allow_custom_value=True,
                            )
                            batch_output_language = gr.Dropdown(
                                TARGET_LANGUAGES,
                                label="Target Language",
                                value=saved_settings.get(
                                    "batch_output_language", "English"
                                ),
                                allow_custom_value=True,
                            )
                        batch_character_profile = build_character_profile_panel("batch")
                        batch_special_instructions = gr.Textbox(
                            label="Special Instructions",
                            placeholder="Give the LLM optional context, formatting instructions, etc.",
                            value=saved_settings.get("batch_special_instructions", ""),
                            lines=8,
                            max_lines=20,
                            elem_id="batch_special_instructions",
                        )
                        batch_parallel_requests = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=int(saved_settings.get("batch_parallel_requests", 1)),
                            step=1,
                            label="Parallel Requests",
                            info="Number of images to process simultaneously. Does not affect translation quality.",
                        )
                        batch_retry_failed_once = gr.Checkbox(
                            value=saved_settings.get(
                                "batch_retry_failed_once", False
                            ),
                            label="Retry Failed Files Once",
                            info=(
                                "After the batch finishes, retry files that failed once "
                                "before creating download ZIPs."
                            ),
                        )
                        batch_previous_context_image_count = gr.Slider(
                            minimum=0,
                            maximum=10,
                            value=int(
                                (
                                    saved_settings.get(
                                        "batch_previous_context_image_count", 0
                                    )
                                    if saved_settings.get(
                                        "send_full_page_context", True
                                    )
                                    else 0
                                )
                            ),
                            step=1,
                            label="Previous Context Images",
                            info=(
                                "Sends up to this many previous source pages "
                                "as visual reference when full-page context is enabled. "
                                "Might improve translation quality."
                            ),
                            interactive=(
                                saved_settings.get("send_full_page_context", True)
                                and saved_settings.get("ocr_method", "LLM") == "LLM"
                            ),
                        )
                        batch_previous_context_text_count = gr.Slider(
                            minimum=0,
                            maximum=50,
                            value=int(
                                saved_settings.get(
                                    "batch_previous_context_text_count", 3
                                )
                            ),
                            step=1,
                            label="Previous Context OCR Text",
                            info=(
                                "Sends up to this many previous pages' OCR text "
                                "transcripts as narrative "
                                "reference. Might improve translation quality."
                            ),
                        )
                        batch_previous_context_text_mode = gr.Dropdown(
                            choices=[
                                ("Wait for previous OCR (quality)", "strict"),
                                ("Use ready OCR only (parallel)", "available"),
                            ],
                            value=saved_settings.get(
                                "batch_previous_context_text_mode", "strict"
                            ),
                            label="Previous OCR Context Mode",
                            info=(
                                "Strict waits for previous pages before translation. "
                                "Ready-only uses completed previous OCR without waiting, "
                                "so API requests can run in parallel but may send fewer "
                                "previous pages."
                            ),
                        )
                    with gr.Column(scale=1):
                        batch_output_gallery = gr.Gallery(
                            label="Translated Images",
                            show_label=True,
                            columns=4,
                            rows=2,
                            height="auto",
                            object_fit="contain",
                        )
                        batch_status_message = gr.Textbox(
                            label="Status",
                            interactive=False,
                            elem_id="batch_status_message",
                        )
                        with gr.Row():
                            batch_success_zip_file = gr.File(
                                label="Successful Files ZIP",
                                interactive=False,
                                visible=True,
                            )
                            batch_failed_zip_file = gr.File(
                                label="Failed Files ZIP",
                                interactive=False,
                                visible=True,
                            )
                        batch_editor_session_select = gr.Dropdown(
                            choices=[],
                            label="고급 편집기 세션",
                            visible=False,
                            interactive=True,
                            filterable=False,
                        )
                        with gr.Row():
                            batch_process_button = gr.Button(
                                "Start Batch Translating", variant="primary"
                            )
                            batch_clear_button = gr.Button("Clear")
                            batch_cancel_button = gr.Button(
                                "Cancel", variant="stop", visible=False
                            )
                            batch_advanced_editor_button = gr.Button(
                                "고급 편집기 열기", visible=False
                            )

            with gr.TabItem("Config", elem_id="settings-tab-container"):
                config_initial_provider = initial_provider
                config_initial_model_name = initial_model_name
                config_initial_models_choices = initial_models_choices

                with gr.Row(elem_id="config-button-row"):
                    save_config_btn = gr.Button(
                        "Save Config", variant="primary", scale=3
                    )
                    reset_defaults_btn = gr.Button(
                        "Reset Defaults", variant="secondary", scale=1
                    )

                # Assign specific ID for JS targeting
                config_status = gr.Markdown(elem_id="config_status_message")

                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, elem_id="settings-nav"):
                        nav_buttons = []
                        setting_groups = []
                        nav_button_detection = gr.Button(
                            "Detection",
                            elem_classes=["nav-button", "nav-button-selected"],
                        )
                        nav_buttons.append(nav_button_detection)
                        nav_button_cleaning = gr.Button(
                            "Cleaning", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_cleaning)
                        nav_button_translation = gr.Button(
                            "Translation", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_translation)
                        nav_button_rendering = gr.Button(
                            "Rendering", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_rendering)
                        nav_button_tag_styles = gr.Button(
                            "태그별 렌더링", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_tag_styles)
                        nav_button_outside_text = gr.Button(
                            "OSB Text", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_outside_text)
                        nav_button_output = gr.Button(
                            "Output", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_output)
                        nav_button_other = gr.Button("Other", elem_classes="nav-button")
                        nav_buttons.append(nav_button_other)

                    with gr.Column(scale=4, elem_id="config-content-area"):
                        # --- Detection Settings ---
                        with gr.Group(
                            visible=True, elem_classes="settings-group"
                        ) as group_detection:
                            gr.Markdown("### Speech Bubble Detection")
                            bubble_detector_model = gr.Radio(
                                choices=["yolo_1", "yolo_2"],
                                value=saved_settings.get(
                                    "bubble_detector_model", "yolo_1"
                                ),
                                label="Bubble Detector Model",
                                info=("Primary YOLO model for bubble detection."),
                            )
                            confidence = gr.Slider(
                                0.1,
                                1.0,
                                value=saved_settings.get("confidence", 0.6),
                                step=0.05,
                                label="Bubble Confidence Threshold",
                                info="Lower values detect more bubbles, but potentially include false positives.",
                            )
                            conjoined_detection_checkbox = gr.Checkbox(
                                value=saved_settings.get("conjoined_detection", True),
                                label="Conjoined Bubble Detection",
                                info=(
                                    "Uses a secondary YOLO model to detect and split "
                                    "conjoined speech bubbles into separate bubbles."
                                ),
                            )
                            conjoined_confidence = gr.Slider(
                                0.1,
                                1.0,
                                value=saved_settings.get("conjoined_confidence", 0.35),
                                step=0.05,
                                label="Conjoined Bubble Confidence Threshold",
                                info="Increase to filter out false positives, but may miss some conjoined bubbles.",
                                interactive=saved_settings.get(
                                    "conjoined_detection", True
                                ),
                            )
                            use_panel_sorting_checkbox = gr.Checkbox(
                                value=saved_settings.get("use_panel_sorting", True),
                                label="Use Panel-aware Sorting",
                                info=(
                                    "Use a panel detection YOLO model to group and sort speech bubbles "
                                    "within each panel for better reading order accuracy."
                                ),
                            )
                            panel_confidence = gr.Slider(
                                0.05,
                                1.0,
                                value=saved_settings.get("panel_confidence", 0.25),
                                step=0.05,
                                label="Panel Confidence Threshold",
                                info="Increase to filter out false positives, but may miss some panels.",
                                interactive=saved_settings.get(
                                    "use_panel_sorting", True
                                ),
                            )
                            seg_model = gr.Radio(
                                choices=["sam3", "sam2", "yolo"],
                                value=saved_settings.get("seg_model", "yolo"),
                                label="Segmentation Model",
                                info=(
                                    "Model to use to segment speech bubbles. "
                                    "SAM 3 requires a HF token (shared with 'OSB Text' section)."
                                ),
                            )
                            osb_text_verification_checkbox = gr.Checkbox(
                                value=saved_settings.get(
                                    "use_osb_text_verification", True
                                ),
                                label="Use AnimeText YOLO model for Bubble Verification",
                                info=(
                                    "Use the AnimeText YOLO model to confirm bubble detections fully cover text. "
                                    "Requires a Hugging Face token (shared with 'OSB Text' section)."
                                ),
                            )
                            config_reading_direction = gr.Radio(
                                choices=["rtl", "ltr"],
                                label="Reading Direction",
                                value=saved_settings.get("reading_direction", "rtl"),
                                info="Order for sorting bubbles (rtl=Manga, ltr=Comic).",
                                elem_id="config_reading_direction",
                            )
                        setting_groups.append(group_detection)

                        # --- Cleaning Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_cleaning:
                            gr.Markdown("### Mask Cleaning & Refinement")
                            thresholding_value = gr.Slider(
                                0,
                                255,
                                value=saved_settings.get("thresholding_value", 200),
                                step=1,
                                label="Fixed Threshold Value",
                                info=(
                                    "Brightness threshold for text detection. Lower helps clean "
                                    "edge-hugging text, but may thin bubble outlines"
                                ),
                                interactive=not saved_settings.get(
                                    "use_otsu_threshold", False
                                ),
                            )
                            use_otsu_threshold = gr.Checkbox(
                                value=saved_settings.get("use_otsu_threshold", False),
                                label="Force Automatic Thresholding (Otsu)",
                                info=(
                                    "Force Otsu's method for thresholding instead of the fixed value (on all bubbles). "
                                    "Recommended for varied lighting. Used as fallback when the fixed "
                                    "value fails, regardless of set value."
                                ),
                            )
                            roi_shrink_px = gr.Slider(
                                0,
                                10,
                                value=saved_settings.get("roi_shrink_px", 5),
                                step=1,
                                label="Shrink Threshold ROI (px)",
                                info=(
                                    "Shrink the threshold ROI inward by N pixels before fill. "
                                    "Lower helps clean edge-hugging text; higher preserves outlines."
                                ),
                            )
                            inpaint_colored_bubbles = gr.Checkbox(
                                value=saved_settings.get(
                                    "inpaint_colored_bubbles", False
                                ),
                                label="Use LAMA Large for Colored Bubbles",
                                info=(
                                    "Use mask-based LAMA Large instead of Flux for text "
                                    "inside colored, grayscale, or gradient speech bubbles."
                                ),
                            )
                            colored_bubble_inpainting_size = gr.Slider(
                                256,
                                4096,
                                value=saved_settings.get(
                                    "colored_bubble_inpainting_size", 2048
                                ),
                                step=256,
                                label="LAMA Inpainting Size",
                                info=(
                                    "Maximum image side used by LAMA. Larger values preserve "
                                    "more detail but require more memory."
                                ),
                                interactive=saved_settings.get(
                                    "inpaint_colored_bubbles", False
                                ),
                            )
                            colored_bubble_inpainting_precision = gr.Dropdown(
                                choices=["fp32", "bf16", "fp16"],
                                value=saved_settings.get(
                                    "colored_bubble_inpainting_precision", "fp32"
                                ),
                                label="LAMA Precision",
                                info=(
                                    "GPU precision for LAMA. FP32 is the compatibility "
                                    "default; BF16 is recommended on supported recent GPUs."
                                ),
                                interactive=saved_settings.get(
                                    "inpaint_colored_bubbles", False
                                ),
                            )
                            colored_bubble_mask_kernel_size = gr.Slider(
                                1,
                                31,
                                value=saved_settings.get(
                                    "colored_bubble_mask_kernel_size", 3
                                ),
                                step=2,
                                label="Text Mask Kernel Size",
                                info=(
                                    "Kernel used to expand the detected text mask. "
                                    "Odd values are used."
                                ),
                                interactive=saved_settings.get(
                                    "inpaint_colored_bubbles", False
                                ),
                            )
                            colored_bubble_mask_dilation_offset = gr.Slider(
                                0,
                                256,
                                value=saved_settings.get(
                                    "colored_bubble_mask_dilation_offset", 50
                                ),
                                step=1,
                                label="Mask Dilation Offset (px)",
                                info=(
                                    "Expands the text mask to include outlines, antialiasing, "
                                    "and attached symbols. Expansion is clipped inside the bubble."
                                ),
                                interactive=saved_settings.get(
                                    "inpaint_colored_bubbles", False
                                ),
                            )

                        setting_groups.append(group_cleaning)

                        # --- Translation Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_translation:
                            gr.Markdown("### OCR & Translation")
                            config_translation_mode = gr.Radio(
                                choices=["one-step", "two-step"],
                                label="Translation Mode",
                                value=saved_settings.get(
                                    "translation_mode",
                                    settings_manager.DEFAULT_SETTINGS[
                                        "translation_mode"
                                    ],
                                ),
                                info=(
                                    "Determines whether to perform OCR and translation together or separately. "
                                    "'two-step' might improve translation quality for less-capable LLMs."
                                ),
                                elem_id="config_translation_mode",
                            )
                            initial_ocr_method = saved_settings.get(
                                "ocr_method",
                                settings_manager.DEFAULT_SETTINGS.get(
                                    "ocr_method", "LLM"
                                ),
                            )
                            ocr_method_radio = gr.Radio(
                                choices=["LLM", "manga-ocr", "paddleocr-vl"],
                                label="OCR Method",
                                value=initial_ocr_method,
                                info=(
                                    "Determines whether to use a vision-capable LLM or a local OCR model for OCR. "
                                    "Local OCR options enable text-only LLMs for translation "
                                    "and must be used in 'two-step' translation mode."
                                ),
                                elem_id="ocr_method_radio",
                                interactive=saved_settings.get(
                                    "translation_mode",
                                    settings_manager.DEFAULT_SETTINGS[
                                        "translation_mode"
                                    ],
                                )
                                != "one-step",
                            )

                            gr.Markdown("### LLM Settings")
                            available_providers = utils.get_available_providers(
                                initial_ocr_method
                            )
                            initial_provider_value = (
                                config_initial_provider
                                if config_initial_provider in available_providers
                                else (
                                    available_providers[0]
                                    if available_providers
                                    else "Google"
                                )
                            )
                            if initial_provider_value != config_initial_provider:
                                config_initial_provider = initial_provider_value
                            provider_selector = gr.Radio(
                                choices=available_providers,
                                label="Translation Provider",
                                value=initial_provider_value,
                                elem_id="provider_selector",
                            )
                            provider_state = gr.State(
                                value=initial_provider_value,
                            )
                            google_api_key = gr.Textbox(
                                label="Google AI Studio API Key",
                                placeholder="Enter Google AI Studio API key (starts with AI...)",
                                type="password",
                                value=saved_settings.get("google_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "Google"),
                                elem_id="google_api_key",
                                info="Stored locally. Or set via GOOGLE_API_KEY env var.",
                            )
                            openai_api_key = gr.Textbox(
                                label="OpenAI API Key",
                                placeholder="Enter OpenAI API key (starts with sk-...)",
                                type="password",
                                value=saved_settings.get("openai_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "OpenAI"),
                                elem_id="openai_api_key",
                                info="Stored locally. Or set via OPENAI_API_KEY env var.",
                            )
                            anthropic_api_key = gr.Textbox(
                                label="Anthropic API Key",
                                placeholder="Enter Anthropic API key (starts with sk-ant-...)",
                                type="password",
                                value=saved_settings.get("anthropic_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "Anthropic"),
                                elem_id="anthropic_api_key",
                                info="Stored locally. Or set via ANTHROPIC_API_KEY env var.",
                            )
                            xai_api_key = gr.Textbox(
                                label="xAI API Key",
                                placeholder="Enter xAI API key (starts with xai-...)",
                                type="password",
                                value=saved_settings.get("xai_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "xAI"),
                                elem_id="xai_api_key",
                                info="Stored locally. Or set via XAI_API_KEY env var.",
                            )
                            deepseek_api_key = gr.Textbox(
                                label="DeepSeek API Key",
                                placeholder="Enter DeepSeek API key (starts with sk-...)",
                                type="password",
                                value=saved_settings.get("deepseek_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "DeepSeek"),
                                elem_id="deepseek_api_key",
                                info="Stored locally. Or set via DEEPSEEK_API_KEY env var.",
                            )
                            zai_api_key = gr.Textbox(
                                label="Z.ai API Key",
                                placeholder="Enter Z.ai API key",
                                type="password",
                                value=saved_settings.get("zai_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "Z.ai"),
                                elem_id="zai_api_key",
                                info="Stored locally. Or set via ZAI_API_KEY env var.",
                            )
                            moonshot_api_key = gr.Textbox(
                                label="Moonshot API Key",
                                placeholder="Enter Moonshot API key (starts with sk-...)",
                                type="password",
                                value=saved_settings.get("moonshot_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "Moonshot AI"),
                                elem_id="moonshot_api_key",
                                info="Stored locally. Or set via MOONSHOT_API_KEY env var.",
                            )
                            openrouter_api_key = gr.Textbox(
                                label="OpenRouter API Key",
                                placeholder="Enter OpenRouter API key (starts with sk-or-...)",
                                type="password",
                                value=saved_settings.get("openrouter_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "OpenRouter"),
                                elem_id="openrouter_api_key",
                                info="Stored locally. Or set via OPENROUTER_API_KEY env var.",
                            )
                            openai_compatible_url_input = gr.Textbox(
                                label="OpenAI-Compatible URL",
                                placeholder="Enter Base URL (e.g., http://localhost:8080/v1)",
                                type="text",
                                value=saved_settings.get(
                                    "openai_compatible_url",
                                    settings_manager.DEFAULT_SETTINGS[
                                        "openai_compatible_url"
                                    ],
                                ),
                                show_copy_button=False,
                                visible=(
                                    config_initial_provider == "OpenAI-Compatible"
                                ),
                                elem_id="openai_compatible_url_input",
                                info="Base URL of your OpenAI-Compatible API endpoint.",
                            )
                            openai_compatible_api_key_input = gr.Textbox(
                                label="OpenAI-Compatible API Key (Optional)",
                                placeholder="Enter API key if required",
                                type="password",
                                value=saved_settings.get(
                                    "openai_compatible_api_key", ""
                                ),
                                show_copy_button=False,
                                visible=(
                                    config_initial_provider == "OpenAI-Compatible"
                                ),
                                elem_id="openai_compatible_api_key_input",
                                info="Stored locally. Or set via OPENAI_COMPATIBLE_API_KEY env var.",
                            )
                            config_model_name = gr.Dropdown(
                                choices=config_initial_models_choices,
                                label="Model",
                                value=config_initial_model_name,
                                info="Select the specific model for the chosen provider.",
                                elem_id="config_model_name",
                                allow_custom_value=True,
                            )
                            (
                                _initial_reasoning_effort_visible,
                                _initial_reasoning_effort_choices,
                                _initial_reasoning_effort_default,
                            ) = utils.get_reasoning_effort_config(
                                config_initial_provider, config_initial_model_name
                            )

                            _initial_reasoning_effort_value = saved_settings.get(
                                "reasoning_effort"
                            )
                            if _initial_reasoning_effort_value is None:
                                _initial_reasoning_effort_value = (
                                    _initial_reasoning_effort_default
                                )
                            elif (
                                _initial_reasoning_effort_choices
                                and _initial_reasoning_effort_value
                                not in _initial_reasoning_effort_choices
                            ):
                                _initial_reasoning_effort_value = (
                                    _initial_reasoning_effort_default
                                )
                            elif not _initial_reasoning_effort_choices:
                                _initial_reasoning_effort_value = None

                            _initial_reasoning_effort_info = (
                                utils.get_reasoning_effort_info_text(
                                    config_initial_provider,
                                    config_initial_model_name,
                                    _initial_reasoning_effort_choices,
                                )
                            )

                            _initial_reasoning_effort_label = (
                                utils.get_reasoning_effort_label(
                                    config_initial_provider,
                                    config_initial_model_name,
                                )
                            )

                            reasoning_effort_dropdown = gr.Radio(
                                choices=_initial_reasoning_effort_choices,
                                label=_initial_reasoning_effort_label,
                                value=_initial_reasoning_effort_value,
                                info=_initial_reasoning_effort_info,
                                visible=_initial_reasoning_effort_visible,
                                elem_id="reasoning_effort_dropdown",
                            )

                            # Effort dropdown (Claude Opus 4.5/4.6/4.7 and Sonnet 4.6)
                            (
                                _initial_effort_visible,
                                _initial_effort_choices,
                                _initial_effort_default,
                            ) = utils.get_effort_config(
                                config_initial_provider, config_initial_model_name
                            )
                            _initial_effort_value = saved_settings.get("effort")
                            if _initial_effort_value is None:
                                _initial_effort_value = _initial_effort_default
                            elif (
                                _initial_effort_choices
                                and _initial_effort_value not in _initial_effort_choices
                            ):
                                _initial_effort_value = _initial_effort_default
                            elif not _initial_effort_choices:
                                _initial_effort_value = None

                            effort_dropdown = gr.Radio(
                                choices=_initial_effort_choices,
                                label="Effort",
                                value=_initial_effort_value,
                                info="Controls token spending eagerness. Opus 4.5/4.6/4.7, Sonnet 4.6 only.",
                                visible=_initial_effort_visible,
                                elem_id="effort_dropdown",
                            )

                            # Verbosity dropdown (GPT-5 series only)
                            (
                                _initial_verbosity_visible,
                                _initial_verbosity_choices,
                                _initial_verbosity_default,
                            ) = utils.get_verbosity_config(
                                config_initial_provider, config_initial_model_name
                            )
                            _initial_verbosity_value = saved_settings.get("verbosity")
                            if _initial_verbosity_value is None:
                                _initial_verbosity_value = _initial_verbosity_default
                            elif (
                                _initial_verbosity_choices
                                and _initial_verbosity_value
                                not in _initial_verbosity_choices
                            ):
                                _initial_verbosity_value = _initial_verbosity_default
                            elif not _initial_verbosity_choices:
                                _initial_verbosity_value = None

                            verbosity_dropdown = gr.Radio(
                                choices=_initial_verbosity_choices,
                                label="Verbosity",
                                value=_initial_verbosity_value,
                                info="Controls response verbosity. GPT-5 series only.",
                                visible=_initial_verbosity_visible,
                                elem_id="verbosity_dropdown",
                            )

                            _initial_enable_web_search_visible = (
                                config_initial_provider
                                not in ("OpenAI-Compatible", "DeepSeek")
                            )
                            (
                                _initial_enable_web_search_label,
                                _initial_enable_web_search_info,
                            ) = utils.get_enable_web_search_label_and_info(
                                config_initial_provider
                                if _initial_enable_web_search_visible
                                else "Google"
                            )

                            enable_web_search_checkbox = gr.Checkbox(
                                label=_initial_enable_web_search_label,
                                value=saved_settings.get("enable_web_search", False),
                                info=_initial_enable_web_search_info,
                                visible=_initial_enable_web_search_visible,
                                elem_id="enable_web_search_checkbox",
                            )

                            # Compute initial visibility for enable_code_execution
                            _initial_enable_code_execution_visible = (
                                utils.is_code_execution_visible(
                                    config_initial_provider,
                                    config_initial_model_name,
                                )
                            )

                            enable_code_execution_checkbox = gr.Checkbox(
                                label="Enable Code Execution with Images",
                                value=saved_settings.get(
                                    "enable_code_execution", False
                                ),
                                info="Allow Gemini 3 Flash to zoom and inspect image details using code execution.",
                                visible=_initial_enable_code_execution_visible,
                                interactive=initial_ocr_method
                                not in ("manga-ocr", "paddleocr-vl"),
                                elem_id="enable_code_execution_checkbox",
                            )

                            (
                                _initial_image_detail_visible,
                                _initial_image_detail_choices,
                                _initial_image_detail_default,
                                _initial_image_detail_info,
                            ) = utils.get_image_detail_config(
                                config_initial_provider, config_initial_model_name
                            )
                            _initial_image_detail_value = saved_settings.get(
                                "image_detail", _initial_image_detail_default
                            )
                            if (
                                _initial_image_detail_choices
                                and _initial_image_detail_value
                                not in _initial_image_detail_choices
                            ):
                                _initial_image_detail_value = (
                                    _initial_image_detail_default
                                )

                            image_detail_dropdown = gr.Radio(
                                label="Image Detail",
                                choices=_initial_image_detail_choices,
                                value=_initial_image_detail_value,
                                info=_initial_image_detail_info,
                                visible=_initial_image_detail_visible,
                                interactive=initial_ocr_method
                                not in ("manga-ocr", "paddleocr-vl"),
                                elem_id="image_detail_dropdown",
                            )

                            # Compute initial visibility for media_resolution (Google/xAI providers only)
                            _mr_bubbles_visible_init, _, _ = (
                                utils.get_media_resolution_config(
                                    config_initial_provider,
                                    config_initial_model_name,
                                )
                            )
                            _initial_media_resolution_visible = (
                                config_initial_provider == "Google"
                                and not _mr_bubbles_visible_init
                            )
                            initial_media_resolution_value = saved_settings.get(
                                "media_resolution", "auto"
                            )

                            media_resolution_dropdown = gr.Radio(
                                label="Media Resolution",
                                choices=["auto", "high", "medium", "low"],
                                value=initial_media_resolution_value,
                                info="Resolution for Gemini to process bubble/context images.",
                                visible=_initial_media_resolution_visible,
                                elem_id="media_resolution_dropdown",
                            )

                            # Compute initial visibility for Gemini 3 and xAI specific media resolution options
                            _mr_bubbles_visible, _mr_choices, _mr_info_base = (
                                utils.get_media_resolution_config(
                                    config_initial_provider, config_initial_model_name
                                )
                            )
                            _mr_bubbles_info = _mr_info_base.replace(
                                "process images", "process bubble images"
                            )
                            _mr_context_info = _mr_info_base.replace(
                                "process images", "process context (full page) images"
                            )

                            initial_media_resolution_bubbles_value = saved_settings.get(
                                "media_resolution_bubbles", "auto"
                            )
                            initial_media_resolution_context_value = saved_settings.get(
                                "media_resolution_context", "auto"
                            )

                            media_resolution_bubbles_dropdown = gr.Radio(
                                label="Media Resolution (Bubbles)",
                                choices=_mr_choices,
                                value=initial_media_resolution_bubbles_value,
                                info=_mr_bubbles_info,
                                visible=_mr_bubbles_visible,
                                elem_id="media_resolution_bubbles_dropdown",
                            )

                            media_resolution_context_dropdown = gr.Radio(
                                label="Media Resolution (Context)",
                                choices=_mr_choices,
                                value=initial_media_resolution_context_value,
                                info=_mr_context_info,
                                visible=_mr_bubbles_visible,
                                elem_id="media_resolution_context_dropdown",
                            )

                            temperature = gr.Slider(
                                0,
                                2.0,
                                value=saved_settings.get("temperature", 0.1),
                                step=0.05,
                                label="Temperature",
                                info="Controls creativity. Lower = deterministic; higher = random.",
                                elem_id="config_temperature",
                            )
                            top_p = gr.Slider(
                                0,
                                1,
                                value=saved_settings.get("top_p", 0.95),
                                step=0.05,
                                label="Top P",
                                info="Controls diversity. Lower = focused; higher = random.",
                                elem_id="config_top_p",
                            )
                            top_k = gr.Slider(
                                0,
                                64,
                                value=saved_settings.get("top_k", 64),
                                step=1,
                                label="Top K",
                                info="Limits sampling pool to top K tokens.",
                                interactive=(
                                    config_initial_provider
                                    not in ("OpenAI", "xAI", "DeepSeek", "Moonshot AI")
                                ),
                                elem_id="config_top_k",
                            )
                            max_tokens = gr.Slider(
                                2048,
                                initial_max_tokens_maximum,
                                value=initial_max_tokens,
                                step=1024,
                                label="Max Tokens",
                                info="Maximum number of tokens in the response.",
                                elem_id="config_max_tokens",
                            )

                            gr.Markdown("### Google / Vertex Request Limits")
                            google_rate_limit_enabled = gr.Checkbox(
                                value=saved_settings.get("google_rate_limit_enabled", False),
                                label="Enable Google / Vertex Request Limits",
                                info=(
                                    "Applies only to Google provider requests, including the Vertex "
                                    "service-account JSON endpoint."
                                ),
                                elem_id="google_rate_limit_enabled",
                            )
                            with gr.Row():
                                google_request_interval_seconds = gr.Number(
                                    value=saved_settings.get("google_request_interval_seconds", 0.0),
                                    label="Minimum Seconds Between Requests",
                                    info="0 disables the minimum spacing rule.",
                                    precision=2,
                                    elem_id="google_request_interval_seconds",
                                )
                                google_requests_per_minute = gr.Number(
                                    value=saved_settings.get("google_requests_per_minute", 0),
                                    label="Requests Per Minute",
                                    info="0 disables the per-minute limit.",
                                    precision=0,
                                    elem_id="google_requests_per_minute",
                                )
                            with gr.Row():
                                google_burst_requests = gr.Number(
                                    value=saved_settings.get("google_burst_requests", 0),
                                    label="Burst Requests",
                                    info="0 disables burst cooldown.",
                                    precision=0,
                                    elem_id="google_burst_requests",
                                )
                                google_burst_cooldown_seconds = gr.Number(
                                    value=saved_settings.get("google_burst_cooldown_seconds", 60.0),
                                    label="Burst Cooldown Seconds",
                                    precision=2,
                                    elem_id="google_burst_cooldown_seconds",
                                )
                            with gr.Row():
                                google_429_cooldown_seconds = gr.Number(
                                    value=saved_settings.get("google_429_cooldown_seconds", 60.0),
                                    label="429 Cooldown Seconds",
                                    info="Minimum retry wait after rate-limit responses.",
                                    precision=2,
                                    elem_id="google_429_cooldown_seconds",
                                )
                                google_max_retries = gr.Number(
                                    value=saved_settings.get("google_max_retries", 3),
                                    label="Max Retries",
                                    precision=0,
                                    elem_id="google_max_retries",
                                )
                            saved_google_429_action = str(
                                saved_settings.get("google_429_hourly_action", "stop")
                            ).strip().lower()
                            if saved_google_429_action not in {"stop", "pause"}:
                                saved_google_429_action = "stop"
                            with gr.Row():
                                google_429_hourly_limit = gr.Number(
                                    value=saved_settings.get("google_429_hourly_limit", 0),
                                    label="429 Errors Per Hour Limit",
                                    info=(
                                        "0 disables auto-stop/rest. The 1-hour counter starts "
                                        "when translation starts and resets after 1 hour."
                                    ),
                                    precision=0,
                                    elem_id="google_429_hourly_limit",
                                )
                                google_429_hourly_action = gr.Radio(
                                    choices=[
                                        ("Stop Translation", "stop"),
                                        ("Rest Then Continue", "pause"),
                                    ],
                                    value=saved_google_429_action,
                                    label="429 Limit Action",
                                    elem_id="google_429_hourly_action",
                                )
                                google_429_pause_minutes = gr.Number(
                                    value=saved_settings.get("google_429_pause_minutes", 5.0),
                                    label="429 Rest Minutes",
                                    info="Used only when the 429 limit action is Rest Then Continue.",
                                    precision=2,
                                    elem_id="google_429_pause_minutes",
                                )

                            gr.Markdown("### Context & Upscaling")
                            send_full_page_context = gr.Checkbox(
                                value=saved_settings.get(
                                    "send_full_page_context", True
                                ),
                                label="Send Full Page to LLM",
                                info=(
                                    "Include full page image as context. Might improve translation quality. "
                                    "Disable if refusals/using less-capable models or to reduce token usage."
                                ),
                                interactive=initial_ocr_method
                                not in ("manga-ocr", "paddleocr-vl"),
                            )
                            parallel_inpaint_translation = gr.Checkbox(
                                value=saved_settings.get(
                                    "parallel_inpaint_translation", False
                                ),
                                label="Parallel Inpaint + LLM Request",
                                info=(
                                    "Start the LLM translation request while inpainting is still running. "
                                    "Only applies to LLM OCR mode."
                                ),
                                interactive=initial_ocr_method
                                not in ("manga-ocr", "paddleocr-vl"),
                            )
                            whiteout_conjoined_bubbles = gr.Checkbox(
                                value=saved_settings.get(
                                    "whiteout_conjoined_bubbles", True
                                ),
                                label="White-out Conjoined Bubbles",
                                info=(
                                    "White-out text from neighboring conjoined bubbles to avoid translating "
                                    "the same text multiple times. Disable if encountering issues."
                                ),
                            )
                            upscale_method = gr.Radio(
                                choices=[
                                    ("Model", "model"),
                                    ("Model (Lite)", "model_lite"),
                                    ("LANCZOS", "lanczos"),
                                    ("None", "none"),
                                ],
                                value=saved_settings.get(
                                    "upscale_method", "model_lite"
                                ),
                                label="Bubble/Context Resizing Method",
                                info=(
                                    "Method to resize cropped bubble images/full page before sending to LLM/OCR model. "
                                    "Model is best quality, Model (Lite) is slightly worse quality but faster/less "
                                    "memory, LANCZOS is worst quality but fastest/least memory."
                                ),
                            )
                            initial_upscale_method = saved_settings.get(
                                "upscale_method", "model_lite"
                            )
                            sliders_interactive = initial_upscale_method != "none"
                            bubble_min_side_pixels = gr.Slider(
                                64,
                                512,
                                value=saved_settings.get("bubble_min_side_pixels", 128),
                                step=16,
                                label="Bubble Min Side Pixels",
                                info=(
                                    "Target minimum side length for speech bubble resizing. "
                                    "Increase for better OCR quality, but may increase token usage."
                                ),
                                elem_id="config_bubble_min_side_pixels",
                                interactive=sliders_interactive,
                            )
                            context_image_max_side_pixels = gr.Slider(
                                512,
                                2560,
                                value=saved_settings.get(
                                    "context_image_max_side_pixels", 1024
                                ),
                                step=128,
                                label="Context Image Max Side Pixels",
                                info=(
                                    "Target maximum side length for full page image resizing. "
                                    "Increase for better OCR quality, but may increase token usage."
                                ),
                                elem_id="config_context_image_max_side_pixels",
                                interactive=sliders_interactive,
                            )
                            osb_min_side_pixels = gr.Slider(
                                64,
                                512,
                                value=saved_settings.get("osb_min_side_pixels", 128),
                                step=16,
                                label="OSB Text Min Side Pixels",
                                info=(
                                    "Target minimum side length for outside speech bubble resizing. "
                                    "Increase for better OCR quality, but may increase token usage."
                                ),
                                elem_id="config_osb_min_side_pixels",
                                interactive=sliders_interactive,
                            )
                        setting_groups.append(group_translation)

                        # --- Rendering Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_rendering:
                            with gr.Column(visible=False):
                                gr.Markdown("### Font Rendering")
                                emphasis_font_pack = gr.Dropdown(
                                    value=saved_emphasis_font_pack,
                                    choices=[""] + font_choices,
                                    label="Emphasis Font",
                                    info=(
                                        "Font for bold/emphasized speech bubble text. "
                                        "Leave empty to use the selected regular font pack's bold variant."
                                    ),
                                )
                                max_font_size = gr.Slider(
                                    5,
                                    50,
                                    value=saved_settings.get("max_font_size", 16),
                                    step=1,
                                    label="Regular Max Font Size (px)",
                                    info="The largest font size used for regular speech bubble text.",
                                )
                                emphasis_max_font_size = gr.Slider(
                                    5,
                                    50,
                                    value=saved_settings.get(
                                        "emphasis_max_font_size",
                                        saved_settings.get("max_font_size", 16),
                                    ),
                                    step=1,
                                    label="Emphasis Max Font Size (px)",
                                    info="The largest font size used when speech bubble text is bold/emphasized.",
                                )
                                min_font_size = gr.Slider(
                                    5,
                                    50,
                                    value=saved_settings.get("min_font_size", 8),
                                    step=1,
                                    label="Min Font Size (px)",
                                    info="The smallest font size the renderer will attempt to use before giving up.",
                                )
                                line_spacing_mult = gr.Slider(
                                    0.5,
                                    2.0,
                                    value=saved_settings.get("line_spacing_mult", 1.0),
                                    step=0.05,
                                    label="Regular Line Spacing Multiplier",
                                    info="Adjusts line spacing for regular speech bubble text (1.0 = standard).",
                                )
                                emphasis_line_spacing_mult = gr.Slider(
                                    0.5,
                                    2.0,
                                    value=saved_settings.get(
                                        "emphasis_line_spacing_mult",
                                        saved_settings.get("line_spacing_mult", 1.0),
                                    ),
                                    step=0.05,
                                    label="Emphasis Line Spacing Multiplier",
                                    info="Adjusts line spacing when speech bubble text is fully bold/emphasized.",
                                )
                                vertical_line_spacing_mult = gr.Slider(
                                    0.5,
                                    2.0,
                                    value=saved_settings.get(
                                        "vertical_line_spacing_mult",
                                        saved_settings.get("line_spacing_mult", 1.0),
                                    ),
                                    step=0.05,
                                    label="Vertical Regular Line Spacing Multiplier",
                                    info="Line spacing used only for regular vertical text inside speech bubbles.",
                                )
                                vertical_emphasis_line_spacing_mult = gr.Slider(
                                    0.5,
                                    2.0,
                                    value=saved_settings.get(
                                        "vertical_emphasis_line_spacing_mult",
                                        saved_settings.get(
                                            "emphasis_line_spacing_mult",
                                            saved_settings.get("vertical_line_spacing_mult", 1.0),
                                        ),
                                    ),
                                    step=0.05,
                                    label="Vertical Emphasis Line Spacing Multiplier",
                                    info="Line spacing used only when vertical speech-bubble text is fully bold/emphasized.",
                                )
                                vertical_font_size_mult = gr.Slider(
                                    0.5,
                                    2.0,
                                    value=saved_settings.get("vertical_font_size_mult", 1.0),
                                    step=0.05,
                                    label="Vertical Text Font Size Multiplier",
                                    info="Scales the max font size search range only for vertical speech-bubble text.",
                                )
                                vertical_max_chars = gr.Slider(
                                    0,
                                    20,
                                    value=saved_settings.get(
                                        "vertical_max_chars",
                                        saved_settings.get("vertical_min_chars", 4),
                                    ),
                                    step=1,
                                    label="Vertical Regular Max Characters",
                                    info="Maximum alphanumeric character count allowed for regular speech-bubble text to auto-switch to vertical layout. Set 0 to disable regular auto vertical text.",
                                )
                                vertical_emphasis_max_chars = gr.Slider(
                                    0,
                                    20,
                                    value=saved_settings.get(
                                        "vertical_emphasis_max_chars",
                                        saved_settings.get(
                                            "vertical_emphasis_min_chars",
                                            saved_settings.get("vertical_max_chars", saved_settings.get("vertical_min_chars", 4)),
                                        ),
                                    ),
                                    step=1,
                                    label="Vertical Emphasis Max Characters",
                                    info="Maximum alphanumeric character count allowed for fully emphasized speech-bubble text to auto-switch to vertical layout. Set 0 to disable emphasized auto vertical text.",
                                )
                            use_subpixel_rendering = gr.Checkbox(
                                value=saved_settings.get(
                                    "use_subpixel_rendering", False
                                ),
                                label="Use Subpixel Rendering",
                                info=(
                                    "Improves text clarity on RGB-based displays. "
                                    "Disable if using a PenTile-based display (i.e., an OLED screen)"
                                ),
                            )
                            font_hinting = gr.Radio(
                                choices=["none", "slight", "normal", "full"],
                                value=saved_settings.get("font_hinting", "none"),
                                label="Font Hinting",
                                info="Adjusts glyph outlines to fit pixel grid. 'None' is often best for "
                                "high-res displays.",
                            )
                            use_ligatures = gr.Checkbox(
                                value=saved_settings.get("use_ligatures", False),
                                label="Use Standard Ligatures (e.g., fi, fl)",
                                info="Enables common letter combinations to be rendered as single glyphs "
                                "(must be supported by the font).",
                            )
                            gr.Markdown("### 글자 색상 및 외곽선")
                            gr.Markdown(
                                "말풍선 내부: 밝은 배경 = 검은 글자 / 어두운 배경 = 흰 글자  \n"
                                "말풍선 외부: 밝은 배경 = 검은 글자 / 어두운 배경 = 흰 글자  \n"
                                "원문이 컬러 글자면 원문 색상을 우선 적용  \n"
                                "외부 글자는 반대색 외곽선 자동 적용"
                            )
                            gr.Markdown("#### 말풍선 내부 · 밝은 배경")
                            with gr.Row():
                                inside_light_text_color = gr.ColorPicker(
                                    value=saved_settings.get(
                                        "inside_light_text_color", "#000000"
                                    ),
                                    label="글자 색",
                                )
                                inside_light_outline_color = gr.ColorPicker(
                                    value=saved_settings.get(
                                        "inside_light_outline_color", "#ffffff"
                                    ),
                                    label="외곽선 색",
                                )
                                inside_light_outline_width = gr.Slider(
                                    0,
                                    10,
                                    value=saved_settings.get(
                                        "inside_light_outline_width", 0.0
                                    ),
                                    step=0.5,
                                    label="외곽선 두께(px)",
                                )
                            gr.Markdown("#### 말풍선 내부 · 어두운 배경")
                            with gr.Row():
                                inside_dark_text_color = gr.ColorPicker(
                                    value=saved_settings.get(
                                        "inside_dark_text_color", "#ffffff"
                                    ),
                                    label="글자 색",
                                )
                                inside_dark_outline_color = gr.ColorPicker(
                                    value=saved_settings.get(
                                        "inside_dark_outline_color", "#000000"
                                    ),
                                    label="외곽선 색",
                                )
                                inside_dark_outline_width = gr.Slider(
                                    0,
                                    10,
                                    value=saved_settings.get(
                                        "inside_dark_outline_width", 0.0
                                    ),
                                    step=0.5,
                                    label="외곽선 두께(px)",
                                )
                            gr.Markdown("#### 말풍선 외부 · 밝은 배경")
                            with gr.Row():
                                outside_light_text_color = gr.ColorPicker(
                                    value=saved_settings.get(
                                        "outside_light_text_color", "#000000"
                                    ),
                                    label="글자 색",
                                )
                                outside_light_outline_color = gr.ColorPicker(
                                    value=saved_settings.get(
                                        "outside_light_outline_color", "#ffffff"
                                    ),
                                    label="외곽선 색",
                                )
                                outside_light_outline_width = gr.Slider(
                                    0,
                                    10,
                                    value=saved_settings.get(
                                        "outside_light_outline_width",
                                        saved_settings.get(
                                            "outside_text_osb_outline_width", 3.0
                                        ),
                                    ),
                                    step=0.5,
                                    label="외곽선 두께(px)",
                                )
                            gr.Markdown("#### 말풍선 외부 · 어두운 배경")
                            with gr.Row():
                                outside_dark_text_color = gr.ColorPicker(
                                    value=saved_settings.get(
                                        "outside_dark_text_color", "#ffffff"
                                    ),
                                    label="글자 색",
                                )
                                outside_dark_outline_color = gr.ColorPicker(
                                    value=saved_settings.get(
                                        "outside_dark_outline_color", "#000000"
                                    ),
                                    label="외곽선 색",
                                )
                                outside_dark_outline_width = gr.Slider(
                                    0,
                                    10,
                                    value=saved_settings.get(
                                        "outside_dark_outline_width",
                                        saved_settings.get(
                                            "outside_text_osb_outline_width", 3.0
                                        ),
                                    ),
                                    step=0.5,
                                    label="외곽선 두께(px)",
                                )
                            gr.Markdown("### Text Layout")
                            detach_trailing_punctuation = gr.Checkbox(
                                value=saved_settings.get(
                                    "detach_trailing_punctuation", True
                                ),
                                label="Detach Trailing Punctuation",
                                info=(
                                    "Move trailing punctuation clusters onto a new line "
                                    "for better text wrapping."
                                ),
                            )
                            auto_vertical_text = gr.Checkbox(
                                value=saved_settings.get("auto_vertical_text", False),
                                label="Auto Vertical Text for Tall Bubbles",
                                info=(
                                    "Stack short translated text vertically in tall speech bubbles "
                                    "when it improves readability."
                                ),
                            )
                            hyphenate_before_scaling = gr.Checkbox(
                                value=saved_settings.get(
                                    "hyphenate_before_scaling", True
                                ),
                                label="Hyphenate Long Words",
                                info="Try inserting hyphens when wrapping before reducing font size.",
                            )
                            hyphen_penalty = gr.Slider(
                                100,
                                2000,
                                value=saved_settings.get("hyphen_penalty", 1000.0),
                                step=100,
                                label="Hyphen Penalty",
                                info="Penalty for hyphenated line breaks in text layout. "
                                "Increase to discourage hyphenation.",
                                interactive=saved_settings.get(
                                    "hyphenate_before_scaling", True
                                ),
                            )
                            hyphenation_min_word_length = gr.Slider(
                                4,
                                10,
                                value=saved_settings.get(
                                    "hyphenation_min_word_length", 8
                                ),
                                step=1,
                                label="Min Word Length for Hyphenation",
                                info="Minimum word length required for hyphenation.",
                                interactive=saved_settings.get(
                                    "hyphenate_before_scaling", True
                                ),
                            )
                            badness_exponent = gr.Slider(
                                2.0,
                                4.0,
                                value=saved_settings.get("badness_exponent", 3.0),
                                step=0.5,
                                label="Badness Exponent",
                                info="Exponent for line badness calculation in text layout. "
                                "Increase to avoid loose lines.",
                            )
                            with gr.Row():
                                padding_horizontal_pixels = gr.Slider(
                                    0,
                                    24,
                                    value=saved_settings.get(
                                        "padding_horizontal_pixels",
                                        saved_settings.get("padding_pixels", 5.0),
                                    ),
                                    step=1,
                                    label="Horizontal Padding (px)",
                                    info="Horizontal padding between text and speech-bubble edges.",
                                )
                                padding_vertical_pixels = gr.Slider(
                                    0,
                                    24,
                                    value=saved_settings.get(
                                        "padding_vertical_pixels",
                                        saved_settings.get("padding_pixels", 5.0),
                                    ),
                                    step=1,
                                    label="Vertical Padding (px)",
                                    info="Vertical padding between text and speech-bubble edges.",
                                )
                            supersampling_factor = gr.Slider(
                                1,
                                16,
                                value=saved_settings.get("supersampling_factor", 4),
                                step=1,
                                label="Supersampling Factor",
                                info="Render text at Nx resolution then downscale for smoother edges. "
                                "Higher values improve quality but use slightly more memory. 1 = disabled.",
                            )
                        setting_groups.append(group_rendering)

                        # --- Tag Style Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_tag_styles:
                            gr.Markdown("### 태그별 글꼴/배치 설정")
                            gr.Markdown(
                                "번역 결과가 `<BN>...</BN>`, `<SFX>...</SFX>`처럼 태그로 감싸져 있으면 "
                                "이 표의 글꼴/크기/줄 간격/세로쓰기 설정을 우선 적용합니다. "
                                "글꼴은 아래 드롭다운으로 선택하거나 표의 글꼴 칸에 직접 입력할 수 있으며, "
                                "fonts 폴더 안의 글꼴 폴더명이 자동으로 표시됩니다. "
                                "빈칸이면 기본 글꼴을 사용합니다."
                            )
                            with gr.Row(
                                equal_height=False,
                                elem_classes="tag-style-font-row",
                            ):
                                tag_style_font_target = gr.Dropdown(
                                    choices=list(TAG_DEFINITIONS.keys()),
                                    value="BN",
                                    label="글꼴을 적용할 태그",
                                    scale=1,
                                )
                                tag_style_font_pack = gr.Dropdown(
                                    choices=[""] + font_choices,
                                    value="",
                                    label="글꼴 선택",
                                    scale=2,
                                )
                                tag_style_font_apply = gr.Button(
                                    "선택한 태그에 글꼴 적용",
                                    variant="secondary",
                                    scale=1,
                                )
                            tag_style_table = gr.Dataframe(
                                headers=TAG_STYLE_COLUMNS,
                                value=saved_tag_style_rows,
                                datatype=[
                                    "str",
                                    "str",
                                    "str",
                                    "number",
                                    "number",
                                    "number",
                                    "number",
                                    "number",
                                    "number",
                                ],
                                column_widths=[
                                    "6%",
                                    "20%",
                                    "15%",
                                    "8%",
                                    "8%",
                                    "8%",
                                    "12%",
                                    "12%",
                                    "11%",
                                ],
                                row_count=(len(saved_tag_style_rows), "fixed"),
                                col_count=(len(TAG_STYLE_COLUMNS), "fixed"),
                                interactive=True,
                                label="태그별 설정 표",
                                wrap=True,
                            )
                            gr.Markdown(
                                "지원 태그: `<BN>`, `<BE>`, `<BT>`, `<BW>`, `<ON>`, `<OT>`, `<OM>`, "
                                "`<OD>`, `<OE>`, `<OH>`, `<SFX>`, `<SFXI>`, `<SFXS>`"
                            )
                        setting_groups.append(group_tag_styles)

                        # --- Outside Text Removal Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_outside_text:
                            gr.Markdown("### Outside Speech Bubble Text")
                            outside_text_huggingface_token = gr.Textbox(
                                value=saved_settings.get(
                                    "outside_text_huggingface_token", ""
                                ),
                                label="HuggingFace Token (Required for certain features)",
                                type="password",
                                info=(
                                    "Required for downloading AnimeText YOLO (required for OSB detection), "
                                    "Flux.1 Kontext Nunchaku, and/or SAM 3 models from HuggingFace Hub. "
                                    "Can also set via HF_TOKEN env var."
                                ),
                            )
                            outside_text_enabled = gr.Checkbox(
                                value=saved_settings.get("outside_text_enabled", False),
                                label="Enable OSB Text Detection",
                                info="Detect, inpaint, and translate text outside speech bubbles.",
                            )

                            # Wrap all settings except the enable checkbox and token in a Column with visibility control
                            with gr.Column(
                                visible=saved_settings.get(
                                    "outside_text_enabled", False
                                )
                            ) as outside_text_settings_wrapper:
                                gr.Markdown("### Detection")
                                outside_text_osb_confidence = gr.Slider(
                                    0.0,
                                    1.0,
                                    value=saved_settings.get(
                                        "outside_text_osb_confidence", 0.6
                                    ),
                                    step=0.05,
                                    label="OSB Text Detection Confidence",
                                    info="Lower values detect more text, but potentially include false positives.",
                                )
                                outside_text_bbox_expansion_percent = gr.Slider(
                                    0.0,
                                    1.0,
                                    value=saved_settings.get(
                                        "outside_text_bbox_expansion_percent", 0.1
                                    ),
                                    step=0.05,
                                    label="Bounding Box Expansion",
                                    info=(
                                        "Percentage to expand bounding boxes for text detection. "
                                        "Higher values capture more context around text."
                                    ),
                                )
                                outside_text_text_box_proximity_ratio = gr.Slider(
                                    0.01,
                                    0.1,
                                    value=saved_settings.get(
                                        "outside_text_text_box_proximity_ratio", 0.02
                                    ),
                                    step=0.01,
                                    label="Text Box Proximity Ratio",
                                    info=(
                                        "Ratio for grouping nearby text boxes (as fraction of image dimension). "
                                        "Increase to group more distant boxes together."
                                    ),
                                )
                                outside_text_enable_page_number_filtering = gr.Checkbox(
                                    value=saved_settings.get(
                                        "outside_text_enable_page_number_filtering",
                                        False,
                                    ),
                                    label="Filter Page Numbers",
                                    info=(
                                        "Use manga-ocr on margin detections to drop likely page numbers. "
                                        "Slightly slower and may detect false positives."
                                    ),
                                )
                                outside_text_page_filter_margin_threshold = gr.Slider(
                                    0.0,
                                    0.3,
                                    value=saved_settings.get(
                                        "outside_text_page_filter_margin_threshold",
                                        0.1,
                                    ),
                                    step=0.01,
                                    label="Page Number Margin Ratio",
                                    info=(
                                        "Maximum vertical margin (ratio of height) for page-number filtering."
                                    ),
                                    interactive=saved_settings.get(
                                        "outside_text_enable_page_number_filtering",
                                        False,
                                    ),
                                )
                                outside_text_page_filter_min_area_ratio = gr.Slider(
                                    0.0,
                                    0.2,
                                    value=saved_settings.get(
                                        "outside_text_page_filter_min_area_ratio",
                                        0.05,
                                    ),
                                    step=0.01,
                                    label="Page Number Min Area Ratio",
                                    info=(
                                        "Minimum area ratio for page-number filtering."
                                    ),
                                    interactive=saved_settings.get(
                                        "outside_text_enable_page_number_filtering",
                                        False,
                                    ),
                                )
                                gr.Markdown("### Inpainting")
                                outside_text_inpainting_method = gr.Radio(
                                    value=saved_settings.get(
                                        "outside_text_inpainting_method",
                                        "flux_klein_4b",
                                    ),
                                    choices=[
                                        ("Flux.2 Klein 9B", "flux_klein_9b"),
                                        ("Flux.2 Klein 4B", "flux_klein_4b"),
                                        ("Flux.1 Kontext (12B)", "flux_kontext"),
                                        ("OpenCV", "opencv"),
                                        ("None (text background)", "none"),
                                    ],
                                    label="Inpainting Method",
                                    info=(
                                        "Klein models are faster and cross-platform, but may introduce minor color "
                                        "shifts. Kontext models do not shift colors, but are larger and slower. "
                                        "Models are downloaded in 4-bit quantization."
                                    ),
                                )
                                _initial_method = saved_settings.get(
                                    "outside_text_inpainting_method", "flux_klein_4b"
                                )
                                _is_kontext = _initial_method == "flux_kontext"
                                outside_text_kontext_backend = gr.Radio(
                                    choices=[
                                        ("SDNQ (cross-platform)", "sdnq"),
                                        ("Nunchaku (CUDA)", "nunchaku"),
                                    ],
                                    value=saved_settings.get(
                                        "outside_text_kontext_backend", "sdnq"
                                    ),
                                    label="Kontext Backend",
                                    info=(
                                        "SDNQ: cross-platform, no HF token. "
                                        "Nunchaku: CUDA-only, HF token required."
                                    ),
                                    visible=_is_kontext,
                                )
                                _is_klein_model = _initial_method in (
                                    "flux_klein_9b",
                                    "flux_klein_4b",
                                )
                                _initial_backend = saved_settings.get(
                                    "outside_text_kontext_backend", "sdnq"
                                )
                                _show_low_vram = _is_klein_model or _is_kontext
                                _initial_offload_mode = saved_settings.get(
                                    "outside_text_flux_offload_mode"
                                )
                                if _initial_offload_mode not in {
                                    "device",
                                    "model",
                                    "sequential",
                                }:
                                    if not saved_settings.get(
                                        "outside_text_flux_cpu_offload", True
                                    ):
                                        _initial_offload_mode = "device"
                                    elif saved_settings.get(
                                        "outside_text_flux_low_vram", False
                                    ):
                                        _initial_offload_mode = "sequential"
                                    else:
                                        _initial_offload_mode = "model"

                                outside_text_flux_offload_mode = gr.Radio(
                                    choices=[
                                        ("GPU 상주 (CPU 오프로딩 안 함)", "device"),
                                        ("일반 CPU 오프로딩", "model"),
                                        ("순차 CPU 오프로딩 (저 VRAM)", "sequential"),
                                    ],
                                    value=_initial_offload_mode,
                                    label="Flux 메모리 모드",
                                    info=(
                                        "RAM 사용량을 줄이려면 GPU 상주를 선택하세요. "
                                        "단, 모델 전체를 담을 충분한 VRAM이 반드시 필요합니다."
                                    ),
                                    visible=_show_low_vram,
                                )
                                outside_text_flux_memory_help = gr.Markdown(
                                    """
**Flux 메모리 모드 사용 설명**

- **GPU 상주 (CPU 오프로딩 안 함)**: Flux 모델 전체를 GPU에 올립니다. 일반 RAM 사용량은 가장 적고 보통 가장 빠르지만, VRAM이 부족하면 모델 로딩 중 `CUDA out of memory` 오류가 발생합니다.
- **일반 CPU 오프로딩**: 현재 기본값입니다. 모델 구성요소를 RAM과 VRAM 사이에서 이동합니다. VRAM 사용량은 줄지만 일반 RAM 사용량이 크게 늘어날 수 있습니다.
- **순차 CPU 오프로딩 (저 VRAM)**: 더 작은 단위로 이동하여 VRAM 사용량을 가장 낮춥니다. 일반 RAM을 계속 사용하며 처리 속도가 가장 느립니다.

모드를 변경하면 다음 Flux 작업을 시작할 때 기존 Flux 모델을 자동으로 내리고 선택한 모드로 다시 로드합니다. GPU 상주 모드는 RAM 부족 해결에 도움이 되지만 VRAM 용량을 먼저 확인해야 합니다.
""",
                                    visible=_show_low_vram,
                                )
                                outside_text_flux_num_inference_steps = gr.Slider(
                                    1,
                                    (
                                        30
                                        if saved_settings.get(
                                            "outside_text_inpainting_method",
                                            "flux_klein_4b",
                                        )
                                        == "flux_kontext"
                                        else 12
                                    ),
                                    value=saved_settings.get(
                                        "outside_text_flux_num_inference_steps", 4
                                    ),
                                    step=1,
                                    label="Steps",
                                    info=(
                                        "Klein: 4 is recommended. "
                                        "Kontext: 6-15 is recommended."
                                    ),
                                    interactive=saved_settings.get(
                                        "outside_text_inpainting_method",
                                        "flux_klein_4b",
                                    )
                                    != "opencv",
                                )
                                outside_text_flux_kontext_prompt = gr.Textbox(
                                    value=saved_settings.get(
                                        "outside_text_flux_kontext_prompt",
                                        settings_manager.DEFAULT_SETTINGS[
                                            "outside_text_flux_kontext_prompt"
                                        ],
                                    ),
                                    label="Flux.1 Kontext Prompt",
                                    info=(
                                        "Prompt used by Flux.1 Kontext for outside-text and "
                                        "colored-bubble inpainting. Empty values use the built-in default."
                                    ),
                                    lines=4,
                                    max_lines=12,
                                )
                                outside_text_flux_klein_prompt = gr.Textbox(
                                    value=saved_settings.get(
                                        "outside_text_flux_klein_prompt",
                                        settings_manager.DEFAULT_SETTINGS[
                                            "outside_text_flux_klein_prompt"
                                        ],
                                    ),
                                    label="Flux.2 Klein Prompt",
                                    info=(
                                        "Prompt shared by Flux.2 Klein 4B and 9B for outside-text "
                                        "and colored-bubble inpainting. Empty values use the built-in default."
                                    ),
                                    lines=6,
                                    max_lines=16,
                                )
                                _is_klein_for_lum = saved_settings.get(
                                    "outside_text_inpainting_method",
                                    "flux_klein_4b",
                                ) in ("flux_klein_9b", "flux_klein_4b")
                                _is_flux_for_klein_options = saved_settings.get(
                                    "outside_text_inpainting_method",
                                    "flux_klein_4b",
                                ) not in ("opencv", "none")
                                _upscale_small_crops_enabled = saved_settings.get(
                                    "outside_text_flux_upscale_small_crops", True
                                )
                                _group_flux_regions_enabled = saved_settings.get(
                                    "outside_text_flux_group_regions", False
                                )
                                outside_text_flux_luminance_correction = gr.Checkbox(
                                    value=saved_settings.get(
                                        "outside_text_flux_luminance_correction",
                                        True,
                                    ),
                                    label="휘도 보정",
                                    info=(
                                        "생성된 인페인팅 패치의 밝기를 주변 배경과 맞춥니다. "
                                        "Klein Flux 영역 묶기를 사용할 때는 원본 OSB 영역별로 보정합니다."
                                    ),
                                    visible=_is_klein_for_lum,
                                    interactive=(
                                        _is_klein_for_lum
                                        and _upscale_small_crops_enabled
                                    ),
                                )
                                outside_text_flux_upscale_small_crops = gr.Checkbox(
                                    value=saved_settings.get(
                                        "outside_text_flux_upscale_small_crops", True
                                    ),
                                    label="Upscale Klein Crops to ~1MP",
                                    info=(
                                        "Scale small Klein inpaint crops before Flux inference for better cleanup."
                                    ),
                                    visible=_is_flux_for_klein_options,
                                    interactive=_is_klein_for_lum,
                                )
                                outside_text_flux_group_regions = gr.Checkbox(
                                    value=saved_settings.get(
                                        "outside_text_flux_group_regions", False
                                    ),
                                    label="Group Flux Regions",
                                    info=(
                                        "Run one Flux pass over a combined expanded mask for all non-solid OSB regions."
                                    ),
                                    visible=_is_flux_for_klein_options,
                                    interactive=saved_settings.get(
                                        "outside_text_inpainting_method",
                                        "flux_klein_4b",
                                    )
                                    not in ("opencv", "none"),
                                )
                                outside_text_flux_residual_diff_threshold = gr.Slider(
                                    0.0,
                                    1.0,
                                    value=saved_settings.get(
                                        "outside_text_flux_residual_diff_threshold",
                                        0.15,
                                    ),
                                    step=0.01,
                                    label="Residual Diff Threshold",
                                    info=(
                                        "First Block Caching threshold for Flux.1 Kontext. "
                                        "Higher = faster, but lower quality."
                                    ),
                                    interactive=saved_settings.get(
                                        "outside_text_inpainting_method",
                                        "flux_klein_4b",
                                    )
                                    == "flux_kontext",
                                    visible=_is_kontext,
                                )
                                outside_text_osb_residual_render_guard_action = gr.Radio(
                                    choices=[
                                        ("처리 안 함", "none"),
                                        ("원본 복구", "restore_original"),
                                        ("LAMA Large 재시도", "lama_large"),
                                    ],
                                    value=saved_settings.get(
                                        "outside_text_osb_residual_render_guard_action",
                                        "restore_original",
                                    ),
                                    label="짧은 효과음 잔여 원문 처리",
                                    info=(
                                        "Flux 처리 후 남은 원문 위에 짧은 효과음 번역이 겹칠 때, "
                                        "처리하지 않고 그대로 렌더링하거나, 원본 영역을 복구하거나, "
                                        "남은 원문 부분만 LAMA Large로 다시 지웁니다."
                                    ),
                                    visible=_is_flux_for_klein_options,
                                    interactive=saved_settings.get(
                                        "outside_text_inpainting_method",
                                        "flux_klein_4b",
                                    )
                                    not in ("opencv", "none"),
                                )
                                outside_text_seed = gr.Number(
                                    value=saved_settings.get("outside_text_seed", 1),
                                    label="Seed",
                                    info="Seed for reproducible inpainting (-1 = random)",
                                    precision=0,
                                    interactive=saved_settings.get(
                                        "outside_text_inpainting_method",
                                        "flux_klein_4b",
                                    )
                                    not in ("opencv", "none"),
                                )
                                gr.Markdown("### OSB \ub80c\ub354\ub9c1 \ubc15\uc2a4")
                                outside_text_osb_tall_box_expand_aspect_ratio = (
                                    gr.Slider(
                                        1.0,
                                        10.0,
                                        value=saved_settings.get(
                                            "outside_text_osb_tall_box_expand_aspect_ratio",
                                            2.0,
                                        ),
                                        step=0.1,
                                        label="\ube44SFX OSB \uac00\ub85c \ud655\uc7a5 \uae30\uc900 (\uc138\ub85c/\uac00\ub85c)",
                                        info=(
                                            "SFX \ud0dc\uadf8\uac00 \uc544\ub2cc OSB \ubc15\uc2a4\uc758 \uc138\ub85c \uae38\uc774\uac00 "
                                            "\uac00\ub85c \uae38\uc774\uc758 \uc774 \ubc30\uc218 \uc774\uc0c1\uc77c \ub54c \uac00\ub85c \ud3ed\uc744 "
                                            "\ud655\uc7a5\ud569\ub2c8\ub2e4."
                                        ),
                                    )
                                )
                                outside_text_osb_tall_box_expand_width_percent = (
                                    gr.Slider(
                                        0,
                                        100,
                                        value=saved_settings.get(
                                            "outside_text_osb_tall_box_expand_width_percent",
                                            20.0,
                                        ),
                                        step=1,
                                        label="\ube44SFX OSB \uac00\ub85c \ud655\uc7a5 (%)",
                                        info=(
                                            "\uc870\uac74\uc744 \ub9cc\uc871\ud55c OSB \ub80c\ub354\ub9c1 \ubc15\uc2a4\uc758 "
                                            "\uc804\uccb4 \uac00\ub85c \ud3ed\uc744 \ub298\ub9ac\ub294 \ube44\uc728\uc785\ub2c8\ub2e4. "
                                            "\uc88c\uc6b0\ub85c \ub098\ub204\uc5b4 \ud655\uc7a5\ud558\uba70, \ud655\uc7a5\ub41c "
                                            "\ubc15\uc2a4\ub85c \uac00\ub85c/\uc138\ub85c \ub80c\ub354\ub9c1\uc744 "
                                            "\ube44\uad50\ud569\ub2c8\ub2e4. 0%\ub294 \uc0ac\uc6a9\ud558\uc9c0 "
                                            "\uc54a\uc74c\uc785\ub2c8\ub2e4."
                                        ),
                                    )
                                )
                                with gr.Column(visible=False):
                                    gr.Markdown("### Font Rendering")
                                    outside_text_osb_render_expansion_narrow_multiplier = gr.Slider(
                                        1.0,
                                        3.0,
                                        value=saved_settings.get(
                                            "outside_text_osb_render_expansion_narrow_multiplier",
                                            1.0,
                                        ),
                                        step=0.1,
                                        label="Narrow/Tall Expansion Multiplier",
                                        info=(
                                            "Expands OSB render boxes whose width/height aspect ratio "
                                            "is at or below the narrow/tall threshold."
                                        ),
                                    )
                                    outside_text_osb_render_expansion_aspect_ratio_threshold = gr.Slider(
                                        0.05,
                                        1.0,
                                        value=saved_settings.get(
                                            "outside_text_osb_render_expansion_aspect_ratio_threshold",
                                            0.4,
                                        ),
                                        step=0.01,
                                        label="Narrow/Tall Aspect Ratio Threshold",
                                        info=(
                                            "Classifies a box as narrow/tall when width divided by "
                                            "height is at or below this value."
                                        ),
                                    )
                                    outside_text_osb_render_expansion_tiny_multiplier = gr.Slider(
                                        1.0,
                                        3.0,
                                        value=saved_settings.get(
                                            "outside_text_osb_render_expansion_tiny_multiplier",
                                            1.0,
                                        ),
                                        step=0.1,
                                        label="Tiny Expansion Multiplier",
                                        info=(
                                            "Expands OSB render boxes whose area is below the tiny "
                                            "bubble area threshold."
                                        ),
                                    )
                                    outside_text_osb_render_expansion_area_threshold_percent = gr.Slider(
                                        0.0,
                                        5.0,
                                        value=saved_settings.get(
                                            "outside_text_osb_render_expansion_area_ratio_threshold",
                                            0.005,
                                        )
                                        * 100.0,
                                        step=0.1,
                                        label="Tiny Area Threshold (%)",
                                        info=(
                                            "Classifies a box as tiny when its area is below this "
                                            "percentage of the full image area."
                                        ),
                                    )
                                    outside_text_osb_font_pack = gr.Dropdown(
                                        value=saved_osb_font_pack,
                                        choices=[""] + font_choices,
                                        label="Text Font",
                                        info="Font for rendering OSB text translations (leave empty to use main font)",
                                    )
                                    outside_text_osb_emphasis_font_pack = gr.Dropdown(
                                        value=saved_osb_emphasis_font_pack,
                                        choices=[""] + font_choices,
                                        label="Emphasis Text Font",
                                        info=(
                                            "Font for bold/emphasized OSB text. "
                                            "Leave empty to use the OSB text font's bold variant."
                                        ),
                                    )
                                    outside_text_osb_max_font_size = gr.Slider(
                                        5,
                                        256,
                                        value=saved_settings.get(
                                            "outside_text_osb_max_font_size", 64
                                        ),
                                        step=1,
                                        label="OSB Regular Max Font Size (px)",
                                        info="The largest font size used for regular OSB text.",
                                    )
                                    outside_text_osb_emphasis_max_font_size = gr.Slider(
                                        5,
                                        256,
                                        value=saved_settings.get(
                                            "outside_text_osb_emphasis_max_font_size",
                                            saved_settings.get(
                                                "outside_text_osb_max_font_size", 64
                                            ),
                                        ),
                                        step=1,
                                        label="OSB Emphasis Max Font Size (px)",
                                        info="The largest font size used when OSB text is bold/emphasized.",
                                    )
                                    outside_text_osb_min_font_size = gr.Slider(
                                        5,
                                        96,
                                        value=saved_settings.get(
                                            "outside_text_osb_min_font_size", 10
                                        ),
                                        step=1,
                                        label="Min Font Size (px)",
                                        info="The smallest font size the renderer will attempt to use for OSB text.",
                                    )
                                    outside_text_osb_line_spacing = gr.Slider(
                                        0.5,
                                        2.0,
                                        value=saved_settings.get(
                                            "outside_text_osb_line_spacing", 1.0
                                        ),
                                        step=0.05,
                                        label="OSB Horizontal Line Spacing Multiplier",
                                        info=(
                                            "Adjusts spacing between horizontal OSB text lines "
                                            "(1.0 = standard). Decrease for tighter horizontal text."
                                        ),
                                    )
                                    outside_text_osb_vertical_line_spacing = gr.Slider(
                                        0.5,
                                        2.0,
                                        value=saved_settings.get(
                                            "outside_text_osb_vertical_line_spacing",
                                            saved_settings.get(
                                                "outside_text_osb_line_spacing", 1.0
                                            ),
                                        ),
                                        step=0.05,
                                        label="OSB Vertical Text Line Spacing Multiplier",
                                        info=(
                                            "Adjusts spacing between vertically stacked OSB characters "
                                            "(1.0 = standard). Decrease for tighter vertical text."
                                        ),
                                    )
                                    outside_text_osb_auto_vertical = gr.Checkbox(
                                        value=saved_settings.get(
                                            "outside_text_osb_auto_vertical", True
                                        ),
                                        label="Auto Vertical Text for Tall OSB Text",
                                        info=(
                                            "Use vertical stacking for outside-bubble text when "
                                            "the detected text box is tall and narrow."
                                        ),
                                    )
                                    outside_text_osb_vertical_aspect_ratio = gr.Slider(
                                        1.0,
                                        4.0,
                                        value=saved_settings.get(
                                            "outside_text_osb_vertical_aspect_ratio", 1.45
                                        ),
                                        step=0.05,
                                        label="OSB Vertical Box Aspect Ratio",
                                        info=(
                                            "Minimum height/width ratio for applying automatic "
                                            "vertical stacking to outside-bubble text."
                                        ),
                                    )
                                    outside_text_osb_vertical_max_chars = gr.Slider(
                                        0,
                                        120,
                                        value=saved_settings.get(
                                            "outside_text_osb_vertical_max_chars", 30
                                        ),
                                        step=1,
                                        label="OSB Vertical Max Characters",
                                        info=(
                                            "Maximum non-space character count for automatic "
                                            "vertical stacking. Set 0 for no character limit."
                                        ),
                                    )
                                    outside_text_osb_rotation_max_degrees = gr.Slider(
                                        0,
                                        45,
                                        value=saved_settings.get(
                                            "outside_text_osb_rotation_max_degrees", 15.0
                                        ),
                                        step=1,
                                        label="OSB Max Auto Tilt Angle (deg)",
                                        info=(
                                            "Maximum angle for automatically tilting rendered "
                                            "outside-bubble text to match the original. Set 0 to disable."
                                        ),
                                    )
                                    outside_text_osb_use_subpixel_rendering = gr.Checkbox(
                                        value=saved_settings.get(
                                            "outside_text_osb_use_subpixel_rendering", True
                                        ),
                                        label="Use Subpixel Rendering",
                                        info=(
                                            "Improves text clarity on RGB-based displays. "
                                            "Disable if using a PenTile-based display (i.e., an OLED screen)"
                                        ),
                                    )
                                    outside_text_osb_font_hinting = gr.Radio(
                                        choices=["none", "slight", "normal", "full"],
                                        value=saved_settings.get(
                                            "outside_text_osb_font_hinting", "none"
                                        ),
                                        label="Font Hinting",
                                        info=(
                                            "Adjusts glyph outlines to fit pixel grid. 'None' is often best for "
                                            "high-res displays."
                                        ),
                                    )
                                    outside_text_osb_use_ligatures = gr.Checkbox(
                                        value=saved_settings.get(
                                            "outside_text_osb_use_ligatures", False
                                        ),
                                        label="Use Standard Ligatures (e.g., fi, fl)",
                                        info=(
                                            "Enables common letter combinations to be rendered as single glyphs "
                                            "(must be supported by the font)."
                                        ),
                                    )
                                    outside_text_osb_outline_width = gr.Slider(
                                        0,
                                        10,
                                        value=saved_settings.get(
                                            "outside_text_osb_outline_width", 3.0
                                        ),
                                        step=0.5,
                                        label="Outline Width (px)",
                                        info="Width of text outline for OSB text.",
                                        visible=False,
                                    )
                                    outside_text_osb_padding_pixels = gr.Slider(
                                        0,
                                        128,
                                        value=saved_settings.get(
                                            "outside_text_osb_padding_pixels", 1.0
                                        ),
                                        step=1,
                                        label="OSB Padding (px)",
                                        info=(
                                            "Padding between rendered outside-bubble text "
                                            "and the detected text box boundary."
                                        ),
                                    )
                        setting_groups.append(group_outside_text)

                        # --- Output Settings ---
                        image_upscale_mode_default = saved_settings.get(
                            "image_upscale_mode", "off"
                        )
                        if image_upscale_mode_default not in {
                            "off",
                            "initial",
                            "final",
                        }:
                            image_upscale_mode_default = "off"
                        image_upscale_factor_default = float(
                            saved_settings.get("image_upscale_factor", 2.0)
                        )

                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_output:
                            gr.Markdown("### 저장 경로")
                            with gr.Row():
                                output_directory = gr.Textbox(
                                    label="저장 폴더 경로",
                                    value=saved_settings.get(
                                        "output_directory", "output"
                                    ),
                                    placeholder=r"C:\path\to\translated",
                                    info=(
                                        "절대 경로를 직접 입력하거나 찾아보기로 선택하세요. "
                                        "상대 경로도 입력할 수 있지만 MangaTranslator 폴더 기준으로 처리됩니다."
                                    ),
                                    scale=8,
                                )
                                browse_output_directory_button = gr.Button(
                                    "찾아보기",
                                    scale=1,
                                )
                            output_path_explorer = gr.FileExplorer(
                                glob="**/*",
                                file_count="single",
                                root_dir=output_browser_root,
                                label=(
                                    "저장 경로 선택 탐색기 "
                                    f"({output_browser_root} 기준)"
                                ),
                                height=320,
                                visible=False,
                                interactive=True,
                            )
                            disable_auto_save = gr.Checkbox(
                                label="번역 자동저장 안함",
                                value=bool(
                                    saved_settings.get("disable_auto_save", False)
                                ),
                                info=(
                                    "켜면 단일 번역과 일괄 번역 결과를 위 저장 폴더에 자동 저장하지 않습니다. "
                                    "화면 표시와 고급 편집기 세션은 유지되며, 일괄 번역의 결과 ZIP은 다운로드용으로 계속 생성됩니다."
                                ),
                            )
                            embed_editor_package_checkbox = gr.Checkbox(
                                label="편집 정보 포함 저장",
                                value=bool(
                                    saved_settings.get("embed_editor_package", True)
                                ),
                                info=(
                                    "켜면 결과 이미지 안에 고급 편집기 세션 정보를 함께 저장합니다. "
                                    "텍스트박스 좌표, 번역문, 마스크, 원본 복구용 이미지가 포함되며, 끄면 편집 정보가 없는 일반 이미지로 저장됩니다."
                                ),
                            )
                            save_plain_image_copy_checkbox = gr.Checkbox(
                                label="일반 이미지도 함께 저장",
                                value=bool(
                                    saved_settings.get("save_plain_image_copy", False)
                                ),
                                info=(
                                    "켜면 편집 정보 포함 저장이 켜져 있을 때 편집 정보가 없는 일반 이미지 사본도 함께 저장합니다. "
                                    "일반 이미지는 파일명 뒤에 _plain이 붙습니다."
                                ),
                            )
                            gr.Markdown("### Output Format")
                            output_format = gr.Radio(
                                choices=["auto", "png", "jpeg"],
                                label="Image Output Format",
                                value=saved_settings.get("output_format", "png"),
                                info="'auto' uses the same format as the input image (defaults to PNG if unknown).",
                            )
                            jpeg_quality = gr.Slider(
                                1,
                                100,
                                value=saved_settings.get("jpeg_quality", 95),
                                step=1,
                                label="JPEG Quality",
                                info="Higher levels result in better quality, but larger file sizes.",
                                interactive=saved_settings.get("output_format", "png")
                                != "png",
                            )
                            png_compression = gr.Slider(
                                0,
                                6,
                                value=saved_settings.get("png_compression", 2),
                                step=1,
                                label="PNG Compression Level",
                                info=(
                                    "Uses OxiPNG. Higher levels result in smaller file sizes, "
                                    "but slower processing times."
                                ),
                                interactive=saved_settings.get("output_format", "png")
                                != "jpeg",
                            )
                            gr.Markdown("### Upscaling")
                            image_upscale_mode = gr.Radio(
                                choices=["off", "initial", "final"],
                                value=image_upscale_mode_default,
                                label="Image Upscaling Method",
                                info=(
                                    "Determines whether to upscale the initial untranslated image or the final "
                                    "translated image. 'Initial' may result in cleaner text, but cause inconsistent "
                                    "results compare to 'final'."
                                ),
                            )
                            image_upscale_model = gr.Radio(
                                choices=[
                                    ("Model", "model"),
                                    ("Model (Lite)", "model_lite"),
                                ],
                                value=saved_settings.get(
                                    "image_upscale_model", "model_lite"
                                ),
                                label="Upscaling Model",
                                info=(
                                    "Model to use for image upscaling. Model is best quality, "
                                    "Model (Lite) is slightly worse quality but faster/less memory."
                                ),
                                interactive=image_upscale_mode_default != "off",
                            )
                            image_upscale_factor = gr.Slider(
                                1.0,
                                8.0,
                                value=image_upscale_factor_default,
                                step=0.1,
                                label="Upscale Factor",
                                info=(
                                    "Factor for the selected upscaling mode. "
                                    "The selected model will perform an upscaling pass every 2x, "
                                    "downscaling to meet the target if needed."
                                ),
                                interactive=image_upscale_mode_default != "off",
                            )
                            auto_scale = gr.Checkbox(
                                value=saved_settings.get("auto_scale", True),
                                label="Auto-Scale to Image Size",
                                info=(
                                    "Automatically scale pipeline parameters (fonts, kernels, etc.) "
                                    "to input image size, treating it as 1MP. Ensures consistent behavior "
                                    "across different image resolutions."
                                ),
                            )
                        setting_groups.append(group_output)

                        # --- Other Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_other:
                            gr.Markdown("### Other")
                            refresh_resources_button = gr.Button(
                                "Refresh Models / Fonts",
                                variant="secondary",
                                elem_classes="config-button",
                            )
                            unload_models_button = gr.Button(
                                "Force Unload Models",
                                variant="secondary",
                                elem_classes="config-button",
                            )
                            verbose = gr.Checkbox(
                                value=saved_settings.get("verbose", False),
                                label="Verbose Logging",
                                info="Enable verbose logging in console.",
                            )
                            cleaning_only_toggle = gr.Checkbox(
                                value=saved_settings.get("cleaning_only", False),
                                label="Cleaning-only Mode",
                                info="Skip translation and text rendering, output only the cleaned speech bubbles.",
                                interactive=not (
                                    saved_settings.get("test_mode", False)
                                    or saved_settings.get("upscaling_only", False)
                                ),
                            )
                            upscaling_only_toggle = gr.Checkbox(
                                value=saved_settings.get("upscaling_only", False),
                                label="Upscaling-only Mode",
                                info="Skip detection and translation, only upscale the image.",
                                interactive=not (
                                    saved_settings.get("cleaning_only", False)
                                    or saved_settings.get("test_mode", False)
                                ),
                            )
                            test_mode_toggle = gr.Checkbox(
                                value=saved_settings.get("test_mode", False),
                                label="Test Mode",
                                info=(
                                    "Skip translation and render placeholder text (lorem ipsum)."
                                ),
                                interactive=not (
                                    saved_settings.get("cleaning_only", False)
                                    or saved_settings.get("upscaling_only", False)
                                ),
                            )
                        setting_groups.append(group_other)

        # --- Define Event Handlers ---
        save_config_inputs = [
            confidence,
            conjoined_confidence,
            panel_confidence,
            seg_model,
            bubble_detector_model,
            conjoined_detection_checkbox,
            osb_text_verification_checkbox,
            use_panel_sorting_checkbox,
            config_reading_direction,
            thresholding_value,
            use_otsu_threshold,
            inpaint_colored_bubbles,
            colored_bubble_inpainting_size,
            colored_bubble_inpainting_precision,
            colored_bubble_mask_kernel_size,
            colored_bubble_mask_dilation_offset,
            roi_shrink_px,
            provider_selector,
            google_api_key,
            google_rate_limit_enabled,
            google_request_interval_seconds,
            google_requests_per_minute,
            google_burst_requests,
            google_burst_cooldown_seconds,
            google_429_cooldown_seconds,
            google_max_retries,
            google_429_hourly_limit,
            google_429_hourly_action,
            google_429_pause_minutes,
            openai_api_key,
            anthropic_api_key,
            xai_api_key,
            deepseek_api_key,
            zai_api_key,
            moonshot_api_key,
            openrouter_api_key,
            openai_compatible_url_input,
            openai_compatible_api_key_input,
            config_model_name,
            temperature,
            top_p,
            top_k,
            max_tokens,
            config_translation_mode,
            ocr_method_radio,
            max_font_size,
            emphasis_max_font_size,
            min_font_size,
            line_spacing_mult,
            emphasis_line_spacing_mult,
            vertical_line_spacing_mult,
            vertical_emphasis_line_spacing_mult,
            vertical_font_size_mult,
            vertical_max_chars,
            vertical_emphasis_max_chars,
            tag_style_table,
            use_subpixel_rendering,
            font_hinting,
            use_ligatures,
            inside_light_text_color,
            inside_light_outline_color,
            inside_light_outline_width,
            inside_dark_text_color,
            inside_dark_outline_color,
            inside_dark_outline_width,
            outside_light_text_color,
            outside_light_outline_color,
            outside_light_outline_width,
            outside_dark_text_color,
            outside_dark_outline_color,
            outside_dark_outline_width,
            output_format,
            jpeg_quality,
            png_compression,
            output_directory,
            disable_auto_save,
            embed_editor_package_checkbox,
            save_plain_image_copy_checkbox,
            verbose,
            cleaning_only_toggle,
            upscaling_only_toggle,
            test_mode_toggle,
            input_language,
            output_language,
            font_dropdown,
            emphasis_font_pack,
            batch_input_language,
            batch_output_language,
            batch_font_dropdown,
            enable_web_search_checkbox,
            enable_code_execution_checkbox,
            image_detail_dropdown,
            media_resolution_dropdown,
            media_resolution_bubbles_dropdown,
            media_resolution_context_dropdown,
            reasoning_effort_dropdown,
            effort_dropdown,
            verbosity_dropdown,
            send_full_page_context,
            parallel_inpaint_translation,
            whiteout_conjoined_bubbles,
            upscale_method,
            bubble_min_side_pixels,
            context_image_max_side_pixels,
            osb_min_side_pixels,
            hyphenate_before_scaling,
            detach_trailing_punctuation,
            auto_vertical_text,
            special_instructions,
            batch_special_instructions,
            hyphen_penalty,
            hyphenation_min_word_length,
            badness_exponent,
            padding_horizontal_pixels,
            padding_vertical_pixels,
            supersampling_factor,
            outside_text_enabled,
            outside_text_seed,
            outside_text_inpainting_method,
            outside_text_kontext_backend,
            outside_text_flux_offload_mode,
            outside_text_flux_num_inference_steps,
            outside_text_flux_kontext_prompt,
            outside_text_flux_klein_prompt,
            outside_text_flux_luminance_correction,
            outside_text_flux_upscale_small_crops,
            outside_text_flux_group_regions,
            outside_text_flux_residual_diff_threshold,
            outside_text_osb_residual_render_guard_action,
            outside_text_osb_confidence,
            outside_text_enable_page_number_filtering,
            outside_text_page_filter_margin_threshold,
            outside_text_page_filter_min_area_ratio,
            outside_text_huggingface_token,
            outside_text_osb_font_pack,
            outside_text_osb_emphasis_font_pack,
            outside_text_osb_max_font_size,
            outside_text_osb_emphasis_max_font_size,
            outside_text_osb_min_font_size,
            outside_text_osb_use_ligatures,
            outside_text_osb_outline_width,
            outside_text_osb_padding_pixels,
            outside_text_osb_line_spacing,
            outside_text_osb_vertical_line_spacing,
            outside_text_osb_auto_vertical,
            outside_text_osb_vertical_aspect_ratio,
            outside_text_osb_vertical_max_chars,
            outside_text_osb_tall_box_expand_aspect_ratio,
            outside_text_osb_tall_box_expand_width_percent,
            outside_text_osb_rotation_max_degrees,
            outside_text_osb_use_subpixel_rendering,
            outside_text_osb_font_hinting,
            outside_text_bbox_expansion_percent,
            outside_text_osb_render_expansion_narrow_multiplier,
            outside_text_osb_render_expansion_aspect_ratio_threshold,
            outside_text_osb_render_expansion_tiny_multiplier,
            outside_text_osb_render_expansion_area_threshold_percent,
            outside_text_text_box_proximity_ratio,
            image_upscale_mode,
            image_upscale_factor,
            image_upscale_model,
            auto_scale,
            batch_parallel_requests,
            batch_retry_failed_once,
            batch_previous_context_image_count,
            batch_previous_context_text_count,
            batch_previous_context_text_mode,
        ]

        reset_outputs = [
            confidence,
            conjoined_confidence,
            panel_confidence,
            seg_model,
            bubble_detector_model,
            conjoined_detection_checkbox,
            osb_text_verification_checkbox,
            use_panel_sorting_checkbox,
            config_reading_direction,
            thresholding_value,
            use_otsu_threshold,
            inpaint_colored_bubbles,
            colored_bubble_inpainting_size,
            colored_bubble_inpainting_precision,
            colored_bubble_mask_kernel_size,
            colored_bubble_mask_dilation_offset,
            roi_shrink_px,
            provider_selector,
            google_api_key,
            google_rate_limit_enabled,
            google_request_interval_seconds,
            google_requests_per_minute,
            google_burst_requests,
            google_burst_cooldown_seconds,
            google_429_cooldown_seconds,
            google_max_retries,
            google_429_hourly_limit,
            google_429_hourly_action,
            google_429_pause_minutes,
            openai_api_key,
            anthropic_api_key,
            xai_api_key,
            deepseek_api_key,
            zai_api_key,
            moonshot_api_key,
            openrouter_api_key,
            openai_compatible_url_input,
            openai_compatible_api_key_input,
            config_model_name,
            temperature,
            top_p,
            top_k,
            max_tokens,
            config_translation_mode,
            ocr_method_radio,
            max_font_size,
            emphasis_max_font_size,
            min_font_size,
            line_spacing_mult,
            emphasis_line_spacing_mult,
            vertical_line_spacing_mult,
            vertical_emphasis_line_spacing_mult,
            vertical_font_size_mult,
            vertical_max_chars,
            vertical_emphasis_max_chars,
            tag_style_table,
            use_subpixel_rendering,
            font_hinting,
            use_ligatures,
            inside_light_text_color,
            inside_light_outline_color,
            inside_light_outline_width,
            inside_dark_text_color,
            inside_dark_outline_color,
            inside_dark_outline_width,
            outside_light_text_color,
            outside_light_outline_color,
            outside_light_outline_width,
            outside_dark_text_color,
            outside_dark_outline_color,
            outside_dark_outline_width,
            output_format,
            jpeg_quality,
            png_compression,
            output_directory,
            disable_auto_save,
            embed_editor_package_checkbox,
            save_plain_image_copy_checkbox,
            verbose,
            cleaning_only_toggle,
            upscaling_only_toggle,
            test_mode_toggle,
            input_language,
            output_language,
            font_dropdown,
            emphasis_font_pack,
            batch_input_language,
            batch_output_language,
            batch_font_dropdown,
            enable_web_search_checkbox,
            enable_code_execution_checkbox,
            image_detail_dropdown,
            media_resolution_dropdown,
            media_resolution_bubbles_dropdown,
            media_resolution_context_dropdown,
            reasoning_effort_dropdown,
            effort_dropdown,
            verbosity_dropdown,
            send_full_page_context,
            parallel_inpaint_translation,
            whiteout_conjoined_bubbles,
            upscale_method,
            bubble_min_side_pixels,
            context_image_max_side_pixels,
            osb_min_side_pixels,
            hyphenate_before_scaling,
            detach_trailing_punctuation,
            auto_vertical_text,
            padding_horizontal_pixels,
            padding_vertical_pixels,
            special_instructions,
            batch_special_instructions,
            outside_text_enabled,
            outside_text_seed,
            outside_text_inpainting_method,
            outside_text_kontext_backend,
            outside_text_flux_offload_mode,
            outside_text_flux_num_inference_steps,
            outside_text_flux_kontext_prompt,
            outside_text_flux_klein_prompt,
            outside_text_flux_luminance_correction,
            outside_text_flux_upscale_small_crops,
            outside_text_flux_group_regions,
            outside_text_flux_residual_diff_threshold,
            outside_text_osb_residual_render_guard_action,
            outside_text_osb_confidence,
            outside_text_enable_page_number_filtering,
            outside_text_page_filter_margin_threshold,
            outside_text_page_filter_min_area_ratio,
            outside_text_huggingface_token,
            outside_text_osb_font_pack,
            outside_text_osb_emphasis_font_pack,
            outside_text_osb_max_font_size,
            outside_text_osb_emphasis_max_font_size,
            outside_text_osb_min_font_size,
            outside_text_osb_use_ligatures,
            outside_text_osb_outline_width,
            outside_text_osb_padding_pixels,
            outside_text_osb_line_spacing,
            outside_text_osb_vertical_line_spacing,
            outside_text_osb_auto_vertical,
            outside_text_osb_vertical_aspect_ratio,
            outside_text_osb_vertical_max_chars,
            outside_text_osb_tall_box_expand_aspect_ratio,
            outside_text_osb_tall_box_expand_width_percent,
            outside_text_osb_rotation_max_degrees,
            outside_text_osb_use_subpixel_rendering,
            outside_text_osb_font_hinting,
            outside_text_bbox_expansion_percent,
            outside_text_osb_render_expansion_narrow_multiplier,
            outside_text_osb_render_expansion_aspect_ratio_threshold,
            outside_text_osb_render_expansion_tiny_multiplier,
            outside_text_osb_render_expansion_area_threshold_percent,
            outside_text_text_box_proximity_ratio,
            image_upscale_mode,
            image_upscale_factor,
            image_upscale_model,
            auto_scale,
            batch_parallel_requests,
            batch_retry_failed_once,
            batch_previous_context_image_count,
            batch_previous_context_text_count,
            batch_previous_context_text_mode,
        ]

        translate_inputs = [
            input_image,
            confidence,
            conjoined_confidence,
            panel_confidence,
            seg_model,
            bubble_detector_model,
            conjoined_detection_checkbox,
            osb_text_verification_checkbox,
            use_panel_sorting_checkbox,
            thresholding_value,
            use_otsu_threshold,
            inpaint_colored_bubbles,
            colored_bubble_inpainting_size,
            colored_bubble_inpainting_precision,
            colored_bubble_mask_kernel_size,
            colored_bubble_mask_dilation_offset,
            roi_shrink_px,
            provider_selector,
            google_api_key,
            openai_api_key,
            anthropic_api_key,
            xai_api_key,
            deepseek_api_key,
            zai_api_key,
            moonshot_api_key,
            openrouter_api_key,
            openai_compatible_url_input,
            openai_compatible_api_key_input,
            config_model_name,
            temperature,
            top_p,
            top_k,
            max_tokens,
            config_reading_direction,
            config_translation_mode,
            ocr_method_radio,
            input_language,
            output_language,
            font_dropdown,
            emphasis_font_pack,
            max_font_size,
            emphasis_max_font_size,
            min_font_size,
            line_spacing_mult,
            emphasis_line_spacing_mult,
            vertical_line_spacing_mult,
            vertical_emphasis_line_spacing_mult,
            vertical_font_size_mult,
            vertical_max_chars,
            vertical_emphasis_max_chars,
            tag_style_table,
            use_subpixel_rendering,
            font_hinting,
            use_ligatures,
            inside_light_text_color,
            inside_light_outline_color,
            inside_light_outline_width,
            inside_dark_text_color,
            inside_dark_outline_color,
            inside_dark_outline_width,
            outside_light_text_color,
            outside_light_outline_color,
            outside_light_outline_width,
            outside_dark_text_color,
            outside_dark_outline_color,
            outside_dark_outline_width,
            output_format,
            jpeg_quality,
            png_compression,
            output_directory,
            disable_auto_save,
            embed_editor_package_checkbox,
            save_plain_image_copy_checkbox,
            verbose,
            cleaning_only_toggle,
            upscaling_only_toggle,
            test_mode_toggle,
            enable_web_search_checkbox,
            enable_code_execution_checkbox,
            image_detail_dropdown,
            media_resolution_dropdown,
            media_resolution_bubbles_dropdown,
            media_resolution_context_dropdown,
            reasoning_effort_dropdown,
            effort_dropdown,
            verbosity_dropdown,
            send_full_page_context,
            parallel_inpaint_translation,
            whiteout_conjoined_bubbles,
            upscale_method,
            bubble_min_side_pixels,
            context_image_max_side_pixels,
            osb_min_side_pixels,
            hyphenate_before_scaling,
            detach_trailing_punctuation,
            auto_vertical_text,
            hyphen_penalty,
            hyphenation_min_word_length,
            badness_exponent,
            padding_horizontal_pixels,
            padding_vertical_pixels,
            supersampling_factor,
            outside_text_enabled,
            outside_text_seed,
            outside_text_inpainting_method,
            outside_text_kontext_backend,
            outside_text_flux_offload_mode,
            outside_text_flux_num_inference_steps,
            outside_text_flux_kontext_prompt,
            outside_text_flux_klein_prompt,
            outside_text_flux_luminance_correction,
            outside_text_flux_upscale_small_crops,
            outside_text_flux_group_regions,
            outside_text_flux_residual_diff_threshold,
            outside_text_osb_residual_render_guard_action,
            outside_text_osb_confidence,
            outside_text_enable_page_number_filtering,
            outside_text_page_filter_margin_threshold,
            outside_text_page_filter_min_area_ratio,
            outside_text_huggingface_token,
            outside_text_osb_font_pack,
            outside_text_osb_emphasis_font_pack,
            outside_text_osb_max_font_size,
            outside_text_osb_emphasis_max_font_size,
            outside_text_osb_min_font_size,
            outside_text_osb_use_ligatures,
            outside_text_osb_outline_width,
            outside_text_osb_padding_pixels,
            outside_text_osb_line_spacing,
            outside_text_osb_vertical_line_spacing,
            outside_text_osb_auto_vertical,
            outside_text_osb_vertical_aspect_ratio,
            outside_text_osb_vertical_max_chars,
            outside_text_osb_tall_box_expand_aspect_ratio,
            outside_text_osb_tall_box_expand_width_percent,
            outside_text_osb_rotation_max_degrees,
            outside_text_osb_use_subpixel_rendering,
            outside_text_osb_font_hinting,
            outside_text_bbox_expansion_percent,
            outside_text_osb_render_expansion_narrow_multiplier,
            outside_text_osb_render_expansion_aspect_ratio_threshold,
            outside_text_osb_render_expansion_tiny_multiplier,
            outside_text_osb_render_expansion_area_threshold_percent,
            outside_text_text_box_proximity_ratio,
            image_upscale_mode,
            image_upscale_factor,
            image_upscale_model,
            auto_scale,
            batch_input_language,
            batch_output_language,
            batch_font_dropdown,
            translator_character_profile["selector"],
            special_instructions,
            batch_special_instructions,
            batch_parallel_requests,
            batch_retry_failed_once,
            batch_previous_context_image_count,
            batch_previous_context_text_count,
            batch_previous_context_text_mode,
        ]

        batch_inputs = [
            input_files,
            input_zip,
            confidence,
            conjoined_confidence,
            panel_confidence,
            seg_model,
            bubble_detector_model,
            conjoined_detection_checkbox,
            osb_text_verification_checkbox,
            use_panel_sorting_checkbox,
            thresholding_value,
            use_otsu_threshold,
            inpaint_colored_bubbles,
            colored_bubble_inpainting_size,
            colored_bubble_inpainting_precision,
            colored_bubble_mask_kernel_size,
            colored_bubble_mask_dilation_offset,
            roi_shrink_px,
            provider_selector,
            google_api_key,
            openai_api_key,
            anthropic_api_key,
            xai_api_key,
            deepseek_api_key,
            zai_api_key,
            moonshot_api_key,
            openrouter_api_key,
            openai_compatible_url_input,
            openai_compatible_api_key_input,
            config_model_name,
            temperature,
            top_p,
            top_k,
            max_tokens,
            config_reading_direction,
            config_translation_mode,
            ocr_method_radio,
            input_language,
            output_language,
            font_dropdown,
            emphasis_font_pack,
            max_font_size,
            emphasis_max_font_size,
            min_font_size,
            line_spacing_mult,
            emphasis_line_spacing_mult,
            vertical_line_spacing_mult,
            vertical_emphasis_line_spacing_mult,
            vertical_font_size_mult,
            vertical_max_chars,
            vertical_emphasis_max_chars,
            tag_style_table,
            use_subpixel_rendering,
            font_hinting,
            use_ligatures,
            inside_light_text_color,
            inside_light_outline_color,
            inside_light_outline_width,
            inside_dark_text_color,
            inside_dark_outline_color,
            inside_dark_outline_width,
            outside_light_text_color,
            outside_light_outline_color,
            outside_light_outline_width,
            outside_dark_text_color,
            outside_dark_outline_color,
            outside_dark_outline_width,
            output_format,
            jpeg_quality,
            png_compression,
            output_directory,
            disable_auto_save,
            embed_editor_package_checkbox,
            save_plain_image_copy_checkbox,
            verbose,
            cleaning_only_toggle,
            upscaling_only_toggle,
            test_mode_toggle,
            enable_web_search_checkbox,
            enable_code_execution_checkbox,
            image_detail_dropdown,
            media_resolution_dropdown,
            media_resolution_bubbles_dropdown,
            media_resolution_context_dropdown,
            reasoning_effort_dropdown,
            effort_dropdown,
            verbosity_dropdown,
            send_full_page_context,
            parallel_inpaint_translation,
            whiteout_conjoined_bubbles,
            upscale_method,
            bubble_min_side_pixels,
            context_image_max_side_pixels,
            osb_min_side_pixels,
            hyphenate_before_scaling,
            detach_trailing_punctuation,
            auto_vertical_text,
            hyphen_penalty,
            hyphenation_min_word_length,
            badness_exponent,
            padding_horizontal_pixels,
            padding_vertical_pixels,
            supersampling_factor,
            outside_text_enabled,
            outside_text_seed,
            outside_text_inpainting_method,
            outside_text_kontext_backend,
            outside_text_flux_offload_mode,
            outside_text_flux_num_inference_steps,
            outside_text_flux_kontext_prompt,
            outside_text_flux_klein_prompt,
            outside_text_flux_luminance_correction,
            outside_text_flux_upscale_small_crops,
            outside_text_flux_group_regions,
            outside_text_flux_residual_diff_threshold,
            outside_text_osb_residual_render_guard_action,
            outside_text_osb_confidence,
            outside_text_enable_page_number_filtering,
            outside_text_page_filter_margin_threshold,
            outside_text_page_filter_min_area_ratio,
            outside_text_huggingface_token,
            outside_text_osb_font_pack,
            outside_text_osb_emphasis_font_pack,
            outside_text_osb_max_font_size,
            outside_text_osb_emphasis_max_font_size,
            outside_text_osb_min_font_size,
            outside_text_osb_use_ligatures,
            outside_text_osb_outline_width,
            outside_text_osb_padding_pixels,
            outside_text_osb_line_spacing,
            outside_text_osb_vertical_line_spacing,
            outside_text_osb_auto_vertical,
            outside_text_osb_vertical_aspect_ratio,
            outside_text_osb_vertical_max_chars,
            outside_text_osb_tall_box_expand_aspect_ratio,
            outside_text_osb_tall_box_expand_width_percent,
            outside_text_osb_rotation_max_degrees,
            outside_text_osb_use_subpixel_rendering,
            outside_text_osb_font_hinting,
            outside_text_bbox_expansion_percent,
            outside_text_osb_render_expansion_narrow_multiplier,
            outside_text_osb_render_expansion_aspect_ratio_threshold,
            outside_text_osb_render_expansion_tiny_multiplier,
            outside_text_osb_render_expansion_area_threshold_percent,
            outside_text_text_box_proximity_ratio,
            image_upscale_mode,
            image_upscale_factor,
            image_upscale_model,
            auto_scale,
            batch_input_language,
            batch_output_language,
            batch_font_dropdown,
            batch_character_profile["selector"],
            special_instructions,
            batch_special_instructions,
            batch_parallel_requests,
            batch_retry_failed_once,
            batch_previous_context_image_count,
            batch_previous_context_text_count,
            batch_previous_context_text_mode,
        ]

        # Config Tab Navigation & Updates
        character_profile_outputs = [
            translator_character_profile["selector"],
            batch_character_profile["selector"],
            translator_character_profile["name"],
            batch_character_profile["name"],
            translator_character_profile["entries"],
            batch_character_profile["entries"],
            translator_character_profile["editor"],
            batch_character_profile["editor"],
            translator_character_profile["status"],
            batch_character_profile["status"],
        ]
        for panel in (translator_character_profile, batch_character_profile):
            panel["selector"].input(
                fn=callbacks.handle_character_profile_select,
                inputs=[panel["selector"]],
                outputs=character_profile_outputs,
                queue=False,
            )
            panel["new"].click(
                fn=callbacks.handle_new_character_profile,
                outputs=character_profile_outputs,
                queue=False,
            )
            panel["save"].click(
                fn=callbacks.handle_save_character_profile,
                inputs=[panel["selector"], panel["name"], panel["entries"]],
                outputs=character_profile_outputs,
                queue=False,
            )
            panel["delete"].click(
                fn=callbacks.handle_delete_character_profile,
                inputs=[panel["selector"]],
                outputs=character_profile_outputs,
                queue=False,
            )

        runtime_log_toggle_button.click(
            fn=callbacks.handle_runtime_log_toggle,
            inputs=[runtime_log_open_state],
            outputs=[
                runtime_log_drawer,
                runtime_log_open_state,
                runtime_log_timer,
                runtime_log_toggle_button,
                runtime_log_output,
            ],
            queue=False,
        ).then(fn=None, js=js_scroll_runtime_log, queue=False)
        runtime_log_refresh_button.click(
            fn=callbacks.handle_runtime_log_refresh,
            outputs=[runtime_log_output],
            queue=False,
        ).then(fn=None, js=js_scroll_runtime_log, queue=False)
        runtime_log_timer.tick(
            fn=callbacks.handle_runtime_log_refresh,
            outputs=[runtime_log_output],
            queue=False,
        ).then(fn=None, js=js_scroll_runtime_log, queue=False)
        runtime_log_copy_button.click(
            fn=None,
            js=js_copy_runtime_log,
            queue=False,
        )
        runtime_log_folder_button.click(
            fn=callbacks.handle_open_runtime_log_directory,
            queue=False,
        )

        output_components_for_switch = setting_groups + nav_buttons
        nav_button_detection.click(
            fn=lambda idx=0: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_cleaning.click(
            fn=lambda idx=1: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_translation.click(
            fn=lambda idx=2: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_rendering.click(
            fn=lambda idx=3: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_tag_styles.click(
            fn=lambda idx=4: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_outside_text.click(
            fn=lambda idx=5: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_output.click(
            fn=lambda idx=6: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_other.click(
            fn=lambda idx=7: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )

        browse_output_directory_button.click(
            fn=lambda: gr.update(visible=True),
            outputs=[output_path_explorer],
            queue=False,
        )
        output_path_explorer.change(
            fn=callbacks.handle_output_path_explorer_select,
            inputs=[output_path_explorer, output_directory],
            outputs=[output_directory],
            queue=False,
        )

        output_format.change(
            fn=callbacks.handle_output_format_change,
            inputs=output_format,
            outputs=[jpeg_quality, png_compression],
            queue=False,
        )

        image_upscale_mode.change(
            fn=lambda mode: (
                gr.update(interactive=mode != "off"),
                gr.update(interactive=mode != "off"),
            ),
            inputs=image_upscale_mode,
            outputs=[image_upscale_factor, image_upscale_model],
            queue=False,
        )

        provider_selector.change(
            fn=callbacks.handle_provider_change,
            inputs=[provider_selector, temperature, ocr_method_radio],
            outputs=[
                google_api_key,
                openai_api_key,
                anthropic_api_key,
                xai_api_key,
                deepseek_api_key,
                zai_api_key,
                moonshot_api_key,
                openrouter_api_key,
                openai_compatible_url_input,
                openai_compatible_api_key_input,
                config_model_name,
                temperature,
                top_p,
                top_k,
                max_tokens,
                enable_web_search_checkbox,
                enable_code_execution_checkbox,
                image_detail_dropdown,
                media_resolution_dropdown,
                media_resolution_bubbles_dropdown,
                media_resolution_context_dropdown,
                reasoning_effort_dropdown,
                effort_dropdown,
                verbosity_dropdown,
            ],
            queue=False,
        ).then(  # Trigger model fetch *after* provider change updates visibility etc.
            fn=lambda prov, url, key, ocr_method: (
                utils.fetch_and_update_compatible_models(url, key, force_refresh=True)
                if prov == "OpenAI-Compatible"
                else (
                    utils.fetch_and_update_openrouter_models(ocr_method=ocr_method)
                    if prov == "OpenRouter"
                    else gr.update()
                )
            ),
            inputs=[
                provider_selector,
                openai_compatible_url_input,
                openai_compatible_api_key_input,
                ocr_method_radio,
            ],
            outputs=[config_model_name],
            queue=True,  # Allow fetching to happen in the background
        )

        # Keep provider_state in sync with provider_selector for manual changes
        provider_selector.change(
            fn=lambda p: p,
            inputs=[provider_selector],
            outputs=[provider_state],
            queue=False,
        )

        config_model_name.change(
            fn=callbacks.handle_model_change,
            inputs=[provider_selector, config_model_name, temperature],
            outputs=[
                temperature,
                top_p,
                top_k,
                max_tokens,
                enable_web_search_checkbox,
                enable_code_execution_checkbox,
                image_detail_dropdown,
                media_resolution_dropdown,
                media_resolution_bubbles_dropdown,
                media_resolution_context_dropdown,
                reasoning_effort_dropdown,
                effort_dropdown,
                verbosity_dropdown,
            ],
            queue=False,
        )

        # Reasoning effort change → update temp/top_p slider interactivity
        reasoning_effort_dropdown.change(
            fn=callbacks.handle_reasoning_effort_change,
            inputs=[provider_selector, config_model_name, reasoning_effort_dropdown],
            outputs=[temperature, top_p],
            queue=False,
        )

        # Thresholding checkbox change handler
        use_otsu_threshold.change(
            fn=callbacks.handle_thresholding_change,
            inputs=use_otsu_threshold,
            outputs=thresholding_value,
            queue=False,
        )

        # Hyphenation checkbox change handler
        hyphenate_before_scaling.change(
            fn=callbacks.handle_hyphenation_change,
            inputs=hyphenate_before_scaling,
            outputs=[hyphen_penalty, hyphenation_min_word_length],
            queue=False,
        )

        # Cleaning-only, Upscaling-only, and Test mode mutual exclusivity handlers
        cleaning_only_toggle.change(
            fn=callbacks.handle_cleaning_only_change,
            inputs=cleaning_only_toggle,
            outputs=[upscaling_only_toggle, test_mode_toggle],
            queue=False,
        )

        # Upscaling-only toggle change handler
        upscaling_only_toggle.change(
            fn=callbacks.handle_upscaling_only_change,
            inputs=upscaling_only_toggle,
            outputs=[cleaning_only_toggle, test_mode_toggle],
            queue=False,
        )

        # Test mode toggle change handler
        test_mode_toggle.change(
            fn=callbacks.handle_test_mode_change,
            inputs=test_mode_toggle,
            outputs=[cleaning_only_toggle, upscaling_only_toggle],
            queue=False,
        )

        # OSB enable/disable handler
        outside_text_enabled.change(
            fn=lambda x: gr.update(visible=x),
            inputs=outside_text_enabled,
            outputs=outside_text_settings_wrapper,
            queue=False,
        )

        # Page-number filtering toggle -> enable/disable related sliders
        outside_text_enable_page_number_filtering.change(
            fn=lambda enabled: (
                gr.update(interactive=enabled),
                gr.update(interactive=enabled),
            ),
            inputs=outside_text_enable_page_number_filtering,
            outputs=[
                outside_text_page_filter_margin_threshold,
                outside_text_page_filter_min_area_ratio,
            ],
            queue=False,
        )

        # Inpainting method change -> enable/disable controls and adjust steps range
        def _update_inpainting_controls(
            method: str,
            current_backend: str,
            current_steps: int,
            upscale_small_crops: bool,
        ):
            """Update controls based on inpainting method selection."""
            is_opencv = method == "opencv"
            is_none = method == "none"
            is_no_flux = is_opencv or is_none
            is_kontext = method == "flux_kontext"
            is_klein = method in ("flux_klein_9b", "flux_klein_4b")

            if is_kontext:
                max_steps = 30
                default_steps = 8
            else:
                max_steps = 12
                default_steps = 4

            show_low_vram = is_klein or is_kontext
            residual_interactive = is_kontext and current_backend == "nunchaku"
            luminance_interactive = is_klein and bool(upscale_small_crops)
            luminance_update = gr.update(visible=False, interactive=False)
            if is_klein:
                luminance_update = gr.update(
                    visible=True,
                    interactive=luminance_interactive,
                )

            return (
                gr.update(visible=is_kontext),
                gr.update(visible=show_low_vram),
                gr.update(visible=show_low_vram),
                gr.update(
                    interactive=(not is_no_flux),
                    maximum=max_steps,
                    value=default_steps,
                ),
                luminance_update,
                gr.update(visible=(not is_no_flux), interactive=is_klein),
                gr.update(visible=(not is_no_flux), interactive=(not is_no_flux)),
                gr.update(visible=is_kontext, interactive=residual_interactive),
                gr.update(visible=(not is_no_flux), interactive=(not is_no_flux)),
                gr.update(interactive=(not is_no_flux)),
            )

        outside_text_inpainting_method.change(
            fn=_update_inpainting_controls,
            inputs=[
                outside_text_inpainting_method,
                outside_text_kontext_backend,
                outside_text_flux_num_inference_steps,
                outside_text_flux_upscale_small_crops,
            ],
            outputs=[
                outside_text_kontext_backend,
                outside_text_flux_offload_mode,
                outside_text_flux_memory_help,
                outside_text_flux_num_inference_steps,
                outside_text_flux_luminance_correction,
                outside_text_flux_upscale_small_crops,
                outside_text_flux_group_regions,
                outside_text_flux_residual_diff_threshold,
                outside_text_osb_residual_render_guard_action,
                outside_text_seed,
            ],
            queue=False,
        )

        def _update_colored_bubble_lama_controls(enabled: bool):
            update = gr.update(interactive=bool(enabled))
            return update, update, update, update

        inpaint_colored_bubbles.change(
            fn=_update_colored_bubble_lama_controls,
            inputs=[inpaint_colored_bubbles],
            outputs=[
                colored_bubble_inpainting_size,
                colored_bubble_inpainting_precision,
                colored_bubble_mask_kernel_size,
                colored_bubble_mask_dilation_offset,
            ],
            queue=False,
        )

        def _update_luminance_interactivity(
            upscale_small_crops: bool, group_regions: bool, method: str
        ):
            """Luminance correction is available for Klein crops scaled to ~1MP."""
            is_klein = method in ("flux_klein_9b", "flux_klein_4b")
            if not is_klein:
                return gr.update(interactive=False)

            luminance_interactive = bool(upscale_small_crops)
            return gr.update(interactive=luminance_interactive)

        for control in (
            outside_text_flux_upscale_small_crops,
            outside_text_flux_group_regions,
        ):
            control.change(
                fn=_update_luminance_interactivity,
                inputs=[
                    outside_text_flux_upscale_small_crops,
                    outside_text_flux_group_regions,
                    outside_text_inpainting_method,
                ],
                outputs=outside_text_flux_luminance_correction,
                queue=False,
            )

        # Kontext backend change -> update Low VRAM and Residual diff visibility
        def _update_kontext_backend_controls(backend: str):
            """Update controls when Kontext backend changes."""
            is_sdnq = backend == "sdnq"
            return (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(interactive=(not is_sdnq)),
            )

        outside_text_kontext_backend.change(
            fn=_update_kontext_backend_controls,
            inputs=outside_text_kontext_backend,
            outputs=[
                outside_text_flux_offload_mode,
                outside_text_flux_memory_help,
                outside_text_flux_residual_diff_threshold,
            ],
            queue=False,
        )

        outside_text_flux_luminance_correction.change(
            fn=callbacks.handle_luminance_correction_change,
            inputs=outside_text_flux_luminance_correction,
            outputs=None,
            queue=False,
        )

        # Conjoined detection change handler - clears SAM cache
        conjoined_detection_checkbox.change(
            fn=callbacks.handle_conjoined_detection_change,
            inputs=conjoined_detection_checkbox,
            outputs=conjoined_confidence,
            queue=False,
        )

        # Panel sorting change handler
        use_panel_sorting_checkbox.change(
            fn=lambda enabled: gr.update(interactive=enabled),
            inputs=use_panel_sorting_checkbox,
            outputs=panel_confidence,
            queue=False,
        )

        # Confidence threshold change handlers - clear YOLO cache
        confidence.change(
            fn=callbacks.handle_confidence_threshold_change,
            inputs=confidence,
            outputs=None,
            queue=False,
        )

        conjoined_confidence.change(
            fn=callbacks.handle_confidence_threshold_change,
            inputs=conjoined_confidence,
            outputs=None,
            queue=False,
        )

        panel_confidence.change(
            fn=callbacks.handle_confidence_threshold_change,
            inputs=panel_confidence,
            outputs=None,
            queue=False,
        )

        outside_text_osb_confidence.change(
            fn=callbacks.handle_confidence_threshold_change,
            inputs=outside_text_osb_confidence,
            outputs=None,
            queue=False,
        )

        # Translation mode change handler - disable OCR selection when one-step
        config_translation_mode.change(
            fn=callbacks.handle_translation_mode_change,
            inputs=[config_translation_mode, ocr_method_radio],
            outputs=ocr_method_radio,
            queue=False,
        )

        # OCR method change handler
        ocr_method_radio.change(
            fn=callbacks.handle_ocr_method_change,
            inputs=[
                ocr_method_radio,
                input_language,
                original_language_state,
                batch_input_language,
                batch_original_language_state,
                provider_state,
                config_model_name,
                openai_compatible_url_input,
                openai_compatible_api_key_input,
            ],
            outputs=[
                provider_selector,
                input_language,
                original_language_state,
                batch_input_language,
                batch_original_language_state,
                send_full_page_context,
                batch_previous_context_image_count,
                parallel_inpaint_translation,
                whiteout_conjoined_bubbles,
                enable_code_execution_checkbox,
                image_detail_dropdown,
                media_resolution_dropdown,
                media_resolution_bubbles_dropdown,
                media_resolution_context_dropdown,
                config_model_name,
                provider_state,
            ],
            queue=False,
        )

        send_full_page_context.change(
            fn=lambda enabled: (
                gr.update(interactive=True)
                if enabled
                else gr.update(value=0, interactive=False)
            ),
            inputs=send_full_page_context,
            outputs=batch_previous_context_image_count,
            queue=False,
        )

        # Upscale method change handler
        upscale_method.change(
            fn=lambda x: [
                gr.update(interactive=x != "none"),
                gr.update(interactive=x != "none"),
                gr.update(interactive=x != "none"),
            ],
            inputs=upscale_method,
            outputs=[
                bubble_min_side_pixels,
                context_image_max_side_pixels,
                osb_min_side_pixels,
            ],
            queue=False,
        )

        tag_style_font_apply.click(
            fn=callbacks.handle_apply_tag_style_font,
            inputs=[tag_style_table, tag_style_font_target, tag_style_font_pack],
            outputs=[tag_style_table],
            queue=False,
        )

        # Config Save/Reset Buttons
        save_config_btn.click(
            fn=callbacks.handle_save_config_click,
            inputs=save_config_inputs,
            outputs=[config_status],
            queue=False,
        ).then(fn=None, inputs=None, outputs=None, js=js_status_fade, queue=False)

        reset_defaults_btn.click(
            fn=functools.partial(
                callbacks.handle_reset_defaults_click,
                fonts_base_dir=fonts_base_dir,
            ),
            inputs=[],
            outputs=reset_outputs,
            queue=False,
        ).then(fn=None, inputs=None, outputs=None, js=js_status_fade, queue=False)

        # Refresh Button
        refresh_outputs = [
            font_dropdown,
            batch_font_dropdown,
            emphasis_font_pack,
            outside_text_osb_font_pack,
            outside_text_osb_emphasis_font_pack,
        ]
        refresh_resources_button.click(
            fn=functools.partial(
                callbacks.handle_refresh_resources_click,
                fonts_base_dir=fonts_base_dir,
            ),
            inputs=[],
            outputs=refresh_outputs,
            js=js_refresh_button_processing,
        ).then(fn=None, inputs=None, outputs=None, js=js_refresh_button_reset)

        # Unload Models Button
        unload_models_button.click(
            fn=callbacks.handle_unload_models_click,
            inputs=[],
            outputs=[],
        )

        # Translator Tab Button
        clear_button.click(
            fn=lambda: (
                None,
                None,
                gr.update(value="", lines=1),
                gr.update(value=None, visible=True),
                gr.update(value=None, visible=True),
                "",
                gr.update(visible=False),
            ),
            outputs=[
                input_image,
                output_image,
                status_message,
                translator_saved_file,
                translator_plain_file,
                editor_session_state,
                open_advanced_editor_button,
            ],
            queue=False,
        ).then(fn=None, js=js_reset_status_height, queue=False)

        open_advanced_editor_button.click(
            fn=callbacks.handle_launch_advanced_editor,
            inputs=[editor_session_state],
            outputs=[status_message],
            queue=False,
        )

        existing_editor_refresh_button.click(
            fn=callbacks.handle_refresh_existing_editor_sessions,
            inputs=[],
            outputs=[
                existing_editor_session_select,
                existing_editor_open_button,
                image_editor_status,
            ],
            queue=False,
            show_progress="hidden",
        )
        existing_editor_session_select.change(
            fn=lambda session_path: gr.update(
                visible=bool(session_path), interactive=bool(session_path)
            ),
            inputs=[existing_editor_session_select],
            outputs=[existing_editor_open_button],
            queue=False,
        )
        existing_editor_open_button.click(
            fn=callbacks.handle_launch_advanced_editor,
            inputs=[existing_editor_session_select],
            outputs=[image_editor_status],
            queue=False,
        )

        image_editor_create_button.click(
            fn=functools.partial(
                callbacks.handle_create_image_editor_sessions,
                fonts_base_dir=fonts_base_dir,
            ),
            inputs=[
                image_editor_folder_path,
                image_editor_font_dropdown,
            ],
            outputs=[
                image_editor_session_select,
                image_editor_open_button,
                image_editor_status,
            ],
            queue=False,
        )
        image_editor_folder_button.click(
            fn=callbacks.handle_browse_image_editor_folder,
            inputs=[image_editor_folder_path],
            outputs=[image_editor_folder_path],
            queue=False,
            show_progress="hidden",
        )
        image_editor_session_select.change(
            fn=lambda session_path: gr.update(
                visible=bool(session_path), interactive=bool(session_path)
            ),
            inputs=[image_editor_session_select],
            outputs=[image_editor_open_button],
            queue=False,
        )
        image_editor_open_button.click(
            fn=callbacks.handle_launch_advanced_editor,
            inputs=[image_editor_session_select],
            outputs=[image_editor_status],
            queue=False,
        )
        image_editor_clear_button.click(
            fn=lambda: (
                "",
                gr.update(choices=[], value=None, visible=False),
                gr.update(visible=False, interactive=False),
                gr.update(value="", lines=1),
            ),
            outputs=[
                image_editor_folder_path,
                image_editor_session_select,
                image_editor_open_button,
                image_editor_status,
            ],
            queue=False,
        )
        batch_clear_button.click(
            fn=lambda: (
                None,
                None,
                None,
                gr.update(value="", lines=1),
                gr.update(value=None, visible=True),
                gr.update(value=None, visible=True),
                gr.update(choices=[], value=None, visible=False),
                gr.update(visible=False),
            ),
            outputs=[
                input_files,
                input_zip,
                batch_output_gallery,
                batch_status_message,
                batch_success_zip_file,
                batch_failed_zip_file,
                batch_editor_session_select,
                batch_advanced_editor_button,
            ],
            queue=False,
        ).then(fn=None, js=js_reset_status_height, queue=False)
        translate_event = translate_button.click(
            fn=functools.partial(
                callbacks.update_process_buttons,
                processing=True,
                button_text_processing="Translating...",
                button_text_idle="Translate",
            ),
            outputs=[
                translate_button,
                clear_button,
                cancel_button,
                batch_process_button,
                batch_clear_button,
                batch_cancel_button,
            ],
            queue=False,
        ).then(
            fn=functools.partial(
                callbacks.handle_translate_click,
                models_dir=models_dir,
                fonts_base_dir=fonts_base_dir,
                target_device=target_device,
            ),
            inputs=translate_inputs,
            outputs=[
                output_image,
                status_message,
                translator_saved_file,
                translator_plain_file,
                editor_session_state,
                open_advanced_editor_button,
            ],
        )
        translate_event.then(
            fn=functools.partial(
                callbacks.update_process_buttons,
                processing=False,
                button_text_processing="Translating...",
                button_text_idle="Translate",
            ),
            outputs=[
                translate_button,
                clear_button,
                cancel_button,
                batch_process_button,
                batch_clear_button,
                batch_cancel_button,
            ],
            queue=False,
        ).then(fn=None, js=js_reset_status_height, queue=False)

        cancel_button.click(
            fn=callbacks.cancel_process,
            cancels=translate_event,
            queue=False,
        )

        # Batch Tab Button
        batch_event = batch_process_button.click(
            fn=functools.partial(
                callbacks.update_process_buttons,
                processing=True,
                button_text_processing="Processing...",
                button_text_idle="Start Batch Translating",
            ),
            outputs=[
                batch_process_button,
                batch_clear_button,
                batch_cancel_button,
                translate_button,
                clear_button,
                cancel_button,
            ],
            queue=False,
        ).then(
            fn=functools.partial(
                callbacks.handle_batch_click,
                models_dir=models_dir,
                fonts_base_dir=fonts_base_dir,
                target_device=target_device,
            ),
            inputs=batch_inputs,
            outputs=[
                batch_output_gallery,
                batch_status_message,
                batch_success_zip_file,
                batch_failed_zip_file,
                batch_editor_session_select,
                batch_advanced_editor_button,
            ],
        )
        batch_editor_session_select.change(
            fn=lambda session_path: gr.update(
                visible=bool(session_path), interactive=bool(session_path)
            ),
            inputs=[batch_editor_session_select],
            outputs=[batch_advanced_editor_button],
            queue=False,
        )
        batch_advanced_editor_button.click(
            fn=callbacks.handle_launch_advanced_editor,
            inputs=[batch_editor_session_select],
            outputs=[batch_status_message],
            queue=False,
        )
        batch_event.then(
            fn=functools.partial(
                callbacks.update_process_buttons,
                processing=False,
                button_text_processing="Processing...",
                button_text_idle="Start Batch Translating",
            ),
            outputs=[
                batch_process_button,
                batch_clear_button,
                batch_cancel_button,
                translate_button,
                clear_button,
                cancel_button,
            ],
            queue=False,
        ).then(fn=None, js=js_reset_status_height, queue=False)

        batch_cancel_button.click(
            fn=callbacks.cancel_process,
            queue=False,
        )

        app.load(
            fn=callbacks.handle_app_load,
            inputs=[
                provider_selector,
                openai_compatible_url_input,
                openai_compatible_api_key_input,
            ],
            outputs=[config_model_name],
            queue=False,
        )
        app.load(
            fn=callbacks.handle_refresh_existing_editor_sessions,
            inputs=[],
            outputs=[
                existing_editor_session_select,
                existing_editor_open_button,
                image_editor_status,
            ],
            queue=False,
        )
        app.load(fn=None, inputs=None, outputs=None, js=js_folder_upload, queue=False)

    return app
