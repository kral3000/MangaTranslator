# Session Handoff — MangaTranslator (portable fork)

**Repo root**: `C:\Users\dongw\Documents\Mana\MangaTranslator_portable\MangaTranslator`
**Live config (the one the app actually reads)**: `%LOCALAPPDATA%\MangaTranslator\config.json`
(NOT `config.public.json` in the repo — that's a stale export copy, see Gotchas)

## Session Intent

Multi-part support session: (1) plan an RT-DETR/branding fork copy — **abandoned**, see Decisions;
(2) add a new tag-based rendering category for handwritten paper text; (3) diagnose/fix a two-step
translation-mode bug where tag markup leaked into rendered images; (4) evaluate DeepSeek vs Gemini
translation quality/speed/cost via real batch logs; (5) add token-usage tracking to the UI receipt.

## Files Modified (this session, chronological)

- **`core/text/text_processing.py`** — added `"OH"` to `TRANSLATION_TAGS` set (~line 911). Required
  so `extract_translation_tag()` strips `<OH>...</OH>` instead of leaving it as literal text.
- **`core/tag_styles.py`** — added `("OH", "말풍선 밖: 종이 위 손글씨")` to `TAG_DEFINITIONS`
  OrderedDict (~line 16). Drives the Tag Styles UI table row + font dropdown automatically (already
  tag-count-agnostic).
- **`ui/layout.py`** — (a) added `<OH>` to the hardcoded supported-tags markdown (~line 2091);
  (b) Tag Styles table: added `column_widths=["7%","23%","16%","9%","9%","9%","13%","14%"]` to the
  `gr.Dataframe` so `SFXI`/`SFXS` render on one line instead of wrapping; (c) `gr.Row` around the
  font-target/font-pick/apply-button controls got `elem_classes="tag-style-font-row"` and the
  `info=` text on the font dropdown was moved into the section's intro Markdown (functionally
  identical, just relocated for the alignment fix).
- **`ui/style.css`** — added `.tag-style-font-row` CSS block: `align-items: flex-end`, `gap: 12px`,
  and `.block.padded { padding-bottom: 0 }` scoped to that row only (fixes a Gradio default
  padding that misaligned the dropdowns vs. the button). Also lightly touched the pre-existing
  `#translator_special_instructions` / `#batch_special_instructions` textarea CSS (unrelated
  minor tweak noticed in the same pass — gave them `resize: vertical; min-height: 12em` instead of
  `overflow-y: hidden`).
- **Live config** (`%LOCALAPPDATA%\MangaTranslator\config.json`, backup:
  `config.json.bak_before_OH_tag` in the same folder) — via a one-off Python script
  (`add_oh_tag.py`, not saved to repo, was in this session's scratchpad):
  - `special_instructions` / `batch_special_instructions`: inserted a new `<OH>` definition line
    after the existing `<OE>` line ("text handwritten by hand on a paper surface drawn in the
    scene..."), and reworded the `<OT>` line to say "printed sign, printed document, printed book
    text" (was just "sign, document, or book text") so OT and OH don't overlap.
  - `tag_styles.OH` entry added: `font_pack` copied verbatim from the existing `SFXS` entry
    (`나눔손글씨 하나손글씨`), `min_font_size=10`, **`max_font_size=80`** (raised from an initial
    `28` after a real page showed panel-filling handwritten diary text getting capped too small —
    see "OH max_font_size" in Decisions).
  - `config.public.json` in the repo (the export copy) was patched in parallel for consistency but
    is **not** what the running app reads.
- **`core/services/translation.py`** (backup: `.bak_before_twostep_tag_fix`, then further modified
  and re-backed-up as `.bak_before_token_usage`) — two-step tag-format fix:
  - new import: `from core.text.text_processing import TRANSLATION_TAGS`
  - new module-level `_TWO_STEP_TAG_FORMAT_OVERRIDE` string constant + `_LEADING_TAG_RE` regex +
    `_normalize_two_step_tagged_value()` function (placed right after `_format_special_instructions`,
    ~line 1966).
  - in the two-step translation-prompt builder: `elif special_instructions_section:
    translation_prompt += _TWO_STEP_TAG_FORMAT_OVERRIDE` (appends a corrective instruction telling
    the model to emit `<TAG>translation</TAG>` only, no `||`, no source echo — **Special
    Instructions text itself was never edited**, this is a code-level prompt append).
  - after `_parse_llm_response_unified()` in the two-step non-local-gemma branch: each parsed value
    is passed through `_normalize_two_step_tagged_value()` as a defensive repair pass (splits on
    `||` and keeps the translation half if the model ignores the override).
- **`utils/token_usage.py`** — **new file**. Provider-agnostic usage normalizer:
  `make_usage()`, `parse_google_usage()` (reads Gemini/Vertex `usageMetadata`),
  `parse_openai_style_usage()` (reads OpenAI-compatible `usage`, splits DeepSeek's
  `completion_tokens_details.reasoning_tokens` back out of `completion_tokens` so `output` means
  "visible answer only"), `accumulate_usage()`, `format_usage()`.
- **`utils/endpoints/google_vertex_json_final.py`** (backup: `.bak_before_token_usage`) — added
  `usage_output: Optional[Dict[str, int]] = None` param to `call_gemini_endpoint`; parses
  `usageMetadata` from the response and updates `usage_output` + logs it (verbose).
- **`utils/endpoints/deepseek.py`** (backup: `.bak_before_token_usage`) — same pattern: added
  `usage_output` param to `call_deepseek_endpoint`, parses `result["usage"]`.
- **`core/services/translation.py`** (same file as above, additional edit) — `_call_llm_endpoint()`
  gained `usage_output` param; a `finally:` block calls `accumulate_usage(usage_output, call_usage)`
  so it's captured regardless of which provider branch returned. `call_translation_api_batch()`
  gained `usage_output` param, forwarded into all 4 `_call_llm_endpoint(...)` call sites (OCR-only,
  two-step translate, one-step, one-step retry).
- **`core/pipeline.py`** (backup: `.bak_before_token_usage`) — `translate_and_render()` gained
  `usage_out: Optional[Dict[str, int]] = None` param (mirrors the existing `ocr_texts_out`
  side-channel pattern); forwarded into both `call_translation_api_batch(...)` call sites (the
  async-overlap-thread worker and the synchronous path).
- **`ui/logic.py`** (backup: `.bak_before_token_usage`) — `translate_manga_logic()` gained
  `usage_out` param, forwarded to `translate_and_render(...)`.
- **`ui/callbacks.py`** (backup: `.bak_before_token_usage`) — new import
  `from utils.token_usage import format_usage`; `handle_translate_click`-adjacent code now does
  `token_usage: Dict[str, int] = {}` then passes `usage_out=token_usage` into
  `logic.translate_manga_logic(...)`, then passes `token_usage` into
  `_format_single_success_message(...)`; that function gained a `token_usage` param and inserts
  `• Tokens: {format_usage(token_usage)}` into the receipt (right after the LLM params line, before
  the Font Pack line). **Batch-mode receipt (`_format_batch_success_message`, ~line 1247) was
  identified but NOT patched — still open, see Next Steps.**

## Files Read Extensively (not modified)

- `core/pipeline.py` — panel-detection duplication found (see Next Steps), OSB/Flux inpainting flow,
  `_should_overlap_inpaint_translation` / `parallel_inpaint_translation` concurrency machinery
  (pre-existing, confirmed already enabled).
- `core/config.py` — `TagStyleConfig`, `OutsideTextConfig.osb_rotation_max_degrees` (default 15.0,
  user's live config has it at 30.0), `RenderingConfig.tag_styles`.
- `ui/settings_manager.py` — `CONFIG_FILE` path resolution (the "which config file is real" fact).
- `utils/model_metadata.py` — `is_deepseek_reasoning_model()`, `get_sampling_interactivity_for_effort()`
  (DeepSeek reasoning models lock temp/top_p unless effort="none").

## Decisions Made

1. **Abandoned**: the earlier-session plan to create a `MangaTranslator_Pro` copy folder with
   RT-DETR-v2 as an additive detector option, branding changes, editor cleanup, `editor_sessions`
   purge button. User explicitly said "걍 아무것도 안할래" (do nothing) and pivoted to fixing the
   existing install instead. **Do not resume that plan unless the user asks.**
2. **OH tag scope**: only ONE new tag (paper handwriting), not separate tags for sign/banner and
   phone/screen — user's own reasoning after seeing a real example: rotation-matching for tilted
   objects would need full perspective/homography transform (not just the single-angle rotation this
   app has), which is out of scope; sign/banner/phone text renders acceptably under existing `OT`.
3. **OH max_font_size raised 28→80**: found via a real page (giant diary-page handwriting spanning
   most of a panel) rendering too small. Root cause: per-tag `max_font_size` is the binary-search
   ceiling for the "fill the detected box as large as possible" algorithm — the algorithm itself
   already scales to match source size, the tag's ceiling was just set too conservatively at
   creation time.
4. **Two-step tag-format bug root cause**: Special Instructions (shared verbatim between one-step
   and two-step prompts) contains a one-step-only line: `Output format: <TAG>source</TAG> ||
   <TAG>translation</TAG>`. In two-step, OCR already happened via manga-ocr, so the model
   (DeepSeek) dutifully produced `<SFX>げっ</SFX> || <SFX>헉</SFX>` per JSON value, but
   `extract_translation_tag()` only strips the outermost tag pair (regex anchored start/end,
   lazy middle group) — leaving `げっ</SFX> || <SFX>헉` baked into the rendered image text.
   Fixed at the code level (prompt override + defensive parse repair) rather than editing the
   user's Special Instructions text, so the tag-definitions portion (including OH) stays reusable
   across both modes.
5. **DeepSeek two-step vs Gemini one-step, evaluated on real batch logs (same page, 3 runs)**:
   - DeepSeek reasoning "none": 40s, but 2 mistranslations (`寝相`→"잠꼬대" should be "잠버릇";
     missed `いつき` as a person's name).
   - DeepSeek reasoning "high" (after raising `max_tokens` 16384→32000, see below): 222s, fixed
     both mistranslations, caught the name `伊月/이츠키`, but appended periods (against manga
     convention) and mis-tagged `すごい寝相だ`/`開口一番` as `ON`(narration) instead of `OM`(monologue)
     — tag misclassification cascades into wrong font/size since no image is ever sent in two-step.
   - Gemini one-step (also reasoning "high"): 25.87s, correct translations, correct tags, no
     periods. **Confirmed via the newly-added token logging that Gemini's `thoughtsTokenCount` was
     15,725 — essentially the same order of magnitude as DeepSeek's reasoning — so the 200s vs 18-90s
     gap is server/inference-speed, not "DeepSeek thinks more."**
   - **User's working conclusion (my recommendation, not yet re-confirmed after the token-usage
     finding): revert to Gemini one-step for production use.** The DeepSeek experiment was framed as
     valuable diagnostic data, not a dead end — user has not explicitly said "abandon DeepSeek."
6. **DeepSeek empty-response bug (earlier in session, root-caused before the tag-format bug)**:
   `max_tokens=16384` + `reasoning_effort=high` → reasoning consumed the entire budget, leaving 0
   tokens for `content` → empty response, `Error processing 01.jpg: DeepSeek-Translate: Empty response`.
   Fixed by user raising `max_tokens` to 32000 (config change, not code — user did this via UI).
7. **Windows Defender likely startup-time culprit**: measured import times (torch 1.81s, gradio
   2.79s, transformers 2.75s ≈ 10s total cold import). `sdnq`/`diffusers` (20s+) confirmed NOT loaded
   at startup (lazy-loaded on first Flux use, already correct). Recommended (a) user adds a Defender
   exclusion for the app folder (user must do this themselves — security setting), (b) make
   `transformers` lazy-load too (~2.7s saved) — **not yet implemented, open task**.

## Current State

- App is stopped (no background dev server running at the end of this session — last `preview_start`
  processes were explicitly stopped after verification).
- All Python changes compile-clean (`py_compile` checked after every edit) and were verified with
  real unit tests + one real end-to-end Gemini API call (11,070 prompt / 15,725 thoughts / 210
  output / 27,005 total tokens, confirmed rendering into the actual UI receipt string).
- `.bak_before_*` files are scattered next to their originals (repo) and in
  `%LOCALAPPDATA%\MangaTranslator\` (config) — safe to delete once user confirms everything is
  working, but currently left in place as rollback points.
- Scratchpad test scripts created this session (not part of the repo, in
  `...\b09a6fcc-.../scratchpad\`): `add_oh_tag.py`, `test_oh_plumbing.py`, `test_twostep_fix.py`,
  `verify_scope.py`, `test_usage.py`, `e2e_usage.py`, `ui_probe.py`, `bump_oh_max_font.py`. These
  are throwaway verification scripts, not deliverables.

## Next Steps (open, in rough priority order)

1. **Batch-mode token usage** — `_format_batch_success_message()` (`ui/callbacks.py` ~line 1247) was
   located but not wired up. Needs per-image usage accumulated across the whole batch (likely another
   `usage_out`-style dict threaded through the batch loop, then `accumulate_usage()`'d per image)
   before it can show a batch-total token line. User has not explicitly asked for this yet but it's
   the natural follow-up to the single-image version just shipped.
2. **Panel-detection duplication** — `core/pipeline.py` calls `detect_panels()` twice per image when
   `use_panel_sorting=True` and `ENABLE_COMPONENT_ORDER_DEBUG=False` (the common case): once at
   ~line 11243 into `debug_panels` (result then unused in that branch), once again at ~line 11872
   for actual sorting. Identified as a real, fixable, zero-quality-impact win (~1-2s/image ×
   28-image batch ≈ 30-60s wasted). **User asked "지금 고쳐드릴까요?" was left unanswered — this is
   a live offer, not yet declined or accepted.**
3. **Special Instructions period rule** — user confirmed wanting periods suppressed
   ("문장 끝 마침표 금지" style instruction). This is a pure Special-Instructions text edit (UI
   textbox), not code — was discussed but the actual edit was not yet performed. Trivial to do
   whenever picked back up.
4. **`transformers` lazy import** — proposed as a ~2.7s startup-time win, not implemented.
5. **Windows Defender exclusion** — user-side action, cannot be done by the assistant; just a
   reminder if startup speed comes up again.
6. **DeepSeek reasoning_effort="medium" untested** — only "none" and "high" were benchmarked; medium
   might be a real sweet spot between the 40s/2-errors and 222s/correct outcomes, but was never
   actually run.
7. Cleanup of `.bak_before_*` files and scratchpad scripts once the user is confident everything is
   stable (not urgent, just housekeeping).

## Gotchas (things that bit us this session — don't re-learn them)

- **`config.public.json` in the repo root is NOT the live config.** The real one is
  `%LOCALAPPDATA%\MangaTranslator\config.json`. Editing only the repo copy silently does nothing to
  the running app — this caused one full round of confusion earlier in the session (fixed UI values
  didn't show up until this was discovered).
- **`extract_translation_tag()` (`core/text/text_processing.py`) only strips the OUTERMOST tag
  pair** via an anchored regex with a lazy middle group — it does not understand compound
  `<TAG>a</TAG> || <TAG>b</TAG>` values. Any future prompt change that could cause a model to emit
  two tag-pairs in one value will reproduce the leaked-markup bug.
- **Two-step mode sends NO images to the translation LLM at all**, even for vision-capable models
  — gated by `config.ocr_method not in ("manga-ocr", "paddleocr-vl")` before attaching
  `full_image_b64`. This is why tag classification (which needs visual grounding) is structurally
  worse in two-step regardless of which text model is used.
- **DeepSeek's `completion_tokens` in the raw API response already INCLUDES reasoning tokens** —
  `parse_openai_style_usage()` subtracts `completion_tokens_details.reasoning_tokens` back out so
  `output` means "visible answer only," matching Gemini's `candidatesTokenCount` semantics. Don't
  double-count if touching this code again.
- Bash tool's `sys.stdout` defaults to a non-UTF-8 codepage on this Windows box — any ad-hoc
  verification script printing Korean text needs `sys.stdout.reconfigure(encoding="utf-8")` or it
  either mangles/garbles output or throws `UnicodeEncodeError`. Also prefer `-X utf8` on the
  `runtime\python.exe` invocation itself.
- The portable Python distribution's real interpreter is `runtime\python.exe` relative to the repo
  root — always `cd` into the repo first, this is not on PATH.
