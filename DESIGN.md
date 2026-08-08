---
name: MangaTranslatorKo Advanced Editor
description: Native PyQt6 manga editing workspace with an ink rail, light inspector, and canvas-first feedback.
colors:
  app-chrome: "#0f172a"
  active-blue: "#2563eb"
  active-blue-deep: "#1d4ed8"
  active-blue-border: "#60a5fa"
  active-blue-focus: "#93c5fd"
  active-blue-soft: "#bfdbfe"
  ink-rail: "#111827"
  ink-hover: "#1f2937"
  chrome-border: "#263244"
  chrome-hover: "#273449"
  chrome-hover-border: "#3b4a60"
  workspace-paper: "#edf2f7"
  inspector-paper: "#f4f7fb"
  session-paper: "#f8fafc"
  field-paper: "#ffffff"
  canvas-charcoal: "#1c222b"
  canvas-bar: "#252d3a"
  canvas-control: "#364152"
  canvas-divider: "#3a4657"
  canvas-control-border: "#4b5a6f"
  body-ink: "#172033"
  label-slate: "#334155"
  quiet-slate: "#475569"
  muted-slate: "#64748b"
  disabled-slate: "#94a3b8"
  scrollbar-slate: "#a8b4c4"
  control-border: "#c4cfdd"
  panel-divider: "#d5dde8"
  control-fill: "#edf1f6"
  control-hover: "#eff6ff"
  control-pressed: "#dbeafe"
  toolbar-text: "#e5e7eb"
  status-paper: "#eef6ff"
  status-ink: "#1e3a8a"
  pure-black: "#000000"
  canvas-selection: "#2196f3"
  canvas-region: "#ffc107"
  canvas-disabled: "#969696"
typography:
  supporting:
    fontFamily: "Segoe UI, Malgun Gothic"
    fontSize: "11px"
    fontWeight: 400
  body:
    fontFamily: "Segoe UI, Malgun Gothic"
    fontSize: "12px"
    fontWeight: 400
  label:
    fontFamily: "Segoe UI, Malgun Gothic"
    fontSize: "12px"
    fontWeight: 500
  action:
    fontFamily: "Segoe UI, Malgun Gothic"
    fontSize: "12px"
    fontWeight: 600
  section:
    fontFamily: "Segoe UI, Malgun Gothic"
    fontSize: "13px"
    fontWeight: 700
  canvas-index:
    fontFamily: "Arial"
    fontSize: "10px"
    fontWeight: 700
rounded:
  micro: "2px"
  compact: "3px"
  small-control: "4px"
  toolbar-control: "5px"
  control: "6px"
  panel: "7px"
  circular-handle: "8px"
spacing:
  tight: "5px"
  compact: "6px"
  control: "8px"
  standard: "10px"
  panel: "14px"
  inspector-inset: "16px"
components:
  topbar-action:
    backgroundColor: "transparent"
    textColor: "{colors.toolbar-text}"
    typography: "{typography.body}"
    rounded: "{rounded.control}"
    padding: "7px 10px"
  toolrail-action:
    backgroundColor: "transparent"
    textColor: "{colors.control-border}"
    typography: "{typography.body}"
    rounded: "{rounded.control}"
    padding: "8px 10px"
    width: "142px"
  toolrail-action-selected:
    backgroundColor: "{colors.active-blue}"
    textColor: "{colors.field-paper}"
    typography: "{typography.body}"
    rounded: "{rounded.control}"
    padding: "8px 10px"
    width: "142px"
  inspector-button:
    backgroundColor: "{colors.control-fill}"
    textColor: "{colors.ink-rail}"
    typography: "{typography.action}"
    rounded: "{rounded.control}"
    padding: "8px 10px"
  inspector-field:
    backgroundColor: "{colors.field-paper}"
    textColor: "{colors.ink-rail}"
    typography: "{typography.body}"
    rounded: "{rounded.control}"
    padding: "6px"
  zoom-button:
    backgroundColor: "{colors.canvas-control}"
    textColor: "{colors.field-paper}"
    typography: "{typography.body}"
    rounded: "{rounded.small-control}"
    padding: "4px 10px"
  status-callout:
    backgroundColor: "{colors.status-paper}"
    textColor: "{colors.status-ink}"
    typography: "{typography.body}"
    rounded: "{rounded.control}"
    padding: "8px"
---

# Design System: MangaTranslatorKo Advanced Editor

## Overview

**Creative North Star: "Ink Rail Workbench"**

The advanced editor is a native Windows Operate surface: a focused workbench where the manga page remains visually dominant and every editing decision stays close to its consequence. Its Tubik/Ribbet pattern is structural rather than decorative—an ink-colored tool rail and minimal top action bar frame a light inspector, while the central canvas holds the user's attention.

The world is practical, dense, and immediate. Dark chrome separates global tools from editable content; pale controls support long inspection sessions; blue appears only for active selection, focus, and direct manipulation. Standard Qt behavior, keyboard shortcuts, tooltips, dialogs, scrollbars, and established editing behavior remain intact.

**Key Characteristics:**

- Canvas-first native desktop workspace
- One-direction tool discovery from the left rail into the adjacent inspector
- Dark ink chrome against a light, compact inspector
- Blue state feedback paired with checked, focused, selected, or textual cues
- Restrained six-pixel controls and standard Qt affordances

## Colors

The palette uses dark ink for application chrome, cool paper neutrals for editable controls, and a single operational blue for active state.

### Primary

- **Operational Blue:** The only primary action-state color; use it for checked tools, focused fields, slider handles, hovered canvas scroll handles, and selected tab text.
- **Operational Blue Border:** The supporting edge color for active and hovered controls.
- **Selection Wash:** The soft selection and pressed-state fill used inside light controls.

### Tertiary

- **Canvas Selection:** The selected region rectangle and resize handles on the canvas.
- **Canvas Region:** The unselected enabled region outline.
- **Canvas Disabled:** The disabled region outline.

### Neutral

- **Ink Rail:** The top bar and left tool rail, establishing the editor's fixed application frame.
- **Workspace Paper:** The base surface behind the inspector and native widgets.
- **Field Paper:** Editable fields and selected tab surfaces.
- **Canvas Charcoal:** The image workspace outside the manga page.
- **Canvas Bar:** The zoom strip immediately above the canvas.
- **Body Ink / Label Slate / Quiet Slate:** The three-step text hierarchy for body copy, labels, and inactive tabs.
- **Control Border / Control Fill:** The compact native field and button vocabulary.

### Named Rules

**The Blue Means State Rule.** Blue marks selection, focus, checked tools, or direct manipulation; it is not ambient decoration.

**The Ink Frames Work Rule.** Dark ink belongs to the tool rail, top action bar, and canvas controls; editable inspector content stays on light paper.

## Typography

**Display Font:** None; this is an Operate surface with no display typography.
**Body Font:** Segoe UI with Malgun Gothic fallback
**Label Font:** Segoe UI with Malgun Gothic fallback

**Character:** The system stack is compact, neutral, and optimized for Korean Windows UI. Hierarchy comes from weight and placement, not from multiple decorative families or oversized headings.

### Hierarchy

- **Body** (400, 12px): Field values, tabs, toolbar actions, and supporting copy.
- **Label** (600, 12px): Inspector labels and compact section identifiers.
- **Action** (700, 12px): Push buttons and the zoom readout.
- **Canvas Index** (700, 10px): Small numbered region badges drawn directly over the image.

### Named Rules

**The System UI Rule.** Keep Segoe UI and Malgun Gothic for editor chrome; do not introduce web fonts or editorial display faces into the native workspace.

## Layout

The first viewport runs left to right as a 160–180px fixed tool rail, a 360–470px inspector, the flexible central canvas, and an optional 190–280px session list when multiple images are present. The 50px top action bar spans the workspace. The horizontal splitter uses one-pixel handles and cannot collapse children; the canvas receives all remaining stretch and retains at least 480px in the initial sizing calculation.

The window opens at 96% of available screen width and 94% of available height, bounded to 1680×1040px and a minimum of 1120×680px, or 1200×680px when the session list is present. The inspector targets 25% of the starting width within a 360–450px range. On screens narrower than 1280px or shorter than 760px, the editor starts maximized.

Inspector content uses 16px outer insets and a 10px vertical rhythm. Related controls keep 6–9px internal gaps; distinct sections receive 8px breathing room and a one-pixel divider below the 13px section title. Tool tabs use 12px internal insets and a 9px rhythm. The zoom bar uses 8px horizontal by 5px vertical insets with 6px spacing. Keep the top action bar minimal and let splitter resizing, scrolling, and native window management handle adaptation.

**The Left-to-Canvas Rule.** A user chooses a tool on the far left, changes its values immediately beside it, and verifies the result in the center without crossing the canvas.

**The Page List Stays Right Rule.** When a job contains multiple images, its page list stays at the far right: tool rail → inspector → manga canvas → image list. A single-image job hides the list without changing the other three positions.

## Elevation & Depth

The editor uses no custom shadows. Depth comes from tonal layering and one-pixel borders: ink chrome over cool workspace paper, white fields over the inspector, and charcoal around the manga page. Native Qt dialogs, menus, and tooltips keep their platform elevation.

**The Flat Workbench Rule.** Resting editor surfaces stay flat; hierarchy comes from adjacency, tone, and borders rather than floating cards.

## Shapes

Controls use gently restrained corners: 6px for most buttons, fields, tabs, help, and status surfaces; 4px for compact zoom controls; 3px for the slider groove; and an 8px circular slider handle. Major panels remain rectilinear and edge-aligned. Borders are one pixel where separation or focus needs to be explicit.

## Components

### Top Action Bar

- **Character:** Minimal global operations in a fixed ink strip. Undo, redo, named history, and brush size stay on the left; save and export stay anchored at the far right.
- **Shape:** Transparent 6px actions with 6px × 9px padding; the filled save action receives 12px horizontal padding.
- **States:** Hover adds a darker slate fill and visible border; shortcuts and tooltips preserve the full operation meaning.
- **Content:** Every action pairs a consistent 1.8px line icon with an explicit Korean label. The history menu names each undoable action and can undo back to a selected point. Save uses the only filled blue toolbar treatment; export remains neutral.
- **Undo/Redo:** Use mirrored near-circular return arrows with a heavier arc and a filled triangular head, matching familiar desktop editor controls at 18px.

### Tool Rail

- **Character:** Narrow, icon-led, and left aligned.
- **Shape:** Every rendered action row uses the same 38px height inside a 204px toolbar. A tighter 6px inset moves icons left and leaves safe breathing room around every full Korean tool name; selected state never changes type weight or geometry.
- **States:** Default is quiet slate on transparent ink; hover brightens the label and darkens the row; checked state fills with Operational Blue and adds a lighter blue border.
- **Behavior:** Actions remain mutually legible through checked state, icons, full visible labels, and tooltips; color is not the sole signal. Never shorten `인페인팅 브러시`, `인페인팅 마스크 지우개`, or the restore/paint tools to an ambiguous `지우기`.
- **Grouping:** Basic, inpainting, painting, and original-restore tools use restrained section labels and one-pixel dividers. Each brush/eraser family has a distinct composite icon instead of repeating the same generic glyph.

### Buttons

- **Shape:** Restrained 6px corners and 8px × 10px padding.
- **Default:** White field paper, dark ink text, one-pixel border, and 600 action weight.
- **Hover / Pressed:** Hover shifts to the pale operational-blue wash with a blue edge; pressed uses the stronger blue wash. Disabled actions use muted slate over disabled paper.

### Inputs / Fields

- **Style:** White field paper, dark ink text, one-pixel cool border, 6px corners, and 6px internal padding.
- **Focus:** The border changes to Operational Blue while native focus and editing behavior remain intact.
- **Selection:** Text selection uses the soft blue wash.
- **Resizable Text:** Source and translated text fields start compact at 68px and 82px. Their lower-right diagonal grip adjusts height from 52–360px without changing the inspector width.
- **Metric Fields:** Position, size, and effect values use a balanced 3×3 grid. Every field reserves the same width for its quiet semantic icon and short label so all value dividers align, then ends with centered stacked increment/decrement controls separated by a subtle rule. The title, value, and stepper surfaces meet inside one uninterrupted one-pixel perimeter with no uneven white inset.
- **Brush Size:** The top-bar brush value uses the same explicit value-and-stacked-stepper language as inspector metrics instead of the native Windows spinbox arrows.
- **Inline Options:** Region rendering and vertical writing use compact icon-led toggle chips with an explicit filled selected state instead of unstyled native Windows checkboxes.

### Tabs

- **Style:** Inactive tabs use control fill and quiet text; the selected tab becomes white with blue text and visually joins the white pane.
- **Shape:** Six-pixel top corners, compact 7px × 12px padding, and a one-pixel pane border.
- **Content:** OCR, inpainting, translation, and editor settings remain grouped in the implemented inspector tab set.

### Canvas Controls

- **Zoom Bar:** A dark strip with compact 4px controls, white labels, and explicit pan, zoom, fit, and 100% actions.
- **Scrollbars:** 18px native bars with 36px minimum handles; handles turn blue on hover.
- **Region Overlays:** Selected regions use a 2px blue outline and visible resize handles; enabled unselected regions use a 1px translucent amber outline; disabled regions use gray. Each number uses a compact dark tab attached directly above the region's top-left corner. When that tab would leave the viewport or collide with another region or badge, the layout scores positions above, below, left, and right, prefers page-exterior canvas space, and uses the least-obstructive visible fallback.

### Status and Help

- **Status:** A pale-blue, blue-ink callout fixed below the inspector scroll area so current tool and operation feedback remains visible.
- **Save Toast:** A compact ink surface appears 18px from the lower-right window edge after a successful save. It pairs the save icon with `<파일명> 저장 완료`, reuses the same surface for repeated saves, and disappears after 2.4 seconds. Failures continue to use the blocking native error dialog.
- **Help:** A light neutral bordered surface inside the inspector; its text is selectable and can be hidden through the existing editor setting.

### Contextual Inspector

- **Context:** A compact blue-tinted header names the active tool and gives one sentence of immediate guidance.
- **Sections:** Text and layout share one compact tab because they are adjusted together; tool settings use the second tab. Position, size, typography, and spacing values use a two-column control grid so the complete text-editing path fits without scrolling at 1200×760. Selecting a tool opens the most relevant tab and nested settings tab without preventing manual navigation.
- **Alignment:** Horizontal text alignment uses a persistent three-button icon group—left, center, right—with a single checked state and tooltips, matching familiar document-editor controls instead of hiding the choices in a dropdown.
- **Typography flow:** Font family and weight sit directly below alignment and render/orientation state. Arrowless numeric fields are grouped into three full-width semantic rows: region (X/Y/W/H), type (size/spacing/padding), and effects (rotation/outline). Each row distributes only its own controls, avoiding empty grid cells and repeated labels.
- **Inline emphasis and color:** Bold, italic, and bold-italic actions show a checked state derived from the selected or whole edited text. Text and outline color controls show both the actual swatch surface and the current uppercase HEX value with contrast-safe labels and icons.
- **Image List:** Multi-image jobs show the current page and selection count, offer select-all and clear-selection actions, and can collapse from the top action bar to return width to the canvas.

## Do's and Don'ts

### Do:

- **Do** keep the advanced editor canvas-first and preserve the left rail → inspector → canvas reading order.
- **Do** keep the multi-image page list at the far right of the canvas.
- **Do** reserve blue for active, focused, selected, or directly manipulated state.
- **Do** preserve native Qt keyboard shortcuts, tooltips, dialogs, scrollbars, focus behavior, and existing editing behavior.
- **Do** use the shared line-icon vocabulary on toolbar actions and inspector buttons while keeping explicit Korean action labels.
- **Do** show the name of the next undo/redo action and keep the named history list collapsible in the top bar.
- **Do** keep user-facing editor labels in clear Korean and verify that they render correctly on Windows.
- **Do** pair state color with checked controls, focus edges, selection handles, labels, or status text.
- **Do** keep inspector controls compact enough for repeated OCR, text, font, mask, inpainting, and translation work.

### Don't:

- **Don't** move the inspector across the canvas or split related tool choice and settings between opposite sides.
- **Don't** place the multi-image page list between the tool rail and inspector.
- **Don't** add gradients, decorative shadows, floating cards, oversized headings, or ornamental chrome.
- **Don't** use Operational Blue as a broad background or decorative accent unrelated to state.
- **Don't** replace the system font stack or imitate browser UI patterns that weaken standard Qt affordances.
- **Don't** shorten a destructive, mask, brush, restore, save, or export action until its target becomes ambiguous.
- **Don't** document or introduce components that are not present in the native editor.
