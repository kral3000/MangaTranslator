---
name: MangaTranslatorKo Advanced Editor
description: Native PyQt6 manga editing workspace with an ink rail, light inspector, and canvas-first feedback.
colors:
  active-blue: "#2563eb"
  active-blue-border: "#60a5fa"
  active-blue-soft: "#bfdbfe"
  ink-rail: "#111827"
  ink-hover: "#1f2937"
  workspace-paper: "#f6f8fb"
  field-paper: "#ffffff"
  canvas-charcoal: "#20242b"
  canvas-bar: "#202733"
  canvas-control: "#374151"
  body-ink: "#172033"
  label-slate: "#334155"
  quiet-slate: "#475569"
  control-border: "#cbd5e1"
  control-fill: "#e8edf5"
  toolbar-text: "#e5e7eb"
  status-paper: "#eef6ff"
  status-ink: "#1e3a8a"
  canvas-selection: "#2196f3"
  canvas-region: "#ffc107"
  canvas-disabled: "#969696"
typography:
  body:
    fontFamily: "Segoe UI, Malgun Gothic"
    fontSize: "12px"
    fontWeight: 400
  label:
    fontFamily: "Segoe UI, Malgun Gothic"
    fontSize: "12px"
    fontWeight: 600
  action:
    fontFamily: "Segoe UI, Malgun Gothic"
    fontSize: "12px"
    fontWeight: 700
  canvas-index:
    fontFamily: "Arial"
    fontSize: "10px"
    fontWeight: 700
rounded:
  compact: "3px"
  small-control: "4px"
  control: "6px"
  circular-handle: "8px"
spacing:
  tight: "5px"
  compact: "6px"
  control: "8px"
  standard: "10px"
  panel: "14px"
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
    padding: "9px 7px"
    width: "68px"
  toolrail-action-selected:
    backgroundColor: "{colors.active-blue}"
    textColor: "{colors.field-paper}"
    typography: "{typography.body}"
    rounded: "{rounded.control}"
    padding: "9px 7px"
    width: "68px"
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

The first viewport runs left to right as an 88–108px fixed tool rail, an optional 190–280px session list when multiple images are present, a 360–470px inspector, and the flexible central canvas. The horizontal splitter cannot collapse children; the canvas receives all remaining stretch and retains at least 480px in the initial sizing calculation.

The window opens at 96% of available screen width and 94% of available height, bounded to 1680×1040px and a minimum of 1120×680px, or 1200×680px when the session list is present. The inspector targets 25% of the starting width within a 360–450px range. On screens narrower than 1280px or shorter than 760px, the editor starts maximized.

Inspector content uses 14px outer insets and an 8px vertical rhythm. The zoom bar uses 8px horizontal by 5px vertical insets with 6px spacing. Keep the top action bar minimal and let splitter resizing, scrolling, and native window management handle adaptation.

**The Left-to-Canvas Rule.** A user chooses a tool on the far left, changes its values immediately beside it, and verifies the result in the center without crossing the canvas.

## Elevation & Depth

The editor uses no custom shadows. Depth comes from tonal layering and one-pixel borders: ink chrome over cool workspace paper, white fields over the inspector, and charcoal around the manga page. Native Qt dialogs, menus, and tooltips keep their platform elevation.

**The Flat Workbench Rule.** Resting editor surfaces stay flat; hierarchy comes from adjacency, tone, and borders rather than floating cards.

## Shapes

Controls use gently restrained corners: 6px for most buttons, fields, tabs, help, and status surfaces; 4px for compact zoom controls; 3px for the slider groove; and an 8px circular slider handle. Major panels remain rectilinear and edge-aligned. Borders are one pixel where separation or focus needs to be explicit.

## Components

### Top Action Bar

- **Character:** Minimal global operations in a fixed ink strip.
- **Shape:** Transparent 6px actions with 7px × 10px padding.
- **States:** Hover adds a darker slate fill and visible border; shortcuts and tooltips preserve the full operation meaning.
- **Content:** The visible Korean actions stay short—`실행 취소`, `다시 실행`, `저장`, and `내보내기`—while save and export tooltips retain their full shortcut descriptions.

### Tool Rail

- **Character:** Narrow, text-only, and left aligned.
- **Shape:** 68px minimum action width inside an 88–108px fixed toolbar; each action uses 9px × 7px padding and a 6px radius.
- **States:** Default is quiet slate on transparent ink; hover brightens the label and darkens the row; checked state fills with Operational Blue and adds a lighter blue border.
- **Behavior:** Actions remain mutually legible through checked state, visible labels, and tooltips; color is not the sole signal.

### Buttons

- **Shape:** Restrained 6px corners and 8px × 10px padding.
- **Default:** Cool control fill, dark ink text, one-pixel border, and bold action weight.
- **Hover / Pressed:** Hover shifts to a pale blue fill with a blue edge; pressed uses the stronger selection wash.

### Inputs / Fields

- **Style:** White field paper, dark ink text, one-pixel cool border, 6px corners, and 6px internal padding.
- **Focus:** The border changes to Operational Blue while native focus and editing behavior remain intact.
- **Selection:** Text selection uses the soft blue wash.

### Tabs

- **Style:** Inactive tabs use control fill and quiet text; the selected tab becomes white with blue text and visually joins the white pane.
- **Shape:** Six-pixel top corners, compact 7px × 12px padding, and a one-pixel pane border.
- **Content:** OCR, inpainting, translation, and editor settings remain grouped in the implemented inspector tab set.

### Canvas Controls

- **Zoom Bar:** A dark strip with compact 4px controls, white labels, and explicit pan, zoom, fit, and 100% actions.
- **Scrollbars:** 18px native bars with 36px minimum handles; handles turn blue on hover.
- **Region Overlays:** Selected regions use blue with thicker outlines and visible resize handles; enabled unselected regions use amber; disabled regions use gray. Number badges pair color with an explicit index.

### Status and Help

- **Status:** A pale-blue, blue-ink callout fixed below the inspector scroll area so current tool and operation feedback remains visible.
- **Help:** A light neutral bordered surface inside the inspector; its text is selectable and can be hidden through the existing editor setting.

## Do's and Don'ts

### Do:

- **Do** keep the advanced editor canvas-first and preserve the left rail → inspector → canvas reading order.
- **Do** reserve blue for active, focused, selected, or directly manipulated state.
- **Do** preserve native Qt keyboard shortcuts, tooltips, dialogs, scrollbars, focus behavior, and existing editing behavior.
- **Do** keep user-facing editor labels in clear Korean and verify that they render correctly on Windows.
- **Do** pair state color with checked controls, focus edges, selection handles, labels, or status text.
- **Do** keep inspector controls compact enough for repeated OCR, text, font, mask, inpainting, and translation work.

### Don't:

- **Don't** move the inspector across the canvas or split related tool choice and settings between opposite sides.
- **Don't** add gradients, decorative shadows, floating cards, oversized headings, or ornamental chrome.
- **Don't** use Operational Blue as a broad background or decorative accent unrelated to state.
- **Don't** replace the system font stack or imitate browser UI patterns that weaken standard Qt affordances.
- **Don't** document or introduce components that are not present in the native editor.
