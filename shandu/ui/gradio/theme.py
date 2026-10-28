from __future__ import annotations

import gradio as gr

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
"""

_PAPER = "#faf9f6"
_SURFACE = "#ffffff"
_INK = "#1c1b19"
_INK_2 = "#3c3a35"
_INK_3 = "#6b6861"
_LINE = "#e3dfd7"
_ACCENT = "#9a3b23"

_PAPER_DARK = "#13120f"
_SURFACE_DARK = "#201d18"
_INK_DARK = "#efeae1"
_INK_2_DARK = "#d8d2c6"
_INK_3_DARK = "#9a9488"
_LINE_DARK = "#35302a"
_ACCENT_DARK = "#d96a45"


def build_theme() -> gr.Theme:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.stone,
        neutral_hue=gr.themes.colors.stone,
        font=("IBM Plex Sans", "system-ui", "sans-serif"),
        font_mono=("IBM Plex Mono", "ui-monospace", "monospace"),
        radius_size=gr.themes.sizes.radius_none,
        spacing_size=gr.themes.sizes.spacing_md,
        text_size=gr.themes.sizes.text_md,
    ).set(
        body_background_fill=_PAPER,
        body_background_fill_dark=_PAPER_DARK,
        body_text_color=_INK,
        body_text_color_dark=_INK_DARK,
        body_text_color_subdued=_INK_3,
        body_text_color_subdued_dark=_INK_3_DARK,
        background_fill_primary=_PAPER,
        background_fill_primary_dark=_PAPER_DARK,
        background_fill_secondary=_SURFACE,
        background_fill_secondary_dark=_SURFACE_DARK,
        border_color_primary=_LINE,
        border_color_primary_dark=_LINE_DARK,
        border_color_accent=_ACCENT,
        border_color_accent_dark=_ACCENT_DARK,
        color_accent=_ACCENT,
        color_accent_soft="rgba(154, 59, 35, 0.09)",
        color_accent_soft_dark="rgba(224, 128, 85, 0.14)",
        slider_color_dark=_ACCENT_DARK,
        block_background_fill=_PAPER,
        block_background_fill_dark=_PAPER_DARK,
        block_border_width="0px",
        block_label_text_color=_INK_3,
        block_label_text_color_dark=_INK_3_DARK,
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_info_text_color=_INK_3,
        block_info_text_color_dark=_INK_3_DARK,
        block_padding="0px",
        block_shadow="none",
        block_shadow_dark="none",
        block_title_text_color=_INK_2,
        block_title_text_color_dark=_INK_2_DARK,
        block_title_text_weight="500",
        panel_background_fill=_PAPER,
        panel_background_fill_dark=_PAPER_DARK,
        panel_border_width="0px",
        input_background_fill=_SURFACE,
        input_background_fill_dark=_SURFACE_DARK,
        input_background_fill_focus=_SURFACE,
        input_background_fill_focus_dark=_SURFACE_DARK,
        input_border_color=_LINE,
        input_border_color_dark="#575046",
        input_border_color_focus=_ACCENT,
        input_border_color_focus_dark=_ACCENT_DARK,
        input_border_width="1px",
        input_placeholder_color=_INK_3,
        input_placeholder_color_dark=_INK_3_DARK,
        input_shadow="none",
        input_shadow_focus="none",
        button_large_radius="2px",
        button_small_radius="2px",
        button_primary_background_fill=_INK,
        button_primary_background_fill_dark="#d8d1c4",
        button_primary_background_fill_hover=_ACCENT,
        button_primary_background_fill_hover_dark=_ACCENT_DARK,
        button_primary_border_color=_INK,
        button_primary_border_color_dark="#d8d1c4",
        button_primary_border_color_hover=_ACCENT,
        button_primary_border_color_hover_dark=_ACCENT_DARK,
        button_primary_text_color=_PAPER,
        button_primary_text_color_dark=_PAPER_DARK,
        button_secondary_background_fill="transparent",
        button_secondary_background_fill_dark="transparent",
        button_secondary_background_fill_hover=_SURFACE,
        button_secondary_background_fill_hover_dark=_SURFACE_DARK,
        button_secondary_border_color=_LINE,
        button_secondary_border_color_dark=_LINE_DARK,
        button_secondary_text_color=_INK_2,
        button_secondary_text_color_dark=_INK_2_DARK,
        table_border_color=_LINE,
        table_border_color_dark=_LINE_DARK,
        table_even_background_fill=_PAPER,
        table_even_background_fill_dark=_PAPER_DARK,
        table_odd_background_fill=_PAPER,
        table_odd_background_fill_dark=_PAPER_DARK,
        table_text_color=_INK_2,
        table_text_color_dark=_INK_2_DARK,
        accordion_text_color=_INK_2,
        accordion_text_color_dark=_INK_2_DARK,
    )


CSS = """
:root {
  --sans: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --serif: 'Source Serif 4', Charter, 'Iowan Old Style', Georgia, serif;
  --mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  --paper: #faf9f6;
  --surface: #ffffff;
  --sunken: #f3f1ec;
  --ink: #1c1b19;
  --ink-2: #3c3a35;
  --ink-3: #6b6861;
  --line: #e3dfd7;
  --line-strong: #cfc9be;
  --accent: #9a3b23;
  --accent-soft: rgba(154, 59, 35, 0.35);
  --ok: #3f6b4e;
  --warn: #8a6520;
  --err: #9b3030;
  --measure: 34rem;
}

body.dark,
.dark {
  --paper: #13120f;
  --surface: #201d18;
  --sunken: #232019;
  --ink: #efeae1;
  --ink-2: #d8d2c6;
  --ink-3: #9a9488;
  --line: #35302a;
  --line-strong: #575046;
  --accent: #d96a45;
  --accent-soft: rgba(217, 106, 69, 0.5);
  --ok: #7fae8b;
  --warn: #cfa055;
  --err: #d98080;
}

.gradio-container {
  font-family: var(--sans) !important;
  font-size: 15px !important;
}

.gradio-container footer {
  color: var(--ink-3) !important;
}

::selection {
  background: var(--accent-soft);
  color: var(--ink);
}

.gradio-container main.contain {
  width: min(100%, 820px) !important;
  max-width: 820px !important;
  margin: 0 auto !important;
  padding-bottom: 96px !important;
}

/* Gradio scopes rules inside @media to elements within .contain, so the page
   gutter, which sits outside it, scales with the viewport instead. */
.gradio-container .main.fillable {
  padding-left: clamp(16px, 4vw, 32px) !important;
  padding-right: clamp(16px, 4vw, 32px) !important;
}

.gradio-container footer .divider {
  display: none !important;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .panel {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.shandu-plain,
.shandu-plain > .wrap,
.shandu-plain .html-container {
  padding: 0 !important;
  margin: 0 !important;
  background: transparent !important;
}

.shandu-plain:empty,
.shandu-plain .html-container:empty {
  display: none !important;
}

/* Masthead */

.shandu-masthead {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 24px;
  padding: 56px 0 20px;
  border-bottom: 1px solid var(--line);
}

.shandu-masthead h1 {
  margin: 0;
  color: var(--ink);
  font-family: var(--serif);
  font-size: 1.9rem;
  font-weight: 600;
  letter-spacing: -0.01em;
}

.shandu-masthead p {
  margin: 6px 0 0;
  max-width: 54ch;
  color: var(--ink-3);
  font-size: 0.92rem;
  line-height: 1.55;
}

.shandu-masthead code {
  color: var(--ink-3);
  font-family: var(--mono);
  font-size: 0.78rem;
}

/* Composer */

.shandu-composer {
  gap: 0 !important;
  margin-top: 30px;
  border: 1px solid var(--line-strong);
  background: var(--surface);
}

.dark .shandu-download button,
.dark .shandu-prose code {
  border: 1px solid var(--line-strong) !important;
}

.dark .shandu-prose blockquote {
  border-left-color: var(--accent) !important;
}

/* Only the question box frames the card; every other control has its own ring. */
.shandu-composer:has(.shandu-question textarea:focus-visible) {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}

.shandu-question textarea {
  border: 0 !important;
  background: transparent !important;
  padding: 20px 20px 14px !important;
  font-family: var(--serif) !important;
  font-size: 1.16rem !important;
  line-height: 1.55 !important;
  color: var(--ink) !important;
  resize: none !important;
}

.shandu-question textarea:focus {
  box-shadow: none !important;
  outline: none !important;
}

#shandu-examples {
  flex: 1 1 auto !important;
  min-width: 0 !important;
  padding: 0 !important;
}

#shandu-examples .label,
#shandu-examples label {
  display: none !important;
}

.shandu-library label,
.shandu-first-run h3 {
  color: var(--ink-3) !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
}

.gradio-container .label svg {
  display: none !important;
}

#shandu-examples .gallery {
  gap: 8px !important;
}

#shandu-examples button,
#shandu-examples .gallery-item {
  width: fit-content !important;
  max-width: 100% !important;
  border: 1px solid var(--line-strong) !important;
  border-radius: 999px !important;
  background: var(--sunken) !important;
  padding: 4px 12px !important;
  color: var(--ink-2) !important;
  font-size: 0.82rem !important;
  font-weight: 400 !important;
  text-align: left !important;
  white-space: nowrap !important;
}

#shandu-examples button:hover,
#shandu-examples .gallery-item:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}

.shandu-actions {
  margin-top: 0 !important;
  padding: 14px 20px !important;
  border-top: 1px solid var(--line) !important;
  gap: 10px !important;
  align-items: center !important;
  justify-content: flex-end !important;
  flex-wrap: nowrap !important;
}

.shandu-actions button.secondary {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}

.shandu-actions button {
  min-width: 0 !important;
  min-height: 38px !important;
  padding: 0 20px !important;
  font-size: 0.9rem !important;
  font-weight: 500 !important;
  flex: 0 0 auto !important;
  width: auto !important;
  cursor: pointer !important;
}

/* Run options: a closed row at the foot of the composer. */

.shandu-options {
  border-top: 1px solid var(--line) !important;
  padding: 0 20px !important;
}

.shandu-options .label-wrap {
  padding: 10px 0 !important;
}

.shandu-options .label-wrap span {
  color: var(--ink-3) !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
}

/* Gradio wraps neighbouring inputs in one .form with no gap between them. */
.shandu-option-row,
.shandu-options .form {
  gap: 12px !important;
}

.shandu-option-row {
  margin-bottom: 4px !important;
}

.shandu-options span.md,
.shandu-options span.md p {
  color: var(--ink-3) !important;
  font-size: 0.76rem !important;
  line-height: 1.45 !important;
  text-wrap: balance;
}

.shandu-options input,
.shandu-options select {
  border-radius: 2px !important;
  font-size: 0.88rem !important;
}

/* A text box and a dropdown in one row share a height at rest; a text box
   still grows with what is pasted into it. */
.shandu-options textarea,
.shandu-options .container > .wrap {
  min-height: 40px !important;
}

.shandu-options textarea {
  padding: 9px 10px !important;
}

.shandu-options ul.options,
.shandu-settings ul.options {
  border: 1px solid var(--line-strong) !important;
}

.shandu-options .shandu-note {
  margin: 2px 0 14px !important;
}

.shandu-options label span,
.shandu-options [data-testid="block-info"],
.shandu-settings label span,
.shandu-settings [data-testid="block-info"] {
  color: var(--ink-3) !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
}

/* Run strip */

.shandu-strip {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 8px 28px;
  margin-top: 32px;
  padding: 14px 0;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
}

.shandu-strip .stage {
  display: inline-flex;
  align-items: baseline;
  gap: 9px;
  color: var(--ink);
  font-size: 0.92rem;
  font-weight: 500;
}

.shandu-strip .stage::before {
  content: '';
  align-self: center;
  width: 6px;
  height: 6px;
  border-radius: 999px;
  background: var(--ink-3);
}

.shandu-strip.is-running .stage::before {
  background: var(--accent);
  animation: shandu-pulse 1.6s ease-in-out infinite;
}

.shandu-strip.is-error .stage::before {
  background: var(--err);
}

.shandu-strip.is-complete .stage::before {
  background: var(--ok);
}

.shandu-strip .field {
  display: inline-flex;
  align-items: baseline;
  gap: 7px;
  color: var(--ink-3);
  font-size: 0.78rem;
}

.shandu-strip .field b {
  color: var(--ink-2);
  font-family: var(--mono);
  font-size: 0.8rem;
  font-weight: 400;
}

.shandu-strip .field.run b {
  overflow-wrap: anywhere;
}

.shandu-fault,
.shandu-hint {
  margin: 16px 0 0;
  padding: 12px 14px;
  border-left: 2px solid var(--err);
  background: var(--sunken);
  color: var(--ink-2);
  font-size: 0.9rem;
  line-height: 1.5;
}

.shandu-fault-detail {
  margin: 8px 0 0;
  padding-left: 16px;
  color: var(--ink-3);
  font-family: var(--mono);
  font-size: 0.74rem;
  line-height: 1.5;
  overflow-wrap: anywhere;
}

.shandu-steps {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 20px;
  width: 100%;
  margin: 7px 0 0;
  padding: 0;
  list-style: none;
  color: var(--ink-3);
  font-size: 0.78rem;
}

.shandu-steps li {
  color: var(--ink-3);
}

.shandu-steps .is-done {
  color: var(--ink-2);
}

.shandu-steps .is-live {
  color: var(--accent);
  font-weight: 500;
}

@keyframes shandu-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.55; }
}

/* Activity */

.shandu-activity {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 0 40px;
  margin-top: 4px;
}

.shandu-activity h3 {
  margin: 0 0 4px;
  padding: 18px 0 8px;
  color: var(--ink-3);
  font-size: 0.78rem;
  font-weight: 500;
  border-bottom: 1px solid var(--line);
}

.shandu-activity ol {
  max-height: 420px;
  overflow: auto;
  scrollbar-color: var(--line-strong) transparent;
  scrollbar-width: thin;
  margin: 0;
  padding: 0;
  list-style: none;
}

.shandu-activity li {
  padding: 9px 0;
  border-bottom: 1px solid var(--line);
}

.shandu-activity li:last-child {
  border-bottom: 0;
}

.shandu-feed li {
  display: grid;
  grid-template-columns: 62px minmax(0, 1fr);
  gap: 2px 12px;
}

.shandu-feed time,
.shandu-tasks time,
.shandu-feed .meta,
.shandu-tasks dd {
  color: var(--ink-3);
  font-family: var(--mono);
  font-size: 0.74rem;
}

.shandu-feed p {
  grid-column: 2;
  margin: 0;
  color: var(--ink-2);
  font-size: 0.86rem;
  line-height: 1.5;
  overflow-wrap: anywhere;
}

.shandu-feed .meta {
  grid-column: 2;
  overflow-wrap: anywhere;
}

.shandu-tasks .head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
}

.shandu-tasks .id {
  color: var(--ink-3);
  font-size: 0.8rem;
}

.shandu-tasks .state {
  color: var(--ink-3);
  font-size: 0.74rem;
}

.shandu-tasks .state.is-running {
  color: var(--accent);
}

.shandu-tasks .state.is-complete {
  color: var(--ok);
}

.shandu-tasks .state.is-error {
  color: var(--err);
}

.shandu-tasks p {
  margin: 5px 0 0;
  color: var(--ink-2);
  font-size: 0.86rem;
  line-height: 1.5;
}

.shandu-tasks dl {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 18px;
  margin: 6px 0 0;
}

.shandu-tasks div {
  display: flex;
  align-items: baseline;
  gap: 6px;
}

.shandu-tasks dt {
  color: var(--ink-3);
  font-size: 0.74rem;
}

.shandu-tasks dd {
  margin: 0;
}

.shandu-tasks .meta {
  display: block;
  margin-top: 6px;
  color: var(--ink-3);
  font-size: 0.8rem;
  line-height: 1.45;
}

.shandu-tasks .state time {
  margin-left: 7px;
}

/* Gradio marks streaming blocks with an accent border; the run strip and the
   feed update on every event, so the border would strobe. */
.gradio-container .generating,
.gradio-container .pending {
  border: 0 !important;
  box-shadow: none !important;
  animation: none !important;
}

/* Recent reports */

.shandu-library {
  margin-top: 48px;
}

.shandu-library .table-wrap,
.shandu-library table {
  width: 100% !important;
  border: 0 !important;
  background: transparent !important;
}

.shandu-library thead {
  display: none !important;
}

.shandu-library tr {
  cursor: pointer !important;
  border-bottom: 1px solid var(--line) !important;
}

.shandu-library tr:hover td {
  color: var(--accent) !important;
}

.shandu-library td {
  padding: 11px 0 !important;
  border: 0 !important;
  background: transparent !important;
  color: var(--ink-2) !important;
  font-size: 0.92rem !important;
}

.shandu-library td,
.shandu-library td > div {
  max-width: none !important;
  width: auto !important;
  text-align: left !important;
  white-space: normal !important;
}

.shandu-library td:nth-child(2) {
  width: 1% !important;
  padding-left: 24px !important;
  color: var(--ink-3) !important;
  font-size: 0.8rem !important;
}

.shandu-library td:nth-child(2) > div,
.shandu-library td:nth-child(3) > div {
  white-space: nowrap !important;
}

.shandu-library td:nth-child(4) {
  display: none !important;
}

.shandu-library td:nth-child(3) {
  width: 1% !important;
  padding-left: 24px !important;
  color: var(--ink-3) !important;
  font-family: var(--mono);
  font-size: 0.76rem !important;
  text-align: right !important;
}

.shandu-library .paginate {
  justify-content: flex-start !important;
  margin-top: 12px !important;
  color: var(--ink-3) !important;
  font-size: 0.78rem !important;
}

.shandu-library .paginate button {
  color: var(--ink-3) !important;
  font-family: var(--mono);
}

.shandu-library .paginate button.current-page {
  color: var(--ink) !important;
  font-weight: 500 !important;
}

/* Recent reports and first run */

.shandu-first-run {
  margin-top: 46px;
  padding-top: 16px;
  border-top: 1px solid var(--line);
}

.shandu-first-run h3 {
  margin: 0;
}

.shandu-first-run ol {
  margin: 12px 0 0;
  padding-left: 20px;
}

.shandu-first-run li {
  margin: 0 0 7px;
  color: var(--ink-2);
  font-size: 0.92rem;
  line-height: 1.5;
}

.shandu-first-run li::marker {
  color: var(--ink-3);
  font-family: var(--mono);
  font-size: 0.78rem;
}

/* Report */

.shandu-report {
  margin-top: 40px;
}

.shandu-report:has(.md:empty) {
  display: none !important;
}

.shandu-prose.padded {
  padding: 0 !important;
}

.shandu-report-head {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  max-width: var(--measure);
  margin: 0 0 30px;
  padding-top: 10px;
  border-top: 1px solid var(--line);
}

.shandu-download {
  flex: 0 0 auto !important;
  width: auto !important;
}

.shandu-download button {
  min-height: 32px !important;
  padding: 0 14px !important;
  font-size: 0.8rem !important;
  font-weight: 400 !important;
}

.shandu-prose {
  max-width: var(--measure);
  margin: 0 !important;
}

.shandu-prose .prose,
.shandu-prose .md {
  font-family: var(--serif) !important;
  font-size: 1.12rem !important;
  line-height: 1.68 !important;
  color: var(--ink-2) !important;
}

.shandu-prose h1,
.shandu-prose h2,
.shandu-prose h3 {
  font-family: var(--serif) !important;
  color: var(--ink) !important;
  font-weight: 600 !important;
  letter-spacing: -0.01em !important;
}

.shandu-prose h1 {
  margin: 0 0 28px !important;
  padding-bottom: 18px !important;
  border-bottom: 1px solid var(--line) !important;
  font-size: 2rem !important;
  line-height: 1.2 !important;
}

.shandu-prose h2 {
  margin: 44px 0 12px !important;
  font-size: 1.32rem !important;
}

.shandu-prose h3 {
  margin: 30px 0 10px !important;
  font-size: 1.2rem !important;
}

.shandu-prose .md > :first-child {
  margin-top: 0 !important;
}

.shandu-prose p,
.shandu-prose li {
  margin: 0 0 18px !important;
  color: var(--ink-2) !important;
}

.shandu-prose strong {
  color: var(--ink) !important;
  font-weight: 600 !important;
}

.shandu-prose ol,
.shandu-prose ul {
  margin: 0 0 18px !important;
  padding-left: 22px !important;
}

.shandu-prose li {
  margin: 0 0 10px !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
}

.shandu-prose li::marker {
  color: var(--ink-3) !important;
}

.shandu-prose a {
  color: var(--accent) !important;
  text-decoration: none !important;
  border-bottom: 1px solid var(--accent-soft) !important;
  overflow-wrap: anywhere;
}

.shandu-prose a:hover {
  border-bottom-color: var(--accent) !important;
}

.shandu-prose blockquote {
  margin: 0 0 18px !important;
  padding-left: 18px !important;
  border-left: 2px solid var(--line-strong) !important;
  color: var(--ink-3) !important;
}

.shandu-prose hr {
  margin: 36px 0 !important;
  border: 0 !important;
  border-top: 1px solid var(--line) !important;
}

.shandu-prose code {
  font-family: var(--mono) !important;
  font-size: 0.86em !important;
  background: var(--sunken) !important;
  color: var(--ink-2) !important;
  padding: 1px 4px !important;
  overflow-wrap: anywhere !important;
}

.shandu-prose table {
  display: block;
  overflow-x: auto;
  border: 0 !important;
  font-family: var(--sans) !important;
  font-size: 0.9rem !important;
}

.shandu-prose tr {
  border: 0 !important;
}

.shandu-prose th,
.shandu-prose td {
  padding: 8px 12px 8px 0 !important;
  border: 0 !important;
  border-bottom: 1px solid var(--line) !important;
  text-align: left !important;
  word-break: normal !important;
}

/* The citation ledger is appended last; set it as a reference list with a
   hanging indent so the numbers form a column. The number shares its run with
   the publisher, so the indent is the width of "[N] " for one and two digits. */
.shandu-prose .md > ul:last-child {
  padding-left: 0 !important;
  list-style: none !important;
}

.shandu-prose .md > ul:last-child li {
  margin: 0 0 9px !important;
  padding-left: 2em !important;
  text-indent: -2em !important;
  font-size: 0.95rem !important;
  line-height: 1.55 !important;
  color: var(--ink-2) !important;
}

.shandu-prose .md > ul:last-child li:nth-child(-n + 9) {
  padding-left: 1.5em !important;
  text-indent: -1.5em !important;
}

.shandu-prose .md > ul:last-child li strong {
  font-weight: 600 !important;
}

.shandu-doc-title {
  max-width: var(--measure);
  margin: 0 0 22px;
  color: var(--ink);
  font-family: var(--serif);
  font-size: 2rem;
  font-weight: 600;
  line-height: 1.2;
  letter-spacing: -0.01em;
}

.shandu-contents {
  max-width: var(--measure);
  margin: 0;
  padding: 14px 0 4px;
  border-top: 1px solid var(--line);
}

.shandu-contents h3 {
  margin: 0 0 9px;
  color: var(--ink-3);
  font-size: 0.78rem;
  font-weight: 500;
}

.shandu-contents ol {
  margin: 0;
  padding: 0;
  columns: 2;
  column-gap: 34px;
  list-style: none;
  counter-reset: contents;
}

.shandu-contents li {
  margin: 0 0 6px;
  color: var(--ink-2);
  font-size: 0.9rem;
  line-height: 1.4;
  break-inside: avoid;
  counter-increment: contents;
}

.shandu-contents li::before {
  content: counter(contents);
  margin-right: 9px;
  color: var(--ink-3);
  font-family: var(--mono);
  font-size: 0.74rem;
}

/* Settings */

/* The edge rule runs the full height of the panel and takes no layout space,
   so the toggle's offsets stay measured from the panel edge. */
.shandu-settings::before {
  content: '';
  position: absolute;
  top: 0;
  bottom: 0;
  left: 0;
  width: 1px;
  background: var(--line-strong);
}

/* The toggle floats over the top of the panel, so the scroller starts below it. */
.shandu-settings .sidebar-content {
  margin-top: 56px;
  height: calc(100% - 56px) !important;
  padding: 8px 24px 40px !important;
  background: var(--surface) !important;
  transition: visibility 0.3s;
}

/* A closed sidebar waits off screen; keep its fields out of the tab order. */
.shandu-settings:not(.open) .sidebar-content {
  visibility: hidden;
}

.shandu-settings .toggle-button {
  width: auto !important;
  height: 32px !important;
  padding: 0 12px 0 8px !important;
  border: 1px solid var(--line-strong) !important;
  border-radius: 999px !important;
  background: var(--paper) !important;
  box-shadow: none !important;
  color: var(--ink-2) !important;
  font-size: 0.8rem !important;
}

.shandu-settings:not(.open) .toggle-button {
  margin-right: 16px;
}

.shandu-settings.open .toggle-button {
  margin-left: 24px;
  background: var(--surface) !important;
}

/* Gradio turns the closed right-hand toggle 180 degrees; the label turns
   back with it, and the open toggle is not turned. */
.shandu-settings .toggle-button::after {
  content: 'Settings';
  display: inline-block;
  margin-left: 4px;
  transform: rotate(180deg);
}

.shandu-settings.open .toggle-button::after {
  content: 'Close';
  transform: none;
}

.shandu-settings h3 {
  margin: 8px 0 16px;
  color: var(--ink);
  font-size: 0.95rem;
  font-weight: 600;
}

.shandu-settings input,
.shandu-settings textarea,
.shandu-settings select {
  border-radius: 2px !important;
  font-size: 0.88rem !important;
}

.shandu-settings .block {
  margin-bottom: 22px !important;
}

.shandu-settings .sidebar-content > .column {
  max-width: 480px;
}

.shandu-settings button.lg {
  min-height: 38px !important;
  padding: 0 20px !important;
  font-size: 0.9rem !important;
  font-weight: 500 !important;
}

.shandu-settings span.md,
.shandu-settings span.md p {
  color: var(--ink-3) !important;
  font-size: 0.76rem !important;
  font-weight: 400 !important;
  line-height: 1.45 !important;
}

/* Gradio paints the unfilled track a fixed light grey and the thumb white,
   which disappear on the dark panel and wash out on the light one. */
.dark .shandu-settings input[type=range] {
  --neutral-200: var(--line-strong);
}

.shandu-settings input[type=range]::-webkit-slider-thumb {
  border: 1.5px solid var(--ink-3) !important;
  background-color: var(--surface) !important;
  box-shadow: none !important;
}

.shandu-settings input[type=range]::-moz-range-thumb {
  border: 1.5px solid var(--ink-3) !important;
  background-color: var(--surface) !important;
  box-shadow: none !important;
}

.shandu-note {
  color: var(--ink-3) !important;
  font-size: 0.78rem !important;
}

.shandu-debug {
  margin-top: 40px;
}

.shandu-debug .label-wrap span,
.shandu-activity-toggle .label-wrap span {
  color: var(--ink-3) !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
}

.shandu-debug .label-wrap,
.shandu-activity-toggle .label-wrap {
  padding: 12px 0 !important;
  border-bottom: 1px solid var(--line) !important;
}

button:focus-visible,
textarea:focus-visible,
input:focus-visible,
select:focus-visible {
  outline: 2px solid var(--accent) !important;
  outline-offset: 2px !important;
}

/* The panels clip an outline on three sides, so a field is ringed inside its
   border. A dropdown's border is on the wrapper around its input. */
.shandu-options :is(textarea, input:not([type=checkbox], [role=listbox])):focus-visible,
.shandu-settings :is(textarea, input:not([type=range], [role=listbox])):focus-visible {
  outline: none !important;
  box-shadow: inset 0 0 0 1px var(--accent);
}

.shandu-options [role=listbox]:focus-visible,
.shandu-settings [role=listbox]:focus-visible {
  outline: none !important;
}

.shandu-options .container > .wrap:focus-within,
.shandu-settings .container > .wrap:focus-within {
  box-shadow: inset 0 0 0 1px var(--accent);
}

@media (max-width: 860px) {
  .shandu-masthead {
    flex-direction: column;
    gap: 10px;
    padding-top: 36px;
  }

  .shandu-activity {
    grid-template-columns: minmax(0, 1fr);
    gap: 0;
  }

  .shandu-prose .prose,
  .shandu-prose .md {
    font-size: 1.06rem !important;
  }

  /* Over a full-width page the labelled toggle would cover text as it scrolls. */
  .shandu-settings:not(.open) .toggle-button {
    background: color-mix(in srgb, var(--paper) 80%, transparent) !important;
    backdrop-filter: blur(6px);
  }

  .shandu-settings:not(.open) .toggle-button::after {
    display: none;
  }
}

@media (max-width: 560px) {
  .shandu-actions {
    flex-wrap: wrap !important;
  }

  .shandu-option-row {
    margin-bottom: 0 !important;
  }

  .shandu-options .column {
    gap: 12px !important;
  }

  #shandu-examples {
    flex-basis: 100% !important;
  }

  #shandu-examples button,
  #shandu-examples .gallery-item {
    white-space: normal !important;
  }

  .shandu-library td:nth-child(2) {
    display: none !important;
  }
}

@media (prefers-reduced-motion: reduce) {
  .shandu-strip.is-running .stage::before {
    animation: none;
  }
}
"""
