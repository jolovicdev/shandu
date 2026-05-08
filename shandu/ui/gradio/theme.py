from __future__ import annotations

import gradio as gr


def build_theme() -> gr.Theme:
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.green,
        secondary_hue=gr.themes.colors.teal,
        neutral_hue=gr.themes.colors.zinc,
        font=("Inter", "system-ui", "sans-serif"),
        font_mono=("JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"),
        radius_size=gr.themes.sizes.radius_sm,
        spacing_size=gr.themes.sizes.spacing_md,
        text_size=gr.themes.sizes.text_md,
    ).set(
        body_background_fill="#070A08",
        body_background_fill_dark="#070A08",
        body_text_color="#F2F5EF",
        body_text_color_dark="#F2F5EF",
        background_fill_primary="#0D130F",
        background_fill_primary_dark="#0D130F",
        background_fill_secondary="#111813",
        background_fill_secondary_dark="#111813",
        border_color_primary="#26312B",
        border_color_primary_dark="#26312B",
        border_color_accent="#2F6F4E",
        border_color_accent_dark="#2F6F4E",
        block_background_fill="#101611",
        block_background_fill_dark="#101611",
        block_border_color="#27332C",
        block_border_color_dark="#27332C",
        block_border_width="1px",
        block_info_text_color="#9BA89E",
        block_info_text_color_dark="#9BA89E",
        block_label_background_fill="#151D17",
        block_label_background_fill_dark="#151D17",
        block_label_text_color="#C9D4CB",
        block_label_text_color_dark="#C9D4CB",
        block_padding="16px",
        block_radius="8px",
        block_shadow="none",
        block_shadow_dark="none",
        input_background_fill="#0B100D",
        input_background_fill_dark="#0B100D",
        input_background_fill_focus="#0E1510",
        input_background_fill_focus_dark="#0E1510",
        input_border_color="#2A352E",
        input_border_color_dark="#2A352E",
        input_border_color_focus="#39D98A",
        input_border_color_focus_dark="#39D98A",
        input_placeholder_color="#6E7A72",
        input_placeholder_color_dark="#6E7A72",
        input_radius="8px",
        button_primary_background_fill="#39D98A",
        button_primary_background_fill_dark="#39D98A",
        button_primary_background_fill_hover="#51E39A",
        button_primary_background_fill_hover_dark="#51E39A",
        button_primary_border_color="#39D98A",
        button_primary_border_color_dark="#39D98A",
        button_primary_text_color="#07100B",
        button_primary_text_color_dark="#07100B",
        button_secondary_background_fill="#151D17",
        button_secondary_background_fill_dark="#151D17",
        button_secondary_background_fill_hover="#1C271F",
        button_secondary_background_fill_hover_dark="#1C271F",
        button_secondary_border_color="#2A352E",
        button_secondary_border_color_dark="#2A352E",
        button_secondary_text_color="#E4ECE5",
        button_secondary_text_color_dark="#E4ECE5",
        table_border_color="#26312B",
        table_border_color_dark="#26312B",
        table_even_background_fill="#0B100D",
        table_even_background_fill_dark="#0B100D",
        table_odd_background_fill="#101611",
        table_odd_background_fill_dark="#101611",
        table_text_color="#E7EEE8",
        table_text_color_dark="#E7EEE8",
        panel_background_fill="#0D130F",
        panel_background_fill_dark="#0D130F",
        panel_border_color="#26312B",
        panel_border_color_dark="#26312B",
    )


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --shandu-bg: #070a08;
  --shandu-panel: #101611;
  --shandu-panel-2: #141c16;
  --shandu-rail: #0d130f;
  --shandu-border: #26312b;
  --shandu-border-strong: #355241;
  --shandu-text: #f2f5ef;
  --shandu-muted: #9ba89e;
  --shandu-dim: #68756d;
  --shandu-green: #39d98a;
  --shandu-teal: #2dd4bf;
  --shandu-amber: #f5b74f;
  --shandu-red: #f66a6a;
  --shandu-section-gap: 14px;
  --shandu-panel-gap: 12px;
  --shandu-panel-padding: 14px;
}

* {
  box-sizing: border-box;
}

html,
body {
  overflow-x: hidden;
}

.gradio-container {
  background: var(--shandu-bg) !important;
  font-size: 14px !important;
}

.gradio-container main.contain {
  width: min(calc(100% - 72px), 1500px) !important;
  max-width: 1500px !important;
  margin: 0 auto !important;
}

.shandu-shell,
.gradio-container.shandu-shell {
  box-sizing: border-box;
  width: 100% !important;
  max-width: none;
  margin: 0;
  padding-left: 0 !important;
  padding-right: 0 !important;
  overflow-x: hidden;
}

.shandu-topbar {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 18px;
  align-items: end;
  padding: 20px 0 14px;
  border-bottom: 1px solid var(--shandu-border);
}

.shandu-brand-kicker,
.shandu-eyebrow,
.shandu-feed header span,
.shandu-status-pill,
.shandu-lane-card h3 {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.shandu-brand h1 {
  margin: 0;
  color: var(--shandu-text);
  font-size: 2.55rem;
  line-height: 0.95;
  letter-spacing: 0;
  font-weight: 800;
}

.shandu-brand p {
  max-width: 720px;
  margin: 10px 0 0;
  color: var(--shandu-muted);
  font-size: 0.92rem;
  line-height: 1.55;
}

.shandu-brand-kicker {
  display: inline-flex;
  align-items: center;
  gap: 9px;
  margin-bottom: 12px;
  color: var(--shandu-teal);
  font-size: 0.72rem;
  font-weight: 600;
}

.shandu-brand-kicker::before {
  content: '';
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: var(--shandu-green);
  box-shadow: 0 0 0 5px rgba(57, 217, 138, 0.1);
}

.shandu-header-metrics {
  display: grid;
  grid-template-columns: repeat(3, minmax(86px, 1fr));
  gap: 10px;
  min-width: 330px;
}

.shandu-header-metrics div {
  border: 1px solid var(--shandu-border);
  border-radius: 8px;
  padding: 10px;
  background: var(--shandu-rail);
}

.shandu-header-metrics span {
  display: block;
  color: var(--shandu-dim);
  font-size: 0.72rem;
}

.shandu-header-metrics b {
  display: block;
  margin-top: 6px;
  color: var(--shandu-text);
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.92rem;
  font-weight: 600;
}

.shandu-runner,
.shandu-run-panel {
  margin-top: 14px;
}

.shandu-run-panel,
.shandu-config-panel {
  min-width: 0;
  max-width: 100%;
  border: 1px solid var(--shandu-border);
  border-radius: 8px;
  background: var(--shandu-panel);
  padding: var(--shandu-panel-padding);
}

.shandu-run-panel textarea {
  min-height: 112px !important;
  font-size: 0.92rem !important;
  line-height: 1.48 !important;
}

.shandu-run-actions {
  align-items: end;
}

.shandu-run-actions button {
  min-height: 42px !important;
  font-weight: 800 !important;
  cursor: pointer !important;
}

.shandu-config-panel {
  margin-top: 12px;
  background: var(--shandu-rail);
}

.shandu-run-panel,
.shandu-config-panel {
  width: 100%;
}

.shandu-config-panel .wrap {
  gap: 10px !important;
}

.shandu-html-reset,
.shandu-html-reset > .wrap,
.shandu-html-reset .html-container {
  border: 0 !important;
  background: transparent !important;
  padding: 0 !important;
  box-shadow: none !important;
}

.shandu-section-title {
  margin: 0;
  color: var(--shandu-text);
  font-size: 1.22rem;
  line-height: 1.15;
  letter-spacing: 0;
  font-weight: 800;
}

.shandu-section-bar,
.shandu-section-bar > .wrap {
  min-height: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
  gap: 12px !important;
  align-items: center !important;
}

.shandu-section-bar > .wrap {
  justify-content: space-between !important;
}

.shandu-download-action {
  flex: 0 0 auto !important;
  width: auto !important;
  margin-left: auto !important;
}

.shandu-save-note {
  min-height: 32px;
}

.shandu-examples {
  margin-top: 12px;
}

.shandu-status-panel {
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(320px, 0.85fr);
  gap: 14px;
  margin: var(--shandu-section-gap) 0 0;
  padding: var(--shandu-panel-padding);
  border: 1px solid var(--shandu-border-strong);
  border-radius: 8px;
  background: linear-gradient(135deg, rgba(57, 217, 138, 0.11), rgba(16, 22, 17, 0.96) 42%);
  max-width: 100%;
}

.shandu-status-panel h2 {
  margin: 10px 0 6px;
  color: var(--shandu-text);
  font-size: 1.18rem;
  letter-spacing: 0;
}

.shandu-status-panel p {
  margin: 0;
  color: var(--shandu-muted);
  line-height: 1.55;
}

.shandu-status-panel dl,
.shandu-lane-card dl {
  display: grid;
  gap: 10px;
  margin: 0;
}

.shandu-status-panel dl {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.shandu-status-panel dt,
.shandu-lane-card dt {
  color: var(--shandu-dim);
  font-size: 0.72rem;
}

.shandu-status-panel dd,
.shandu-lane-card dd {
  margin: 2px 0 0;
  color: var(--shandu-text);
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.94rem;
  overflow-wrap: anywhere;
}

.shandu-status-pill {
  display: inline-flex;
  width: fit-content;
  padding: 5px 9px;
  border-radius: 999px;
  color: #07100b;
  background: var(--shandu-green);
  font-size: 0.7rem;
  font-weight: 700;
}

.shandu-status-pill.is-error {
  background: var(--shandu-red);
}

.shandu-status-pill.is-complete {
  background: var(--shandu-teal);
}

.shandu-status-errors {
  grid-column: 1 / -1;
  margin: 0;
  padding: 12px 16px 12px 30px;
  border: 1px solid rgba(246, 106, 106, 0.35);
  border-radius: 8px;
  color: #ffd8d8;
  background: rgba(246, 106, 106, 0.1);
}

.shandu-lane-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 12px;
  margin: var(--shandu-section-gap) 0 0;
}

.shandu-lane-card {
  min-height: 132px;
  min-width: 0;
  border: 1px solid var(--shandu-border);
  border-radius: 8px;
  background: var(--shandu-panel);
  padding: 12px;
}

.shandu-lane-card header {
  display: flex;
  align-items: center;
  gap: 9px;
  margin-bottom: 14px;
}

.shandu-lane-card header span {
  width: 8px;
  height: 28px;
  border-radius: 999px;
  background: var(--shandu-green);
}

.shandu-lane-card h3 {
  margin: 0;
  color: var(--shandu-muted);
  font-size: 0.72rem;
  font-weight: 700;
}

.lane-search header span {
  background: var(--shandu-teal);
}

.lane-scrape header span {
  background: var(--shandu-amber);
}

.lane-cite header span {
  background: var(--shandu-red);
}

.shandu-main-stack {
  margin-top: var(--shandu-section-gap);
  gap: var(--shandu-section-gap) !important;
}

.shandu-main-stack > .wrap {
  gap: var(--shandu-section-gap) !important;
}

.shandu-report-panel,
.shandu-telemetry-panel {
  min-width: 0;
  max-width: 100%;
}

.shandu-report-panel,
.shandu-telemetry-panel {
  border: 1px solid var(--shandu-border);
  border-radius: 8px;
  background: #0c120e;
  padding: var(--shandu-panel-padding);
}

.shandu-telemetry-panel {
  background: var(--shandu-panel);
}

.shandu-report-panel > .wrap,
.shandu-telemetry-panel > .wrap {
  gap: var(--shandu-panel-gap) !important;
}

.shandu-report-copy,
.shandu-report-copy > .wrap,
.shandu-report-copy .prose,
.shandu-report-copy .markdown {
  border: 0 !important;
  background: transparent !important;
  padding: 0 !important;
  box-shadow: none !important;
}

.shandu-report-copy p {
  margin: 0 !important;
}

.shandu-report-copy .markdown h2,
.shandu-report-copy .prose h2 {
  margin: 22px 0 10px !important;
  color: var(--shandu-text) !important;
  font-size: 1.1rem !important;
}

.shandu-report-copy .markdown ul,
.shandu-report-copy .prose ul {
  display: grid;
  gap: 8px;
  margin: 10px 0 0 !important;
  padding: 0 !important;
  list-style: none;
}

.shandu-report-copy .markdown li,
.shandu-report-copy .prose li {
  margin: 0 !important;
  padding: 9px 11px;
  border: 1px solid var(--shandu-border);
  border-radius: 8px;
  background: var(--shandu-rail);
  color: var(--shandu-muted);
  line-height: 1.45;
  overflow-wrap: anywhere;
}

.shandu-report-copy .markdown li strong,
.shandu-report-copy .prose li strong {
  color: var(--shandu-text);
}

.shandu-report-copy .markdown a,
.shandu-report-copy .prose a {
  color: var(--shandu-teal) !important;
  text-decoration-color: rgba(45, 212, 191, 0.45) !important;
}

.shandu-citations-panel {
  margin-top: 0 !important;
  overflow: hidden !important;
  padding: 0 !important;
}

.shandu-citations-panel > .label-wrap,
.shandu-citations-panel button.label-wrap {
  min-height: 38px !important;
  padding: 8px 14px !important;
  border-bottom: 1px solid var(--shandu-border) !important;
  background: rgba(16, 22, 17, 0.96) !important;
  color: var(--shandu-text) !important;
  font-size: 0.92rem !important;
  font-weight: 700 !important;
}

.shandu-citations-panel > .wrap {
  padding: 12px 14px 14px !important;
}

.shandu-citations-panel > .wrap > .block {
  margin: 0 !important;
}

.shandu-telemetry-grid {
  align-items: stretch;
}

.shandu-task-panel {
  min-width: 0;
  min-height: 210px;
  border: 1px solid var(--shandu-border);
  border-radius: 8px;
  background: var(--shandu-rail);
  overflow: hidden;
  padding: 0;
}

.shandu-task-panel::before {
  content: 'Tasks';
  display: block;
  margin: 0;
  padding: 12px 14px;
  border-bottom: 1px solid var(--shandu-border);
  background: rgba(16, 22, 17, 0.96);
  color: var(--shandu-text);
  font-size: 0.92rem;
  font-weight: 700;
}

.shandu-task-panel > .wrap {
  padding: 0 !important;
}

.shandu-task-board {
  max-height: 360px;
  overflow: auto;
}

.shandu-task-board.empty p {
  margin: 0;
  padding: 18px 14px;
  color: var(--shandu-muted);
}

.shandu-task-board ol {
  display: grid;
  gap: 0;
  margin: 0;
  padding: 0;
  list-style: none;
}

.shandu-task-board li {
  display: grid;
  gap: 9px;
  padding: 12px 14px;
  border-bottom: 1px solid rgba(38, 49, 43, 0.72);
}

.shandu-task-board li:last-child {
  border-bottom: 0;
}

.shandu-task-board li:nth-child(even) {
  background: rgba(255, 255, 255, 0.015);
}

.task-row-head {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto auto;
  gap: 10px;
  align-items: center;
}

.shandu-task-board .task-id,
.shandu-task-board time,
.shandu-task-board .task-status {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
}

.shandu-task-board .task-id {
  color: var(--shandu-teal);
  font-size: 0.72rem;
  font-weight: 600;
  overflow-wrap: anywhere;
}

.shandu-task-board time {
  color: var(--shandu-dim);
  font-size: 0.72rem;
}

.shandu-task-board p {
  margin: 0;
  color: var(--shandu-text);
  font-size: 0.92rem;
  line-height: 1.42;
}

.shandu-task-board dl {
  display: grid;
  grid-template-columns: minmax(0, 1fr) repeat(3, minmax(54px, auto));
  gap: 8px;
  margin: 0;
}

.shandu-task-board dt {
  color: var(--shandu-dim);
  font-size: 0.68rem;
}

.shandu-task-board dd {
  margin: 2px 0 0;
  color: var(--shandu-muted);
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.72rem;
  overflow-wrap: anywhere;
}

.task-status {
  display: inline-flex;
  width: fit-content;
  max-width: 100%;
  padding: 3px 7px;
  border-radius: 999px;
  color: #07100b;
  background: var(--shandu-green);
  font-size: 0.66rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.task-status.is-idle {
  color: var(--shandu-muted);
  background: #172119;
}

.task-status.is-error {
  background: var(--shandu-red);
}

.shandu-report-panel .prose,
.shandu-report-panel .markdown {
  color: var(--shandu-text) !important;
}

.shandu-feed {
  min-height: 210px;
  max-height: 360px;
  overflow: auto;
  border: 1px solid var(--shandu-border);
  border-radius: 8px;
  background: var(--shandu-rail);
}

.shandu-feed header {
  position: sticky;
  top: 0;
  z-index: 2;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 14px;
  border-bottom: 1px solid var(--shandu-border);
  background: rgba(16, 22, 17, 0.96);
}

.shandu-feed h3 {
  margin: 0;
  color: var(--shandu-text);
  font-size: 0.92rem;
}

.shandu-feed header span {
  color: var(--shandu-dim);
  font-size: 0.68rem;
}

.shandu-feed.empty p {
  margin: 18px 14px;
  color: var(--shandu-muted);
}

.shandu-feed ol {
  display: grid;
  gap: 0;
  margin: 0;
  padding: 0;
  list-style: none;
}

.shandu-feed li {
  display: grid;
  grid-template-columns: 66px 86px minmax(0, 1fr);
  gap: 8px 10px;
  padding: 12px 14px;
  border-bottom: 1px solid rgba(38, 49, 43, 0.72);
}

.shandu-feed time,
.shandu-feed b,
.shandu-feed span,
.shandu-feed small {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.72rem;
}

.shandu-feed time {
  color: var(--shandu-dim);
}

.shandu-feed b {
  color: var(--shandu-green);
}

.shandu-feed span {
  color: var(--shandu-teal);
}

.shandu-feed p {
  grid-column: 3;
  margin: 0;
  color: var(--shandu-text);
  line-height: 1.45;
}

.shandu-feed small {
  grid-column: 3;
  color: var(--shandu-dim);
  overflow-wrap: anywhere;
}

.shandu-tabs {
  margin-top: var(--shandu-section-gap);
}

.shandu-advanced {
  border: 1px solid var(--shandu-border);
  border-radius: 8px;
}

.shandu-report-panel .table-wrap,
.shandu-telemetry-panel .table-wrap,
.shandu-tabs .table-wrap {
  overflow-x: auto !important;
}

button,
.gr-button {
  transition: background-color 160ms ease, border-color 160ms ease, color 160ms ease, opacity 160ms ease !important;
}

button:focus-visible,
textarea:focus-visible,
input:focus-visible,
select:focus-visible {
  outline: 2px solid var(--shandu-green) !important;
  outline-offset: 2px !important;
}

@media (max-width: 980px) {
  .shandu-topbar,
  .shandu-status-panel {
    grid-template-columns: 1fr;
  }

  .shandu-header-metrics {
    min-width: 0;
  }

  .shandu-runner,
  .shandu-telemetry-grid {
    flex-direction: column !important;
  }

  .shandu-runner > *,
  .shandu-telemetry-grid > * {
    width: 100% !important;
    max-width: 100% !important;
    flex: 1 1 auto !important;
  }
}

@media (max-width: 640px) {
  .gradio-container main.contain {
    width: calc(100% - 24px) !important;
  }

  .shandu-topbar {
    padding-top: 18px;
  }

  .shandu-brand h1 {
    font-size: 2.2rem;
  }

  .shandu-header-metrics,
  .shandu-status-panel dl {
    grid-template-columns: 1fr;
  }

  .shandu-run-panel,
  .shandu-config-panel,
  .shandu-status-panel,
  .shandu-report-panel,
  .shandu-telemetry-panel,
  .shandu-lane-card {
    padding: 12px;
  }

  .shandu-run-panel textarea {
    min-height: 112px !important;
  }

  .shandu-status-panel {
    gap: 14px;
  }

  .shandu-feed li {
    grid-template-columns: 56px minmax(0, 1fr);
  }

  .shandu-feed b,
  .shandu-feed span,
  .shandu-feed p,
  .shandu-feed small {
    grid-column: 2;
  }

  .task-row-head {
    grid-template-columns: minmax(0, 1fr) auto;
  }

  .shandu-task-board time {
    grid-column: 1 / -1;
  }

  .shandu-task-board dl {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .shandu-task-board .task-query {
    grid-column: 1 / -1;
  }
}

@media (prefers-reduced-motion: reduce) {
  button,
  .gr-button {
    transition: none !important;
  }
}
"""
