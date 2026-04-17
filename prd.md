# DataPilot — Product Requirements Document

## 1. Overview

**DataPilot** is a terminal-native AI copilot for data analysis and data science. It launches with a single command (`datapilot`) and guides the user through an end-to-end analytical workflow — from raw CSV to a fully reproducible Jupyter notebook — with an AI agent that *explains* every step before executing it.

It is modelled after the Claude Code UX: one command starts the session, a one-time setup picks the AI provider, and everything else happens inside an interactive TUI.

---

## 2. Problem Statement

- Data scientists and analysts spend significant time on boilerplate: profiling, preprocessing, trying model baselines.
- Beginners don't always understand *why* a given transformation or model is the right choice for their data.
- Jupyter notebooks are powerful but force the user to write every line themselves.
- AutoML tools train a model but don't explain their decisions — they are opaque.

DataPilot sits between "write it yourself" and "hand it to AutoML": the agent does the work **and** explains each step, producing a reproducible notebook the user can re-run, edit, or share.

---

## 3. Target Users

| Persona | Primary Need |
|---|---|
| Junior data scientist | Wants a teacher — explain *why* median imputation, *why* random forest here |
| Analyst (non-ML) | Wants quick EDA + insights without writing pandas from scratch |
| Senior data scientist | Wants a speed-up for boilerplate; freedom to override any step |
| Student | Wants to learn pandas / sklearn through guided, hands-on practice |

---

## 4. Goals

### Primary
- Launch to first agent response in under 5 seconds (local LLM).
- Complete a full analysis (upload → EDA → model) in under 10 minutes.
- Every generated step is auto-saved; zero data loss on crash or network drop.
- Exported notebook is runnable standalone, without DataPilot installed.

### Secondary
- Support 4+ LLM providers out of the box (Ollama local + 3 cloud).
- Work fully offline with Ollama.
- First-time setup completes in under 60 seconds.

---

## 5. Non-Goals

- Not a general-purpose notebook editor (Jupyter is better for that).
- Not replacing hand-written ML code — only the exploratory / boilerplate phase.
- No cloud-hosted service; runs entirely on the user's machine.
- No real-time multi-user collaboration.

---

## 6. User Flows

### 6.1 First-time setup (one-time agent selection)
1. User installs: `pip install datapilot-cli`.
2. Runs: `datapilot`.
3. Welcome panel is displayed.
4. **Agent-selection wizard runs exactly once**:
   - Provider: Ollama / Anthropic Claude / OpenAI GPT / Google Gemini.
   - Model (per provider).
   - API key (cloud only) — stored in the OS keyring, never in plain text.
5. Selection is written to `~/.datapilot/config.json` as the user's default.
6. **The wizard is never shown again** until the user explicitly runs `/agent`.

### 6.2 Returning user (every subsequent launch)
1. `datapilot` — reads the saved default, skips the selection wizard entirely.
2. User picks project type: **Data Analysis** or **Data Science**.
3. Provides CSV path.
4. Agent loop begins: propose → explain → approve → run → save.
5. User exits with `/quit` or Ctrl+D — session auto-saved to disk.

### 6.3 Change agent (opt-in)
- `/agent` re-runs the selection wizard.
- New choice overwrites the default. Persists across sessions.

### 6.4 Resume an interrupted session
1. `datapilot` detects an unfinished session and offers to resume.
2. DataFrame + cell history are restored from local snapshots.
3. Agent continues from the last saved cell.

### 6.5 Edit a prior cell mid-session
1. `/edit 5` opens cell 5 in `$EDITOR` (vim / nano / VS Code — user's default).
2. On save, DataPilot asks: *rerun only this cell*, *rerun from this cell forward*, *rerun + let the agent react to the change*, or *cancel*.
3. Downstream DataFrame state is restored from the parquet snapshot taken before cell 5 originally ran.

### 6.6 Export
1. `/export` at any point, or automatically offered at session end.
2. Formats: `.ipynb`, `.py`, `.html`, or all.
3. Files written under the project directory.

---

## 7. Functional Requirements

### FR1 — Project modes
- **Data Analysis**: EDA, statistics, visualizations, insight report. No ML model.
- **Data Science**: full ML pipeline — problem detection, profiling, EDA, preprocessing, feature engineering, training, evaluation, SHAP explainability.
- Mode is chosen per project. `/mode` switches it mid-session (with user confirmation).

### FR2 — Agent selection (persisted)
- Selection wizard shown **only on first run** or via explicit `/agent` command.
- Persisted to `~/.datapilot/config.json`.
- API keys stored in OS keyring; **never** written to the config file.
- `DATAPILOT_AGENT=ollama:llama3.1` environment variable can override the default for a single run without altering the persisted config (useful for scripting).

### FR3 — Interactive cells
Every pipeline step must:
- Show the code **before** executing it.
- Show plain-language reasoning for the choice.
- Prompt the user for: approve, skip, edit, explain-more, or show-alternatives.
- Be editable after execution (`/edit <n>`) and re-runnable (`/rerun <n>` or `/rerun-from <n>`).

### FR4 — Auto-save on every cell
After every generated cell, DataPilot must:
- Write `session.json` atomically (write → `fsync` → rename).
- Regenerate `notebook.ipynb`.
- Save a Parquet snapshot of the DataFrame state taken **before** the cell ran.
- Guarantee no data loss on crash, SIGKILL, power loss, or network drop.

### FR5 — Resume
- Detect incomplete sessions on startup.
- Restore DataFrame + model state from snapshots.
- Agent context primed with a compacted summary of prior cells.

### FR6 — Export
- `.ipynb` — Jupyter notebook (markdown + code + outputs).
- `.py` — plain Python script.
- `.html` — rendered report (Jinja2 + matplotlib).
- All formats re-runnable standalone without DataPilot installed.

### FR7 — Slash commands
Minimum set: `/help`, `/mode`, `/agent`, `/save`, `/resume`, `/export`, `/edit`, `/rerun`, `/rerun-from`, `/revert`, `/cells`, `/diff`, `/why`, `/code`, `/undo`, `/quit`.

---

## 8. Non-Functional Requirements

### NFR1 — Security  *(see also: §10 Security Model)*
- API keys in OS keyring; never plaintext on disk.
- `~/.datapilot/` directory is mode `0700`.
- Agent-generated code is AST-validated before execution; dangerous nodes blocked.
- User CSV paths validated against path traversal.
- `$EDITOR` invocation sanitized (no shell interpretation).
- Cloud-LLM sends default to "schema-only" — raw data rows are not sent unless the user opts in.
- PII-named columns are masked before any cloud egress.
- No telemetry; no outbound calls except to the configured LLM provider.

### NFR2 — Performance
- Cold start under 2 seconds.
- First agent response under 5 seconds (local Ollama, warm model).
- Per-cell save under 200 ms.

### NFR3 — Reliability
- Atomic file writes (tempfile + `fsync` + `rename`).
- Session integrity validated on load; corrupted sessions flagged, not silently loaded.
- Parquet snapshots use zstd compression.

### NFR4 — Portability
- Works on macOS, Linux, Windows 10+.
- Python 3.10+.
- No admin / root required.

### NFR5 — Accessibility
- Respects `NO_COLOR`.
- Works in 80×24 terminals.
- Fully keyboard-driven (no mouse required).

---

## 9. Success Metrics

- Time to first cell generated: < 10 seconds median.
- Session completion rate: > 80% (users who start complete a project).
- Export rate: > 60% of completed sessions export to `.ipynb`.
- Resume success rate: > 99% (no corrupted sessions).
- Zero security incidents related to generated-code execution in first 6 months.

---

## 10. Security Model (summary)

See [agent.md](agent.md) and [ARCHITECTURE.md](ARCHITECTURE.md) for full detail. At a glance:

| Surface | Control |
|---|---|
| API keys | OS keyring (Keychain / Credential Manager / libsecret); encrypted fallback if unavailable |
| Config file | `~/.datapilot/config.json` chmod 0600; no secrets |
| Project files | `~/.datapilot/projects/` chmod 0700 |
| Generated code | AST deny-list + restricted namespace + timeout |
| Cloud-LLM payloads | Schema-only by default; opt-in for full-data mode; PII-column auto-masking |
| Path handling | Traversal + symlink-escape validated |
| Editor spawn | `shlex.split`, no shell |
| Logs | API keys, raw data, and PII scrubbed |
| Telemetry | None |

---

## 11. Milestones

| Milestone | Scope |
|---|---|
| M1 | CLI entry, TUI shell, one-time agent wizard, config persistence |
| M2 | Session + atomic save, notebook export, 3-tool flow (profile, EDA, preprocess) |
| M3 | Full Data Science mode (8 tools) + Data Analysis mode |
| M4 | Mid-session cell editing, rerun cascade, crash recovery |
| M5 | Security hardening (AST validator, keyring, PII masking, schema-only default) |
| M6 | Public release on PyPI |

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| LLM generates unsafe code | AST validator + user approval before every execution |
| Sensitive user data leaks to cloud LLM | Default schema-only mode; PII-column masking; explicit opt-in for full-data |
| Session file corruption | Atomic writes + integrity check on load |
| Large datasets exhaust memory | Sample-first mode for files > 500 MB |
| API key leaked via logs or screen share | Keyring storage + log scrubbing + key never echoed |
| Prompt injection via CSV content | Data is delimited in tool observations; agent instructed to treat data as data |
