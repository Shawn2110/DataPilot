# DataPilot

**Terminal-native AI copilot for data analysis and data science.**

One command (`datapilot`) launches an interactive TUI session where an AI agent guides you through your workflow — upload a CSV, pick a mode, and the agent reasons through every step, explaining its choices before running any code. Every step is saved as a Jupyter cell, auto-persisted to disk after generation, and exportable as a reproducible notebook.

---

## What it does

```
$ datapilot

  +----------------------------------------+
  |   DataPilot                            |
  |   AI Data Science & Analysis Copilot  |
  +----------------------------------------+

  What are you working on?
    [1] Data Analysis    (EDA, stats, visualizations, insights)
    [2] Data Science     (full ML pipeline + trained model)
  >
```

The agent:
- **Explains before it acts** — shows reasoning and code *before* executing.
- **Asks at every stage** — you approve, skip, edit, or ask for alternatives.
- **Saves continuously** — every cell is auto-saved. Crash, disconnect, power loss: no data lost.
- **Teaches as it works** — plain-language reasoning for *why* each choice was made.

---

## Features

- **Two modes** — Data Analysis (EDA-focused) or Data Science (full ML pipeline).
- **Pick your agent once** — choose Ollama / Claude / GPT-4 / Gemini on first run; never asked again unless you use `/agent`.
- **Local-first** — works fully offline with Ollama; no cloud required.
- **Auto-save on every cell** — atomic writes + parquet DataFrame snapshots. Resume any session with `datapilot`.
- **Edit cells mid-session** — `/edit 5` opens the cell in `$EDITOR`; re-run only that cell or cascade downstream.
- **Reproducible export** — `.ipynb`, `.py`, or `.html`. All re-runnable standalone.
- **Security-hardened** — API keys in OS keyring, AST-validated execution, schema-only cloud mode by default, PII masking, no telemetry.

---

## Installation

```bash
pip install datapilot-cli
```

**Requirements**
- Python 3.10+
- For local LLM: [Ollama](https://ollama.com/download) with a model pulled (e.g. `ollama pull llama3.1`)
- For cloud LLMs: an API key for Anthropic, OpenAI, or Google

---

## Quick start

### First launch (one-time setup)

```bash
$ datapilot
```

You'll be asked once to pick a default agent:

```
 Pick your default AI agent (asked once — change any time with /agent):

   [1] Ollama — Llama 3.1     (free, local, offline)
   [2] Anthropic — Claude
   [3] OpenAI — GPT-4
   [4] Google — Gemini
 >
```

Your selection is saved to `~/.datapilot/config.json`. API keys are stored in your OS keyring — **never** in plain text.

### Every launch after

```bash
$ datapilot
```

Skips the wizard entirely. You go straight to project creation or session resume.

### Change your agent

Inside a session:
```
 › /agent
```

Re-runs the selection wizard. The new choice becomes your default.

---

## Example session

```
 › /new customer_churn

 Which mode?
   [1] Data Analysis
   [2] Data Science
 › 2

 Path to your CSV:
 › ~/Downloads/customers.csv

 Agent: Loaded 10,000 rows x 18 columns. I see a 'Churn' column that looks
        like your target (binary 0/1, 26% positive). Shall I treat this as
        binary classification?

   [y] yes   [n] no, let me pick   [w] explain
 › y

 Agent: Age has 19% missing values. Distribution is right-skewed, so median
        imputation is robust. Here's the code:

        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        df['Age'] = imputer.fit_transform(df[['Age']])

   [y] run  [n] skip  [e] edit  [w] why  [a] alternatives
 › y

 ✓ Cell 3 saved. notebook.ipynb updated.
```

---

## Slash commands

| Command | What it does |
|---|---|
| `/help` | List all commands |
| `/mode` | Switch Data Analysis / Data Science |
| `/agent` | Change the AI provider + model |
| `/edit <n>` | Edit cell `n` in `$EDITOR` |
| `/rerun <n>` | Re-execute cell `n` |
| `/rerun-from <n>` | Re-execute cell `n` and every cell after it |
| `/revert <n>` | Discard cell `n` and all cells after it |
| `/cells` | List all cells and their status |
| `/diff <n>` | Show what an edited cell changed |
| `/why` | Ask the agent to re-explain the last decision |
| `/code` | Show the code for the last action |
| `/save` | Force-save (auto-save already runs after every cell) |
| `/export` | Export as `.ipynb`, `.py`, or `.html` |
| `/resume <name>` | Resume a saved project |
| `/quit` | Exit |

---

## Project layout on disk

Everything lives under `~/.datapilot/` (mode 0700):

```
~/.datapilot/
  config.json                # provider + preferences (no secrets)
  projects/
    customer_churn/
      session.json           # atomic-saved session state
      notebook.ipynb         # regenerated every cell
      data/
        raw.csv              # your uploaded dataset
        snapshots/           # parquet: DataFrame state before each cell
      models/                # saved .joblib models
      charts/                # generated PNGs
      export/                # /export outputs
```

---

## Architecture (at a glance)

```
  Terminal  ->  DataPilot TUI  ->  Session Controller
                     |                      |
                     v                      +-- Agent Runner (LangGraph) --> LLM
                Rich + prompt_toolkit       |
                                            +-- Tool Registry (the ML/EDA tools)
                                            |
                                            +-- Storage Manager (atomic save, snapshots)
                                            |
                                            +-- Code Executor (AST-validated, sandboxed)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for full details.

---

## Security

- **API keys** — stored in the OS keyring (macOS Keychain, Windows Credential Manager, Linux Secret Service). Never in `config.json`, never in logs.
- **Agent-generated code** — AST-validated against a deny-list; runs in a restricted namespace with a 60-second timeout.
- **Cloud-LLM privacy** — default mode sends schema + summary stats only. Raw data rows never leave your machine unless you explicitly opt in per session.
- **PII auto-masking** — columns matching `email`, `ssn`, `phone`, `credit_card`, etc. are masked before any cloud egress.
- **No telemetry** — DataPilot makes no outbound calls other than to your configured LLM provider.

Full security model: [agent.md](agent.md) §9 and [ARCHITECTURE.md](ARCHITECTURE.md) §6.

---

## Documentation

- [prd.md](prd.md) — Product requirements, user flows, success metrics
- [agent.md](agent.md) — Agent internals, provider support, tool registry, security
- [ARCHITECTURE.md](ARCHITECTURE.md) — Components, data model, file layout, threat model

---

## License

MIT

---

Built by [Shawn2110](https://github.com/Shawn2110).
