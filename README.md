# Micro-Learning Curriculum Builder

Transforms a technical PDF book into a structured feed of 15-minute reading sessions — sequentially ordered, dependency-sorted, and enriched with internet context. Built for fragmented daily learning in 5–15 minute windows.

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) running locally (for evaluator and clustering stages)
- A [Cerebras](https://cloud.cerebras.ai) API key (primary generation)
- A [Groq](https://console.groq.com) API key (optional — used as fallback for smaller tasks)

---

## Installation

```bash
# Navigate to the project directory
cd productive

# Create and activate a virtual environment
python -m venv env
env\Scripts\activate        # Windows
# source env/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

Pull the local evaluator model via Ollama:

```bash
ollama pull llama3.2:3b
```

---

## Configuration

All runtime configuration lives in `config.json` at the project root. **Never hardcode keys in source files.**

```json
{
  "cerebras_api_key": "your-key-here",
  "cerebras_model": "qwen-3-235b-a22b-instruct-2507",
  "groq_api_key": "",
  "groq_model": "llama-3.3-70b-versatile",
  "ollama_host": "http://localhost:11434",
  "ollama_model": "llama3.2:3b"
}
```

**Multiple Cerebras keys:** the pipeline supports up to 2 Cerebras API keys for automatic rotation when the daily token quota on one key is exhausted. Add a second key via the UI settings panel or directly in `config.json` as `"cerebras_api_key_2"`. Daily quota usage is tracked in `state/key_quota.json`.

**Groq:** optional. When configured, Groq (`llama-3.3-70b-versatile`) is used as a fallback for tasks where Cerebras is rate-limited. If no Groq key is set, the pipeline runs Cerebras-only.

If `cerebras_api_key` is empty on first run, the CLI prompts for it and writes it to `config.json` automatically. Keys are never logged or printed.

---

## Usage

### Run the pipeline (process a PDF)

```bash
python main.py --pipeline path/to/book.pdf
```

Runs all 9 stages: `Ingest → Pass1 → Cluster → Pass2 → GraphBuild → IndexGen → AwaitApproval → SessionGen → Complete`

- Progress is saved after every stage — safe to interrupt with `Ctrl+C`
- Resume an interrupted run by re-running the same command; it will offer to pick up where it stopped
- Typical wall-clock time: 2–4 hours for a ~300-page book (mostly API throughput)

### Approve the session index

After `IndexGen`, the pipeline pauses and prints the proposed curriculum. You can:

- Type `approve` to begin generation
- Type edit instructions to revise it (loops until you type `approve`)

### Start the reading interface

```bash
python main.py --serve
```

Opens `http://localhost:5000`.

- **Pipeline view** — stage progress, live session counter, index approval UI
- **Feed view** — full session list + reading interface (appears when pipeline completes)

### Export / import sessions

```bash
python main.py --export
python main.py --import curriculum_export_20260321_120000.json
```

### Reset all state

```bash
python main.py --reset
```

Prompts for confirmation, then wipes `state/pipeline_state.json`, `output/sessions.json`, and `output/progress.json`.

---

## Project Structure

```
productive/
├── main.py                     # CLI entrypoint (--pipeline, --serve, --export, --import, --reset)
├── config.json                 # API keys and model settings
├── requirements.txt
├── eval.py                     # Offline session quality evaluator
├── pipeline/
│   ├── ingest.py               # PDF extraction and cleaning (PyMuPDF, deterministic)
│   ├── pass1.py                # Concept inventory — one large Cerebras call
│   ├── cluster.py              # Theme clustering — Ollama
│   ├── pass2.py                # Deep concept extraction — parallel Cerebras calls per cluster
│   ├── graph.py                # Dependency graph + topological sort
│   ├── planner.py              # Session planning, REVISIT assignment, index generation
│   ├── generator.py            # 3-pass session text generation (sequential)
│   ├── evaluator.py            # Ollama-based structural evaluator (5 checks per session)
│   └── orchestrator.py         # Stage management, resume logic, progress display
├── api/
│   ├── cerebras.py             # Cerebras wrapper: retry, token logging, dual-key rotation, quota tracking
│   ├── groq_client.py          # Groq wrapper: fallback for rate-limited tasks
│   ├── ollama.py               # Ollama wrapper: binary evaluation calls
│   └── rate_limiter.py         # Sliding-window rate limiter (shared by all API clients)
├── state/
│   ├── manager.py              # StateManager: atomic JSON writes, stage output storage
│   ├── pipeline_state.json     # Auto-written after every stage
│   └── key_quota.json          # Daily token quota tracking per Cerebras key
├── utils/
│   ├── errors.py               # Custom exception hierarchy (PipelineError, APIError, etc.)
│   ├── pipeline_logger.py      # Structured logger with stage context
│   └── validators.py           # Input validators (PDF path, config, session count)
├── output/
│   ├── sessions.json           # All generated sessions (final product)
│   └── progress.json           # Per-session completion status and word-level bookmarks
├── tests/
│   └── *.py                    # Unit tests for all components
└── ui/
    ├── index.html              # Single-page app (pipeline + feed views)
    ├── style.css
    └── app.js
```

---

## How Session Generation Works

Sessions are generated using a **3-pass architecture** per session (sequential — never parallel):

1. **Pass 1 — Tension plan** (~80 tokens): Given the session's concept list and the next session's first concept, generate a draft closing question. This gives the body a destination to build toward.
2. **Pass 2 — Concepts body** (~2000–2400 words): Generate all teaching content. The model knows the TENSION destination and builds toward it. REVISIT markers are embedded inline with `[REVISIT]...[/REVISIT]` tags.
3. **Pass 3 — Anchor + Tension** (~150 + 50 words): Given the previous session's exact closing question and the just-generated body, write the opening ANCHOR paragraph and refine the TENSION question.

Final session = Pass 3 ANCHOR + Pass 2 body (with REVISIT parsed and placed) + Pass 3 TENSION.

After generation, 5 structural checks run via Ollama: TENSION, ANCHOR, COHERENCE, LENGTH, REVISIT. Failures trigger targeted section-level regeneration (not full session redo), up to 2 retries per check.

---

## Data Files

| File | Purpose |
|------|---------|
| `config.json` | API keys, model names, Ollama host |
| `state/pipeline_state.json` | Current stage + all stage outputs — auto-written by pipeline |
| `state/key_quota.json` | Daily token usage per Cerebras API key |
| `output/sessions.json` | All generated sessions — the final product |
| `output/progress.json` | Per-session completion status + word-level bookmark position |

The original PDF is never stored. Only the cleaned tagged text and all downstream outputs are persisted.

---

## Notes

- If Ollama is unreachable on startup, the pipeline warns and offers to continue without evaluation or wait.
- All API calls log token usage to stdout for debugging.
- API keys are never logged or printed anywhere.
- The Cerebras free tier has a ~900k token/day quota per key. For a 300-page book, SessionGen consumes ~500k tokens. With two keys, a full run fits in one day.
