# Technical Overview: How the Pipeline Works End to End

This document traces data from raw PDF to reading-ready sessions, stage by stage.

---

## The Big Picture

```
PDF
 │
 ▼
[Ingest] ──► tagged clean text
 │
 ▼
[Pass1] ──► rough concept inventory (135 concepts for 259-page book)
 │
 ▼
[Cluster] ──► 5–8 thematic clusters
 │
 ▼
[Pass2] ──► deep enriched concept objects (parallel, one call per cluster)
 │
 ▼
[GraphBuild] ──► directed dependency DAG + topologically sorted sequence
 │
 ▼
[IndexGen] ──► session plan + human-readable index
 │
 ▼
[AwaitApproval] ──► (human edits loop here) ──► approved index
 │
 ▼
[SessionGen] ──► 89 prose sessions (3 passes + evaluator per session)
 │
 ▼
state/pipeline_state.json + output/sessions.json
 │
 ▼
[UI / Feed] ──► reading interface
```

All intermediate outputs are written to `state/pipeline_state.json` keyed by stage name. The `orchestrator.py` reads this on startup to offer resume.

---

## Stage 1: Ingest (`pipeline/ingest.py`)

**Input:** PDF file path  
**Output:** single clean string with every block prefixed by a type tag

PyMuPDF (`fitz`) processes the PDF page by page. Each block is tagged:

```
HEADER | SUBHEADER | BODY | CODE | PSEUDOCODE | TABLE | FIGURE-CAPTION | FOOTNOTE | PAGE-NUMBER
```

Tagging uses font size, weight, indentation, and whitespace signals. Dropped entirely: figure captions, footnotes, page numbers, running headers. Code and pseudocode blocks are preserved exactly — no reflowing.

Post-processing fixes: hyphenated line-break words rejoined, unicode ligatures normalized (`ﬁ→fi`), garbled math approximated with ASCII + `[MATH]` flag, multi-column layouts reordered left-to-right.

**Key stat (InferenceEngineering.pdf):** 259 pages → 303,931 characters of tagged text.

**Saved to state:** `outputs["ingest"]` = tagged text string.

---

## Stage 2: Pass1 (`pipeline/pass1.py`)

**Input:** tagged clean text from Ingest  
**Output:** rough concept inventory

One large Cerebras call (or chunked into 9 calls for free-tier TPM limits). The model is framed as a curriculum designer, not a summarizer.

The model produces four running lists simultaneously:
1. **Concepts** — name, one-line description, originating section, whether recurring
2. **Explicit dependency signals** — verbatim "recall that", "as we saw" instances with location
3. **Implicit assumptions** — terms used without definition
4. **Author anchor moments** — the author's crispest one-liners per concept

Volume check: fewer than 40 concepts → too coarse; more than 250 → too granular. Expected range for a 300-page book: 60–200.

**Key stat:** 135 concepts extracted from InferenceEngineering.pdf.

**Saved to state:** `outputs["pass1"]` = raw LLM output string.

---

## Stage 3: Cluster (`pipeline/cluster.py`)

**Input:** concept names + one-liners from Pass1  
**Output:** 5–8 named thematic clusters

One Cerebras (or Ollama) call with just the concept list — not the full book text. Each concept assigned to exactly one cluster. Cluster names must be domain-meaningful ("Memory & Hardware", not "Group A").

Cluster descriptions generated here are injected into every downstream session generation prompt to orient the model within the book's intellectual architecture.

**Key stat:** 6 clusters for InferenceEngineering.pdf.

**Saved to state:** `outputs["cluster"]` = list of cluster objects.

---

## Stage 4: Pass2 (`pipeline/pass2.py`)

**Input:** full tagged book text + full concept list + per-cluster concept lists  
**Output:** richly annotated concept objects

One Cerebras call per cluster, run in parallel via `concurrent.futures.ThreadPoolExecutor`. Each call receives the entire book text — concepts in one cluster may be most deeply explained in chapters nominally about another cluster.

Per-concept output fields:
- `canonical_name`, `description` (2–4 sentences as the book presents it)
- `primary_passage` (best verbatim excerpt + chapter ref)
- `secondary_passages` (other locations)
- `dependency_signals` (explicit cross-references verbatim)
- `implicit_prerequisites` (undefined terms used in explanations)
- `author_anchor` (author's crispest one-liner)
- `enrichment_flag` (boolean — is book treatment thin/dated? triggers web search during generation)
- `concept_weight` (`light` / `medium` / `heavy` — determines how many concepts share a session)
- `cross_theme_deps` (dependencies on concepts in other clusters)

**Saved to state:** `outputs["pass2"]` = list of enriched concept objects.

---

## Stage 5: GraphBuild (`pipeline/graph.py`)

**Input:** all Pass2 concept objects  
**Output:** directed acyclic dependency graph + topologically sorted concept sequence

Three sub-steps:

1. **Deduplication** — one Cerebras call flags likely duplicates across cluster boundaries. Richer extraction kept, thinner discarded.
2. **DAG construction** — edges built from (in priority order): explicit dependency signals, implicit prerequisites, cross-theme flags, and one final Cerebras call to infer remaining logical dependencies. Each edge stores its reason.
3. **Cycle detection & resolution** — if A→B and B→A, a Cerebras call decomposes the coarser concept into sub-concepts that break the cycle.
4. **Topological sort** — `graphlib.TopologicalSorter` (Python 3.9+ stdlib). Where multiple valid orderings exist, same-cluster concepts are kept contiguous.

**Key stat:** 135 concepts, 223 dependency edges for InferenceEngineering.pdf.

**Saved to state:** `outputs["graph"]` = DAG metadata + sorted concept list.

---

## Stage 6: IndexGen (`pipeline/planner.py`)

**Input:** topologically sorted concept sequence  
**Output:** session plan + human-readable index

**Session grouping** — consecutive concepts packed into sessions using three heuristics (in priority order):
1. Concept weight: one heavy → 1–2 light companions max
2. Same-cluster contiguity where topological order permits
3. Never split a concept across sessions; target 12–13 min for concepts portion

**REVISIT assignment (spaced repetition layer)** — for each session, scans 3–7 sessions back for concepts with a specific, statable new connection to current session material. Connection must be one concrete sentence. If no meaningful connection exists, `REVISIT: none`. A concept can only be revisited once across the entire curriculum.

**Title generation** — one Ollama call per session title. Evocative and specific ("Why Your GPU Is Starving", not "GPU Utilization"). Titles name the insight or tension, not the topic.

**Index format** printed to terminal:
```
Session 01 — Why Closed Models Can't Match Open Source Flexibility   (~14 min)
  → Kubernetes: container orchestration for ML deployments
  → Closed Models: capabilities and restrictions
  → Open Models: flexibility and operational tradeoffs
  ↺ [none]

Session 02 — The Hidden Cost of Autoregression   (~13 min)
  → Autoregressive generation: token-by-token decoding
  → KV-Cache: why memory layout determines throughput
  ↺ Kubernetes [connection: ...]
```

**Key stat:** 89 sessions planned, 41/89 REVISIT assigned for InferenceEngineering.pdf.

**Saved to state:** `outputs["index_gen"]` = session plan objects + approved index text.

---

## Stage 7: AwaitApproval (interactive)

Human reviews the printed index. Edit instructions are accepted as free text and sent to Cerebras to revise the index (topological order strictly preserved). Loop continues until the user types `approve`. Session generation cannot begin until this gate is passed — enforced in `orchestrator.py`.

---

## Stage 8: SessionGen (`pipeline/generator.py` + `pipeline/evaluator.py`)

**Input:** approved session plan (89 sessions)  
**Output:** 89 prose sessions in `output/sessions.json`

Sessions are generated **sequentially** — each session receives the last 200 words of the previous session's actual output as context (not a plan excerpt), so the ANCHOR resolves something that was actually written.

### 3-Pass Generation Per Session

**Pass 1 — Tension plan** (`_generate_tension_plan`):
- Input: concept list + next session's first concept
- Output: draft 50-word TENSION question
- Purpose: gives Pass 2 a destination to build toward
- ~80 tokens out, temp=0.4

**Pass 2 — Concepts body** (`_generate_concepts_body`):
- Input: all concept metadata + cluster description + enrichment flags + draft TENSION
- Output: ~2000–2400 words of teaching prose
- Model embeds `[REVISIT]...[/REVISIT]` markers at the most natural transition point
- System prompt explicitly forbids writing the ANCHOR or TENSION
- temp=0.55

**Pass 3 — Anchor + Tension** (`_generate_anchor_and_tension`):
- Input: previous session's exact closing question + full concepts body + draft TENSION
- Output: 150-word ANCHOR paragraph + refined 50-word TENSION
- First sentence of ANCHOR must directly resolve the previous session's question
- temp=0.4

**Stitching** (`_stitch_session`): Pass3 ANCHOR + Pass2 body (REVISIT blocks parsed and placed) + Pass3 TENSION.

### 5-Check Evaluator (per session)

Runs via Ollama after stitching. All prompts are binary (YES/NO + one sentence of evidence):

| Check | What it tests |
|-------|--------------|
| **TENSION** | Is the closing question unanswerable from this session alone? Does it end the session? Is it concrete? |
| **ANCHOR** | Does the first sentence directly resolve the previous TENSION? Is the bridge ≤3 sentences? |
| **COHERENCE** | No forward references? REVISIT paragraph makes a new connection (not re-explanation)? |
| **LENGTH** | Word count 1800–2400. Hard fail <1500 or >2600. Pure code check, no LLM. |
| **REVISIT** | Deterministic marker search first; if found, Ollama verifies it's a new connection. |

**Retry logic:** on failure, the failing section is extracted and sent back to Cerebras with a one-sentence failure reason. Only the failing section is regenerated; it's then spliced back. Max 2 retries per check. After 2 failures: `REVIEW` flag set, pipeline continues.

**Saved to state:** `session_results[session_id]` = evaluator results per session.  
**Written to output:** `output/sessions.json` keyed by session ID.

---

## Stage 9: Complete

`current_stage` set to `Complete` in state. `orchestrator.py` stops. The UI's Feed View activates automatically (polling-based check in `app.js`).

---

## Reading Interface (`ui/`)

Served by Flask via `main.py --serve` on port 5000.

**Feed View:**
- Full session list with completion status (not started / in-progress dot / complete checkmark)
- Click any session to open regardless of completion state

**Reading View:**
- 680px centered column, serif body, 17–18px, line-height 1.75–1.85
- No visible section structure — prose flows without ANCHOR/REVISIT/TENSION labels
- Code blocks: syntax-highlighted, horizontally scrollable, plain-language explanation below with thin separator
- Session completes on "Mark complete →" click or 3-second scroll-bottom dwell
- Prev/Next navigation at bottom only

**Word-level bookmarking:**
- Every word wrapped in `<span data-word-index="N">`
- Click sets bookmark for that session; stored in `output/progress.json`
- On reopen: auto-scroll to bookmarked word at ~30% from viewport top
- One bookmark per session; completing a session clears its bookmark

---

## Data Flow Summary

```
PDF
 └─► ingest.py ──────────────────────────────────────────────────────┐
                                                                      │ tagged text
pass1.py ◄──────────────────────────────────────────────────────────┘
 └─► concept inventory
      └─► cluster.py ──► clusters
           └─► pass2.py (×6, parallel) ──► enriched concept objects
                └─► graph.py ──► DAG + sorted sequence
                     └─► planner.py ──► session plan + index
                          └─► [human approval]
                               └─► generator.py (×89, sequential)
                                    └─► evaluator.py (per session)
                                         └─► output/sessions.json
                                              └─► ui/ (reading interface)
```

State written at every `└─►` arrow. Any stage can be resumed independently.
