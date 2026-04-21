# Building ConceptFlow: A Dependency-Aware Curriculum Engine for Technical Books

Technical books are not articles. They build on themselves. A chapter on attention mechanisms assumes you absorbed the matrix multiplication chapter, which assumed the linear algebra chapter. Read linearly over weeks with gaps, the mental model never stabilizes. You finish the book and retain a fraction of it.

ConceptFlow is a pipeline that preprocesses a technical PDF into a library of 15-minute reading sessions: dependency-sorted, spaced-repetition-seeded, and served through a clean reader. The pipeline runs once per book. The output is a static session library you consume over days.

This post covers the architecture and the decisions behind it.

---

## The Core Problem

The naive approach to summarizing a book for daily reading is chunking: split by chapter, summarize each chunk, serve sequentially. This fails because:

- Chapter boundaries do not align with concept boundaries
- A concept introduced in chapter 3 may be explained most clearly in chapter 7
- Summaries lose the explanatory texture that makes a concept stick
- There is no mechanism to surface connections between concepts separated by many sessions

ConceptFlow treats the book as a concept graph, not a document. The pipeline extracts concepts, builds explicit dependency edges, topologically sorts them, assigns spaced repetition links across sessions, and generates prose that respects the resulting ordering.

---

## Pipeline Overview

```
PDF
 └─ Ingest          deterministic extraction and cleaning
 └─ Pass1           rough concept inventory (one LLM call, full book)
 └─ Cluster         group concepts into 5-8 thematic clusters
 └─ Pass2           deep per-concept extraction (parallel, one call per cluster)
 └─ GraphBuild      dependency graph + topological sort
 └─ IndexGen        session planning + REVISIT assignment + index generation
 └─ AwaitApproval   human review gate
 └─ SessionGen      session text generation (sequential, 3-pass per session)
 └─ Complete
```

State is written to disk after every stage. The pipeline is safe to interrupt and resume. All generation uses Cerebras (`qwen-3-235b`) for its 1M token context window. A local Ollama model (`llama3.2:3b`) handles structural evaluation only.

---

## Stage 1: Ingestion

PDF extraction is fully deterministic. PyMuPDF extracts raw text blocks with font metadata. Each block is classified into a structural type: `HEADER`, `SUBHEADER`, `BODY`, `CODE`, `PSEUDOCODE`, `TABLE`, `FIGURE-CAPTION`, `FOOTNOTE`, `PAGE-NUMBER`.

Several transformations happen before the text reaches the LLM:

- Dropped: figure captions, footnotes, page numbers, running headers, image-only blocks
- Preserved verbatim: `CODE`, `PSEUDOCODE`, `ALGORITHM` blocks with indentation intact
- Reconstructed: multi-column tables converted to prose comparison sentences
- Fixed: hyphenated line-break words rejoined, unicode ligatures normalized, garbled math approximated with ASCII and a `[MATH]` flag
- Multi-column layouts: columns detected and reordered left-to-right before linearization

The output is a single clean tagged string. Feeding the LLM structured type tags lets the generation prompts make decisions based on block type without re-parsing.

---

## Stage 2 and 3: Concept Extraction

Pass 1 sends the entire cleaned book in one call and asks the model to act as a curriculum designer, not a summarizer. The prompt enforces a specific concept definition: an atomic, teachable idea. "Flash attention uses tiling to avoid materializing the full NxN attention matrix in HBM" is a concept. "Attention mechanisms" is a topic.

The call collects four lists simultaneously: concepts with one-line descriptions, explicit dependency signals (verbatim "as we saw", "recall that" instances with locations), implicit assumptions (terms used without definition), and author anchor moments (crisp explanatory one-liners the author themselves wrote).

Pass 2 runs one call per cluster in parallel using `ThreadPoolExecutor`. Each call gets the full book text plus the target cluster's concept list. Sending the full book to every cluster call is intentional: a concept in the "Memory" cluster may have its clearest explanation in the "Scheduling" chapter. Restricting context to the cluster's chapters would miss this.

Each concept in the output carries: a canonical name, a 2-4 sentence description, primary and secondary passage references, dependency signals, implicit prerequisites, an author anchor quote, an enrichment flag (whether web search is mandatory during generation), and a weight (`light`, `medium`, `heavy`) used in session planning.

---

## Stage 4: The Dependency Graph

Concepts are merged across clusters and deduplicated via a single Cerebras call. Dependency edges are built from four sources in priority order:

1. Explicit dependency signals captured in Pass 2
2. Implicit prerequisites captured in Pass 2
3. Cross-theme flags set during cluster extraction
4. A final Cerebras call to infer remaining logical dependencies

Every edge stores the reason for its existence. This reason text is passed directly into session generation prompts as connective tissue.

Two structural checks run after graph construction:

**Circular dependency resolution.** If A depends on B and B depends on A, one of them is too coarse. A Cerebras call decomposes the offending concept into sub-concepts that break the cycle.

**Topological sort.** `graphlib.TopologicalSorter` from Python's standard library computes a valid linear ordering. Where multiple valid orderings exist, same-cluster concepts are kept contiguous to minimize cognitive context-switching between sessions.

---

## Stage 5: Session Planning and REVISIT Assignment

Concepts are grouped into sessions of 3-5 based on weight. One heavy concept gets at most two light companions. The target is 12-13 minutes of concept content; the structural elements (ANCHOR, REVISIT, TENSION) fill the remaining 2-3.

The REVISIT mechanism implements spaced repetition without a scheduling algorithm. For each session, the planner scans concepts from 3-7 sessions back and looks for a substantive new connection to the current session's concepts. The connection must be articulable in a single specific sentence. "Paged attention reappears here because continuous batching's slot management is what makes paged attention's block structure necessary" is a connection. "Both relate to memory" is not.

A concept can only be revisited once across the entire curriculum. If no meaningful connection exists in the 3-7 session window, the session gets no REVISIT rather than a fabricated one. The one-sentence connection reason is stored as a field and passed to the generator, which expands it into a paragraph.

---

## Stage 6: Session Generation

Each session is generated in three sequential passes. Parallelism is intentionally excluded: each session's ANCHOR opening paragraph must respond to the previous session's actual closing question. Parallel generation would make this impossible.

**Pass 1 (tension plan, ~80 tokens).** Given the session's concept list and the first concept of the next session, generate a draft closing question. This gives the body content a destination.

**Pass 2 (concepts body, ~2000-2400 words).** Generate all teaching content. The model knows the TENSION destination and builds toward it. REVISIT markers are embedded inline with `[REVISIT]...[/REVISIT]` tags at the point of natural connection, not appended at the end.

**Pass 3 (anchor and tension, ~200 words).** Given the previous session's exact closing question and the just-generated body, write the opening ANCHOR paragraph and refine the TENSION question.

Final session = Pass 3 ANCHOR + Pass 2 body (REVISIT parsed and placed) + Pass 3 TENSION.

The three-pass structure forces internal consistency. The body knows where it is going. The anchor knows what it is resolving. The tension is written after seeing the body it emerges from.

---

## Stage 7: Structural Evaluation

After each session, a local Ollama model runs four binary checks. Using a small local model here is deliberate: the evaluation scope is narrow enough that a 3B model is reliable if the prompts are constrained. All prompts are closed-ended: "Does condition X hold? Answer YES or NO, then one sentence of evidence."

**TENSION:** Is the closing question unanswerable from this session alone? Is it concrete enough to think about during the day? Not definitional, not rhetorical, not so broad it provides no handle?

**ANCHOR:** Does the opening paragraph resolve the previous session's tension? Does it transition within three sentences? Does it avoid re-explaining concepts from the previous session?

**COHERENCE:** Are all concepts introduced before they are used? Does the REVISIT paragraph avoid re-explaining core mechanics from scratch? Does the session stay within its approved concept list?

**LENGTH:** Word count check. No LLM involved. Target 1800-2400 words. Hard fail below 1500 or above 2600.

On failure, only the failing section is extracted and sent back for targeted regeneration with the specific failure reason. Full session regeneration is never triggered. Each check has a maximum of two retries. After two failures, the session is flagged for human review and the pipeline continues.

---

## The Reading Interface

The reader is a single HTML file served by a minimal Python HTTP server. No framework, no build step. The UI has two states: pipeline view (during generation) and feed view (after completion).

The reading view targets a 680px column, 19px body text, 1.87 line height. Sessions render as flowing prose with no visible section structure. There are no headers, no labels, no "ANCHOR:" prefixes. The structural elements are invisible bones.

Word-level bookmarking wraps every word in a span with a data attribute. Clicking a word sets a bookmark stored server-side in `progress.json`. On next open, the session scrolls to position the bookmarked word at 30% from the top.

Session completion triggers on "Mark complete" or on a 3-second dwell at scroll bottom. Completion data is stored per-session alongside the bookmark.

---

## Deployment

The pipeline and reader are designed to run locally. For static sharing, `build.py` strips evaluation metadata from `sessions.json` and copies all UI assets into `dist/`. The resulting directory deploys as a static site to any host. The live version is at [learn-everyday-delta.vercel.app](https://learn-everyday-delta.vercel.app).

---

## Numbers

A 300-page technical book produces roughly 60-90 sessions at an average of 2100 words each. Pipeline wall-clock time is 2-4 hours, dominated by API throughput. Session generation consumes approximately 500k Cerebras tokens for a book of that size.

Eval scores across six pipeline runs have converged around 8.5/10 on structural checks. The main remaining gap is TENSION quality: roughly 80% of sessions produce questions that are genuinely non-answerable from the current session and concrete enough to think about. Getting that to 95%+ is the primary focus of ongoing prompt iteration.

---

## Source

[github.com/kaushall13/ConceptFlow](https://github.com/kaushall13/ConceptFlow)
