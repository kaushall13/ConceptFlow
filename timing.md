# Pipeline Timing

Observed timings for **InferenceEngineering.pdf** (259 pages, 303,931 chars extracted, 89 sessions).  
Model: Cerebras `qwen-3-235b-a22b-instruct-2507` (generation) + Ollama `qwen2.5:7b` (evaluation/clustering).

---

## Per-Stage Breakdown

| Stage | Wall-clock time | API calls | Notes |
|-------|-----------------|-----------|-------|
| **Ingest** | ~2–3 min | 0 | PyMuPDF page-by-page extraction. Pure CPU. |
| **Pass1** | ~8–12 min | 9 Cerebras | Full book text chunked into 9 calls (free-tier TPM limit). Each call ~1–2 min including rate-limit waits. |
| **Cluster** | ~1–2 min | 1 Cerebras | Concept names only, fast call. |
| **Pass2** | ~8–15 min | 6 Cerebras (parallel) | One call per cluster. 6 clusters run concurrently via `ThreadPoolExecutor`. Wall clock = slowest cluster call. |
| **GraphBuild** | ~5–8 min | 2–3 Cerebras | Deduplication call + dependency inference call + optional cycle-breaking call. Topo sort is instantaneous (stdlib `graphlib`). |
| **IndexGen** | ~15–25 min | 89 Ollama | Title generation batched via Ollama. ~15–20s per batch. Session grouping logic is deterministic. |
| **AwaitApproval** | Variable | 0 | Human review of the printed index. Auto-approve mode: ~0s. |
| **SessionGen** | ~3–6 hours | ~350–450 Cerebras + 89–150 Ollama | Sequential: 89 sessions × 3 Cerebras passes + 1 Ollama evaluation. Each session ~2–4 min at free-tier rate limits; retries and 60s rate-limit waits add variance. |

---

## SessionGen Breakdown (Per Session)

| Sub-step | Cerebras calls | Approx time |
|----------|---------------|-------------|
| Pass 1 — tension plan | 1 | ~15–30s |
| Pass 2 — concepts body | 1 (+ 1 if extension needed) | ~45–90s |
| Pass 3 — anchor + tension | 1 | ~20–40s |
| Evaluator (Ollama, 5 checks) | 0 Cerebras | ~20–60s |
| Retry (per failing check, up to 2×) | 1 Cerebras each | ~30–60s each |
| **Total per session (no retries)** | **3** | **~2–3 min** |
| **Total per session (with retries)** | **3–9** | **~3–8 min** |

With 60-second rate-limit waits on the Cerebras free tier, sessions that hit 429 errors add 1–2 minutes each. Across 89 sessions, this dominates total SessionGen time.

---

## Total Wall-Clock Time

| Scenario | Estimated total |
|----------|----------------|
| Fast path (no rate limiting, no retries) | ~45–60 min |
| Typical free-tier run (rate limiting, some retries) | **3–5 hours** |
| Actual InferenceEngineering.pdf run | **~4–6 hours across 5 pipeline sessions** (spread over multiple days due to 900k/day Cerebras quota) |

---

## Token Budget (InferenceEngineering.pdf)

| Stage | Approximate tokens |
|-------|-------------------|
| Pass1 (9 chunks) | ~120k–180k |
| Cluster | ~5k |
| Pass2 (6 clusters) | ~80k–120k |
| GraphBuild | ~20k–40k |
| SessionGen (89 sessions × 3 passes) | ~400k–550k |
| Evaluator retries (Cerebras-side) | ~30k–60k |
| **Total** | **~650k–950k tokens** |

The Cerebras free tier provides ~900k tokens/day per API key. With two keys and key rotation, a full pipeline run fits in one day. With one key, it typically requires 2 days.

---

## Scaling Estimates

| Book size | Sessions (est.) | SessionGen time (est.) | Total pipeline |
|-----------|----------------|----------------------|----------------|
| 100 pages | ~35 sessions | 1–2 hours | ~2–3 hours |
| 200 pages | ~60 sessions | 2–3.5 hours | ~3–4.5 hours |
| 300 pages | ~89 sessions | 3–5 hours | ~4–6 hours |
| 500 pages | ~140 sessions | 5–8 hours | ~6–9 hours (may need 2-day run) |

Ingest, Pass1, and Pass2 scale roughly linearly with page count. SessionGen scales with session count, which scales with concept count, which scales with page count but sub-linearly.
