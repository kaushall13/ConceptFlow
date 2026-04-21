"""
Pass 1: Structural Read - Extract rough concept inventory from full book text.
Uses chunked processing to work within Cerebras free-tier TPM limits.
"""

import time
from typing import Dict, List, Any

CHUNK_SIZE = 40_000  # chars per chunk (~10k tokens, safely under free-tier TPM)


def perform_pass1(clean_text: str, cerebras_client) -> Dict[str, Any]:
    """
    Perform Pass 1: Structural read to extract concept inventory.
    Splits the book into chunks, extracts concepts per chunk, then merges.

    Returns:
        Dictionary containing concepts, dependencies, assumptions, and anchors
    """
    chunks = _chunk_text(clean_text)
    total = len(chunks)
    print(f"  Book split into {total} chunks (~{CHUNK_SIZE//1000}k chars each)")

    partial_results = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  Processing chunk {i}/{total}...")
        partial = _extract_chunk(chunk, i, total, cerebras_client)
        partial_results.append(partial)
        if i < total:
            print(f"  Waiting 65s for TPM window reset...")
            time.sleep(65)

    if total == 1:
        result = partial_results[0]
    else:
        print(f"  Merging {total} partial concept inventories...")
        result = _merge_partials(partial_results, cerebras_client)

    # Volume check
    concept_count = len(result.get("concepts", []))
    if concept_count < 40:
        print(f"  [WARN] Only {concept_count} concepts — too coarse, splitting...")
        result = _adjust_concept_volume(result, "split", cerebras_client)
    elif concept_count > 250:
        print(f"  [WARN] {concept_count} concepts — too granular, merging...")
        result = _adjust_concept_volume(result, "merge", cerebras_client)
    else:
        print(f"  [PASS] Found {concept_count} concepts (within target range)")

    return result


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into chunks at paragraph boundaries."""
    chunks = []
    while len(text) > chunk_size:
        split_at = text.rfind('\n\n', 0, chunk_size)
        if split_at == -1:
            split_at = chunk_size
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        chunks.append(text)
    return chunks


def _extract_chunk(chunk: str, chunk_num: int, total_chunks: int, cerebras_client) -> Dict[str, Any]:
    """Extract concept inventory from a single chunk."""
    system_prompt = """You are a curriculum designer whose job is to teach a technical book to a working engineer in 15-minute sessions.

Your task is to read this PORTION of the book and extract every distinct technical concept you find.

CONCEPT DEFINITION:
A concept is an atomic, teachable idea — something explainable in isolation.

Good examples:
- "Flash attention uses tiling to avoid materializing the full NxN attention matrix in HBM"
- "Continuous batching reuses GPU memory slots across multiple requests"
- "KV cache stores computed key-value pairs to avoid recomputation"

Bad examples (topics, not concepts):
- "Attention mechanisms" (too broad)
- "GPU utilization" (a metric, not a concept)

EXTRACT FOUR THINGS from this portion:

1. CONCEPTS — name, one-line description, location (section/chapter), recurring (true/false)
2. EXPLICIT DEPENDENCY SIGNALS — every "recall that", "as we saw", "building on", "this requires" verbatim + location
3. IMPLICIT ASSUMPTIONS — terms used without definition
4. AUTHOR ANCHOR MOMENTS — crisp one-liners the author gives

ANTI-PATTERNS: Do NOT produce chapter summaries. Do NOT produce one concept per heading. Be atomic.

Respond with valid JSON only. No markdown, no preamble."""

    user_prompt = f"""This is part {chunk_num} of {total_chunks} of the book.

Extract all distinct technical concepts from THIS PORTION ONLY. Be thorough — it's better to include too many than to miss concepts.

{chunk}

Return valid JSON:
{{
  "concepts": [
    {{
      "name": "concept name",
      "description": "one-line description",
      "location": "Section X.Y or chapter reference",
      "recurring": true
    }}
  ],
  "explicit_dependencies": [
    {{
      "signal": "exact phrase",
      "location": "Section X.Y",
      "refers_to": "concept being referenced"
    }}
  ],
  "implicit_assumptions": [
    {{
      "term": "assumed term",
      "context": "where used",
      "location": "Section X.Y"
    }}
  ],
  "author_anchors": [
    {{
      "anchor": "author's crisp one-liner",
      "location": "Section X.Y"
    }}
  ]
}}"""

    return cerebras_client.generate_json(system_prompt, user_prompt, max_tokens=6000)


def _merge_partials(partials: List[Dict[str, Any]], cerebras_client) -> Dict[str, Any]:
    """Merge partial concept inventories from all chunks into one unified list."""
    import json

    all_concepts = []
    all_deps = []
    all_assumptions = []
    all_anchors = []

    for p in partials:
        all_concepts.extend(p.get("concepts", []))
        all_deps.extend(p.get("explicit_dependencies", []))
        all_assumptions.extend(p.get("implicit_assumptions", []))
        all_anchors.extend(p.get("author_anchors", []))

    print(f"  Total before merge: {len(all_concepts)} concepts from {len(partials)} chunks")

    system_prompt = """You are a curriculum designer merging concept inventories extracted from different portions of the same book.

Your task: deduplicate and unify the concept list into a single clean inventory.

Rules:
- If the same concept appears under different names, keep the most precise name and richer description
- If very similar concepts differ slightly (e.g. different locations), merge into one with the most informative description
- Do NOT add new concepts or change meanings
- Final target: 60-200 concepts total
- Keep all explicit_dependencies, implicit_assumptions, and author_anchors (deduplicate obvious duplicates)

Respond with valid JSON only."""

    user_prompt = f"""Here are concept inventories extracted from {len(partials)} portions of a technical book.

Merge these into a single unified, deduplicated inventory. Target 60-200 concepts.

ALL CONCEPTS ({len(all_concepts)} total before dedup):
{json.dumps(all_concepts, indent=2)[:30000]}

ALL EXPLICIT DEPENDENCIES:
{json.dumps(all_deps, indent=2)[:5000]}

ALL IMPLICIT ASSUMPTIONS:
{json.dumps(all_assumptions, indent=2)[:5000]}

ALL AUTHOR ANCHORS:
{json.dumps(all_anchors, indent=2)[:5000]}

Return the merged result in this JSON structure:
{{
  "concepts": [...],
  "explicit_dependencies": [...],
  "implicit_assumptions": [...],
  "author_anchors": [...]
}}"""

    print("  Waiting 65s before merge call...")
    time.sleep(65)
    return cerebras_client.generate_json(system_prompt, user_prompt, max_tokens=8000)


def _adjust_concept_volume(result: Dict[str, Any], adjustment: str, cerebras_client) -> Dict[str, Any]:
    """Adjust concept volume by splitting (if too coarse) or merging (if too granular)."""
    import json
    print(f"  Adjusting concepts: {adjustment}...")

    system_prompt = "You are a curriculum designer refining a concept inventory. Adjust granularity as instructed."

    if adjustment == "split":
        instruction = "Split each broad concept into 2-3 more atomic, teachable sub-concepts. Target 60-200 total."
    else:
        instruction = "Merge closely related concepts. Target 60-200 total."

    user_prompt = f"""Current concept inventory:

{json.dumps(result.get('concepts', []), indent=2)[:20000]}

{instruction}

Return only the concepts list in JSON: {{"concepts": [...]}}"""

    time.sleep(65)
    adjusted = cerebras_client.generate_json(system_prompt, user_prompt, max_tokens=6000)

    if "concepts" in adjusted:
        result["concepts"] = adjusted["concepts"]

    print(f"  [PASS] Adjusted to {len(result['concepts'])} concepts")
    return result
