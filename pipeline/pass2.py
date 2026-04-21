"""
Pass 2: Deep Extraction - Extract detailed information per concept cluster.
Runs sequentially (not parallel) to respect free-tier rate limits.
"""

import time
from typing import Dict, List, Any


def perform_pass2(clean_text: str,
                  all_concepts: List[Dict[str, Any]],
                  clusters: List[Dict[str, Any]],
                  cerebras_client) -> Dict[str, Dict[str, Any]]:
    """
    Perform Pass 2: Deep extraction per cluster, sequentially to respect rate limits.

    Returns:
        Dictionary mapping cluster names to their deep extraction results
    """
    print("  Performing deep extraction per cluster (sequential, rate-limit safe)...")

    global_concepts_summary = "\n".join([
        f"- {c['name']}: {c['description']}"
        for c in all_concepts
    ])

    extraction_results = {}
    total = len(clusters)

    for idx, cluster in enumerate(clusters, 1):
        cluster_name = cluster["name"]
        print(f"  [{idx}/{total}] Extracting cluster: {cluster_name}")

        try:
            result = _extract_cluster_concepts(
                global_concepts_summary,
                cluster,
                cerebras_client
            )
            extraction_results[cluster_name] = result
            concept_count = len(result.get('concepts', []))
            print(f"  [{idx}/{total}] Extracted {concept_count} concepts from '{cluster_name}'")
        except Exception as e:
            print(f"  [{idx}/{total}] Failed to extract '{cluster_name}': {e}")
            import traceback
            traceback.print_exc()
            extraction_results[cluster_name] = {"concepts": [], "error": str(e)}

    total_extracted = sum(len(r.get('concepts', [])) for r in extraction_results.values())
    print(f"  Total: deep extracted {total_extracted} concepts across {len(extraction_results)} clusters")

    return extraction_results


def _extract_cluster_concepts(global_concepts_summary: str,
                               cluster: Dict[str, Any],
                               cerebras_client) -> Dict[str, Any]:
    """Extract deep information for concepts in a single cluster."""

    cluster_concepts_list = "\n".join([
        f"- {name}"
        for name in cluster["concepts"]
    ])

    system_prompt = """You are a curriculum designer performing deep extraction for a cluster of related technical concepts.

Your task is to extract detailed information for each concept in the target cluster.

IMPORTANT CONTEXT:
- The full book text is provided - concepts may be explained in chapters nominally about other topics
- Use the global concept list to understand the full scope

For each concept extract:
1. canonical_name: Clearest, most precise name
2. description: 2-4 sentences as the book presents it (not loose paraphrase)
3. primary_passage: Single best verbatim/close-paraphrase excerpt with chapter ref
4. secondary_passages: List of other chapter/section references where elaborated
5. dependency_signals: Explicit "recall that", "as we saw", "building on" instances with location
6. implicit_prerequisites: Terms used without definition
7. author_anchor: Author's crispest one-liner for this concept
8. enrichment_flag: true if book treatment is thin or dated
9. concept_weight: "light" / "medium" / "heavy"
10. cross_theme_deps: Dependencies on concepts in other clusters with relationship type

Respond with valid JSON only."""

    user_prompt = f"""GLOBAL CONTEXT - All 135 concepts in this book (Inference Engineering):

{global_concepts_summary}

TARGET CLUSTER:
Name: {cluster['name']}
Description: {cluster.get('description', '')}
Concepts to extract:
{cluster_concepts_list}

Extract detailed information for each concept in the target cluster.
Use the global concept list and your knowledge of inference engineering to produce accurate, book-faithful extractions.
Focus on production systems — how these concepts manifest in real LLM serving infrastructure.

Respond with valid JSON:
{{
  "concepts": [
    {{
      "original_name": "concept name from cluster",
      "canonical_name": "clearest name",
      "description": "2-4 sentences as book presents it",
      "primary_passage": "best excerpt with reference",
      "secondary_passages": ["other location references"],
      "dependency_signals": [
        {{"signal": "exact phrase", "location": "Chapter X", "refers_to": "concept"}}
      ],
      "implicit_prerequisites": ["term"],
      "author_anchor": "author's crispest one-liner or empty string",
      "enrichment_flag": false,
      "concept_weight": "medium",
      "cross_theme_deps": [
        {{"concept": "name", "relationship": "requires"}}
      ]
    }}
  ]
}}"""

    result = cerebras_client.generate_json(system_prompt, user_prompt, max_tokens=12000)

    if "concepts" not in result:
        raise ValueError(f"Invalid Pass 2 response for cluster '{cluster['name']}': missing 'concepts' key")

    # Add placeholder for any missing concepts
    extracted_names = {c["original_name"] for c in result["concepts"]}
    missing_names = set(cluster["concepts"]) - extracted_names

    if missing_names:
        print(f"    Warning: Missing extractions for: {', '.join(missing_names)}")
        for name in missing_names:
            result["concepts"].append({
                "original_name": name,
                "canonical_name": name,
                "description": "Concept identified in book (extraction incomplete)",
                "primary_passage": "",
                "secondary_passages": [],
                "dependency_signals": [],
                "implicit_prerequisites": [],
                "author_anchor": "",
                "enrichment_flag": False,
                "concept_weight": "medium",
                "cross_theme_deps": []
            })

    return result
