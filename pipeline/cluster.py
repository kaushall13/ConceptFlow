"""
Theme Clustering - Group concepts into thematic clusters
"""

from typing import Dict, List, Any


def cluster_concepts(concepts: List[Dict[str, Any]], ollama_client) -> Dict[str, Any]:
    """
    Group concepts into 5-8 thematic clusters.

    Args:
        concepts: List of concept dictionaries from Pass 1
        ollama_client: OllamaAPI client instance

    Returns:
        Dictionary containing clusters with names and descriptions
    """
    print("  Clustering concepts into themes...")

    # Prepare concept list for clustering (names + one-liners only)
    concept_summary = "\n".join([
        f"- {c['name']}: {c['description']}"
        for c in concepts
    ])

    system_prompt = """You are a curriculum designer organizing technical concepts into thematic domains.

Your task is to group the provided concepts into 5-8 thematic clusters representing the book's major intellectual domains.

CLUSTERING RULES:
- Each concept must belong to exactly one cluster
- Cluster names must be domain-meaningful (e.g., "Memory & Hardware", not "Cluster A")
- Flag any cluster with fewer than 5 or more than 30 concepts
- Concepts in a cluster should share deep conceptual connections, not just keywords

For each cluster, provide:
- name: Domain-meaningful name
- concepts: List of concept names in this cluster
- description: One-paragraph description explaining what this cluster covers and why these concepts belong together

CLUSTERING STRATEGY:
Think about the fundamental questions each concept answers:
- How does X work? (Implementation details)
- Why does X matter? (Performance, correctness)
- Where is X used? (Application domains)
- What are the trade-offs? (Design decisions)

Group concepts that answer similar questions or operate in the same conceptual space.

Respond with valid JSON only. No markdown, no additional text."""

    user_prompt = f"""Here are all the concepts extracted from the book:

{concept_summary}

Group these concepts into 5-8 thematic clusters following the rules above.

Ensure your response is valid JSON with this structure:
{{
  "clusters": [
    {{
      "name": "Domain-Meaningful Cluster Name",
      "concepts": ["concept1", "concept2", ...],
      "description": "One paragraph explaining the cluster's focus and why these concepts belong together."
    }}
  ]
}}

Remember: Each concept must be in exactly one cluster. Target 5-8 clusters total."""

    # Call Ollama API
    result = ollama_client.generate_json(user_prompt, system_prompt=system_prompt)

    # Validate result
    if "clusters" not in result:
        raise ValueError("Invalid clustering response: missing 'clusters' key")

    clusters = result["clusters"]

    # Validate clustering constraints
    cluster_count = len(clusters)

    if cluster_count < 5 or cluster_count > 8:
        print(f"  [WARN] {cluster_count} clusters found - outside target range (5-8)")
        result = _adjust_cluster_count(result, ollama_client)
        clusters = result["clusters"]

    # Validate each cluster
    for cluster in clusters:
        concept_count = len(cluster["concepts"])

        if concept_count < 5:
            print(f"  [WARN] Cluster '{cluster['name']}' has only {concept_count} concepts (minimum 5)")
        elif concept_count > 30:
            print(f"  [WARN] Cluster '{cluster['name']}' has {concept_count} concepts (maximum 30)")
        else:
            print(f"  [PASS] Cluster '{cluster['name']}': {concept_count} concepts")

    # Verify all concepts are assigned
    assigned_concepts = set()
    for cluster in clusters:
        assigned_concepts.update(cluster["concepts"])

    all_concepts = {c["name"] for c in concepts}
    missing_concepts = all_concepts - assigned_concepts

    if missing_concepts:
        print(f"  [WARN] {len(missing_concepts)} concepts not assigned to any cluster")
        # Assign missing concepts to a "Miscellaneous" cluster
        if clusters:
            # Add to first cluster or create new one
            for concept in missing_concepts:
                clusters[0]["concepts"].append(concept)
            print(f"  [PASS] Assigned missing concepts to '{clusters[0]['name']}'")

    print(f"  [PASS] Created {len(clusters)} thematic clusters")

    return result


def _adjust_cluster_count(result: Dict[str, Any], ollama_client) -> Dict[str, Any]:
    """
    Adjust number of clusters to fit target range (5-8).

    Args:
        result: Current clustering result
        ollama_client: OllamaAPI client

    Returns:
        Adjusted clustering result
    """
    print("  Adjusting cluster count to target range (5-8)...")

    clusters = result["clusters"]
    current_count = len(clusters)

    if current_count < 5:
        instruction = f"You have only {current_count} clusters. Merge some clusters to create exactly 5-8 clusters."
    else:  # current_count > 8
        instruction = f"You have {current_count} clusters. Split some clusters to create exactly 5-8 clusters."

    system_prompt = """You are a curriculum designer adjusting cluster granularity.

Your task is to adjust the number of clusters to exactly 5-8 while maintaining meaningful groupings."""

    user_prompt = f"""Here is the current clustering:

{result}

{instruction}

Return the same JSON structure with adjusted clusters.
Each concept must still belong to exactly one cluster.
Maintain meaningful domain connections."""

    adjusted_result = ollama_client.generate_json(user_prompt, system_prompt=system_prompt)

    new_count = len(adjusted_result["clusters"])
    print(f"  [PASS] Adjusted to {new_count} clusters")

    return adjusted_result