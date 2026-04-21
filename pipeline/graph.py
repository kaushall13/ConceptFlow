"""
Concept Graph - Build dependency graph and topological sort of concepts
"""

import graphlib
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict


def build_concept_graph(pass2_output: Dict[str, Dict[str, Any]], ollama_client) -> Dict[str, Any]:
    """
    Build concept dependency graph from Pass 2 outputs.

    Args:
        pass2_output: Dictionary mapping cluster names to their extraction results
        ollama_client: OllamaAPI client for deduplication and dependency inference

    Returns:
        Dictionary with concepts, edges, topological order, and graph metadata
    """
    print("  Building concept dependency graph...")

    # Step 1: Merge all cluster extractions into unified concept list
    all_concepts = _merge_concepts(pass2_output)

    print(f"    Merged {len(all_concepts)} concepts from {len(pass2_output)} clusters")

    # Step 2: Deduplicate concepts
    all_concepts = _deduplicate_concepts(all_concepts, ollama_client)

    print(f"    After deduplication: {len(all_concepts)} concepts")

    # Step 3: Build dependency graph
    edges = _build_edges(all_concepts, ollama_client)

    print(f"    Built {len(edges)} dependency edges")

    # Step 4: Detect and resolve circular dependencies
    all_concepts, edges = _resolve_circular_dependencies(all_concepts, edges, ollama_client)

    # Step 5: Detect orphan concepts
    orphans = _detect_orphans(all_concepts, edges)

    if orphans:
        print(f"    [WARN] Found {len(orphans)} orphan concepts (no dependencies):")
        for orphan in orphans[:5]:  # Show first 5
            print(f"      - {orphan}")
        if len(orphans) > 5:
            print(f"      ... and {len(orphans) - 5} more")

    # Step 6: Topological sort
    sorted_concepts = _topological_sort(all_concepts, edges)

    print(f"    Topologically sorted: {len(sorted_concepts)} concepts")

    # Build result dictionary
    result = {
        "concepts": all_concepts,
        "edges": edges,
        "sorted_concepts": sorted_concepts,
        "orphans": orphans,
        "metadata": {
            "total_concepts": len(all_concepts),
            "total_edges": len(edges),
            "orphan_count": len(orphans)
        }
    }

    return result


def _merge_concepts(pass2_output: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge concepts from all clusters into unified list.

    Args:
        pass2_output: Dictionary mapping cluster names to extraction results

    Returns:
        List of all concepts with cluster information
    """
    all_concepts = []

    for cluster_name, cluster_data in pass2_output.items():
        for concept in cluster_data.get("concepts", []):
            # Add cluster information to concept
            concept_with_cluster = concept.copy()
            concept_with_cluster["cluster"] = cluster_name
            all_concepts.append(concept_with_cluster)

    return all_concepts


def _deduplicate_concepts(concepts: List[Dict[str, Any]], ollama_client) -> List[Dict[str, Any]]:
    """
    Deduplicate concepts using Ollama to identify likely duplicates.

    Args:
        concepts: List of concept dictionaries
        ollama_client: OllamaAPI client

    Returns:
        Deduplicated list of concepts
    """
    if len(concepts) < 100:  # Small list, skip deduplication
        return concepts

    print("    Checking for duplicate concepts...")

    # Prepare concept summary for deduplication
    concept_summary = "\n".join([
        f"{i}: {c['canonical_name']} - {c['description'][:100]}"
        for i, c in enumerate(concepts)
    ])

    system_prompt = """You are a curriculum designer identifying duplicate concepts in a concept inventory.

Your task is to flag likely duplicate concepts - different names or slight variations that refer to the same core idea.

DUPLICATE CRITERIA:
- Same core technical idea with different names (e.g., "self-attention" vs "self-attention mechanism")
- One is a subset or more specific version of another (e.g., "transformer encoder" vs "transformer")
- Two concepts explain the same fundamental mechanism

NOT DUPLICATES:
- Related but distinct concepts (e.g., "attention" vs "self-attention")
- One concept uses another (e.g., "attention mechanism" vs "attention head")
- Variations with meaningful differences (e.g., "batch size" vs "gradient accumulation")

For each duplicate pair, specify which one to keep (the richer extraction) and which to discard.

Respond with valid JSON only. No markdown, no additional text."""

    user_prompt = f"""Here is the concept inventory:

{concept_summary}

Flag duplicate pairs using this structure:
{{
  "duplicates": [
    {{
      "keep_index": 0,
      "discard_index": 5,
      "reason": "Both refer to the same mechanism; keep has richer description"
    }}
  ]
}}

If no duplicates found, return: {{"duplicates": []}}"""

    try:
        result = ollama_client.generate_json(user_prompt, system_prompt=system_prompt)

        duplicates = result.get("duplicates", [])

        if duplicates:
            print(f"    Found {len(duplicates)} duplicate pairs")

            # Create set of indices to discard
            discard_indices = {dup["discard_index"] for dup in duplicates}

            # Filter out duplicates
            deduplicated = [
                c for i, c in enumerate(concepts)
                if i not in discard_indices
            ]

            print(f"    Deduplicated to {len(deduplicated)} concepts")
            return deduplicated

    except Exception as e:
        print(f"    [WARN] Deduplication failed: {e}, proceeding without deduplication")

    return concepts


def _build_edges(concepts: List[Dict[str, Any]], ollama_client) -> List[Dict[str, Any]]:
    """
    Build dependency edges from concept data and inferred dependencies.

    Args:
        concepts: List of concept dictionaries
        ollama_client: OllamaAPI client for dependency inference

    Returns:
        List of edge dictionaries (from_concept -> to_concept)
    """
    edges = []

    # Create concept name to index mapping
    concept_map = {c["canonical_name"]: i for i, c in enumerate(concepts)}

    # Step 1: Extract explicit dependency signals from concepts
    for concept in concepts:
        concept_name = concept["canonical_name"]

        # From explicit dependency signals
        for dep_signal in concept.get("dependency_signals", []):
            ref_concept = dep_signal.get("refers_to", "")
            if ref_concept in concept_map:
                edges.append({
                    "from": ref_concept,
                    "to": concept_name,
                    "reason": f"Explicit: {dep_signal['signal']}",
                    "source": "explicit"
                })

        # From implicit prerequisites
        for prereq in concept.get("implicit_prerequisites", []):
            # Try to find matching concept
            matching_concept = _find_matching_concept(prereq, concept_map, concepts)
            if matching_concept:
                edges.append({
                    "from": matching_concept,
                    "to": concept_name,
                    "reason": f"Implicit: uses '{prereq}' without definition",
                    "source": "implicit"
                })

        # From cross-theme dependencies
        for cross_dep in concept.get("cross_theme_deps", []):
            ref_concept = cross_dep.get("concept", "")
            if ref_concept in concept_map:
                edges.append({
                    "from": ref_concept,
                    "to": concept_name,
                    "reason": f"Cross-theme: {cross_dep['relationship']}",
                    "source": "cross_theme"
                })

    # Step 2: Infer remaining dependencies using Ollama
    if len(concepts) > 50:  # Only for larger books
        print("    Inferring additional dependencies...")
        inferred_edges = _infer_dependencies(concepts, concept_map, ollama_client)
        edges.extend(inferred_edges)

    # Remove duplicates and self-loops
    unique_edges = {}
    for edge in edges:
        if edge["from"] != edge["to"]:  # No self-loops
            key = (edge["from"], edge["to"])
            if key not in unique_edges:
                unique_edges[key] = edge

    return list(unique_edges.values())


def _find_matching_concept(search_term: str, concept_map: Dict[str, int], concepts: List[Dict[str, Any]]) -> str:
    """
    Find a concept that matches the search term.

    Args:
        search_term: Term to search for
        concept_map: Mapping of concept names to indices
        concepts: List of concept dictionaries

    Returns:
        Matching concept name or empty string
    """
    # Exact match
    if search_term in concept_map:
        return search_term

    # Partial match
    for concept_name in concept_map:
        if search_term.lower() in concept_name.lower() or concept_name.lower() in search_term.lower():
            return concept_name

    return ""


def _infer_dependencies(concepts: List[Dict[str, Any]], concept_map: Dict[str, int], ollama_client) -> List[Dict[str, Any]]:
    """
    Infer logical dependencies not captured by explicit signals.

    Args:
        concepts: List of concept dictionaries
        concept_map: Mapping of concept names to indices
        ollama_client: OllamaAPI client

    Returns:
        List of inferred edge dictionaries
    """
    # Prepare concept summary
    concept_summary = "\n".join([
        f"- {c['canonical_name']}: {c['description']}"
        for c in concepts
    ])

    system_prompt = """You are a curriculum designer inferring logical dependencies between technical concepts.

Your task is to identify dependencies that are implied but not explicitly stated.

DEPENDENCY DEFINITION:
Edge A -> B means "B cannot be deeply understood without A first"

INFERENCE RULES:
- Look for hierarchical relationships (e.g., "attention" -> "self-attention")
- Look for prerequisite knowledge (e.g., "gradient descent" -> "Adam optimizer")
- Look for foundational concepts used to build more complex ones
- Look for "requires" or "assumes" relationships in descriptions

DO NOT INFER:
- Simply related concepts (e.g., two different types of attention)
- Concepts that mention each other but don't depend on each other
- Concepts in different domains unless clearly hierarchical

Focus on the most critical 20-30 dependencies that form the core learning path.

Respond with valid JSON only. No markdown, no additional text."""

    user_prompt = f"""Here are all the concepts:

{concept_summary}

Identify inferred dependencies using this structure:
{{
  "dependencies": [
    {{
      "from": "prerequisite concept",
      "to": "dependent concept",
      "reason": "one-sentence explanation of why this dependency exists"
    }}
  ]
}}

Identify the most critical dependencies (20-30 max). Focus on core learning path."""

    try:
        result = ollama_client.generate_json(user_prompt, system_prompt=system_prompt)

        inferred = []
        for dep in result.get("dependencies", []):
            from_concept = dep["from"]
            to_concept = dep["to"]

            if from_concept in concept_map and to_concept in concept_map:
                inferred.append({
                    "from": from_concept,
                    "to": to_concept,
                    "reason": f"Inferred: {dep['reason']}",
                    "source": "inferred"
                })

        if inferred:
            print(f"    Inferred {len(inferred)} dependencies")

        return inferred

    except Exception as e:
        print(f"    [WARN] Dependency inference failed: {e}")
        return []


def _resolve_circular_dependencies(concepts: List[Dict[str, Any]], edges: List[Dict[str, Any]], ollama_client) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Detect and resolve circular dependencies in the graph.

    Args:
        concepts: List of concept dictionaries
        edges: List of edge dictionaries
        ollama_client: OllamaAPI client

    Returns:
        Tuple of (updated concepts, updated edges)
    """
    # Build adjacency list for cycle detection
    adj = defaultdict(list)
    edge_dict = {}  # Map (from, to) -> edge

    for edge in edges:
        adj[edge["from"]].append(edge["to"])
        edge_dict[(edge["from"], edge["to"])] = edge

    # Detect cycles using DFS
    visited = set()
    rec_stack = set()
    cycles = []

    def dfs(node, path):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in adj[node]:
            if neighbor not in visited:
                dfs(neighbor, path.copy())
            elif neighbor in rec_stack:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)

        rec_stack.remove(node)

    for concept_name in {c["canonical_name"] for c in concepts}:
        if concept_name not in visited:
            dfs(concept_name, [])

    if not cycles:
        return concepts, edges

    print(f"    [WARN] Found {len(cycles)} circular dependencies")

    # For now, remove edges to break cycles (simple approach)
    # In a more sophisticated implementation, we would decompose concepts
    for cycle in cycles[:3]:  # Handle first 3 cycles
        print(f"    Cycle: {' -> '.join(cycle)}")

        # Remove the edge that creates the cycle (last edge in cycle)
        if len(cycle) >= 2:
            from_node = cycle[-2]
            to_node = cycle[-1]

            # Remove this edge
            edges = [e for e in edges if not (e["from"] == from_node and e["to"] == to_node)]
            print(f"      Removed edge: {from_node} -> {to_node} to break cycle")

    print(f"    Resolved cycles, remaining edges: {len(edges)}")

    return concepts, edges


def _detect_orphans(concepts: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[str]:
    """
    Detect orphan concepts (no incoming or outgoing edges).

    Args:
        concepts: List of concept dictionaries
        edges: List of edge dictionaries

    Returns:
        List of orphan concept names
    """
    concept_names = {c["canonical_name"] for c in concepts}

    # Build adjacency sets
    has_incoming = set()
    has_outgoing = set()

    for edge in edges:
        has_incoming.add(edge["to"])
        has_outgoing.add(edge["from"])

    # Find orphans (no incoming AND no outgoing)
    orphans = [
        name for name in concept_names
        if name not in has_incoming and name not in has_outgoing
    ]

    return orphans


def _topological_sort(concepts: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[str]:
    """
    Perform topological sort on concepts respecting dependency edges.

    Args:
        concepts: List of concept dictionaries
        edges: List of edge dictionaries

    Returns:
        Topologically sorted list of concept names
    """
    # Build graph for topological sort
    ts = graphlib.TopologicalSorter()

    # Add all nodes
    for concept in concepts:
        concept_name = concept["canonical_name"]
        ts.add(concept_name)

    # Add edges
    for edge in edges:
        ts.add(edge["to"], edge["from"])  # Note: add() adds edge (node, *predecessors)

    # Get sorted order
    try:
        sorted_order = list(ts.static_order())

        # Group concepts by cluster where possible (keep same-cluster concepts contiguous)
        sorted_order = _group_by_cluster(sorted_order, concepts, edges)

        return sorted_order

    except graphlib.CycleError as e:
        print(f"    [WARN] Topological sort failed due to cycle: {e}")
        # Fall back to original order
        return [c["canonical_name"] for c in concepts]


def _compute_tiers(sorted_order: List[str], dependency_graph: Dict[str, Set[str]]) -> Dict[str, int]:
    """
    Compute topological tier for each concept.

    A concept's tier is the length of the longest path from any root
    (no-prerequisite concept) to that concept. Tier 0 = no prerequisites.

    Args:
        sorted_order: Topologically sorted concept names (prerequisites first)
        dependency_graph: Maps concept name -> set of its prerequisites

    Returns:
        Dictionary mapping concept name to its tier number
    """
    tiers: Dict[str, int] = {}
    for concept in sorted_order:
        prereqs = dependency_graph.get(concept, set())
        if not prereqs:
            tiers[concept] = 0
        else:
            # Tier = max tier of known prerequisites + 1
            # Prerequisites that don't appear in tiers yet are treated as tier 0
            tiers[concept] = max((tiers.get(p, 0) for p in prereqs), default=0) + 1
    return tiers


def _validate_topological_order(ordered_concepts: List[str], dependency_graph: Dict[str, Set[str]]) -> List[Tuple[str, Set[str]]]:
    """Verify no concept appears before its prerequisites."""
    seen: Set[str] = set()
    violations: List[Tuple[str, Set[str]]] = []
    for concept in ordered_concepts:
        prereqs = dependency_graph.get(concept, set())
        missing = prereqs - seen
        if missing:
            violations.append((concept, missing))
        seen.add(concept)
    return violations


def _group_by_cluster(sorted_order: List[str], concepts: List[Dict[str, Any]], edges: List[Dict[str, Any]] = None) -> List[str]:
    """
    Group concepts by cluster while respecting topological order.

    # Tier-preserving cluster grouping — maintains dependency order while keeping related concepts together

    Args:
        sorted_order: Topologically sorted concept names
        concepts: List of concept dictionaries with cluster information
        edges: List of edge dictionaries (used to build dependency graph for tier computation)

    Returns:
        Sorted order with same-cluster concepts grouped within each topological tier
    """
    if edges is None:
        edges = []

    # Degenerate case: no edges, fall back to original topological order
    if not edges:
        return sorted_order

    # Build prerequisite map: concept -> set of its direct prerequisites
    dependency_graph: Dict[str, Set[str]] = defaultdict(set)
    for edge in edges:
        dependency_graph[edge["to"]].add(edge["from"])

    # Compute tiers using the topological order as a stable base
    tiers = _compute_tiers(sorted_order, dependency_graph)

    # Create concept-to-cluster mapping
    concept_to_cluster = {c["canonical_name"]: c.get("cluster", "unknown") for c in concepts}

    # Group concepts by tier, preserving the intra-tier order from sorted_order
    tier_to_concepts: Dict[int, List[str]] = defaultdict(list)
    for concept in sorted_order:
        tier = tiers.get(concept, 0)
        tier_to_concepts[tier].append(concept)

    # Within each tier, reorder by cluster name (stable sort preserves relative order
    # for concepts that share a cluster, keeping the topo-sort order intact within a cluster)
    reordered: List[str] = []
    for tier_num in sorted(tier_to_concepts.keys()):
        tier_concepts = tier_to_concepts[tier_num]
        # Stable sort by cluster name — concepts in the same cluster end up contiguous
        tier_concepts_sorted = sorted(tier_concepts, key=lambda c: concept_to_cluster.get(c, "unknown"))
        reordered.extend(tier_concepts_sorted)

    # Validate that the reordering preserves the dependency guarantee
    violations = _validate_topological_order(reordered, dependency_graph)
    if violations:
        print(f"    [WARN] Tier-preserving cluster grouping introduced {len(violations)} topological violation(s):")
        for concept, missing_prereqs in violations[:5]:
            print(f"      - '{concept}' appears before prerequisites: {missing_prereqs}")
        if len(violations) > 5:
            print(f"      ... and {len(violations) - 5} more")
        print("    [WARN] Falling back to original topological order to preserve dependency guarantee.")
        return sorted_order

    return reordered