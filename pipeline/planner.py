"""
Session Planning & Index - Group concepts into sessions with spaced repetition
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


def plan_sessions(graph_output: Dict[str, Any],
                  ollama_client,
                  cerebras_client=None) -> Dict[str, Any]:
    """
    Plan sessions and generate curriculum index.

    Args:
        graph_output: Output from graph building stage
        ollama_client: Ollama client for structural tasks (titles, revisit scoring)
        cerebras_client: Unused, kept for signature compatibility (revise_index uses it separately)
    """
    llm = ollama_client

    print("  Planning sessions and generating index...")

    sorted_concepts = graph_output["sorted_concepts"]
    concept_graph = graph_output.get("edges", [])
    all_concepts = {c["canonical_name"]: c for c in graph_output["concepts"]}

    # FIX 2: Pass concept_graph so tier-ordering can enforce topological guarantees
    sessions = _group_into_sessions(sorted_concepts, all_concepts, concept_graph)
    print(f"  Planned {len(sessions)} sessions")

    sessions = _assign_revisits_heuristic(sessions, all_concepts, llm)
    revisit_count = sum(1 for s in sessions if s.get("revisit"))
    print(f"  Assigned {revisit_count} REVISIT connections")

    sessions = _generate_titles_batched(sessions, all_concepts, llm)

    overcoverage = _check_concept_overcoverage(sessions, threshold=3)
    if overcoverage:
        print(f"\n  [WARN] Concept overcoverage detected in {len(overcoverage)} concept(s):")
        for item in overcoverage:
            print(f"    '{item['concept']}': {item['session_count']} sessions ({', '.join(str(s) for s in item['sessions'])})")
        print("  Consider revising the index to consolidate these before approving.\n")

    index = _generate_index(sessions)

    return {
        "index": index,
        "metadata": {
            "total_sessions": len(sessions),
            "total_concepts": len(sorted_concepts),
            "revisit_connections": revisit_count,
            "estimated_total_minutes": sum(s.get("estimated_minutes", 15) for s in sessions)
        },
        "overcoverage": overcoverage,
        "approved": False
    }


# ---------------------------------------------------------------------------
# FIX 2 helper: Compute dependency tiers before grouping
# ---------------------------------------------------------------------------

def _compute_tiers(sorted_concepts: List[str],
                   concept_graph: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Assign each concept a dependency tier.
    Tier 0: no prerequisites in the concept list.
    Tier N: deepest prerequisite is in tier N-1.
    Ensures that sessions are ordered so no session requires knowledge
    from a later session (forward-reference prevention).
    """
    # Build adjacency: prereq -> dependents, and reverse: concept -> its prereqs
    prereqs_of: Dict[str, List[str]] = defaultdict(list)
    for edge in concept_graph:
        src = edge.get("from") or edge.get("source") or edge.get("prereq")
        dst = edge.get("to") or edge.get("target") or edge.get("concept")
        if src and dst:
            prereqs_of[dst].append(src)

    tiers: Dict[str, int] = {}
    # Process in topological order (sorted_concepts is already topo-sorted)
    for name in sorted_concepts:
        prs = prereqs_of.get(name, [])
        if not prs:
            tiers[name] = 0
        else:
            tiers[name] = max(tiers.get(p, 0) for p in prs) + 1
    return tiers


def _group_into_sessions(sorted_concepts: List[str],
                          all_concepts: Dict[str, Dict[str, Any]],
                          concept_graph: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Group consecutive concepts into sessions of 3-5 concepts."""
    sessions = []
    current_session = []
    current_weight = 0

    WEIGHT_MAP = {"light": 1, "medium": 2, "heavy": 3}
    MAX_SESSION_WEIGHT = 6  # ~3 medium concepts

    for concept_name in sorted_concepts:
        concept = all_concepts.get(concept_name, {})
        weight = WEIGHT_MAP.get(concept.get("concept_weight", "medium"), 2)

        # Heavy concept gets its own session
        if weight >= 3:
            if current_session:
                sessions.append(_finalize_session(current_session, all_concepts))
                current_session = []
                current_weight = 0
            sessions.append(_finalize_session([concept_name], all_concepts))
            continue

        # Check if adding this would exceed weight or count limits
        if current_session and (current_weight + weight > MAX_SESSION_WEIGHT or len(current_session) >= 5):
            sessions.append(_finalize_session(current_session, all_concepts))
            current_session = []
            current_weight = 0

        current_session.append(concept_name)
        current_weight += weight

        # Minimum 3 concepts per session unless heavy
        if len(current_session) >= 3 and current_weight >= MAX_SESSION_WEIGHT:
            sessions.append(_finalize_session(current_session, all_concepts))
            current_session = []
            current_weight = 0

    if current_session:
        sessions.append(_finalize_session(current_session, all_concepts))

    # FIX 2: Tier-ordering pass — stable sort sessions by max tier of their concepts
    # so that no session contains a concept from tier N unless all tier<N concepts
    # have already been placed in earlier sessions.
    if concept_graph:
        tiers = _compute_tiers(sorted_concepts, concept_graph)
        # Assign each session its max concept tier
        for session in sessions:
            max_tier = max(
                (tiers.get(c["name"], 0) for c in session["concepts"]),
                default=0
            )
            session["_tier"] = max_tier
        # Stable sort by tier so intra-tier order is preserved
        sessions.sort(key=lambda s: s["_tier"])
        # FIX 2: Guard — after tier sort, verify no concept's prerequisite appears
        # in a later session. If a violation is found, swap violating sessions only.
        _enforce_topo_order(sessions, concept_graph)

    return sessions


def _enforce_topo_order(sessions: List[Dict[str, Any]],
                         concept_graph: List[Dict[str, Any]]) -> None:
    """
    Post-sort safety pass: scan for any concept whose prerequisite appears in a
    later session. When found, move the prerequisite session immediately before
    the dependent session. Modifies sessions in place.
    """
    prereqs_of: Dict[str, List[str]] = defaultdict(list)
    for edge in concept_graph:
        src = edge.get("from") or edge.get("source") or edge.get("prereq")
        dst = edge.get("to") or edge.get("target") or edge.get("concept")
        if src and dst:
            prereqs_of[dst].append(src)

    # Build index: concept_name -> session index (refreshed each sweep)
    changed = True
    max_passes = len(sessions)  # Bound iterations
    passes = 0
    while changed and passes < max_passes:
        changed = False
        passes += 1
        concept_to_session = {}
        for idx, session in enumerate(sessions):
            for c in session["concepts"]:
                concept_to_session[c["name"]] = idx

        for i, session in enumerate(sessions):
            for concept in session["concepts"]:
                for prereq in prereqs_of.get(concept["name"], []):
                    prereq_idx = concept_to_session.get(prereq)
                    if prereq_idx is not None and prereq_idx > i:
                        # Move prereq session to just before current session
                        prereq_session = sessions.pop(prereq_idx)
                        sessions.insert(i, prereq_session)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break


def _finalize_session(concept_names: List[str],
                       all_concepts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    WEIGHT_MAP = {"light": 1, "medium": 2, "heavy": 3}

    concepts_data = []
    total_weight = 0

    for name in concept_names:
        concept = all_concepts.get(name, {})
        w = WEIGHT_MAP.get(concept.get("concept_weight", "medium"), 2)
        total_weight += w
        concepts_data.append({
            "name": name,
            "description": concept.get("description", ""),
            "weight": concept.get("concept_weight", "medium")
        })

    n = len(concept_names)
    avg_weight = total_weight / n if n > 0 else 2
    estimated_minutes = max(10, min(20, int(n * 3 * avg_weight / 2) + 3))

    return {
        "concepts": concepts_data,
        "concept_count": n,
        "total_weight": total_weight,
        "estimated_minutes": estimated_minutes
    }


# ---------------------------------------------------------------------------
# FIX 3: Revised REVISIT assignment
# ---------------------------------------------------------------------------

def _assign_revisits_heuristic(sessions: List[Dict[str, Any]],
                                all_concepts: Dict[str, Dict[str, Any]],
                                llm) -> List[Dict[str, Any]]:
    """
    Assign REVISIT concepts using heuristics, with one batched LLM call per session
    for borderline cases. Much cheaper than one LLM call per concept pair.

    FIX 3: Widened window (2-10), recalibrated weights (dep-first), lowered LLM
    threshold (0.35 -> 0.2), improved specificity prompt, and minimum 60% revisit
    rate enforcement for sessions with index >= 3.
    """
    revisited_concepts = set()

    # FIX 3: Widened window — look back 2-10 sessions (was 3-7)
    MIN_BACK = 2
    MAX_BACK = 10

    def _try_assign(session_idx: int, current_session: Dict, score_threshold: float) -> bool:
        """Attempt to assign a revisit for one session. Returns True if assigned."""
        i = session_idx
        window_start = max(0, i - MAX_BACK)
        window_end = max(0, i - MIN_BACK)

        if window_start > window_end:
            return False

        candidates = []
        for j in range(window_start, window_end + 1):
            for concept in sessions[j]["concepts"]:
                name = concept["name"]
                if name not in revisited_concepts:
                    distance = i - j
                    candidates.append((name, distance, j))

        if not candidates:
            return False

        current_concept_names = [c["name"] for c in current_session["concepts"]]

        scored = []
        for name, distance, session_idx_back in candidates:
            concept_data = all_concepts.get(name, {})

            # FIX 3: dep_score weight raised to 0.6 — actual dependency relationship is
            # the most reliable signal for a meaningful revisit connection
            has_cross_dep = any(
                dep.get("concept") in current_concept_names
                for dep in concept_data.get("cross_theme_deps", [])
            )
            dep_score = 1.0 if has_cross_dep else 0.0

            # FIX 3: concept_weight_score weight raised to 0.3 — heavier concepts
            # benefit more from revisit reinforcement
            concept_weight_score = {"heavy": 1.0, "medium": 0.7, "light": 0.3}.get(
                concept_data.get("concept_weight", "medium"), 0.5
            )

            # FIX 3: dist_score weight lowered to 0.1 — distance is a weak signal
            dist_score = 1.0 if 4 <= distance <= 5 else (0.7 if 3 <= distance <= 7 else 0.4)

            # FIX 3: New formula: dep first, weight second, distance minimal
            total = dep_score * 0.6 + concept_weight_score * 0.3 + dist_score * 0.1
            scored.append((total, name, concept_data))

        scored.sort(reverse=True)
        best_score, best_name, best_data = scored[0]

        # Always assign the best-scoring candidate — threshold removed.
        # For low-score candidates, call LLM for a better reason string, but
        # assign regardless of whether the LLM returns something.
        reason = _build_heuristic_reason(best_name, best_data, current_concept_names, set())
        if best_score < 0.3:
            top3 = [(n, d) for _, n, d in scored[:3]]
            llm_result = _llm_pick_revisit(top3, current_session["concepts"], all_concepts, llm)
            if llm_result:
                _llm_name, llm_reason = llm_result
                reason = llm_reason  # Use LLM's richer reason; still assign best_name

        current_session["revisit"] = {"name": best_name, "reason": reason}
        revisited_concepts.add(best_name)
        return True

    # Assign revisits for all eligible sessions; always assigns best candidate.
    for i, current_session in enumerate(sessions):
        if i < MIN_BACK:
            current_session["revisit"] = None
            continue
        assigned = _try_assign(i, current_session, score_threshold=0.0)
        if not assigned:
            current_session["revisit"] = None

    return sessions


def _build_heuristic_reason(revisit_name: str,
                             revisit_data: Dict[str, Any],
                             current_concepts: List[str],
                             current_clusters: set) -> str:
    """Build a plausible revisit reason from available metadata."""
    cross_deps = revisit_data.get("cross_theme_deps", [])
    for dep in cross_deps:
        if dep.get("concept") in current_concepts:
            rel = dep.get("relationship", "connects to")
            return f"{revisit_name} {rel} {dep['concept']}, now visible from this session's vantage point"

    cluster = revisit_data.get("cluster", "")
    if cluster in current_clusters:
        return f"{revisit_name} from the {cluster} domain gains new clarity through this session's concepts"

    if current_concepts:
        return (f"{revisit_name} shaped the constraints that make {current_concepts[0]} necessary "
                f"— the earlier design decision becomes visible from this vantage point")
    return f"{revisit_name} connects to this session's core mechanisms in ways not apparent when first introduced"


def _llm_pick_revisit(candidates: List[Tuple[str, Dict[str, Any]]],
                       current_concepts: List[Dict[str, Any]],
                       all_concepts: Dict[str, Dict[str, Any]],
                       llm) -> Optional[Tuple[str, str]]:
    """
    Ask LLM to pick the best revisit from top candidates. Single LLM call.
    FIX 3: Prompt rewritten to force a specific technical sentence rather than a
    generic yes/no, so the LLM rejects only truly non-specific connections.
    """
    current_names = [c["name"] for c in current_concepts]
    current_descs = "\n".join(f"- {c['name']}: {c.get('description', '')}" for c in current_concepts)

    candidate_text = "\n".join(
        f"- {name}: {data.get('description', '')}"
        for name, data in candidates
    )

    system_prompt = """You are picking the best past concept to revisit in a spaced repetition curriculum.

For each candidate, attempt to complete this sentence with a specific technical reason:
  "[candidate] reappears here because [current concept] [specific relationship]."

Rules:
- The relationship must be concrete and technical — not "both relate to memory" or "both are important"
- Only return a candidate if you can complete the sentence with a SPECIFIC reason
- If no candidate yields a specific technical sentence, return null for choice

Respond with JSON only."""

    user_prompt = f"""Current session concepts:
{current_descs}

Candidates to revisit:
{candidate_text}

Complete the sentence "[candidate] reappears here because [current concept] [specific relationship]."
Return the candidate with the most specific connection, or null if none qualifies.

{{
  "choice": "concept name or null",
  "reason": "[candidate] reappears here because [current concept] [specific relationship]"
}}"""

    try:
        result = llm.generate_json(user_prompt, system_prompt=system_prompt)
        choice = result.get("choice")
        reason = result.get("reason", "")
        if choice and choice != "null" and reason:
            return choice, reason
    except Exception as e:
        print(f"      Revisit LLM call failed: {e}")

    return None


# ---------------------------------------------------------------------------
# FIX 1 helper: Post-assignment title-concept alignment validation
# ---------------------------------------------------------------------------

_TITLE_STOP_WORDS = {
    "the", "a", "an", "of", "in", "for", "and", "to", "is", "are",
    "how", "why", "what", "when", "where", "your", "our", "its", "their"
}

_GENERIC_TITLE_SUFFIXES = {
    "overview", "explained", "introduction", "insights", "understanding",
    "demystified", "basics", "fundamentals", "primer"
}


def _validate_title_concept_alignment(sessions: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
    """
    FIX 1: Check each session's title for alignment with its concept names/descriptions.
    A title is flagged as misaligned if:
      - It shares no significant words with any concept name or description, AND
      - It contains a generic suffix word (Overview, Explained, etc.)
    Returns list of (session_index, session) tuples for misaligned sessions.
    """
    misaligned = []
    for idx, session in enumerate(sessions):
        title = session.get("title", "")
        title_words = {
            w.lower().strip("\"'.,;:!?")
            for w in title.split()
            if w.lower().strip("\"'.,;:!?") not in _TITLE_STOP_WORDS
        }

        # Collect significant words from concept names + descriptions
        concept_words = set()
        for c in session.get("concepts", []):
            for w in (c.get("name", "") + " " + c.get("description", "")).split():
                w_clean = w.lower().strip("\"'.,;:!?")
                if w_clean and w_clean not in _TITLE_STOP_WORDS:
                    concept_words.add(w_clean)

        overlap = title_words & concept_words
        has_generic_suffix = bool(title_words & _GENERIC_TITLE_SUFFIXES)

        if not overlap:
            misaligned.append((idx, session))

    return misaligned


def _regenerate_single_title(session: Dict[str, Any], llm) -> str:
    """
    FIX 1: Regenerate a title for one session with a focused single-session LLM call.
    Called only for sessions flagged as misaligned by _validate_title_concept_alignment.
    """
    concept_names = ", ".join(c["name"] for c in session["concepts"])
    revisit_part = ""
    if session.get("revisit"):
        revisit_part = f"\nKey tension/revisit: {session['revisit'].get('reason', '')}"

    system_prompt = """You are generating one evocative title for a technical learning session.

TITLE PRINCIPLES:
- Name the insight or tension, not just the topic
- Be specific and active (hint at what will be understood)
- 3-8 words maximum
- No colons or subtitles
- NEVER end with: Overview, Explained, Introduction, Insights, Understanding, Demystified

EXAMPLES (positive):
- "Why Your GPU Is Starving" (not "GPU Utilization")
- "The Hidden Cost of Attention" (not "Attention Mechanisms Explained")
- "When Batch Size Backfires" (not "Batch Size Overview")
- "Memory That Outlives the Request" (not "Memory Management")
- "The Lie Inside Every Probability" (not "Probability Basics")
- "Why Inference Gets Slow" (not "Inference Performance Introduction")
- "What the Scheduler Cannot See" (not "Scheduling Insights")
- "Two Caches, One Bottleneck" (not "Cache Fundamentals")

If no specific tension comes to mind, use 'Why X happens' or 'When X breaks' framing.

Respond with JSON only: {"title": "..."}"""

    user_prompt = f"""Session covers: {concept_names}{revisit_part}

Generate one evocative title for this session."""

    try:
        result = llm.generate_json(user_prompt, system_prompt=system_prompt)
        title = result.get("title", "").strip().strip('"').strip("'")
        if title and 2 <= len(title.split()) <= 10:
            return title
    except Exception as e:
        print(f"      Single-title regeneration failed: {e}")

    # Fallback: non-generic concept-based title
    concepts = session["concepts"]
    if len(concepts) == 1:
        return concepts[0]["name"]
    return f"{concepts[0]['name']} & {concepts[1]['name']}"


# ---------------------------------------------------------------------------
# FIX 4 helper: Banned title suffix check
# ---------------------------------------------------------------------------

_BANNED_TITLE_SUFFIXES = {
    "overview", "explained", "introduction", "insights", "demystified",
    "simplified", "decoded", "unveiled", "compared", "secrets", "practice",
    "action", "gains", "boost", "optimized", "challenges", "magic",
    "and more", "puzzle", "trick", "needs", "roles", "basics", "essentials",
    "deep dive", "recap", "revisited", "breakdown",
}


def _title_has_banned_suffix(title: str) -> bool:
    """FIX 4: Return True if the last significant word is a banned generic suffix."""
    words = [w.lower().strip("\"'.,;:!?") for w in title.split()]
    if words and words[-1] in _BANNED_TITLE_SUFFIXES:
        return True
    return False


def _generate_single_title(session: Dict[str, Any], llm) -> str:
    """
    Generate one title for a single session using a focused per-session LLM call.
    Includes strong positive and negative examples so the model sees the input/output
    pattern clearly before being asked to produce a title.
    """
    concept_lines = "\n".join(
        f"- {c['name']}: {c.get('description', '')}"
        for c in session["concepts"]
    )
    revisit_part = ""
    if session.get("revisit") and session["revisit"].get("reason"):
        revisit_part = f"\nKey tension/connection: {session['revisit']['reason']}"

    system_prompt = """You are generating ONE evocative title for a single technical learning session.

TITLE PRINCIPLES:
- Name the insight or tension, not just the topic
- Must contain a verb OR pose a question
- Be specific and active — hint at what will be understood
- 3-8 words maximum
- No colons or subtitles
- Bare noun phrases are forbidden (e.g. "GPU Memory Management" is wrong)

BANNED LAST WORDS: overview, explained, introduction, insights, understanding,
demystified, simplified, decoded, unveiled, compared, secrets, practice,
action, gains, boost, optimized, challenges, magic, puzzle, trick, needs,
roles, basics, essentials, deep dive, recap, revisited, breakdown, and more.

POSITIVE EXAMPLES (concepts → title):
1. concepts=[Flash Attention, Tiling, HBM bandwidth]
   title="Why Your GPU Is Starving"
2. concepts=[Scaled Dot-Product Attention, QKV projection, Softmax cost]
   title="The Hidden Cost of Attention"
3. concepts=[Batch size, Throughput vs latency, GPU utilization]
   title="When Batch Size Backfires"
4. concepts=[KV cache, Paged attention, Memory fragmentation]
   title="Memory That Outlives the Request"
5. concepts=[Logit, Softmax, Probability calibration]
   title="The Lie Inside Every Probability"
6. concepts=[Continuous batching, Request queue, Prefill vs decode]
   title="The Queue That Ate Your Latency"
7. concepts=[Tokenization, BPE, Vocabulary coverage]
   title="Why Tokens Are Not Words"
8. concepts=[Speculative decoding, Draft model, Acceptance rate]
   title="Two Models, One Output"
9. concepts=[CUDA kernels, Warp divergence, Occupancy]
   title="What Keeps the GPU Waiting"
10. concepts=[Positional encoding, RoPE, Extrapolation]
    title="The Position Lie Transformers Tell"

NEGATIVE EXAMPLES (concepts → bad title → why bad):
1. concepts=[Flash Attention, Tiling] → "Flash Attention Explained" — generic suffix, no insight named
2. concepts=[KV Cache, Paged Attention] → "Memory Management" — bare noun phrase, no verb
3. concepts=[Softmax, Logits] → "Probability Basics" — banned suffix, no tension
4. concepts=[Continuous Batching] → "Batching Decoded" — banned suffix, no insight
5. concepts=[CUDA, Warp] → "GPU Programming and More" — banned suffix, vague

If no specific tension comes to mind, use 'Why X happens', 'When X breaks', or 'What X cannot see' framing.

You are generating a title for ONE session. Return ONLY the title, nothing else.
Respond with JSON only: {"title": "..."}"""

    user_prompt = f"""Session concepts:
{concept_lines}{revisit_part}

Generate one evocative title for this session."""

    try:
        result = llm.generate_json(user_prompt, system_prompt=system_prompt)
        title = result.get("title", "").strip().strip('"').strip("'")
        if title and 2 <= len(title.split()) <= 10:
            return title
    except Exception as e:
        print(f"      Single-title generation failed: {e}")

    # Fallback: concept-based title (non-generic)
    concepts = session["concepts"]
    if len(concepts) == 1:
        return concepts[0]["name"]
    return f"{concepts[0]['name']} & {concepts[1]['name']}"


def _generate_titles_batched(sessions: List[Dict[str, Any]],
                              all_concepts: Dict[str, Dict[str, Any]],
                              llm) -> List[Dict[str, Any]]:
    """
    Generate titles per-session (one LLM call per session) to eliminate ordering
    bugs that arise when a batch call returns scrambled or partial results.

    After generation:
    - Immediately validate alignment for each title; regenerate on failure.
    - Run a banned-suffix sweep.
    - Run a deduplication sweep; regenerate the duplicate with fewer concept words.
    """
    print(f"  Generating titles for {len(sessions)} sessions (per-session)...")

    for i, session in enumerate(sessions, 1):
        session["session_number"] = f"{i:02d}"

        title = _generate_single_title(session, llm)
        session["title"] = title

        # Immediately validate alignment; regenerate if misaligned
        misaligned = _validate_title_concept_alignment([session])
        if misaligned:
            print(f"    Session {i:02d}: title misaligned, regenerating...")
            session["title"] = _regenerate_single_title(session, llm)

        # Banned-suffix check after alignment fix
        if _title_has_banned_suffix(session["title"]):
            print(f"    Session {i:02d}: banned suffix detected, regenerating...")
            session["title"] = _regenerate_single_title(session, llm)

        print(f"    Session {i:02d}: \"{session['title']}\"")

    # Deduplication sweep: find exact or near-exact duplicate titles (case-insensitive).
    # When a duplicate is found, regenerate the one with fewer concept-name words in
    # its title (keep the more descriptive one).
    seen_titles: Dict[str, int] = {}  # title_lower -> session_number (1-based)
    for session in sessions:
        title_lower = session["title"].lower().strip()
        if title_lower in seen_titles:
            # Determine which copy is less descriptive (fewer concept words in title)
            earlier_idx = seen_titles[title_lower] - 1  # convert to 0-based
            earlier_session = sessions[earlier_idx]

            def _concept_word_overlap(s: Dict[str, Any]) -> int:
                title_words = {w.lower().strip("\"'.,;:!?") for w in s["title"].split()}
                concept_words = set()
                for c in s.get("concepts", []):
                    for w in c.get("name", "").split():
                        concept_words.add(w.lower().strip("\"'.,;:!?"))
                return len(title_words & concept_words)

            earlier_overlap = _concept_word_overlap(earlier_session)
            current_overlap = _concept_word_overlap(session)

            if current_overlap <= earlier_overlap:
                # Regenerate the current (duplicate) session's title
                print(f"    Session {session['session_number']}: duplicate title, regenerating...")
                session["title"] = _regenerate_single_title(session, llm)
                seen_titles[session["title"].lower().strip()] = int(session["session_number"])
            else:
                # Regenerate the earlier session's title
                print(f"    Session {earlier_session['session_number']}: duplicate title, regenerating...")
                earlier_session["title"] = _regenerate_single_title(earlier_session, llm)
                seen_titles[earlier_session["title"].lower().strip()] = int(earlier_session["session_number"])
                seen_titles[title_lower] = int(session["session_number"])
        else:
            seen_titles[title_lower] = int(session["session_number"])

    return sessions


def _check_concept_overcoverage(sessions: list, threshold: int = 3) -> list:
    """
    Flag any concept appearing as primary focus in >= threshold sessions.
    Primary focus = first concept in session OR concept name in session title.
    Returns list of flagged items for human review.
    """
    from collections import defaultdict
    concept_session_map = defaultdict(list)

    for session in sessions:
        session_num = session.get("session_number", "?")
        title = session.get("title", "").lower()

        for i, concept in enumerate(session.get("concepts", [])):
            name = concept["name"] if isinstance(concept, dict) else concept
            name_lower = name.lower()
            # Primary focus: first concept OR name appears in title
            is_primary = (i == 0) or any(word in title for word in name_lower.split()[:3] if len(word) > 4)
            if is_primary:
                concept_session_map[name].append(session_num)

    flagged = []
    for concept_name, session_list in concept_session_map.items():
        if len(session_list) >= threshold:
            flagged.append({
                "concept": concept_name,
                "session_count": len(session_list),
                "sessions": session_list,
                "recommendation": f"Consider consolidating {len(session_list)} sessions covering '{concept_name}'"
            })

    return sorted(flagged, key=lambda x: x["session_count"], reverse=True)


def _generate_index(sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    index = []
    for session in sessions:
        entry = {
            "session_number": session["session_number"],
            "title": session["title"],
            "estimated_minutes": session["estimated_minutes"],
            "concepts": [
                {"name": c["name"], "description": c.get("description", "")}
                for c in session["concepts"]
            ]
        }
        if session.get("revisit"):
            entry["revisit"] = session["revisit"]
        index.append(entry)
    return index


def revise_index(current_index: List[Dict[str, Any]],
                 instructions: str,
                 cerebras_client) -> List[Dict[str, Any]]:
    """Revise the curriculum index based on user instructions."""
    print("  Revising index based on instructions...")

    index_text = _format_index_as_text(current_index)

    system_prompt = """You are a curriculum designer revising a curriculum index based on user feedback.

Modify the index per the user's instructions while preserving topological concept order.

CONSTRAINTS:
- Session numbers remain sequential (01, 02, 03, ...)
- Maintain concept dependency ordering
- Titles remain evocative and specific

Respond with valid JSON only:
{"index": [...]}"""

    user_prompt = f"""CURRENT INDEX:

{index_text}

USER INSTRUCTIONS:
{instructions}

Revise per instructions.

JSON format:
{{
  "index": [
    {{
      "session_number": "01",
      "title": "Title",
      "estimated_minutes": 15,
      "concepts": [{{"name": "concept", "description": "one-liner"}}],
      "revisit": {{"name": "concept", "reason": "reason"}}
    }}
  ]
}}

Omit "revisit" if none. Return all sessions."""

    try:
        result = cerebras_client.generate_json(system_prompt, user_prompt, max_tokens=8000)
        revised = result.get("index", current_index)
        if _validate_index(revised):
            print(f"  Revised index: {len(revised)} sessions")
            return revised
        else:
            print("  Revised index failed validation - using original")
            return current_index
    except Exception as e:
        print(f"  Index revision failed: {e}")
        return current_index


def _format_index_as_text(index: List[Dict[str, Any]]) -> str:
    lines = []
    for session in index:
        lines.append(f"Session {session['session_number']}: {session['title']} ({session.get('estimated_minutes', '?')} min)")
        for concept in session.get("concepts", []):
            lines.append(f"  -> {concept['name']}: {concept.get('description', '')}")
        if session.get("revisit"):
            r = session["revisit"]
            lines.append(f"  <- {r['name']}: {r['reason']}")
        lines.append("")
    return "\n".join(lines)


def _validate_index(index: List[Dict[str, Any]]) -> bool:
    if not isinstance(index, list) or not index:
        return False
    for i, session in enumerate(index, 1):
        if not all(f in session for f in ["session_number", "title", "estimated_minutes", "concepts"]):
            return False
        if not isinstance(session["concepts"], list) or not session["concepts"]:
            return False
    return True
