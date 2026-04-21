"""
Session Generation - 3-pass generation architecture for flowing prose sessions
"""

import re
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _trim_to_last_complete_sentence(text: str, approx_words: int = 200) -> str:
    if not text:
        return ""
    # Take a generous window then trim to last sentence boundary
    words = text.split()
    window = " ".join(words[-max(approx_words * 2, 400):])
    # Find last sentence-ending punctuation
    for punct in ("? ", ". ", "! "):
        idx = window.rfind(punct)
        if idx != -1:
            window = window[:idx + 1]
            break
    # Now take the last approx_words words of that
    return " ".join(window.split()[-approx_words:])


def _trim_to_words(text: str, target: int) -> str:
    """Trim text to approximately target words, preserving complete sentences."""
    words = text.split()
    if len(words) <= target:
        return text

    # Trim to target but try to end at a sentence boundary
    truncated = " ".join(words[:target])

    # Find last sentence end
    for end_char in ['. ', '? ', '! ']:
        last_idx = truncated.rfind(end_char)
        if last_idx > len(truncated) * 0.85:
            return truncated[:last_idx + 1].strip()

    return truncated


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

def _get_previous_context(session_num: str, state_manager) -> Dict[str, Any]:
    if session_num == "01":
        return {"is_first": True, "text": "", "concepts": [], "tension_excerpt": ""}

    prev_num = f"{int(session_num) - 1:02d}"
    session_results = state_manager.get_session_results()

    if prev_num in session_results:
        prev_result = session_results[prev_num]
        prev_text = prev_result.get("content") or ""
        last_200_words = _trim_to_last_complete_sentence(prev_text, 200)
        tension_excerpt = prev_result.get("tension_excerpt", "")
        return {
            "is_first": False,
            "text": last_200_words,
            "concepts": [c.get("name", "") for c in prev_result.get("concepts", [])],
            "tension_excerpt": tension_excerpt,
        }

    return {"is_first": False, "text": "", "concepts": [], "tension_excerpt": ""}


def _prepare_concept_metadata(concepts: List[Dict[str, Any]],
                               revisit: Optional[Dict[str, Any]],
                               all_concepts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}

    for concept in concepts:
        name = concept["name"]
        full_data = all_concepts.get(name, {})

        # Cross-theme deps: extract "contrasts_with" relationships for CONTRAST fodder
        cross_deps = full_data.get("cross_theme_deps", [])
        contrasts = []
        if isinstance(cross_deps, list):
            for dep in cross_deps:
                if isinstance(dep, dict) and dep.get("relationship_type") in ("contrasts_with", "distinguished_from"):
                    contrasts.append(dep.get("concept", ""))
        elif isinstance(cross_deps, dict):
            for rel_type, names in cross_deps.items():
                if rel_type in ("contrasts_with", "distinguished_from"):
                    contrasts.extend(names if isinstance(names, list) else [names])

        metadata[name] = {
            "canonical_name": name,
            "description": full_data.get("description", concept.get("description", "")),
            "primary_passage": full_data.get("primary_passage", ""),
            "secondary_passages": full_data.get("secondary_passages", []),
            "author_anchor": full_data.get("author_anchor", ""),
            "weight": full_data.get("concept_weight", concept.get("weight", "medium")),
            "enrichment_flag": full_data.get("enrichment_flag", False),
            # Previously unused — now injected for richer depth
            "implicit_prerequisites": full_data.get("implicit_prerequisites", []),
            "dependency_signals": full_data.get("dependency_signals", []),
            "contrasts_with": [c for c in contrasts if c],
        }

    if revisit:
        revisit_name = revisit["name"]
        revisit_data = all_concepts.get(revisit_name, {})
        metadata["__revisit__"] = {
            "name": revisit_name,
            "reason": revisit["reason"],
            "description": revisit_data.get("description", ""),
            "primary_passage": revisit_data.get("primary_passage", ""),
        }

    return metadata


def _prepare_dependency_context(concepts: List[Dict[str, Any]],
                                 all_concepts: Dict[str, Dict[str, Any]],
                                 graph_output: Dict[str, Any]) -> Dict[str, Any]:
    concept_names = {c["name"] for c in concepts}
    edges = graph_output.get("edges", [])

    prerequisites: Dict[str, list] = defaultdict(list)
    dependents: Dict[str, list] = defaultdict(list)

    for edge in edges:
        if edge["to"] in concept_names:
            prerequisites[edge["to"]].append(edge["from"])
        elif edge["from"] in concept_names:
            dependents[edge["from"]].append(edge["to"])

    def get_desc(name: str) -> str:
        return all_concepts.get(name, {}).get("description", name)

    return {
        "prerequisites": {name: [get_desc(p) for p in prereqs] for name, prereqs in prerequisites.items()},
        "dependents": {name: [get_desc(d) for d in deps] for name, deps in dependents.items()},
    }


def _get_cluster_description(concepts: List[Dict[str, Any]],
                              all_concepts: Dict[str, Dict[str, Any]],
                              graph_output: Dict[str, Any]) -> str:
    cluster_counts: Dict[str, int] = defaultdict(int)
    for concept in concepts:
        cluster = all_concepts.get(concept["name"], {}).get("cluster", "")
        if cluster:
            cluster_counts[cluster] += 1

    if not cluster_counts:
        return ""

    dominant = max(cluster_counts, key=cluster_counts.get)  # type: ignore[arg-type]
    return f"This session focuses on the {dominant} domain."


# ---------------------------------------------------------------------------
# Anti-fabrication block (shared across prompts)
# ---------------------------------------------------------------------------

_ANTI_FABRICATION = """CRITICAL — DO NOT FABRICATE:
- Never invent first-person anecdotes ("We ran this at a client...", "I saw this in production once...")
- Never fabricate specific metrics ("latency dropped from 850ms to 210ms", "cost fell 65%")
- When you need a concrete example, use: "A typical production setup...", "Systems like vLLM or TGI...", "In a real deployment you might see..."
- You may reference well-known systems (vLLM, TensorRT-LLM, TGI, Triton, SGLang) by name
- You may state general ranges ("LLM inference typically runs at 10-100 tok/s depending on...") without fabricating specific measurements
- Never attribute specific metrics or findings to named companies or teams (e.g., "Meta reported that...", "Google's PaLM team found...", "OpenAI showed..."). These claims are unverifiable and likely invented.
- If you need a concrete example with numbers, use clearly hypothetical framing: "In a typical production deployment, you might see..." or "Consider a setup where...\""""


# ---------------------------------------------------------------------------
# Pass 1: Tension pre-plan
# ---------------------------------------------------------------------------

def _generate_tension_plan(session_plan: Dict[str, Any],
                            graph_output: Dict[str, Any],
                            cerebras_client) -> str:
    """
    Pass 1 — Generate a single concrete tension question to aim the session toward.
    max_tokens=200, temperature=0.4
    """
    concepts = session_plan["concepts"]
    concept_names = [c["name"] for c in concepts]
    next_concept = session_plan.get("next_session_first_concept", "")

    system_prompt = (
        "You are a curriculum architect. Given technical concepts and the first concept of the next "
        "session, write a single concrete tension question — the question a reader should be actively "
        "thinking about after finishing this session. 50 words maximum. The question must be: specific "
        "(not rhetorical), unanswerable from this session alone, answerable from the next session, "
        "concrete enough to think about during the day. Return ONLY the question, nothing else."
    )

    user_prompt = (
        f"Session concepts: {', '.join(concept_names)}\n"
        f"Next session opens with: {next_concept}\n\n"
        "Write the tension question."
    )

    result = cerebras_client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=200,
        temperature=0.4,
    )
    return result.strip()


# ---------------------------------------------------------------------------
# Explanation quality rules (injected into every per-concept call)
# ---------------------------------------------------------------------------

_CONCEPT_QUALITY_RULES = """EXPLANATION QUALITY — FOLLOW EXACTLY:

ANALOGY:
Before writing any analogy, identify three relational correspondences (not surface attributes) you are mapping. If you cannot name three, discard the analogy and find one you can map fully.
Source domains must be familiar to working engineers: OS scheduling, network protocols, compiler passes, database internals, memory allocators, CPU cache hierarchies. Not general-public domains (factories, rivers, traffic jams) unless the structural mapping is formally precise.
After the analogy, make one structural correspondence explicit in one sentence: "[source] maps to [target] because [relation in source] corresponds to [relation in target]."
The analogy must enable at least one correct prediction the reader hasn't been told. If it cannot, it is decoration, not explanation.

CONCEPT INTRODUCTION — FIRST PRINCIPLES:
Never open a concept with its name or a definition. Open with the problem it solves.
The concept's name appears only after the reader understands why it needs to exist.
For any concept that improved on something simpler: name the naive approach and give its specific failure mode with scale numbers ("at 32K sequence length, KV cache exceeds VRAM by 10x" — not "it doesn't scale").
Before introducing the mechanism, state the necessary property: "any correct solution must [do X]." The concept then arrives as the thing that satisfies that property.

PRODUCTION INSTANTIATION:
Every concept requires at least ONE of the following: (a) a named config parameter with its default value, (b) a specific observable metric at a named percentile, (c) a named error message or failure mode, or (d) a named function/file in a real open-source codebase (vLLM, TensorRT-LLM, SGLang, Triton).
"X is used in vLLM" is never sufficient. State where in vLLM, what it controls, and what you observe when you change it.
Production failure modes are more instructive than happy-path descriptions. Ask: what breaks first when this mechanism is stressed?"""


# ---------------------------------------------------------------------------
# Pass 2: Concepts body — per-concept generation
# ---------------------------------------------------------------------------

def _build_concepts_user_prompt(session_plan: Dict[str, Any],
                                  graph_output: Dict[str, Any],
                                  concept_metadata: Dict[str, Any],
                                  dependency_context: Dict[str, Any],
                                  cluster_description: str,
                                  needs_web_search: bool,
                                  tension_plan: str) -> str:
    concepts = session_plan["concepts"]
    concept_list = [
        {"name": name, **meta}
        for name, meta in concept_metadata.items()
        if name != "__revisit__"
    ]

    parts = []

    if cluster_description:
        parts.append(f"DOMAIN CONTEXT:\n{cluster_description}")

    prereq_lines = []
    for concept_name, prereqs in dependency_context.get("prerequisites", {}).items():
        if prereqs:
            prereq_lines.append(f"  - {concept_name} builds on: {', '.join(prereqs[:2])}")
    if prereq_lines:
        parts.append("DEPENDENCY CONTEXT:\n" + "\n".join(prereq_lines))

    parts.append("CONCEPTS TO COVER (in this order):\n")
    for concept in concept_list:
        block = f"{concept['canonical_name']}:\n"
        block += f"  - Description: {concept['description']}\n"
        block += f"  - Weight: {concept['weight']}\n"
        if concept.get("enrichment_flag"):
            block += "  - Note: Enrich with current implementations/examples\n"
        if concept.get("primary_passage"):
            passage = concept["primary_passage"][:300]
            block += f"  - Key passage: {passage}...\n"
        if concept.get("secondary_passages"):
            for sp in concept["secondary_passages"][:2]:
                block += f"  - Secondary passage: {str(sp)[:200]}...\n"
        if concept.get("author_anchor"):
            block += f"  - Author's insight: {concept['author_anchor']}\n"
        parts.append(block)

    parts.append(
        "Write the teaching body now. Start directly with the first concept. "
        "No opening bridge paragraph. No closing question."
    )

    return "\n\n".join(parts)


def _generate_single_concept(
    concept_meta: Dict[str, Any],
    next_concept_name: Optional[str],
    is_last_concept: bool,
    is_first_concept: bool,
    previous_tail: str,
    tension_plan: str,
    cluster_description: str,
    needs_web_search: bool,
    revisit_meta: Optional[Dict[str, Any]],
    cerebras_client,
) -> str:
    """
    Generate one concept's prose section (~700-900 words).
    Feeds previous concept's tail for coherence across per-concept calls.
    max_tokens=1200, temperature=0.65
    """
    name = concept_meta["canonical_name"]
    weight = concept_meta.get("weight", "medium")

    # Build transition instruction
    if is_last_concept:
        transition_instruction = (
            f"This is the LAST concept in the session. "
            f"The session's tension question will be: [{tension_plan}]. "
            f"Your final sentences must make this question feel inevitable — the reader should almost see the answer but not quite. "
            f"Do NOT write the question itself."
        )
    else:
        transition_instruction = (
            f"This concept is followed by: {next_concept_name}. "
            f"Your final 1-2 sentences must use causal structure to make that concept's arrival feel necessary: "
            f"'But [constraint this concept creates], which means [gap], which is precisely where {next_concept_name} enters.' "
            f"Never write 'Now let's look at' or 'Next, we'll cover'."
        )

    # Build REVISIT instruction if applicable
    revisit_instruction = ""
    if revisit_meta:
        revisit_instruction = f"""
MANDATORY REVISIT (~150 words, woven in naturally after your main explanation):
Write one paragraph that reconnects to {revisit_meta['name']}.
Connection reason: {revisit_meta['reason']}
Rules: (1) name {revisit_meta['name']} explicitly, (2) identify the specific constraint from that earlier concept that makes THIS concept's existence necessary, (3) show the hinge where old and new lock together. Never re-explain the earlier concept's mechanism. 130-160 words."""

    # Build previous context instruction
    prev_instruction = ""
    if previous_tail:
        prev_instruction = f"""The previous concept section ended with:
---
{previous_tail}
---
Continue directly from this natural stopping point. Do not summarize, do not reintroduce. The prose flows as one seamless document."""

    system_prompt = f"""You are a senior engineer explaining a single technical concept to a sharp junior engineer new to this domain but not to engineering. Direct. Warm. Uses "you". No hedging.

You are writing EXACTLY ONE CONCEPT SECTION: {name}
Target: 900–1100 words. MINIMUM 900 words — do not stop before reaching 900 words. No headers, no labels. Pure prose.

{_CONCEPT_QUALITY_RULES}

CONTRAST (for {'this concept — it is marked ' + weight if weight == 'heavy' else 'skip — this is a ' + weight + ' concept'}):
{'Write one sentence: "This is NOT [most plausible confusion] — the difference is [specific mechanism that distinguishes them]."' if weight == 'heavy' else 'No contrast sentence needed.'}

{transition_instruction}

{revisit_instruction}

Code blocks must be followed by a plain-language explanation of what the code demonstrates.

{_ANTI_FABRICATION}

No section headers. No labels. No [REVISIT] tags — write it as seamless prose."""

    # Build user prompt: concept data + context
    user_parts = []

    if cluster_description:
        user_parts.append(f"Domain: {cluster_description}")

    user_parts.append(f"CONCEPT: {name}")
    user_parts.append(f"Description: {concept_meta.get('description', '')}")
    user_parts.append(f"Weight: {weight}")

    if concept_meta.get("author_anchor"):
        user_parts.append(f"Author's crispest insight: {concept_meta['author_anchor']}")

    if concept_meta.get("primary_passage"):
        user_parts.append(f"Key passage from book: {concept_meta['primary_passage'][:400]}")

    if concept_meta.get("secondary_passages"):
        for sp in concept_meta["secondary_passages"][:2]:
            user_parts.append(f"Supporting passage: {str(sp)[:250]}")

    if concept_meta.get("implicit_prerequisites"):
        prereqs = concept_meta["implicit_prerequisites"]
        if isinstance(prereqs, list) and prereqs:
            user_parts.append(f"Reader already knows: {', '.join(str(p) for p in prereqs[:4])}")

    if concept_meta.get("contrasts_with"):
        user_parts.append(f"Plausible confusions to contrast against: {', '.join(concept_meta['contrasts_with'][:3])}")

    if concept_meta.get("enrichment_flag"):
        user_parts.append("Note: Enrich with current real-world implementations and production examples.")

    if prev_instruction:
        user_parts.append(prev_instruction)

    if is_first_concept:
        scene_setter = concept_meta.get("_scene_setter", "")
        if scene_setter:
            user_parts.append(scene_setter)
        user_parts.append("This is the FIRST concept of the session. Start teaching immediately — no preamble.")

    user_parts.append("Write the concept section now.")

    user_prompt = "\n\n".join(user_parts)

    result = cerebras_client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=1800,
        temperature=0.65,
        enable_web_search=needs_web_search and bool(concept_meta.get("enrichment_flag")),
    )
    return result.strip()


def _generate_concepts_body(session_plan: Dict[str, Any],
                              graph_output: Dict[str, Any],
                              tension_plan: str,
                              previous_context: Dict[str, Any],
                              concept_metadata: Dict[str, Any],
                              dependency_context: Dict[str, Any],
                              cluster_description: str,
                              needs_web_search: bool,
                              cerebras_client) -> str:
    """
    Pass 2 — Generate ONLY the teaching body (no ANCHOR, no TENSION).
    Per-concept architecture: one focused call per concept for depth.
    Coherence maintained by feeding each concept's tail to the next.
    """
    revisit_meta = concept_metadata.get("__revisit__")

    concept_list = [
        {"name": name, **meta}
        for name, meta in concept_metadata.items()
        if name != "__revisit__"
    ]

    session_num = session_plan.get("session_number", "01")

    # Scene-setter for session 01 first concept
    scene_setter_prefix = ""
    if session_num == "01" and concept_list:
        first_concept = concept_list[0]["canonical_name"]
        scene_setter_prefix = (
            f"You are writing the very first session of this curriculum. "
            f"Before this concept's explanation, open with exactly 2 sentences: "
            f"(1) the concrete engineering problem this domain solves (latency, throughput, cost, memory — be specific), "
            f"(2) name {first_concept} as the foundation everything else builds on. "
            f"Then begin the concept explanation.\n\n"
        )

    # REVISIT goes into the second-to-last concept (or last if only 1-2 concepts)
    revisit_target_idx = max(0, len(concept_list) - 2) if revisit_meta else -1

    sections: List[str] = []
    previous_tail = ""

    for i, concept in enumerate(concept_list):
        is_first = (i == 0)
        is_last = (i == len(concept_list) - 1)
        next_name = concept_list[i + 1]["canonical_name"] if not is_last else None
        include_revisit = (revisit_meta is not None and i == revisit_target_idx)

        # Inject scene-setter into first concept's system prompt via a modified meta
        concept_meta_for_call = dict(concept)
        if is_first and scene_setter_prefix:
            # Prepend scene-setter to the description field so system prompt picks it up
            concept_meta_for_call["_scene_setter"] = scene_setter_prefix

        section = _generate_single_concept(
            concept_meta=concept_meta_for_call,
            next_concept_name=next_name,
            is_last_concept=is_last,
            is_first_concept=is_first,
            previous_tail=previous_tail,
            tension_plan=tension_plan,
            cluster_description=cluster_description,
            needs_web_search=needs_web_search,
            revisit_meta=revisit_meta if include_revisit else None,
            cerebras_client=cerebras_client,
        )

        sections.append(section)

        # Feed last 200 words of this section to next concept
        words = section.split()
        previous_tail = " ".join(words[-200:]) if len(words) >= 200 else section

    session_text = "\n\n".join(sections)

    # Strip any accidental REVISIT planning headers
    session_text = re.sub(
        r'^REVISIT TARGET:[^\n]*\n(?:[A-Z][A-Z ]+:[^\n]*\n)*\s*\n',
        '',
        session_text.strip(),
    )

    return session_text.strip()


# ---------------------------------------------------------------------------
# Pass 3: ANCHOR + TENSION
# ---------------------------------------------------------------------------

def _generate_anchor_and_tension(concepts_body: str,
                                  tension_plan: str,
                                  previous_context: Dict[str, Any],
                                  session_plan: Dict[str, Any],
                                  concept_metadata: Dict[str, Any],
                                  cerebras_client) -> Tuple[str, str]:
    """
    Pass 3 — Generate ANCHOR (~150w) and final TENSION (~50w).
    max_tokens=800, temperature=0.4
    Returns: (anchor_text, tension_text)
    """
    is_first = previous_context.get("is_first", False)

    concept_list = [
        {"name": name, **meta}
        for name, meta in concept_metadata.items()
        if name != "__revisit__"
    ]
    concept_names = [c["name"] for c in concept_list]
    first_concept_name = concept_names[0] if concept_names else ""

    system_prompt = f"""You are writing two structural elements for a technical learning session. Write ONLY these two elements, one after the other, with a blank line between. No labels, no headers, no other content.

ELEMENT 1 — ANCHOR (~150 words):
- First sentence directly resolves the prior question shown below
- Sentences 2-3 bridge to this session: name or strongly imply the session's primary concept ({first_concept_name}) as the destination
- Sentence 4 is where the session begins — ANCHOR is over, do not extend beyond ~150 words
- Must NOT copy or quote the prior session's text — write fresh
- Must NOT re-explain the prior session's concept in depth

ELEMENT 2 — TENSION (~50 words):
- Single concrete question, stated directly
- Unanswerable from this session alone, answerable from the next
- NOT yes/no answerable by intuition. NOT "Why is X important?" (too vague). NOT rhetorical.
- GOOD format: "What happens when [constraint from this session] collides with [limitation] at scale?" — specific enough to replay during the day, concrete enough to generate counterexamples.
- After the question, nothing — no summary, no wrap-up, no "In the next session..."

For Session 01: write ONLY a brief 1-sentence scene-setting opening (no ANCHOR) followed by a blank line, then TENSION."""

    # Build user prompt
    user_parts = []

    if not is_first:
        resolve_text = previous_context.get("tension_excerpt") or previous_context.get("text", "")
        if resolve_text:
            # Extract just the tension question (last ?-containing sentence) if present
            sentences = re.split(r'(?<=[.!?])\s+', resolve_text.strip())
            question_sentences = [s for s in sentences if '?' in s]
            if question_sentences:
                resolve_text = question_sentences[-1].strip()
            else:
                # Fall back to last 80 words
                resolve_words = resolve_text.split()
                resolve_text = " ".join(resolve_words[-80:]) if len(resolve_words) >= 80 else resolve_text
        user_parts.append(
            f'PREVIOUS SESSION ENDED WITH THIS QUESTION — your ANCHOR must answer it within the first 1-2 sentences:\n"{resolve_text}"\n\nA good ANCHOR starts with the direct answer: "Because..." / "The answer is..." / "It turns out..." — then bridges to this session\'s material.'
        )

    user_parts.append(f"Session concepts: {', '.join(concept_names)}")
    user_parts.append(f"Primary concept: {first_concept_name}")

    # Last 1200 chars of concepts body
    body_tail = concepts_body[-1200:]
    user_parts.append(
        f"Teaching body you just wrote (use this to calibrate TENSION):\n{body_tail}"
    )

    user_parts.append(
        f"Draft tension question to refine (refine if session content suggests a more specific formulation):\n{tension_plan}"
    )

    user_prompt = "\n\n".join(user_parts)

    raw = cerebras_client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=800,
        temperature=0.4,
    )
    raw = raw.strip()

    # Split on the blank line separating ANCHOR from TENSION
    # Find the last blank-line boundary and treat everything after as tension
    parts = re.split(r'\n\s*\n', raw)
    if len(parts) >= 2:
        anchor_text = "\n\n".join(parts[:-1]).strip()
        tension_text = parts[-1].strip()
    else:
        # Fallback: use half/half split
        mid = len(raw) // 2
        anchor_text = raw[:mid].strip()
        tension_text = raw[mid:].strip()

    return anchor_text, tension_text


# ---------------------------------------------------------------------------
# Stitch + extend
# ---------------------------------------------------------------------------

def _stitch_session(anchor: str, concepts_body: str, tension: str) -> str:
    """
    Parse [REVISIT]...[/REVISIT] markers from concepts_body (keep content, strip markers),
    then stitch anchor + body + tension into the final session text.
    """
    # Strip [REVISIT] / [/REVISIT] markers but keep inner content
    cleaned_body = re.sub(r'\[REVISIT\]\s*', '', concepts_body)
    cleaned_body = re.sub(r'\s*\[/REVISIT\]', '', cleaned_body)
    cleaned_body = cleaned_body.strip()

    return f"{anchor}\n\n{cleaned_body}\n\n{tension}"


def _extend_concepts_body(concepts_body: str,
                           session_plan: Dict[str, Any],
                           cerebras_client) -> str:
    """
    Extend concepts_body when the stitched result is too short.
    max_tokens=1000, temperature=0.65
    """
    system_prompt = (
        "You are extending a teaching body. Write 300-500 words of additional explanation "
        "that adds depth to the thinnest concept already covered — go deeper on the mechanism, "
        "add a more precise analogy, or add a production example with a specific named config "
        "parameter, observable metric, or real codebase reference. "
        "Do NOT add a new concept. Do NOT write a closing question. "
        "Output ONLY the new paragraph(s) to append — do not repeat the existing body."
    )

    user_prompt = f"Teaching body to extend:\n\n{concepts_body}"

    try:
        extension = cerebras_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1000,
            temperature=0.65,
        )
        return concepts_body.rstrip() + "\n\n" + extension.strip()
    except Exception as e:
        print(f"      Body extension failed: {e}")
        return concepts_body


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------

def _generate_session_text(session_plan: Dict[str, Any],
                            graph_output: Dict[str, Any],
                            state_manager,
                            cerebras_client) -> str:
    """
    Orchestrate the 3-pass generation for a single session.
    """
    session_num = session_plan["session_number"]
    concepts = session_plan["concepts"]
    revisit = session_plan.get("revisit")

    previous_context = _get_previous_context(session_num, state_manager)
    is_first = previous_context.get("is_first", False)

    all_concepts = {c["canonical_name"]: c for c in graph_output["concepts"]}
    concept_metadata = _prepare_concept_metadata(concepts, revisit, all_concepts)
    dependency_context = _prepare_dependency_context(concepts, all_concepts, graph_output)
    cluster_description = _get_cluster_description(concepts, all_concepts, graph_output)

    needs_web_search = any(
        all_concepts.get(c["name"], {}).get("enrichment_flag", False)
        for c in concepts
    )

    # Pass 1: tension pre-plan
    print("      Pass 1: generating tension plan...")
    tension_plan = _generate_tension_plan(session_plan, graph_output, cerebras_client)
    print(f"      Tension: {tension_plan[:80]}...")

    # Pass 2: concepts body
    print("      Pass 2: generating concepts body...")
    concepts_body = _generate_concepts_body(
        session_plan=session_plan,
        graph_output=graph_output,
        tension_plan=tension_plan,
        previous_context=previous_context,
        concept_metadata=concept_metadata,
        dependency_context=dependency_context,
        cluster_description=cluster_description,
        needs_web_search=needs_web_search,
        cerebras_client=cerebras_client,
    )
    print(f"      Body: {len(concepts_body.split())} words")

    # Pass 3: anchor + tension
    print("      Pass 3: generating anchor and tension...")
    anchor, tension = _generate_anchor_and_tension(
        concepts_body=concepts_body,
        tension_plan=tension_plan,
        previous_context=previous_context,
        session_plan=session_plan,
        concept_metadata=concept_metadata,
        cerebras_client=cerebras_client,
    )

    # Stitch
    session_text = _stitch_session(anchor, concepts_body, tension)
    word_count = len(session_text.split())
    print(f"      Stitched: {word_count} words")

    # Extend if too short — up to 2 passes targeting 1800+ words
    for _ext_pass in range(2):
        if word_count >= 1800:
            break
        print(f"      Too short ({word_count} words) — extending concepts body (pass {_ext_pass + 1})...")
        concepts_body = _extend_concepts_body(concepts_body, session_plan, cerebras_client)
        session_text = _stitch_session(anchor, concepts_body, tension)
        word_count = len(session_text.split())
        print(f"      After extension pass {_ext_pass + 1}: {word_count} words")

    if word_count > 2600:
        print(f"      Too long ({word_count} words) — trimming to 2400...")
        session_text = _trim_to_words(session_text, 2400)

    return session_text


# ---------------------------------------------------------------------------
# Summary card
# ---------------------------------------------------------------------------

def _generate_summary_card(session_text: str,
                            concept_metadata: Dict[str, Any],
                            cerebras_client) -> str:
    """
    Generate a 3-5 line concept summary card appended after TENSION.
    Uses a short Cerebras call: max_tokens=200, temperature=0.3
    Returns the summary card text (plain text, no headers).
    """
    concept_list = [
        name for name in concept_metadata.keys()
        if name != "__revisit__"
    ]
    if len(concept_list) < 2:
        return ""  # Skip for single-concept sessions

    concept_names_str = ", ".join(concept_list)

    # Last 800 chars of session for context
    session_tail = session_text[-800:]

    system_prompt = """You are writing a brief concept summary card. Write ONLY a plain-text list of 3-5 one-liners. No headers, no labels, no "Key Takeaways:" prefix. Each line starts with an em-dash (—). Each line names one concept and states the single most important thing the reader now understands about it that they didn't before this session. Maximum 100 words total. No blank lines between items."""

    user_prompt = f"""Concepts covered: {concept_names_str}

Session ending (for context):
{session_tail}

Write the 3-5 line summary card now."""

    try:
        result = cerebras_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=200,
            temperature=0.3,
        )
        return result.strip()
    except Exception as e:
        print(f"      [SummaryCard] Generation failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_session_content(session_plan: Dict[str, Any],
                              graph_output: Dict[str, Any],
                              state_manager,
                              cerebras_client) -> Tuple[str, str]:
    """
    Generate a single session's content as flowing prose.

    Returns:
        (session_text, tension_excerpt)
        session_text: 1800-2400 words of flowing prose
        tension_excerpt: last 80 words of session_text, stored for next session's ANCHOR
    """
    session_num = session_plan["session_number"]
    concepts = session_plan["concepts"]

    print(f"    Concepts: {[c['name'] for c in concepts]}")

    session_text = _generate_session_text(
        session_plan=session_plan,
        graph_output=graph_output,
        state_manager=state_manager,
        cerebras_client=cerebras_client,
    )

    # Build tension_excerpt BEFORE appending summary card so it doesn't bleed into next session's ANCHOR
    words = session_text.split()
    tension_excerpt = " ".join(words[-80:]) if len(words) >= 80 else session_text

    # Generate and append concept summary card
    concept_metadata = _prepare_concept_metadata(
        concepts,
        session_plan.get("revisit"),
        {c["canonical_name"]: c for c in graph_output["concepts"]},
    )
    summary_card = _generate_summary_card(session_text, concept_metadata, cerebras_client)
    if summary_card:
        session_text = session_text + "\n\n" + summary_card

    total_words = len(session_text.split())
    print(f"      Generated {total_words} words (total, including summary card)")

    return session_text, tension_excerpt
