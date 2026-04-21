"""
Local LLM Evaluator - Binary structural checks with Cerebras-powered retry
"""

import re as _re
from typing import Dict, Any, Optional


def evaluate_session(session_text: str,
                     session_plan: Dict[str, Any],
                     state_manager,
                     ollama_client,
                     cerebras_client=None) -> tuple:
    """
    Evaluate a session with five structural checks.
    Uses Ollama for checks (fast, free, local).
    Uses Cerebras to regenerate failing sections on retry.

    Returns:
        Tuple of (evaluation_results dict, session_text str).
        session_text may be modified if truncation repair succeeded.
    """
    print("    Running structural evaluation...")

    if ollama_client is None:
        print("      Ollama unavailable - using length-only fallback evaluation")
        return _fallback_evaluation(session_text), session_text

    evaluation_results = {}

    evaluation_results["TENSION"] = _check_tension(
        session_text, session_plan, state_manager, ollama_client, cerebras_client
    )
    evaluation_results["ANCHOR"] = _check_anchor(
        session_text, session_plan, state_manager, ollama_client, cerebras_client
    )
    evaluation_results["COHERENCE"] = _check_coherence(
        session_text, session_plan, ollama_client, cerebras_client
    )
    evaluation_results["TRANSITIONS"] = _check_transitions(
        session_text, session_plan, ollama_client
    )
    evaluation_results["LENGTH"] = _check_length(session_text)

    # Truncation recovery — if ends mid-sentence, attempt surgical TENSION append
    length_result = evaluation_results.get("LENGTH", {})
    if (not length_result.get("passed", True) and
            "ends mid-sentence" in length_result.get("reason", "") and
            cerebras_client):
        print("      Attempting surgical TENSION append for truncated session...")
        repaired = _complete_truncated_session(session_text, session_plan, cerebras_client)
        if repaired and repaired != session_text:
            new_length = _check_length(repaired)
            new_length["retry_count"] = 1
            if new_length.get("passed"):
                session_text = repaired
                evaluation_results["LENGTH"] = new_length
                print("      Truncation repaired successfully")
            else:
                print("      Truncation repair did not resolve LENGTH check")

    evaluation_results["REVISIT"] = _check_revisit(
        session_text, session_plan, ollama_client, cerebras_client
    )

    passed = sum(1 for r in evaluation_results.values() if r["passed"])
    print(f"    Evaluation: {passed}/{len(evaluation_results)} checks passed")

    return evaluation_results, session_text


def _check_tension(session_text: str,
                   session_plan: Dict[str, Any],
                   state_manager,
                   ollama_client,
                   cerebras_client) -> Dict[str, Any]:
    # Strip trailing summary card (em-dash bullets appended after TENSION in generator)
    # Format: blank line then 3-5 lines each starting with — or –
    stripped = _re.sub(r'\n\n[—–][^\n]+(\n[—–][^\n]+)*\s*$', '', session_text, flags=_re.DOTALL).rstrip()
    if stripped:
        session_text = stripped

    # --- Deterministic pre-gate ---
    words = session_text.split()
    last_300_words = " ".join(words[-300:]) if len(words) >= 300 else session_text

    # Gate 1: Must have at least one '?' in last 300 words
    question_positions = [m.start() for m in _re.finditer(r'\?', last_300_words)]
    if not question_positions:
        return {
            "check": "TENSION", "passed": False,
            "evidence": "No question mark found in final 300 words — TENSION question is absent or session is truncated",
            "retry_count": 0, "needs_review": True
        }

    # Gate 2: The last '?' must be in a sentence of ≥5 words (not a parenthetical/code artifact)
    last_q_pos = question_positions[-1]
    fragment = last_300_words[:last_q_pos + 1]
    prev_terminal = max(
        fragment.rfind('. ', 0, last_q_pos),
        fragment.rfind('! ', 0, last_q_pos),
        fragment.rfind('? ', 0, last_q_pos),
    )
    sentence_start = prev_terminal + 2 if prev_terminal >= 0 else 0
    tension_sentence = last_300_words[sentence_start:last_q_pos + 1].strip()
    if len(tension_sentence.split()) < 5:
        return {
            "check": "TENSION", "passed": False,
            "evidence": f"Question mark found but in a {len(tension_sentence.split())}-word sentence — likely a parenthetical or code artifact, not a TENSION question",
            "retry_count": 0, "needs_review": True
        }

    # Gate 3: Session must end on or near the question (≤20 words after the final '?')
    words_after_q = last_300_words[last_q_pos + 1:].split()
    if len(words_after_q) > 20:
        return {
            "check": "TENSION", "passed": False,
            "evidence": f"Question mark found but {len(words_after_q)} words follow it — session continues past TENSION instead of ending there",
            "retry_count": 0, "needs_review": True
        }
    # Pre-gate passed — continue to Ollama check

    # The next session's first concept is a better target than using current session's first
    next_concept = ""
    if session_plan.get("concepts"):
        next_concept = session_plan["concepts"][-1]["name"]

    result = {"check": "TENSION", "passed": False, "evidence": "", "retry_count": 0, "needs_review": False}
    max_retries = 2

    current_text = session_text

    for attempt in range(max_retries + 1):
        try:
            words_cur = current_text.split()
            session_ending = " ".join(words_cur[-100:]) if len(words_cur) >= 100 else current_text
            passed, evidence = ollama_client.check_tension(session_ending, next_concept)

            if passed:
                result["passed"] = True
                result["evidence"] = evidence
                return result

            if attempt < max_retries:
                print(f"      TENSION failed (attempt {attempt+1}): {evidence[:80]}")
                if cerebras_client:
                    print(f"      [Generator] Regenerating TENSION (attempt {attempt+1}): {evidence[:120]}")
                    current_text = _regenerate_tension(current_text, session_plan, evidence, cerebras_client)
                result["retry_count"] += 1
            else:
                result["evidence"] = evidence
                result["needs_review"] = True
                print(f"      TENSION failed after {max_retries} retries")

        except Exception as e:
            print(f"      TENSION check error: {e}")
            if attempt == max_retries:
                result["evidence"] = f"Error: {e}"
                result["needs_review"] = True
                return result

    return result


def _extract_tension_question(text: str) -> str:
    """Extract the TENSION question from the last 300 words of a session."""
    if not text:
        return ""
    words = text.split()
    last_300 = " ".join(words[-300:]) if len(words) >= 300 else text
    q_positions = [m.start() for m in _re.finditer(r'\?', last_300)]
    if not q_positions:
        return ""
    last_q = q_positions[-1]
    fragment = last_300[:last_q + 1]
    prev_terminal = max(fragment.rfind('. '), fragment.rfind('! '), fragment.rfind('? ', 0, last_q))
    start = prev_terminal + 2 if prev_terminal >= 0 else 0
    return last_300[start:last_q + 1].strip()


def _check_anchor(session_text: str,
                  session_plan: Dict[str, Any],
                  state_manager,
                  ollama_client,
                  cerebras_client) -> Dict[str, Any]:
    session_num = session_plan.get("session_number", "01")

    if session_num == "01":
        return {
            "check": "ANCHOR", "passed": True,
            "evidence": "First session - ANCHOR not required",
            "retry_count": 0, "needs_review": False
        }

    words = session_text.split()
    session_opening = " ".join(words[:200]) if len(words) >= 200 else session_text
    previous_context = _get_previous_session_data(session_num, state_manager)

    # Deterministic pre-check: opening must not be verbatim copy of previous session
    if previous_context and previous_context.get("ending"):
        session_start = session_text[:300].lower().split()
        prev_end = previous_context["ending"][-300:].lower().split()
        if len(session_start) > 10 and len(prev_end) > 10:
            overlap = len(set(session_start[:30]) & set(prev_end[-30:]))
            overlap_ratio = overlap / min(30, len(session_start), len(prev_end))
            if overlap_ratio > 0.65:
                return {
                    "check": "ANCHOR", "passed": False,
                    "evidence": "ANCHOR opens with verbatim or near-verbatim text from the previous session (copy-paste detected)",
                    "retry_count": 0, "needs_review": True
                }

    # Extract tension question from previous content to give Ollama the exact text
    prev_content = previous_context.get('text', '')
    if not prev_content:
        # Fall back to 'ending' field which holds the last paragraph
        prev_content = previous_context.get('ending', '')
    tension_question = _extract_tension_question(prev_content)

    result = {"check": "ANCHOR", "passed": False, "evidence": "", "retry_count": 0, "needs_review": False}
    max_retries = 2
    current_text = session_text

    for attempt in range(max_retries + 1):
        try:
            passed, evidence = ollama_client.check_anchor(
                session_opening,
                previous_context.get("ending", ""),
                previous_context.get("concepts", []),
                tension_question
            )

            if passed:
                result["passed"] = True
                result["evidence"] = evidence
                return result

            if attempt < max_retries:
                print(f"      ANCHOR failed (attempt {attempt+1}): {evidence[:80]}")
                if cerebras_client:
                    print(f"      [Generator] Regenerating ANCHOR (attempt {attempt+1}): {evidence[:120]}")
                    current_text = _regenerate_anchor(current_text, previous_context, evidence, cerebras_client)
                    words = current_text.split()
                    session_opening = " ".join(words[:200]) if len(words) >= 200 else current_text
                result["retry_count"] += 1
            else:
                result["evidence"] = evidence
                result["needs_review"] = True
                print(f"      ANCHOR failed after {max_retries} retries")

        except Exception as e:
            print(f"      ANCHOR check error: {e}")
            if attempt == max_retries:
                result["evidence"] = f"Error: {e}"
                result["needs_review"] = True
                return result

    return result


def _check_coherence(session_text: str,
                     session_plan: Dict[str, Any],
                     ollama_client,
                     cerebras_client) -> Dict[str, Any]:
    session_concepts = [c["name"] for c in session_plan.get("concepts", [])]
    revisit_concept = session_plan.get("revisit", {}).get("name") if session_plan.get("revisit") else None
    session_title = session_plan.get("title", "")

    result = {"check": "COHERENCE", "passed": False, "evidence": "", "retry_count": 0, "needs_review": False}
    max_retries = 2

    for attempt in range(max_retries + 1):
        try:
            passed, evidence = ollama_client.check_coherence(
                session_text, session_concepts, revisit_concept, session_title
            )

            if passed:
                result["passed"] = True
                result["evidence"] = evidence
                return result

            if attempt < max_retries:
                print(f"      COHERENCE failed (attempt {attempt+1}): {evidence[:80]}")
                if cerebras_client:
                    print(f"      [Generator] Regenerating COHERENCE (attempt {attempt+1}): {evidence[:120]}")
                    session_text = _regenerate_coherence(
                        session_text, evidence, session_concepts, revisit_concept, cerebras_client
                    )
                result["retry_count"] += 1
            else:
                result["evidence"] = evidence
                result["needs_review"] = True
                print(f"      COHERENCE failed after {max_retries} retries")

        except Exception as e:
            print(f"      COHERENCE check error: {e}")
            if attempt == max_retries:
                result["evidence"] = f"Error: {e}"
                result["needs_review"] = True
                return result

    return result


def _check_transitions(session_text: str,
                        session_plan: Dict[str, Any],
                        ollama_client) -> Dict[str, Any]:
    """
    Check concept-to-concept transitions for causal bridging.
    Checks each adjacent concept boundary in the session.
    """
    result = {"check": "TRANSITIONS", "passed": True, "evidence": "", "retry_count": 0, "needs_review": False}

    concepts = session_plan.get("concepts", [])
    if len(concepts) < 2:
        result["evidence"] = "Single concept session — no transitions to check"
        return result

    # Split session into paragraphs for rough concept boundary detection
    paragraphs = [p.strip() for p in session_text.split('\n\n') if p.strip()]
    if len(paragraphs) < 3:
        result["evidence"] = "Too few paragraphs to detect transitions"
        return result

    failures = []

    for i in range(len(concepts) - 1):
        concept_a = concepts[i]["name"]
        concept_b = concepts[i + 1]["name"]

        # Estimate boundary: split paragraphs roughly evenly across concepts
        # Use middle third of session as the transition zone
        n = len(paragraphs)
        # Find approximate boundary paragraph index for this transition
        boundary_idx = int(n * (i + 1) / len(concepts))
        boundary_idx = max(1, min(boundary_idx, n - 2))

        # Get tail of concept A and head of concept B
        tail_para = paragraphs[boundary_idx - 1]
        head_para = paragraphs[boundary_idx]

        tail_words = tail_para.split()
        head_words = head_para.split()
        concept_a_tail = " ".join(tail_words[-80:]) if len(tail_words) > 80 else tail_para
        concept_b_head = " ".join(head_words[:80]) if len(head_words) > 80 else head_para

        try:
            passed, evidence = ollama_client.check_transition(
                concept_a_tail, concept_b_head, concept_a, concept_b
            )
            if not passed:
                failures.append(f"{concept_a}→{concept_b}: {evidence}")
        except Exception as e:
            print(f"      TRANSITIONS check error at {concept_a}→{concept_b}: {e}")

    if failures:
        result["passed"] = False
        result["evidence"] = "; ".join(failures[:2])  # Report first 2 failures max
        result["needs_review"] = True
    else:
        result["evidence"] = f"All {len(concepts)-1} transitions pass causal bridge check"

    return result


def _check_length(session_text: str) -> Dict[str, Any]:
    word_count = len(session_text.split())

    # Hard failures: outside acceptable word-count range — return immediately
    # so these don't get a misleading "mid-sentence" message
    if word_count < 1500:
        return {
            "check": "LENGTH", "passed": False,
            "evidence": f"{word_count} words (too short, hard minimum 1500, target 1800)",
            "retry_count": 0, "needs_review": True
        }
    if word_count > 2600:
        return {
            "check": "LENGTH", "passed": False,
            "evidence": f"{word_count} words (too long, maximum 2600)",
            "retry_count": 0, "needs_review": True
        }

    # Completeness check: session must end on a complete sentence
    stripped = session_text.rstrip()
    if stripped and stripped[-1] not in '.?!':
        return {
            "check": "LENGTH", "passed": False,
            "reason": "Session ends mid-sentence (likely truncated by API token limit)",
            "evidence": f"{word_count} words but ends mid-sentence (likely truncated by API token limit)",
            "word_count": word_count,
            "retry_count": 0,
            "needs_review": True
        }

    # Word count is within acceptable range and session is complete
    if 1800 <= word_count <= 2600:
        return {
            "check": "LENGTH", "passed": True,
            "evidence": f"{word_count} words (within 1800-2600 target range)",
            "retry_count": 0, "needs_review": False
        }
    else:
        # 1500–1799: passes but warn
        print(f"      [WARN] Session is {word_count} words — below ideal 1800 target")
        return {
            "check": "LENGTH", "passed": True,
            "warning": "Below ideal 1800-word target",
            "evidence": f"{word_count} words (below ideal 1800-2600 range, minimum acceptable 1500)",
            "retry_count": 0, "needs_review": False
        }


def _complete_truncated_session(session_text: str,
                                session_plan: Dict[str, Any],
                                cerebras_client) -> str:
    """
    Surgical repair for truncated sessions.
    Finds last complete sentence, discards fragment, appends a TENSION question.
    """
    stripped = session_text.rstrip()

    # Find the last complete sentence boundary
    last_terminal = -1
    for i in range(len(stripped) - 1, -1, -1):
        if stripped[i] in '.!?':
            after = stripped[i+1:i+2] if i + 1 < len(stripped) else ''
            before = stripped[i-1:i] if i > 0 else ''
            if (after == '' or after in ' \n\r\t') and before.strip():
                last_terminal = i
                break

    if last_terminal < 0:
        return session_text  # Cannot repair

    clean_body = stripped[:last_terminal + 1]

    # Get the last complete sentence for context
    last_sent_start = max(
        clean_body.rfind('. ', 0, last_terminal),
        clean_body.rfind('! ', 0, last_terminal),
        clean_body.rfind('? ', 0, last_terminal),
    )
    last_complete_sentence = clean_body[last_sent_start + 2:].strip() if last_sent_start >= 0 else clean_body[-200:].strip()

    concepts = session_plan.get('concepts', [])
    concept_names = [c['name'] if isinstance(c, dict) else c for c in concepts]
    next_concept = session_plan.get('next_session_first_concept', '')

    system_prompt = """You are writing a TENSION question to close a technical reading session.
The TENSION must:
- Be 40-60 words, a single concrete question
- Be unanswerable from what the reader just read
- Be specific enough to think about during the day
- Follow naturally from the sentence provided
- End with a question mark
- Be the FINAL sentence — write nothing after it
Write ONLY the tension question. No preamble."""

    user_prompt = f"""The session ends with this sentence:
"{last_complete_sentence}"

Session concepts: {', '.join(concept_names)}
Next session's first concept: {next_concept}

Write a 40-60 word TENSION question that follows naturally."""

    try:
        if not cerebras_client:
            return session_text
        tension = cerebras_client.generate(system_prompt, user_prompt, max_tokens=150, temperature=0.8)
        tension = tension.strip()
        if not tension.endswith('?'):
            tension = tension.rstrip('.!') + '?'
        return f"{clean_body}\n\n{tension}"
    except Exception as e:
        print(f"        Truncation repair failed: {e}")
        return session_text


def _check_revisit(session_text: str,
                   session_plan: Dict[str, Any],
                   ollama_client,
                   cerebras_client) -> Dict[str, Any]:
    """5th check: REVISIT paragraph present and making a new connection."""
    revisit_plan = session_plan.get('revisit')
    if not revisit_plan or not revisit_plan.get('name'):
        return {"check": "REVISIT", "passed": True, "evidence": "No REVISIT planned for this session", "retry_count": 0, "needs_review": False}

    revisit_concept = revisit_plan.get('name', '')

    REVISIT_MARKERS = ["earlier", "recall", "back in session", "revisit", "building on",
                       "as we saw", "we discussed", "introduced earlier", "remember when",
                       "comes back", "reappears", "return to", "first saw"]
    content_lower = session_text.lower()
    found_markers = [m for m in REVISIT_MARKERS if m in content_lower]

    if not found_markers:
        return {
            "check": "REVISIT", "passed": False,
            "evidence": f"No REVISIT paragraph found. Expected revisit of '{revisit_concept}'. No marker words detected.",
            "retry_count": 0, "needs_review": True
        }

    # Find the paragraph with the revisit content
    paragraphs = [p for p in session_text.split("\n\n") if p.strip()]
    revisit_paragraph = ""
    for p in paragraphs:
        p_lower = p.lower()
        if any(m in p_lower for m in found_markers) and revisit_concept.lower()[:10] in p_lower:
            revisit_paragraph = p
            break
    if not revisit_paragraph:
        revisit_paragraph = max(paragraphs, key=lambda p: sum(1 for m in found_markers if m in p.lower()), default="")

    result = {"check": "REVISIT", "passed": False, "evidence": "", "retry_count": 0, "needs_review": False}
    concept_names = [c['name'] if isinstance(c, dict) else c for c in session_plan.get('concepts', [])]

    current_text = session_text
    for attempt in range(3):  # max 2 retries
        try:
            passed, evidence = ollama_client.check_revisit(revisit_paragraph, revisit_concept, concept_names)
            if passed:
                # Secondary check: is the connection specific (not vague)?
                if session_plan.get("revisit"):
                    revisit_reason = session_plan["revisit"].get("reason", "")
                    if revisit_reason:
                        try:
                            spec_passed, spec_evidence = ollama_client.check_revisit_specificity(
                                revisit_paragraph, revisit_concept, revisit_reason
                            )
                            if not spec_passed:
                                passed = False
                                evidence = f"REVISIT present but connection is vague: {spec_evidence}"
                        except Exception as e:
                            print(f"      REVISIT specificity check error: {e}")
                            # Don't fail the check on error — just skip specificity sub-check
            if passed:
                result["passed"] = True
                result["evidence"] = evidence
                return result

            if attempt < 2:
                print(f"      REVISIT failed (attempt {attempt+1}): {evidence[:80]}")
                result["retry_count"] += 1
                if cerebras_client:
                    current_text = _regenerate_revisit_paragraph(current_text, session_plan, evidence, cerebras_client)
                    paragraphs = [p for p in current_text.split("\n\n") if p.strip()]
                    for p in paragraphs:
                        if any(m in p.lower() for m in REVISIT_MARKERS):
                            revisit_paragraph = p
                            break
            else:
                result["evidence"] = evidence
                result["needs_review"] = True
                print(f"      REVISIT failed after 2 retries")
        except Exception as e:
            result["evidence"] = f"Error: {e}"
            if attempt == 2:
                result["needs_review"] = True
            break

    return result


def _regenerate_revisit_paragraph(session_text: str, session_plan: Dict[str, Any],
                                  failure_reason: str, cerebras_client) -> str:
    revisit_plan = session_plan.get('revisit', {})
    revisit_name = revisit_plan.get('name', '')
    revisit_reason = revisit_plan.get('reason', '')
    concept_names = [c['name'] if isinstance(c, dict) else c for c in session_plan.get('concepts', [])]

    MARKERS = ["earlier", "recall", "revisit", "building on", "as we saw", "we discussed", "introduced earlier"]
    paragraphs = [p for p in session_text.split("\n\n") if p.strip()]

    # Find the failing paragraph or insert before TENSION (last paragraph)
    target_idx = len(paragraphs) - 2 if len(paragraphs) >= 2 else 0
    for i, p in enumerate(paragraphs):
        if any(m in p.lower() for m in MARKERS):
            target_idx = i
            break

    context_para = paragraphs[target_idx - 1] if target_idx > 0 else ""

    system_prompt = """Write a REVISIT paragraph (~140 words) for a technical reading session.
Must: name the earlier concept explicitly, draw ONE specific new connection to current material, NOT re-explain core mechanics from scratch.
Write ONLY the paragraph. No preamble."""

    user_prompt = f"""Earlier concept: {revisit_name}
Connection reason: {revisit_reason}
Current session concepts: {', '.join(concept_names)}
Failure reason to fix: {failure_reason}

Preceding paragraph context:
{context_para[-400:] if context_para else '[start of session]'}

Write replacement REVISIT paragraph."""

    try:
        new_para = cerebras_client.generate(system_prompt, user_prompt, max_tokens=350, temperature=0.7)
        paragraphs[target_idx] = new_para.strip()
        return "\n\n".join(paragraphs)
    except Exception as e:
        print(f"        REVISIT regeneration error: {e}")
        return session_text


def _get_previous_session_data(session_num: str, state_manager) -> Dict[str, Any]:
    if session_num == "01":
        return {"ending": "", "concepts": []}

    prev_num = f"{int(session_num) - 1:02d}"
    session_results = state_manager.get_session_results()

    if prev_num in session_results:
        prev_result = session_results[prev_num]
        prev_text = prev_result.get("content") or ""
        paragraphs = [p for p in prev_text.split("\n\n") if p.strip()]
        final_paragraph = paragraphs[-1] if paragraphs else ""
        prev_concepts = [c["name"] for c in prev_result.get("concepts", [])]
        return {"ending": final_paragraph, "concepts": prev_concepts, "text": prev_text}

    return {"ending": "", "concepts": [], "text": ""}


def _regenerate_tension(session_text: str,
                        session_plan: Dict[str, Any],
                        failure_reason: str,
                        cerebras_client) -> str:
    """Regenerate only the TENSION (final paragraph) using Cerebras."""
    paragraphs = [p for p in session_text.split("\n\n") if p.strip()]
    if len(paragraphs) < 2:
        return session_text

    main_content = "\n\n".join(paragraphs[:-1])
    old_ending = paragraphs[-1]

    concepts = session_plan.get("concepts", [])
    concept_names = [c["name"] for c in concepts]

    system_prompt = """You are rewriting the final TENSION question of a technical reading session.

The TENSION must:
- Be a single concrete question (NOT rhetorical)
- Be unanswerable from the current session alone
- Be specific enough to think about during the day
- NOT have an obvious yes/no answer
- NOT be definitional ("what is X?")
- End the session - write nothing after it

Write ONLY the replacement final paragraph. No preamble."""

    user_prompt = f"""The current ending FAILED evaluation.

Failure reason: {failure_reason}

Current (failing) ending:
{old_ending}

Session concepts covered: {', '.join(concept_names)}

Write a replacement final paragraph that ends with a properly formed TENSION question."""

    try:
        new_ending = cerebras_client.generate(system_prompt, user_prompt, max_tokens=200, temperature=0.8)
        return f"{main_content}\n\n{new_ending.strip()}"
    except Exception as e:
        print(f"        TENSION regeneration failed: {e}")
        return session_text


def _regenerate_anchor(session_text: str,
                       previous_context: Dict[str, Any],
                       failure_reason: str,
                       cerebras_client) -> str:
    """Regenerate only the ANCHOR (opening paragraph) using Cerebras."""
    paragraphs = [p for p in session_text.split("\n\n") if p.strip()]
    if len(paragraphs) < 2:
        return session_text

    old_opening = paragraphs[0]
    rest = "\n\n".join(paragraphs[1:])

    prev_ending = previous_context.get("ending", "")
    prev_concepts = previous_context.get("concepts", [])

    system_prompt = """You are rewriting the ANCHOR (opening paragraph) of a technical reading session.

The ANCHOR must:
- First sentence directly resolves the previous session's final question
- Transition to new content within 3 sentences
- NOT re-explain previous concepts in depth
- NOT reference terms not in the previous session
- Be ~100-150 words total

Write ONLY the replacement opening paragraph. No preamble."""

    user_prompt = f"""The current opening FAILED evaluation.

Failure reason: {failure_reason}

Previous session concepts: {', '.join(prev_concepts) if prev_concepts else 'none'}

Previous session ending (the TENSION question):
{prev_ending}

Current (failing) opening:
{old_opening}

Write a replacement opening paragraph that properly anchors to the previous session's tension."""

    try:
        new_opening = cerebras_client.generate(system_prompt, user_prompt, max_tokens=250, temperature=0.7)
        return f"{new_opening.strip()}\n\n{rest}"
    except Exception as e:
        print(f"        ANCHOR regeneration failed: {e}")
        return session_text


def _regenerate_coherence(session_text: str,
                          failure_reason: str,
                          concept_list: list,
                          revisit_concept: Optional[str],
                          cerebras_client) -> str:
    """Regenerate the full session to fix a COHERENCE failure using Cerebras."""
    concepts_str = ", ".join(concept_list) if concept_list else "none"
    revisit_str = revisit_concept if revisit_concept else "none"

    system_prompt = """You are fixing a structural coherence problem in a technical reading session.

The session is flowing prose with no section labels. You must correct only the specific issue described.

COHERENCE rules:
- Every concept must be introduced before it is used (no forward references)
- The REVISIT paragraph must name the earlier concept explicitly and draw one new connection — it must NOT re-explain that concept's core mechanics from scratch
- The session must cover only the concepts listed, introducing no others
- Each concept should be explained before the next begins

Return the COMPLETE corrected session text. No preamble, no explanation — just the corrected prose."""

    user_prompt = f"""This session FAILED the COHERENCE check.

Failure reason: {failure_reason}

Concepts introduced in this session (in order): {concepts_str}
REVISIT concept: {revisit_str}

Current session text:
{session_text}

Fix the specific coherence issue described above. Return the complete corrected session text. Maximum 2600 words."""

    try:
        corrected = cerebras_client.generate(system_prompt, user_prompt, max_tokens=4500, temperature=0.6)
        return corrected.strip()
    except Exception as e:
        print(f"        COHERENCE regeneration failed: {e}")
        return session_text


def _fallback_evaluation(session_text: str) -> Dict[str, Any]:
    """Fallback when Ollama is unavailable - only check length."""
    word_count = len(session_text.split())
    length_ok = 1800 <= word_count <= 2600

    return {
        "TENSION": {"check": "TENSION", "passed": True, "evidence": "Ollama unavailable - assumed pass", "retry_count": 0, "needs_review": False},
        "ANCHOR": {"check": "ANCHOR", "passed": True, "evidence": "Ollama unavailable - assumed pass", "retry_count": 0, "needs_review": False},
        "COHERENCE": {"check": "COHERENCE", "passed": True, "evidence": "Ollama unavailable - assumed pass", "retry_count": 0, "needs_review": False},
        "TRANSITIONS": {"check": "TRANSITIONS", "passed": True, "evidence": "Ollama unavailable - assumed pass", "retry_count": 0, "needs_review": False},
        "LENGTH": {
            "check": "LENGTH",
            "passed": length_ok,
            "evidence": f"{word_count} words",
            "retry_count": 0,
            "needs_review": not length_ok
        },
        "REVISIT": {"check": "REVISIT", "passed": True, "evidence": "Ollama unavailable - assumed pass", "retry_count": 0, "needs_review": False},
    }
