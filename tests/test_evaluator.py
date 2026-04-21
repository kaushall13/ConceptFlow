"""
Tests for pipeline/evaluator.py - evaluate_session and all check functions
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.evaluator import (
    evaluate_session,
    _check_tension,
    _check_anchor,
    _check_coherence,
    _check_length,
    _get_previous_session_data,
    _regenerate_tension,
    _regenerate_anchor,
    _fallback_evaluation,
)


def _make_session_plan(session_number="02", concepts=None, revisit=None):
    if concepts is None:
        concepts = [
            {"name": "ConceptA", "description": "Desc A", "weight": "medium"},
            {"name": "ConceptB", "description": "Desc B", "weight": "light"},
        ]
    plan = {
        "session_number": session_number,
        "title": "Test",
        "concepts": concepts,
        "estimated_minutes": 15,
    }
    if revisit:
        plan["revisit"] = revisit
    return plan


def _make_long_text(words=2000):
    return ("This is a test word. " * (words // 5)).strip()


def _make_state_manager(sessions=None):
    sm = MagicMock()
    sm.get_session_results.return_value = sessions or {}
    return sm


class TestEvaluateSession:
    def test_happy_path_all_checks_present(self):
        ollama = MagicMock()
        ollama.check_tension.return_value = (True, "Good tension")
        ollama.check_anchor.return_value = (True, "Good anchor")
        ollama.check_coherence.return_value = (True, "Coherent")
        sm = _make_state_manager()
        result = evaluate_session(
            _make_long_text(2000),
            _make_session_plan("02"),
            sm, ollama
        )
        assert "TENSION" in result
        assert "ANCHOR" in result
        assert "COHERENCE" in result
        assert "LENGTH" in result

    def test_each_result_has_required_fields(self):
        ollama = MagicMock()
        ollama.check_tension.return_value = (True, "ok")
        ollama.check_anchor.return_value = (True, "ok")
        ollama.check_coherence.return_value = (True, "ok")
        sm = _make_state_manager()
        result = evaluate_session(_make_long_text(2000), _make_session_plan("02"), sm, ollama)
        for check in result.values():
            assert "check" in check
            assert "passed" in check
            assert "evidence" in check
            assert "retry_count" in check
            assert "needs_review" in check

    def test_ollama_none_returns_fallback(self):
        sm = _make_state_manager()
        result = evaluate_session(_make_long_text(2000), _make_session_plan("02"), sm, None)
        # Fallback: tension, anchor, coherence all assumed pass
        assert result["TENSION"]["passed"] is True
        assert result["ANCHOR"]["passed"] is True
        assert result["COHERENCE"]["passed"] is True


class TestCheckLength:
    def test_within_range_passes(self):
        text = "word " * 2000
        result = _check_length(text)
        assert result["passed"] is True
        assert "2000" in result["evidence"]

    def test_too_short_fails(self):
        text = "word " * 500
        result = _check_length(text)
        assert result["passed"] is False
        assert result["needs_review"] is True
        assert "too short" in result["evidence"]

    def test_too_long_fails(self):
        text = "word " * 3000
        result = _check_length(text)
        assert result["passed"] is False
        assert result["needs_review"] is True
        assert "too long" in result["evidence"]

    def test_exactly_1500_words_passes(self):
        text = "word " * 1500
        result = _check_length(text)
        assert result["passed"] is True

    def test_exactly_2600_words_passes(self):
        text = "word " * 2600
        result = _check_length(text)
        assert result["passed"] is True

    def test_1499_words_fails(self):
        text = "word " * 1499
        result = _check_length(text)
        assert result["passed"] is False

    def test_2601_words_fails(self):
        text = "word " * 2601
        result = _check_length(text)
        assert result["passed"] is False


class TestCheckTension:
    def test_passes_on_first_attempt(self):
        ollama = MagicMock()
        ollama.check_tension.return_value = (True, "Tension is well-formed")
        sm = _make_state_manager()
        result = _check_tension(_make_long_text(), _make_session_plan(), sm, ollama, None)
        assert result["passed"] is True
        assert result["retry_count"] == 0

    def test_fails_after_max_retries(self):
        ollama = MagicMock()
        ollama.check_tension.return_value = (False, "Tension is too vague")
        sm = _make_state_manager()
        result = _check_tension(_make_long_text(), _make_session_plan(), sm, ollama, None)
        assert result["passed"] is False
        assert result["needs_review"] is True
        assert result["retry_count"] == 2  # max_retries

    def test_retries_with_cerebras_regeneration(self):
        ollama = MagicMock()
        ollama.check_tension.side_effect = [
            (False, "Failure reason"),
            (True, "Now it passes"),
        ]
        cerebras = MagicMock()
        cerebras.generate.return_value = "New ending with a better question?"
        sm = _make_state_manager()
        # Text needs 2+ paragraphs for regeneration to work (otherwise _regenerate_tension returns unchanged)
        multi_para_text = "First paragraph content.\n\nFinal tension question?"
        result = _check_tension(multi_para_text, _make_session_plan(), sm, ollama, cerebras)
        cerebras.generate.assert_called_once()
        assert result["passed"] is True
        assert result["retry_count"] == 1

    def test_no_regeneration_when_no_cerebras(self):
        ollama = MagicMock()
        ollama.check_tension.return_value = (False, "Failed")
        sm = _make_state_manager()
        cerebras = None
        result = _check_tension(_make_long_text(), _make_session_plan(), sm, ollama, cerebras)
        # Should still retry with same text (no regeneration)
        assert result["passed"] is False

    def test_exception_marks_needs_review(self):
        ollama = MagicMock()
        ollama.check_tension.side_effect = Exception("Ollama down")
        sm = _make_state_manager()
        result = _check_tension(_make_long_text(), _make_session_plan(), sm, ollama, None)
        assert result["needs_review"] is True


class TestCheckAnchor:
    def test_session_01_auto_passes(self):
        ollama = MagicMock()
        sm = _make_state_manager()
        result = _check_anchor(_make_long_text(), _make_session_plan("01"), sm, ollama, None)
        assert result["passed"] is True
        assert "First session" in result["evidence"]
        ollama.check_anchor.assert_not_called()

    def test_passes_on_first_attempt(self):
        ollama = MagicMock()
        ollama.check_anchor.return_value = (True, "Anchor is good")
        sm = _make_state_manager({"01": {"content": "prev text " * 200, "concepts": [{"name": "C1"}]}})
        result = _check_anchor(_make_long_text(), _make_session_plan("02"), sm, ollama, None)
        assert result["passed"] is True

    def test_fails_and_retries_with_cerebras(self):
        ollama = MagicMock()
        ollama.check_anchor.side_effect = [
            (False, "Anchor references unknown term"),
            (True, "Fixed"),
        ]
        cerebras = MagicMock()
        cerebras.generate.return_value = "Fixed opening paragraph."
        sm = _make_state_manager({"01": {"content": "prev " * 200, "concepts": [{"name": "C1"}]}})
        # Text needs 2+ paragraphs for regeneration to work
        multi_para_text = "Old opening anchor paragraph.\n\nMiddle content.\n\nFinal tension?"
        result = _check_anchor(multi_para_text, _make_session_plan("02"), sm, ollama, cerebras)
        cerebras.generate.assert_called_once()
        assert result["passed"] is True

    def test_max_retries_marks_needs_review(self):
        ollama = MagicMock()
        ollama.check_anchor.return_value = (False, "Always fails")
        sm = _make_state_manager({"01": {"content": "prev " * 200, "concepts": []}})
        result = _check_anchor(_make_long_text(), _make_session_plan("02"), sm, ollama, None)
        assert result["passed"] is False
        assert result["needs_review"] is True
        assert result["retry_count"] == 2


class TestCheckCoherence:
    def test_passes_on_first_attempt(self):
        ollama = MagicMock()
        ollama.check_coherence.return_value = (True, "Coherent")
        result = _check_coherence(_make_long_text(), _make_session_plan(), ollama, None)
        assert result["passed"] is True
        assert result["retry_count"] == 0

    def test_fails_after_retries(self):
        ollama = MagicMock()
        ollama.check_coherence.return_value = (False, "Forward reference detected")
        result = _check_coherence(_make_long_text(), _make_session_plan(), ollama, None)
        assert result["passed"] is False
        assert result["needs_review"] is True

    def test_revisit_concept_extracted_from_plan(self):
        ollama = MagicMock()
        ollama.check_coherence.return_value = (True, "ok")
        plan = _make_session_plan(revisit={"name": "RevisitName", "reason": "reason"})
        _check_coherence(_make_long_text(), plan, ollama, None)
        call_args = ollama.check_coherence.call_args
        assert call_args[0][2] == "RevisitName"  # Third positional arg

    def test_no_revisit_passes_none(self):
        ollama = MagicMock()
        ollama.check_coherence.return_value = (True, "ok")
        _check_coherence(_make_long_text(), _make_session_plan(), ollama, None)
        call_args = ollama.check_coherence.call_args
        assert call_args[0][2] is None

    def test_exception_marks_needs_review(self):
        ollama = MagicMock()
        ollama.check_coherence.side_effect = Exception("Connection error")
        result = _check_coherence(_make_long_text(), _make_session_plan(), ollama, None)
        assert result["needs_review"] is True


class TestGetPreviousSessionData:
    def test_session_01_returns_empty(self):
        sm = _make_state_manager()
        data = _get_previous_session_data("01", sm)
        assert data["ending"] == ""
        assert data["concepts"] == []

    def test_extracts_ending_from_previous_session(self):
        prev_content = "First paragraph.\n\nSecond paragraph.\n\nFinal paragraph with tension."
        sm = _make_state_manager({
            "02": {
                "content": prev_content,
                "concepts": [{"name": "C1"}, {"name": "C2"}]
            }
        })
        data = _get_previous_session_data("03", sm)
        assert "Final paragraph" in data["ending"]

    def test_extracts_concept_names(self):
        sm = _make_state_manager({
            "04": {"content": "text", "concepts": [{"name": "Alpha"}, {"name": "Beta"}]}
        })
        data = _get_previous_session_data("05", sm)
        assert "Alpha" in data["concepts"]
        assert "Beta" in data["concepts"]

    def test_missing_previous_returns_empty(self):
        sm = _make_state_manager({})
        data = _get_previous_session_data("05", sm)
        assert data["ending"] == ""
        assert data["concepts"] == []


class TestRegenerateTension:
    def test_replaces_final_paragraph(self):
        text = "Para one.\n\nPara two.\n\nOld ending question?"
        cerebras = MagicMock()
        cerebras.generate.return_value = "New ending question?"
        plan = _make_session_plan()
        result = _regenerate_tension(text, plan, "Too vague", cerebras)
        assert "New ending question?" in result
        assert "Para one" in result

    def test_returns_original_on_failure(self):
        text = "Para one.\n\nPara two.\n\nOld ending."
        cerebras = MagicMock()
        cerebras.generate.side_effect = Exception("API error")
        plan = _make_session_plan()
        result = _regenerate_tension(text, plan, "Failed", cerebras)
        assert result == text

    def test_single_paragraph_returns_unchanged(self):
        text = "Only one paragraph here."
        cerebras = MagicMock()
        plan = _make_session_plan()
        result = _regenerate_tension(text, plan, "reason", cerebras)
        assert result == text

    def test_failure_reason_in_cerebras_prompt(self):
        text = "Para one.\n\nOld ending."
        cerebras = MagicMock()
        cerebras.generate.return_value = "New ending."
        plan = _make_session_plan()
        _regenerate_tension(text, plan, "UNIQUE_FAILURE_REASON_XYZ", cerebras)
        call_args = cerebras.generate.call_args
        user_prompt = call_args[0][1] if call_args[0] else ""
        assert "UNIQUE_FAILURE_REASON_XYZ" in user_prompt


class TestRegenerateAnchor:
    def test_replaces_opening_paragraph(self):
        text = "Old opening paragraph.\n\nMiddle content.\n\nFinal tension."
        cerebras = MagicMock()
        cerebras.generate.return_value = "New fixed opening."
        previous_context = {"ending": "Tension question?", "concepts": ["C1"]}
        result = _regenerate_anchor(text, previous_context, "Bad anchor", cerebras)
        assert "New fixed opening." in result
        assert "Middle content" in result

    def test_returns_original_on_failure(self):
        text = "Para one.\n\nRest."
        cerebras = MagicMock()
        cerebras.generate.side_effect = Exception("Error")
        result = _regenerate_anchor(text, {}, "reason", cerebras)
        assert result == text

    def test_single_paragraph_returns_unchanged(self):
        text = "Only one paragraph."
        cerebras = MagicMock()
        result = _regenerate_anchor(text, {}, "reason", cerebras)
        assert result == text


class TestFallbackEvaluation:
    def test_returns_all_four_checks(self):
        result = _fallback_evaluation("word " * 2000)
        assert "TENSION" in result
        assert "ANCHOR" in result
        assert "COHERENCE" in result
        assert "LENGTH" in result

    def test_tension_anchor_coherence_assumed_pass(self):
        result = _fallback_evaluation("word " * 2000)
        assert result["TENSION"]["passed"] is True
        assert result["ANCHOR"]["passed"] is True
        assert result["COHERENCE"]["passed"] is True

    def test_length_check_is_real(self):
        long_ok = _fallback_evaluation("word " * 2000)
        assert long_ok["LENGTH"]["passed"] is True

        too_short = _fallback_evaluation("word " * 100)
        assert too_short["LENGTH"]["passed"] is False
