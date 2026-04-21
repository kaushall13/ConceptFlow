"""
Tests for pipeline/pass1.py - perform_pass1 and _adjust_concept_volume
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.pass1 import perform_pass1, _adjust_concept_volume


def _make_pass1_result(concept_count=80):
    return {
        "concepts": [
            {"name": f"Concept{i}", "description": f"Desc {i}", "location": "S1", "recurring": False}
            for i in range(concept_count)
        ],
        "explicit_dependencies": [
            {"signal": "recall that attention", "location": "S2", "refers_to": "Concept0"}
        ],
        "implicit_assumptions": [
            {"term": "GPU", "context": "used without definition", "location": "S1"}
        ],
        "author_anchors": [
            {"anchor": "Tiling avoids HBM materialisation", "location": "S3"}
        ],
    }


class TestPerformPass1:
    def test_happy_path_returns_structured_result(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = _make_pass1_result(100)
        result = perform_pass1("clean book text", cerebras)
        assert "concepts" in result
        assert "explicit_dependencies" in result
        assert "implicit_assumptions" in result
        assert "author_anchors" in result

    def test_calls_generate_json_once_in_normal_range(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = _make_pass1_result(100)
        perform_pass1("text", cerebras)
        assert cerebras.generate_json.call_count == 1

    def test_full_text_passed_to_api(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = _make_pass1_result(100)
        perform_pass1("UNIQUE_BOOK_CONTENT_XYZ", cerebras)
        call_args = cerebras.generate_json.call_args
        user_prompt = call_args[0][1] if call_args[0] else call_args[1].get("user_prompt", "")
        assert "UNIQUE_BOOK_CONTENT_XYZ" in user_prompt

    def test_raises_when_missing_required_keys(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = {"concepts": []}  # Missing other keys
        with pytest.raises(ValueError, match="missing required keys"):
            perform_pass1("text", cerebras)

    def test_too_few_concepts_triggers_split(self, capsys):
        cerebras = MagicMock()
        few_concepts_result = _make_pass1_result(20)  # < 40
        adjusted_result = _make_pass1_result(80)
        cerebras.generate_json.side_effect = [few_concepts_result, adjusted_result]
        result = perform_pass1("text", cerebras)
        assert cerebras.generate_json.call_count == 2
        assert len(result["concepts"]) == 80

    def test_too_many_concepts_triggers_merge(self, capsys):
        cerebras = MagicMock()
        many_concepts_result = _make_pass1_result(300)  # > 250
        adjusted_result = _make_pass1_result(120)
        cerebras.generate_json.side_effect = [many_concepts_result, adjusted_result]
        result = perform_pass1("text", cerebras)
        assert cerebras.generate_json.call_count == 2
        assert len(result["concepts"]) == 120

    def test_exact_40_concepts_is_not_too_few(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = _make_pass1_result(40)
        perform_pass1("text", cerebras)
        assert cerebras.generate_json.call_count == 1

    def test_exact_250_concepts_is_not_too_many(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = _make_pass1_result(250)
        perform_pass1("text", cerebras)
        assert cerebras.generate_json.call_count == 1

    def test_prints_concept_count(self, capsys):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = _make_pass1_result(75)
        perform_pass1("text", cerebras)
        captured = capsys.readouterr()
        assert "75" in captured.out


class TestAdjustConceptVolume:
    def test_split_calls_generate_json(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = _make_pass1_result(80)
        result = _adjust_concept_volume(_make_pass1_result(20), "split", cerebras)
        cerebras.generate_json.assert_called_once()

    def test_merge_calls_generate_json(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = _make_pass1_result(100)
        result = _adjust_concept_volume(_make_pass1_result(300), "merge", cerebras)
        cerebras.generate_json.assert_called_once()

    def test_split_instruction_appears_in_prompt(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = _make_pass1_result(80)
        _adjust_concept_volume(_make_pass1_result(20), "split", cerebras)
        call_args = cerebras.generate_json.call_args
        user_prompt = call_args[0][1] if call_args[0] else ""
        assert "split" in user_prompt.lower() or "atomic" in user_prompt.lower()

    def test_merge_instruction_appears_in_prompt(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = _make_pass1_result(100)
        _adjust_concept_volume(_make_pass1_result(300), "merge", cerebras)
        call_args = cerebras.generate_json.call_args
        user_prompt = call_args[0][1] if call_args[0] else ""
        assert "merge" in user_prompt.lower()

    def test_raises_if_adjusted_result_missing_keys(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = {"concepts": []}  # Missing keys
        with pytest.raises(ValueError):
            _adjust_concept_volume(_make_pass1_result(20), "split", cerebras)

    def test_returns_adjusted_result(self):
        cerebras = MagicMock()
        expected = _make_pass1_result(90)
        cerebras.generate_json.return_value = expected
        result = _adjust_concept_volume(_make_pass1_result(20), "split", cerebras)
        assert len(result["concepts"]) == 90
