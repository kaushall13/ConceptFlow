"""
Tests for pipeline/pass2.py - perform_pass2 and _extract_cluster_concepts
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.pass2 import perform_pass2, _extract_cluster_concepts


def _make_all_concepts(n=10):
    return [
        {"name": f"Concept{i}", "description": f"Desc {i}"}
        for i in range(n)
    ]


def _make_cluster(name, concept_names):
    return {"name": name, "concepts": concept_names, "description": f"Desc of {name}"}


def _make_extraction_result(concept_names):
    return {
        "concepts": [
            {
                "original_name": name,
                "canonical_name": name,
                "description": f"2-4 sentence description of {name}",
                "primary_passage": f"Passage for {name}",
                "secondary_passages": [],
                "dependency_signals": [],
                "implicit_prerequisites": [],
                "author_anchor": "",
                "enrichment_flag": False,
                "concept_weight": "medium",
                "cross_theme_deps": [],
            }
            for name in concept_names
        ]
    }


class TestPerformPass2:
    def test_happy_path_returns_dict_keyed_by_cluster(self):
        cerebras = MagicMock()
        clusters = [
            _make_cluster("Memory", ["ConceptA", "ConceptB"]),
            _make_cluster("Inference", ["ConceptC"]),
        ]
        cerebras.generate_json.side_effect = [
            _make_extraction_result(["ConceptA", "ConceptB"]),
            _make_extraction_result(["ConceptC"]),
        ]
        result = perform_pass2("book text", _make_all_concepts(), clusters, cerebras)
        assert "Memory" in result
        assert "Inference" in result

    def test_calls_generate_json_once_per_cluster(self):
        cerebras = MagicMock()
        clusters = [
            _make_cluster("C1", ["Concept0"]),
            _make_cluster("C2", ["Concept1"]),
            _make_cluster("C3", ["Concept2"]),
        ]
        cerebras.generate_json.side_effect = [
            _make_extraction_result(["Concept0"]),
            _make_extraction_result(["Concept1"]),
            _make_extraction_result(["Concept2"]),
        ]
        perform_pass2("text", _make_all_concepts(), clusters, cerebras)
        assert cerebras.generate_json.call_count == 3

    def test_full_book_text_included_in_each_call(self):
        cerebras = MagicMock()
        clusters = [_make_cluster("C1", ["ConceptA"])]
        cerebras.generate_json.return_value = _make_extraction_result(["ConceptA"])
        perform_pass2("UNIQUE_BOOK_TEXT_ABC", _make_all_concepts(), clusters, cerebras)
        call_args = cerebras.generate_json.call_args
        user_prompt = call_args[0][1]
        assert "UNIQUE_BOOK_TEXT_ABC" in user_prompt

    def test_global_concepts_included_in_prompt(self):
        cerebras = MagicMock()
        clusters = [_make_cluster("C1", ["Concept0"])]
        cerebras.generate_json.return_value = _make_extraction_result(["Concept0"])
        all_concepts = [{"name": "GlobalMarkerConcept", "description": "test"}]
        perform_pass2("text", all_concepts, clusters, cerebras)
        call_args = cerebras.generate_json.call_args
        user_prompt = call_args[0][1]
        assert "GlobalMarkerConcept" in user_prompt

    def test_cluster_extraction_failure_stored_with_error(self):
        cerebras = MagicMock()
        clusters = [
            _make_cluster("Good", ["ConceptA"]),
            _make_cluster("Bad", ["ConceptB"]),
        ]
        cerebras.generate_json.side_effect = [
            _make_extraction_result(["ConceptA"]),
            Exception("API timeout"),
        ]
        result = perform_pass2("text", _make_all_concepts(), clusters, cerebras)
        assert "Bad" in result
        assert result["Bad"]["error"] == "API timeout"
        assert result["Bad"]["concepts"] == []

    def test_empty_clusters_returns_empty_dict(self):
        cerebras = MagicMock()
        result = perform_pass2("text", _make_all_concepts(), [], cerebras)
        assert result == {}

    def test_result_has_concepts_list(self):
        cerebras = MagicMock()
        clusters = [_make_cluster("Memory", ["ConceptA", "ConceptB"])]
        cerebras.generate_json.return_value = _make_extraction_result(["ConceptA", "ConceptB"])
        result = perform_pass2("text", _make_all_concepts(), clusters, cerebras)
        assert isinstance(result["Memory"]["concepts"], list)
        assert len(result["Memory"]["concepts"]) == 2


class TestExtractClusterConcepts:
    def test_happy_path_returns_concepts(self):
        cerebras = MagicMock()
        cluster = _make_cluster("Memory", ["ConceptA", "ConceptB"])
        cerebras.generate_json.return_value = _make_extraction_result(["ConceptA", "ConceptB"])
        result = _extract_cluster_concepts("text", "global summary", cluster, cerebras)
        assert "concepts" in result
        assert len(result["concepts"]) == 2

    def test_raises_when_missing_concepts_key(self):
        cerebras = MagicMock()
        cluster = _make_cluster("Memory", ["ConceptA"])
        cerebras.generate_json.return_value = {"other_key": "data"}
        with pytest.raises(ValueError, match="missing 'concepts' key"):
            _extract_cluster_concepts("text", "global", cluster, cerebras)

    def test_adds_placeholder_for_missing_concepts(self):
        cerebras = MagicMock()
        cluster = _make_cluster("Memory", ["ConceptA", "ConceptB", "ConceptC"])
        # Only extracts ConceptA and ConceptB — ConceptC is missing
        cerebras.generate_json.return_value = _make_extraction_result(["ConceptA", "ConceptB"])
        result = _extract_cluster_concepts("text", "global", cluster, cerebras)
        extracted_names = {c["original_name"] for c in result["concepts"]}
        assert "ConceptC" in extracted_names

    def test_placeholder_has_required_fields(self):
        cerebras = MagicMock()
        cluster = _make_cluster("Memory", ["ConceptA", "Missing"])
        cerebras.generate_json.return_value = _make_extraction_result(["ConceptA"])
        result = _extract_cluster_concepts("text", "global", cluster, cerebras)
        missing = next(c for c in result["concepts"] if c["original_name"] == "Missing")
        assert "canonical_name" in missing
        assert "description" in missing
        assert missing["enrichment_flag"] is False
        assert missing["concept_weight"] == "medium"

    def test_cluster_description_included_in_prompt(self):
        cerebras = MagicMock()
        cluster = {"name": "Memory", "concepts": ["ConceptA"], "description": "UNIQUE_CLUSTER_DESC_XYZ"}
        cerebras.generate_json.return_value = _make_extraction_result(["ConceptA"])
        _extract_cluster_concepts("text", "global", cluster, cerebras)
        call_args = cerebras.generate_json.call_args
        user_prompt = call_args[0][1]
        assert "UNIQUE_CLUSTER_DESC_XYZ" in user_prompt

    def test_cluster_concept_names_in_prompt(self):
        cerebras = MagicMock()
        cluster = _make_cluster("Memory", ["MARKER_CONCEPT_XYZ"])
        cerebras.generate_json.return_value = _make_extraction_result(["MARKER_CONCEPT_XYZ"])
        _extract_cluster_concepts("text", "global", cluster, cerebras)
        call_args = cerebras.generate_json.call_args
        user_prompt = call_args[0][1]
        assert "MARKER_CONCEPT_XYZ" in user_prompt
